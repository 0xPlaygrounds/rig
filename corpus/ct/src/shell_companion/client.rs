//! `rig shell-ask`, the snippet's client: it reads the shell's last
//! command from the snippet's state directory, hands it to the session
//! that serves the project, else to a one-shot session of its own, and
//! prints what comes back. A suggested command goes in `DIR/suggested`,
//! for the snippet to put in the edit buffer.

use std::io::{self, BufRead, BufReader, Write};
use std::os::unix::net::UnixStream;
use std::path::{Path, PathBuf};

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_basics::print::{AppHeadlessExt, KeepRunning};
use rig_harness::prelude::*;

use crate::shell_companion::{Event, OneShot, Request, Server, ShellAsk, Sink, queue, socket_path};

/// The file the suggested command is written to, in the state directory.
pub const SUGGESTED: &str = "suggested";

/// What the shell's words say: the state directory, whether a command is
/// asked for, and the question.
#[derive(Debug, Default, PartialEq, Eq)]
pub struct Words {
    /// The snippet's state directory.
    pub state: Option<PathBuf>,
    /// `--cmd`.
    pub cmd: bool,
    /// The rest.
    pub question: String,
}

/// Reads `--state DIR [--cmd] [question…]`.
pub fn parse(words: &[String]) -> Words {
    let mut parsed = Words::default();
    let mut rest = Vec::new();
    let mut words = words.iter();
    while let Some(word) = words.next() {
        match word.as_str() {
            "--state" if rest.is_empty() => parsed.state = words.next().map(PathBuf::from),
            "--cmd" if rest.is_empty() => parsed.cmd = true,
            _ => rest.push(word.as_str()),
        }
    }
    parsed.question = rest.join(" ");
    parsed
}

/// The request for `words`, from the state the snippet recorded in
/// `state`: `command`, `exit`, `cwd` and `output`.
pub fn read_ask(state: &Path, words: &Words) -> ShellAsk {
    let read = |name: &str| std::fs::read_to_string(state.join(name)).unwrap_or_default();
    ShellAsk {
        question: words.question.clone(),
        cmd: words.cmd,
        command: read("command").trim_end().to_owned(),
        exit: read("exit").trim().parse().ok(),
        cwd: read("cwd").trim_end().to_owned(),
        tail: read("output"),
    }
}

/// The socket of the session serving `cwd` or the nearest directory above
/// it, if one listens.
pub fn find_socket(home: &Path, cwd: &Path) -> Option<PathBuf> {
    cwd.ancestors()
        .map(|dir| socket_path(home, dir))
        .find(|path| UnixStream::connect(path).is_ok())
}

/// Prints `event` as the shell shows it.
///
/// # Errors
///
/// When `out` cannot be written.
pub fn render(event: &Event, out: &mut impl Write) -> io::Result<()> {
    match event {
        Event::Queued { ahead: 0 } | Event::Sent => Ok(()),
        Event::Queued { ahead } => writeln!(out, "(rig is busy: {ahead} ahead of this)"),
        Event::Text { text } => write!(out, "{text}"),
        Event::Tool { line } => writeln!(out, "\n  > {line}"),
        Event::Approval => writeln!(out, "\n  waiting for approval in rig"),
        Event::Done { command, .. } => {
            writeln!(out)?;
            match command {
                Some(command) => writeln!(out, "(in your edit buffer: {command})"),
                None => Ok(()),
            }
        }
        Event::Failed { why } => writeln!(out, "\nrig: {why}"),
    }?;
    out.flush()
}

/// Asks the session listening on `socket`, printing to `out` what comes
/// back; the suggested command, if any.
///
/// # Errors
///
/// When the connection fails, or the turn failed.
pub fn ask(socket: &Path, ask: &ShellAsk, out: &mut impl Write) -> io::Result<Option<String>> {
    let mut stream = UnixStream::connect(socket)?;
    writeln!(
        stream,
        "{}",
        serde_json::to_string(ask).map_err(io::Error::other)?
    )?;
    for line in BufReader::new(stream).lines() {
        let event: Event = serde_json::from_str(&line?).map_err(io::Error::other)?;
        render(&event, out)?;
        match event {
            Event::Done { command, .. } => return Ok(command),
            Event::Failed { why } => return Err(io::Error::other(why)),
            _ => {}
        }
    }
    Err(io::Error::other("the session went away"))
}

/// Writes `command` where the snippet reads it.
pub fn suggest(state: &Path, command: Option<&str>) {
    if let Some(command) = command {
        std::fs::write(state.join(SUGGESTED), command).ok();
    }
}

/// How a relayed request ended: the suggested command, or why it failed.
struct Relayed(Result<Option<String>, String>);

/// Sets up a `rig shell-ask` run.
pub(crate) fn build(app: &mut App, words: &[String]) {
    let words = parse(words);
    let Some(state) = words.state.clone() else {
        app.add_systems(Startup, |mut commands: Commands| {
            eprintln!("rig shell-ask: --state DIR is required (the snippet passes it)");
            commands.trigger(Exit(ExitReason::Failed(2)));
        });
        return;
    };
    let ask = read_ask(&state, &words);
    let home = RigHome::of(app.world());
    let cwd = PathBuf::from(&ask.cwd);
    match find_socket(home.root(), &cwd) {
        Some(socket) => relay(app, socket, ask, state),
        None => {
            eprintln!("rig: no session serves this project; asking a one-shot session");
            // Spawned now, not in `Startup`: the headless run decides in its
            // first frame whether anything keeps it running.
            app.run_headless(None);
            app.world_mut().spawn((OneShot, KeepRunning));
            let mut server = app.world_mut().resource_mut::<Server>();
            queue(
                &mut server,
                Request {
                    ask,
                    sink: Sink::Local(state),
                },
            );
        }
    }
}

/// Relays `ask` to the session on `socket` off the main thread, then
/// exits.
fn relay(app: &mut App, socket: PathBuf, ask: ShellAsk, state: PathBuf) {
    app.add_systems(Startup, move |wake: Res<Wake>, mut commands: Commands| {
        let (sender, relayed) = async_channel::bounded(1);
        let (socket, ask) = (socket.clone(), ask.clone());
        let spawned = std::thread::Builder::new()
            .name("rig-shell-ask".to_owned())
            .spawn(move || {
                let result = self::ask(&socket, &ask, &mut io::stdout());
                sender
                    .send_blocking(Relayed(result.map_err(|why| why.to_string())))
                    .ok();
            });
        if spawned.is_err() {
            commands.trigger(Exit(ExitReason::Failed(1)));
            return;
        }
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        commands.spawn(Running::stream(pool, &wake, relayed));
    })
    .add_observer(
        move |inserted: On<Insert<Done<Relayed>>>,
              relayed: Query<&Done<Relayed>>,
              mut commands: Commands| {
            let Ok(Done(Relayed(result))) = relayed.get(inserted.entity) else {
                return;
            };
            let exit = match result {
                Ok(command) => {
                    suggest(&state, command.as_deref());
                    ExitReason::Quit
                }
                Err(why) => {
                    eprintln!("rig: {why}");
                    ExitReason::Failed(1)
                }
            };
            commands.trigger(Exit(exit));
        },
    );
}
