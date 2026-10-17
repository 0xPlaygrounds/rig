//! Plugin corpus (held-out): H22: Terminal title and attention bell (spec `H22-terminal-title.md`).
//!
//! Keeps the terminal's title (OSC 2) at `rig · <session> · <state>`:
//! `idle`, `working` with a spinner that turns at most twice a second,
//! `needs you` while a picker or question is open, or `error` after a
//! failed turn. Inside tmux it also renames the window, and gives it its
//! name back on exit. It rings the bell when a long turn ends or a picker
//! opens while the terminal is not known to have the user's focus. The
//! terminal's title is pushed on its title stack at the first write and
//! popped as the app exits, whatever asked it to (the first [`ExitRequest`]
//! effect settling is the exit).
//!
//! The escapes go through the terminal view's writer ([`TerminalWrite`]);
//! a `--print` run writes them to stdout when it is a terminal, and no
//! other run writes any. The `[title]` table of the settings:
//! `enabled = true`, `bell_seconds = 10`.

use std::ffi::OsString;
use std::io::{IsTerminal, Write};
use std::process::Stdio;
use std::sync::mpsc::{Sender, channel};
use std::sync::{Arc, Mutex};
use std::thread::JoinHandle;

use bevy_ecs::system::SystemParam;
use rig_basics::sessions::SessionTitle;
use rig_harness::prelude::*;
use rig_tui::{TerminalFocus, TerminalWrite};
use serde::Deserialize;

/// How long the spinner shows each of its characters.
const SPIN: Duration = Duration::from_millis(500);

/// The spinner's characters.
const SPINNER: [char; 4] = ['⠋', '⠙', '⠹', '⠸'];

/// Saves the terminal's title on its title stack (xterm's `CSI 22 ; 0 t`).
const PUSH_TITLE: &[u8] = b"\x1b[22;0t";

/// Restores the title saved last (`CSI 23 ; 0 t`).
const POP_TITLE: &[u8] = b"\x1b[23;0t";

/// The bell.
const BELL: &[u8] = b"\x07";

/// The `[title]` settings.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct TitleSettings {
    /// Whether the title, the window name and the bell are kept at all.
    pub enabled: bool,
    /// The shortest turn, in seconds, whose end rings the bell.
    pub bell_seconds: u64,
}

impl Default for TitleSettings {
    fn default() -> Self {
        Self {
            enabled: true,
            bell_seconds: 10,
        }
    }
}

/// Where the escapes go. Inserted before the app is finished, it replaces
/// the one the run's front implies.
#[derive(Resource, Clone, Debug)]
pub enum TitleOutput {
    /// The terminal view's own writer, between frames.
    Terminal,
    /// Stdout, in a `--print` run on a terminal.
    Stdout,
    /// A buffer, for a test.
    Buffer(Arc<Mutex<Vec<u8>>>),
}

/// Writes escapes to the [`TitleOutput`].
#[derive(SystemParam)]
struct Out<'w> {
    output: Res<'w, TitleOutput>,
    terminal: MessageWriter<'w, TerminalWrite>,
    focus: Option<Res<'w, TerminalFocus>>,
}

impl Out<'_> {
    fn write(&mut self, bytes: &[u8]) {
        match &*self.output {
            TitleOutput::Terminal => {
                self.terminal.write(TerminalWrite(bytes.to_vec()));
            }
            TitleOutput::Stdout => {
                let mut stdout = std::io::stdout().lock();
                stdout.write_all(bytes).and_then(|()| stdout.flush()).ok();
            }
            TitleOutput::Buffer(buffer) => {
                if let Ok(mut buffer) = buffer.lock() {
                    buffer.extend_from_slice(bytes);
                }
            }
        }
    }

    /// Rings the bell unless the user is known to look at the terminal.
    fn bell(&mut self) {
        if self.focus.as_deref() != Some(&TerminalFocus(Some(true))) {
            self.write(BELL);
        }
    }
}

/// What the title shows.
#[derive(Resource, Default)]
struct Title {
    /// The title written last; `None` before the first, which pushes the
    /// terminal's own.
    written: Option<String>,
    /// When the agents started working, on the app's clock.
    working_since: Option<Duration>,
    /// Whether the last top-level turn failed.
    failed: bool,
    /// The app exits: the title was given back, and nothing more is
    /// written.
    restored: bool,
}

/// When the turn on this entity started, on the app's clock.
#[derive(Component)]
struct TurnClock(Duration);

/// Keeps the terminal's title and rings its bell.
#[derive(Default)]
pub struct TitlePlugin;

impl Plugin for TitlePlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<TitleSettings>("title")
            .add_message::<TerminalWrite>();
    }

    fn finish(&self, app: &mut App) {
        let view = app.runs_front(rig_tui::FRONT);
        let world = app.world();
        if !world
            .get_resource::<TitleSettings>()
            .is_some_and(|settings| settings.enabled)
        {
            return;
        }
        let print = world
            .get_resource::<Invoked>()
            .is_some_and(|invoked| invoked.args.print.is_some());
        let output = match world.get_resource::<TitleOutput>() {
            Some(output) => output.clone(),
            None if view => TitleOutput::Terminal,
            None if print && std::io::stdout().is_terminal() => TitleOutput::Stdout,
            None => return,
        };
        app.insert_resource(output)
            .init_resource::<Title>()
            .init_resource::<Processes>()
            .add_systems(Startup, start_tmux)
            .add_systems(PostUpdate, show_title)
            .add_observer(restore)
            .add_observer(turn_started)
            .add_observer(turn_ended)
            .add_observer(picker_opened)
            .add_observer(asked);
    }
}

fn turn_started(turn: On<Add<TurnOf>>, time: Res<Time<Real>>, mut commands: Commands) {
    // Frames for the spinner while the turn runs.
    commands
        .entity(turn.entity)
        .try_insert((TurnClock(time.elapsed()), KeepAwake(SPIN)));
}

/// A top-level agent's turn ended: the title says whether it failed, and
/// a long one rings the bell.
fn turn_ended(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent, &TurnClock)>,
    subagents: Query<(), With<SpawnedBy>>,
    time: Res<Time<Real>>,
    settings: Res<TitleSettings>,
    mut title: ResMut<Title>,
    mut out: Out,
) {
    let Ok((outcome, &ForAgent(agent), TurnClock(since))) = turns.get(settled.entity) else {
        return;
    };
    if subagents.contains(agent) {
        return;
    }
    title.failed = matches!(outcome, TurnOutcome::Failed(_));
    let took = time.elapsed().saturating_sub(*since);
    if took >= Duration::from_secs(settings.bell_seconds) {
        out.bell();
    }
}

fn picker_opened(_: On<Add<Picker>>, mut out: Out) {
    out.bell();
}

fn asked(_: On<Add<Ask>>, mut out: Out) {
    out.bell();
}

/// Writes the title when what it says changed.
fn show_title(
    turns: Query<(), (With<TurnOf>, Without<Settled>)>,
    pickers: Query<(), With<Picker>>,
    asks: Query<(), (With<Ask>, Without<Outcome>)>,
    session: Option<Res<SessionTitle>>,
    time: Res<Time<Real>>,
    mut title: ResMut<Title>,
    tmux: Option<ResMut<Tmux>>,
    mut out: Out,
) {
    if title.restored {
        return;
    }
    let working = !turns.is_empty();
    title.working_since = match (working, title.working_since) {
        (true, None) => Some(time.elapsed()),
        (true, since) => since,
        (false, _) => None,
    };
    let state = if !pickers.is_empty() || !asks.is_empty() {
        "needs you"
    } else if working {
        "working"
    } else if title.failed {
        "error"
    } else {
        "idle"
    };
    let mut shown = state.to_owned();
    if let Some(since) = title.working_since.filter(|_| state == "working") {
        let turned = time.elapsed().saturating_sub(since).as_millis() / SPIN.as_millis();
        let index = usize::try_from(turned).unwrap_or_default() % SPINNER.len();
        if let Some(spinner) = SPINNER.get(index) {
            shown = format!("{state} {spinner}");
        }
    }
    let name = session
        .map(|session| {
            session
                .name
                .clone()
                .unwrap_or_else(|| session.typed.clone())
        })
        .unwrap_or_default();
    let name: String = name.chars().filter(|c| !c.is_control()).collect();
    let text = match name.trim() {
        "" => format!("rig · {shown}"),
        name => format!("rig · {name} · {shown}"),
    };
    if let Some(mut tmux) = tmux {
        tmux.rename(format!("rig · {state}"));
    }
    if title.written.as_deref() == Some(text.as_str()) {
        return;
    }
    if title.written.is_none() {
        out.write(PUSH_TITLE);
    }
    out.write(format!("\x1b]2;{text}\x1b\\").as_bytes());
    title.written = Some(text);
}

/// Gives the terminal its title back, and tmux its window's name, once
/// the app exits: the first exit effect settled, which nothing undoes.
fn restore(
    settled: On<Add<Settled>>,
    exits: Query<(), With<ExitRequest>>,
    mut title: ResMut<Title>,
    tmux: Option<ResMut<Tmux>>,
    mut out: Out,
) {
    if title.restored || !exits.contains(settled.entity) {
        return;
    }
    title.restored = true;
    if title.written.is_some() {
        out.write(POP_TITLE);
    }
    if let Some(mut tmux) = tmux {
        tmux.finish();
    }
}

/// The tmux window rig runs in: its commands run in order on a thread of
/// their own.
#[derive(Resource)]
struct Tmux {
    /// `-t <pane>`, when tmux named the pane.
    target: Vec<OsString>,
    jobs: Option<Sender<Vec<OsString>>>,
    worker: Option<JoinHandle<()>>,
    /// What gives the window its own name back.
    restore: Vec<OsString>,
    /// The name it was given last.
    shown: String,
}

impl Tmux {
    fn run(&self, args: Vec<OsString>) {
        if let Some(jobs) = &self.jobs {
            jobs.send(args).ok();
        }
    }

    fn rename(&mut self, name: String) {
        if self.shown == name {
            return;
        }
        let mut args = vec![OsString::from("rename-window")];
        args.extend(self.target.iter().cloned());
        args.push(OsString::from(&name));
        self.run(args);
        self.shown = name;
    }

    /// Gives the window its name back and waits until tmux did.
    fn finish(&mut self) {
        let restore = std::mem::take(&mut self.restore);
        self.run(restore);
        self.jobs = None;
        if let Some(worker) = self.worker.take() {
            worker.join().ok();
        }
    }
}

/// Inside tmux (`$TMUX`), reads the window's name and whether tmux names
/// it itself, so it can be given back, and starts the thread that runs
/// tmux.
fn start_tmux(processes: Res<Processes>, mut commands: Commands) {
    if std::env::var_os("TMUX").is_none() {
        return;
    }
    let target: Vec<OsString> = match std::env::var_os("TMUX_PANE") {
        Some(pane) => vec!["-t".into(), pane],
        None => Vec::new(),
    };
    let launcher = processes.launcher();
    let asked = launcher
        .command("tmux")
        .arg("display-message")
        .arg("-p")
        .args(&target)
        .arg("#{window_name}\t#{automatic-rename}")
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .output();
    let Ok(asked) = asked.map(|output| String::from_utf8_lossy(&output.stdout).into_owned()) else {
        return;
    };
    let Some((name, automatic)) = asked.trim_end_matches('\n').split_once('\t') else {
        return;
    };
    let restore: Vec<OsString> = match automatic {
        "1" | "on" => ["set-option", "-w"]
            .into_iter()
            .map(OsString::from)
            .chain(target.iter().cloned())
            .chain(["automatic-rename", "on"].map(OsString::from))
            .collect(),
        _ => std::iter::once(OsString::from("rename-window"))
            .chain(target.iter().cloned())
            .chain(std::iter::once(OsString::from(name)))
            .collect(),
    };
    let (jobs, received) = channel::<Vec<OsString>>();
    let worker = std::thread::spawn(move || {
        for args in received {
            launcher
                .command("tmux")
                .args(args)
                .stdin(Stdio::null())
                .stdout(Stdio::null())
                .stderr(Stdio::null())
                .status()
                .ok();
        }
    });
    commands.insert_resource(Tmux {
        target,
        jobs: Some(jobs),
        worker: Some(worker),
        restore,
        shown: name.to_owned(),
    });
}

#[cfg(test)]
mod tests;
