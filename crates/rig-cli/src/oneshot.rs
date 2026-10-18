//! `rig -p`: one shot. It starts a session (or sends to `-r ID`), waits for
//! the answer and prints it on stdout, nothing else; it exits as `rig wait`
//! does. The prompt comes from the words, or from stdin when there are none
//! or it is `-`; with a prompt given, stdin is never read.
//!
//! With `--json` (or progress on a chatty stderr) a second connection
//! follows the session's events while the first waits; the run ends once
//! the events through the wait's `seq` are out.

use std::sync::mpsc;
use std::time::Duration;

use rig_client::Client;
use rig_protocol::methods::{self, Unload, WaitResult};
use rig_protocol::{CommandSeq, EventRecord, Seq, SessionId};

use crate::args::Args;
use crate::out::Failure;
use crate::verbs::{self, Ctx};

enum Msg {
    Event(EventRecord),
    Done(Result<WaitResult, Failure>),
}

pub fn run(args: &Args) -> Result<u8, Failure> {
    let ctx = Ctx::new(args)?;
    let timeout = args.duration("timeout", "-p")?.unwrap_or(None);
    let stdin_is_terminal = std::io::IsTerminal::is_terminal(&std::io::stdin());
    let reads = !stdin_is_terminal || args.words.first().is_some_and(|word| word == "-");
    let prompt = verbs::prompt(&args.words, reads, verbs::read_stdin)
        .map_err(|error| Failure::usage(format!("could not read stdin: {error}"), "-p"))?
        .filter(|prompt| !prompt.trim().is_empty())
        .ok_or_else(|| {
            Failure::usage(
                "rig -p needs a prompt: rig -p \"...\", or `-` to read stdin",
                "-p",
            )
        })?;
    let client = ctx.connect()?;
    let model = args.value("model");
    let (id, command) = match args.value("resume") {
        Some(session) if model.is_some() => {
            let id = verbs::resolve(&client, session);
            let command =
                verbs::choose_model(&client, &id, model.unwrap_or_default(), Some(prompt), &ctx)?;
            (id, command)
        }
        Some(session) => {
            let id = verbs::resolve(&client, session);
            let sent = client
                .call::<methods::Send>(&methods::Send {
                    session_id: id.clone(),
                    text: Some(prompt),
                    content: None,
                    mode: None,
                    key: verbs::key(args),
                })
                .map_err(|error| {
                    Failure::client(error, format!("rig agents --all{}", ctx.host_flag()))
                })?;
            (id, Some(sent.command_seq))
        }
        None => {
            let started = client
                .call::<methods::Start>(&methods::Start {
                    cwd: verbs::cwd(args, "-p")?,
                    prompt: if model.is_some() {
                        None
                    } else {
                        Some(prompt.clone())
                    },
                    name: args.value("name").map(str::to_owned),
                    host: None,
                    build: verbs::build(args, "-p")?,
                    grant: verbs::grant(args, "-p")?,
                    policy: None,
                    budget: verbs::budget(args, "-p")?,
                    context: None,
                    unload: Some(Unload::Turn),
                    view: false,
                    key: verbs::key(args),
                })
                .map_err(|error| Failure::client(error, "rig help -p"))?;
            match model {
                Some(model) => {
                    let command = verbs::choose_model(
                        &client,
                        &started.session_id,
                        model,
                        Some(prompt),
                        &ctx,
                    )?;
                    (started.session_id, command)
                }
                None => (started.session_id, started.command_seq),
            }
        }
    };
    stop_on_interrupt(&ctx, &id);
    let progress = !ctx.json && rig_harness_protocol::chatty();
    if !ctx.json && !progress {
        let result = verbs::call_wait(&client, &id, command, timeout, &ctx)?;
        return Ok(verbs::print_wait(&ctx, &id, &result, timeout, "-p"));
    }
    let result = follow(&ctx, &id, command, timeout, |record| {
        if ctx.json {
            verbs::print_event(&ctx, record);
        } else if let Some(line) = progress_line(record) {
            eprintln!("{line}");
        }
    })?;
    Ok(verbs::print_wait(&ctx, &id, &result, timeout, "-p"))
}

/// Waits for command `command` of `id` on one connection while another
/// passes the session's events to `each`, through the wait's `seq`.
fn follow(
    ctx: &Ctx,
    id: &SessionId,
    command: Option<CommandSeq>,
    timeout: Option<Duration>,
    mut each: impl FnMut(&EventRecord),
) -> Result<WaitResult, Failure> {
    let (sent, got) = mpsc::channel();
    {
        let sent = sent.clone();
        let target = ctx.target.clone();
        let id = id.clone();
        std::thread::spawn(move || {
            let events = Client::connect(&target, "rig-cli").and_then(|client| {
                let events = client.subscribe(&id, Seq(0))?;
                Ok((client, events))
            });
            let Ok((_client, events)) = events else {
                return;
            };
            for record in events.flatten() {
                if sent.send(Msg::Event(record)).is_err() {
                    return;
                }
            }
        });
    }
    {
        let target = ctx.target.clone();
        let id = id.clone();
        let json = ctx.json;
        std::thread::spawn(move || {
            let ctx = Ctx { target, json };
            let result = ctx
                .connect()
                .and_then(|client| verbs::call_wait(&client, &id, command, timeout, &ctx));
            sent.send(Msg::Done(result)).ok();
        });
    }
    let mut last = Seq(0);
    let mut done: Option<WaitResult> = None;
    loop {
        let message = match &done {
            // The wait is over: the events through its seq are on their
            // way; a stream that stalls longer than this has ended.
            Some(_) => got.recv_timeout(Duration::from_secs(2)).ok(),
            None => got.recv().ok(),
        };
        match message {
            Some(Msg::Event(record)) => {
                if record.seq > last {
                    last = record.seq;
                    each(&record);
                }
            }
            Some(Msg::Done(result)) => done = Some(result?),
            None => break,
        }
        if let Some(result) = &done
            && last >= result.seq
        {
            break;
        }
    }
    done.ok_or_else(|| {
        Failure::new(
            rig_protocol::Exit::Unavailable,
            "lost the rig server while waiting",
            format!("rig wait {}", id.0),
        )
    })
}

/// What a chatty stderr shows of the run: each tool call as it starts.
fn progress_line(record: &EventRecord) -> Option<String> {
    (record.kind == "call.started")
        .then(|| verbs::transcript(record))
        .flatten()
}

/// On Ctrl+C or SIGTERM, stops `id`'s turn and exits 130: a one-shot run
/// leaves no turn behind.
#[cfg(unix)]
fn stop_on_interrupt(ctx: &Ctx, id: &SessionId) {
    use signal_hook::consts::{SIGINT, SIGTERM};
    let Ok(mut signals) = signal_hook::iterator::Signals::new([SIGINT, SIGTERM]) else {
        return;
    };
    let target = ctx.target.clone();
    let id = id.clone();
    std::thread::spawn(move || {
        if signals.forever().next().is_some() {
            if let Ok(client) = Client::connect(&target, "rig-cli") {
                client
                    .call::<methods::Stop>(&methods::Stop {
                        session_id: id.clone(),
                        scope: methods::StopScope::Turn,
                        key: rig_protocol::Key(uuid::Uuid::new_v4().to_string()),
                    })
                    .ok();
            }
            eprintln!(
                "rig: interrupted; stopped the turn of session {}\nnext: rig log {}",
                id.0, id.0
            );
            std::process::exit(i32::from(rig_protocol::Exit::Interrupted.code()));
        }
    });
}

#[cfg(not(unix))]
fn stop_on_interrupt(_ctx: &Ctx, _id: &SessionId) {}
