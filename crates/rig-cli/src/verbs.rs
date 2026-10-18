//! The verbs that talk to a session: start, send, wait, answer, log,
//! agents, stop.

use std::path::PathBuf;
use std::sync::mpsc;
use std::time::Duration;

use rig_client::{Client, Target};
use rig_protocol::event::State;
use rig_protocol::methods::Budget;
use rig_protocol::methods::{
    self, AskInfo, Disposition, SendMode, SessionInfo, StopScope, WaitResult, WaitStatus,
};
use rig_protocol::{AskId, CommandSeq, EventRecord, Exit, Grant, Key, Rule, Seq, SessionId};
use serde_json::{Value, json};

use crate::args::Args;
use crate::out::{self, Failure};

/// The default `rig wait --timeout`.
pub const WAIT_TIMEOUT: Duration = Duration::from_secs(600);
/// How long `rig stop` waits for the tree to stop.
const STOP_TIMEOUT: Duration = Duration::from_secs(60);

/// What every verb shares: where the server is and how to print.
pub struct Ctx {
    pub target: Target,
    pub json: bool,
}

impl Ctx {
    pub fn new(args: &Args) -> Result<Ctx, Failure> {
        let rig_home = rig_harness_protocol::Home::from_env().root().to_path_buf();
        let target = match args.value("host") {
            Some(host) => remote_target(&rig_home, host)?,
            None => Target::Local { rig_home },
        };
        Ok(Ctx {
            target,
            json: args.has("json"),
        })
    }

    /// A connection to the server, started if needed.
    pub fn connect(&self) -> Result<Client, Failure> {
        Client::connect(&self.target, "rig-cli")
            .map_err(|error| Failure::client(error, "rig server status"))
    }

    /// `rig ... --host H` when this context is remote, for `next:` lines.
    pub fn host_flag(&self) -> String {
        match &self.target {
            Target::Remote { host, .. } => format!(" --host {host}"),
            Target::Local { .. } => String::new(),
        }
    }
}

/// The `[remote.hosts]` entry `host` of `rig_home`'s settings.
/// A host with no entry is a usage error whose next: is the entry to add.
pub fn remote_target(rig_home: &std::path::Path, host: &str) -> Result<Target, Failure> {
    Target::remote(rig_home, host).map_err(|error| {
        let settings = rig_home.join("settings.toml");
        match &error {
            rig_client::ClientError::Unavailable { next } if next.starts_with("add ") => {
                Failure::usage(
                    format!(
                        "no host `{host}` in [remote.hosts] of {}",
                        settings.display()
                    ),
                    "start",
                )
                .next_is(&format!(
                    "add to {}:\n[remote.hosts]\n{host} = [\"ssh\", \"{host}\"]",
                    settings.display()
                ))
            }
            _ => Failure::client(error, "rig help-agents remote"),
        }
    })
}

/// `--key K`, or a new random key.
pub fn key(args: &Args) -> Key {
    Key(args
        .value("key")
        .map_or_else(|| uuid::Uuid::new_v4().to_string(), str::to_owned))
}

/// The prompt in `words`: their text joined by spaces; `-` reads stdin
/// (through `stdin`), and so does no prompt at all when `absent_reads`.
/// stdin is never touched otherwise, so a driver whose stdin is an open
/// pipe never hangs here.
pub fn prompt(
    words: &[String],
    absent_reads: bool,
    stdin: impl FnOnce() -> std::io::Result<String>,
) -> std::io::Result<Option<String>> {
    match words {
        [dash] if dash == "-" => stdin().map(Some),
        [] if absent_reads => stdin().map(Some),
        [] => Ok(None),
        words => Ok(Some(words.join(" "))),
    }
}

/// stdin to its end.
pub fn read_stdin() -> std::io::Result<String> {
    let mut text = String::new();
    std::io::Read::read_to_string(&mut std::io::stdin(), &mut text)?;
    Ok(text)
}

/// The grant `--ask`, `--allow` and `--deny` ask for, `None` without any.
pub fn grant(args: &Args, verb: &str) -> Result<Option<Grant>, Failure> {
    let mut approve = Vec::new();
    for rule in args.values("allow") {
        approve.push(parse_rule(rule).ok_or_else(|| {
            Failure::usage(
                format!("--allow takes TOOL, shell:PREFIX or edits:DIR, not `{rule}`"),
                verb,
            )
        })?);
    }
    approve.extend(args.values("deny").map(|name| Rule::Deny {
        name: name.to_owned(),
    }));
    let mut ask = Vec::new();
    for list in args.values("ask") {
        for what in list.split(',').map(str::trim) {
            if what.is_empty() {
                return Err(Failure::usage(
                    format!(
                        "--ask takes shell, edits or tool names, comma-separated, not `{list}`"
                    ),
                    verb,
                ));
            }
            ask.push(what.to_owned());
        }
    }
    Ok((!approve.is_empty() || !ask.is_empty()).then(|| Grant {
        // An owner's session keeps every tool, folder and the network;
        // --ask says what asks, the rules answer up front.
        net: true,
        approve,
        ask,
        ..Grant::default()
    }))
}

/// `TOOL`, `shell:PREFIX` or `edits:DIR` (made absolute).
pub fn parse_rule(rule: &str) -> Option<Rule> {
    if let Some(prefix) = rule.strip_prefix("shell:") {
        return (!prefix.is_empty()).then(|| Rule::ShellPrefix {
            prefix: prefix.to_owned(),
        });
    }
    if let Some(path) = rule.strip_prefix("edits:") {
        return (!path.is_empty()).then(|| Rule::EditsUnder {
            path: std::path::absolute(path).unwrap_or_else(|_| PathBuf::from(path)),
        });
    }
    let tool = !rule.is_empty()
        && rule
            .chars()
            .all(|c| c.is_alphanumeric() || "_-./".contains(c));
    tool.then(|| Rule::Tool {
        name: rule.to_owned(),
    })
}

/// `--cwd DIR` made absolute, else the current directory.
pub fn cwd(args: &Args, verb: &str) -> Result<PathBuf, Failure> {
    let dir = match args.value("cwd") {
        Some(dir) => std::path::absolute(dir),
        None => std::env::current_dir(),
    };
    dir.map_err(|error| Failure::usage(format!("no working directory: {error}"), verb))
}

/// The session `name_or_id` names: a session with that id, else the one
/// with that name, else it as given (the server then says it is unknown).
pub fn resolve(client: &Client, name_or_id: &str) -> SessionId {
    let sessions = client
        .call::<methods::List>(&methods::List {
            tree: false,
            all: true,
        })
        .unwrap_or_default();
    sessions
        .iter()
        .find(|info| info.session_id.0 == name_or_id)
        .or_else(|| {
            sessions
                .iter()
                .find(|info| info.name.as_deref() == Some(name_or_id))
        })
        .map_or_else(
            || SessionId(name_or_id.to_owned()),
            |info| info.session_id.clone(),
        )
}

/// The `next:` for an unknown session.
fn unknown(ctx: &Ctx) -> String {
    format!("rig agents --all{}", ctx.host_flag())
}

pub fn start(args: &Args) -> Result<u8, Failure> {
    let ctx = Ctx::new(args)?;
    let prompt = prompt(&args.words, false, read_stdin)
        .map_err(|error| Failure::usage(format!("could not read stdin: {error}"), "start"))?;
    let model = args.value("model");
    let params = methods::Start {
        cwd: cwd(args, "start")?,
        // With a model, the prompt follows the model's choice.
        prompt: if model.is_some() {
            None
        } else {
            prompt.clone()
        },
        name: args.value("name").map(str::to_owned),
        host: None,
        build: build(args, "start")?,
        grant: grant(args, "start")?,
        policy: None,
        budget: budget(args, "start")?,
        context: None,
        unload: None,
        view: false,
        key: key(args),
    };
    let client = ctx.connect()?;
    let mut started = client
        .call::<methods::Start>(&params)
        .map_err(|error| Failure::client(error, "rig help start"))?;
    if let Some(model) = model {
        let sent = choose_model(&client, &started.session_id, model, prompt, &ctx)?;
        started.command_seq = sent.or(started.command_seq);
    }
    let id = &started.session_id.0;
    let next = format!("rig wait {id}{}", ctx.host_flag());
    if ctx.json {
        out::line(&out::object(
            "started",
            json!({
                "sessionId": id,
                "seq": started.seq,
                "commandSeq": started.command_seq,
                "next": next,
            }),
        ));
    } else {
        out::text(id);
    }
    Ok(0)
}

/// The build this binary is: the name of its directory under
/// `bin/builds/`, where the launcher installs every build.
pub fn own_build() -> Option<rig_protocol::BuildId> {
    let exe = std::env::current_exe().ok()?;
    let directory = exe.parent()?;
    let builds = directory.parent()?;
    if builds.file_name()? != "builds" {
        return None;
    }
    let id = directory.file_name()?.to_string_lossy().into_owned();
    Some(rig_protocol::BuildId(id))
}

/// The build a new session runs: the one `--plugins FILE` names, which
/// the launcher builds (once; later it reuses it), else this binary's own,
/// so a server started on an older build still runs the session on the
/// one the launcher just chose.
pub fn build(args: &Args, verb: &str) -> Result<Option<rig_protocol::BuildId>, Failure> {
    let Some(list) = args.value("plugins") else {
        return Ok(own_build());
    };
    let list = std::path::absolute(list)
        .map_err(|error| Failure::usage(format!("--plugins {list}: {error}"), verb))?;
    let launcher = std::env::var_os(rig_harness_protocol::env::LAUNCHER).ok_or_else(|| {
        Failure::usage(
            "--plugins needs the rig launcher: run this through `rig`",
            verb,
        )
    })?;
    let output = std::process::Command::new(launcher)
        .args(["build", "--print-build-id", "--plugins"])
        .arg(&list)
        .stdin(std::process::Stdio::null())
        .stderr(std::process::Stdio::inherit())
        .output();
    let failed = |why: String| {
        Failure::new(
            Exit::Failed,
            format!("could not build {}: {why}", list.display()),
            format!(
                "RIG_VERBOSE=1 rig build --print-build-id --plugins {}",
                list.display()
            ),
        )
    };
    let output = output.map_err(|error| failed(error.to_string()))?;
    if !output.status.success() {
        return Err(failed(output.status.to_string()));
    }
    let id = String::from_utf8_lossy(&output.stdout).trim().to_owned();
    if id.is_empty() {
        return Err(failed("the launcher printed no build id".to_owned()));
    }
    Ok(Some(rig_protocol::BuildId(id)))
}

/// Sends `id` the slash command `/model MODEL`, then `prompt` if any, and
/// returns the prompt's command number: commands apply in order, so the
/// prompt runs on that model.
pub fn choose_model(
    client: &Client,
    id: &SessionId,
    model: &str,
    prompt: Option<String>,
    ctx: &Ctx,
) -> Result<Option<CommandSeq>, Failure> {
    let send = |text: String| {
        client
            .call::<methods::Send>(&methods::Send {
                session_id: id.clone(),
                text: Some(text),
                content: None,
                mode: Some(SendMode::Queue),
                key: Key(uuid::Uuid::new_v4().to_string()),
            })
            .map_err(|error| Failure::client(error, unknown(ctx)))
    };
    send(format!("/model {model}"))?;
    prompt
        .map(|prompt| send(prompt).map(|sent| sent.command_seq))
        .transpose()
}

pub fn budget(args: &Args, verb: &str) -> Result<Option<Budget>, Failure> {
    Ok(args.number("budget-tokens", verb)?.map(|tokens| Budget {
        tokens: Some(tokens),
        cost_usd: None,
    }))
}

pub fn send(args: &Args) -> Result<u8, Failure> {
    let ctx = Ctx::new(args)?;
    let session = args.session("send")?;
    let words = args.words.get(1..).unwrap_or_default();
    let text = prompt(words, false, read_stdin)
        .map_err(|error| Failure::usage(format!("could not read stdin: {error}"), "send"))?
        .ok_or_else(|| {
            Failure::usage(
                "rig send needs the text to send, or `-` to read it from stdin",
                "send",
            )
        })?;
    let mode = match (args.has("queue"), args.has("interrupt")) {
        (true, true) => {
            return Err(Failure::usage(
                "--queue and --interrupt exclude each other",
                "send",
            ));
        }
        (true, false) => Some(SendMode::Queue),
        (false, true) => Some(SendMode::Interrupt),
        (false, false) => None,
    };
    let client = ctx.connect()?;
    let id = resolve(&client, session);
    let sent = client
        .call::<methods::Send>(&methods::Send {
            session_id: id.clone(),
            text: Some(text),
            content: None,
            mode,
            key: key(args),
        })
        .map_err(|error| Failure::client(error, unknown(&ctx)))?;
    let disposition = disposition(sent.disposition);
    let next = format!("rig wait {}{}", id.0, ctx.host_flag());
    if ctx.json {
        out::line(&out::object(
            "sent",
            json!({
                "sessionId": id.0,
                "commandSeq": sent.command_seq,
                "disposition": disposition,
                "next": next,
            }),
        ));
    } else {
        out::text(&format!("{} {disposition}", sent.command_seq.0));
    }
    Ok(0)
}

fn disposition(disposition: Disposition) -> &'static str {
    match disposition {
        Disposition::Started => "started",
        Disposition::Steered => "steered",
        Disposition::Queued => "queued",
    }
}

pub fn wait(args: &Args) -> Result<u8, Failure> {
    let ctx = Ctx::new(args)?;
    args.session("wait")?;
    let timeout = args
        .duration("timeout", "wait")?
        .unwrap_or(Some(WAIT_TIMEOUT));
    let after = args.number("after", "wait")?.map(CommandSeq);
    let client = ctx.connect()?;
    let ids: Vec<SessionId> = args
        .words
        .iter()
        .map(|word| resolve(&client, word))
        .collect();
    let [id] = ids.as_slice() else {
        if after.is_some() {
            return Err(Failure::usage(
                "--after names a command of one session; wait on one id with it",
                "wait",
            ));
        }
        return wait_many(&ctx, &ids, timeout, args.has("all"));
    };
    let result = call_wait(&client, id, after, timeout, &ctx)?;
    Ok(print_wait(&ctx, id, &result, timeout, "wait"))
}

/// `rig wait A B ...`: each session's wait on a connection of its own; the
/// first to settle (not time out) is printed, its id first, or with `all`
/// every one is, in the order given. A timeout leaves the rest working.
fn wait_many(
    ctx: &Ctx,
    ids: &[SessionId],
    timeout: Option<Duration>,
    all: bool,
) -> Result<u8, Failure> {
    let (sent, got) = mpsc::channel();
    for (index, id) in ids.iter().enumerate() {
        let sent = sent.clone();
        let id = id.clone();
        let ctx = Ctx {
            target: ctx.target.clone(),
            json: ctx.json,
        };
        std::thread::spawn(move || {
            let result = ctx
                .connect()
                .and_then(|client| call_wait(&client, &id, None, timeout, &ctx));
            sent.send((index, result)).ok();
        });
    }
    drop(sent);
    let mut results: Vec<Option<WaitResult>> = vec![None; ids.len()];
    for (index, result) in got.iter() {
        let result = result?;
        if !all && result.status != WaitStatus::TimedOut {
            let id = ids
                .get(index)
                .cloned()
                .unwrap_or_else(|| SessionId(String::new()));
            if !ctx.json {
                out::text(&id.0);
            }
            return Ok(print_wait(ctx, &id, &result, timeout, "wait"));
        }
        if let Some(slot) = results.get_mut(index) {
            *slot = Some(result);
        }
    }
    let mut exit = 0;
    let mut working: Vec<&str> = Vec::new();
    for (id, result) in ids.iter().zip(&results) {
        let Some(result) = result else { continue };
        if result.status == WaitStatus::TimedOut {
            working.push(&id.0);
            continue;
        }
        if !ctx.json {
            out::text(&id.0);
        }
        let code = print_wait(ctx, id, result, timeout, "wait");
        if exit == 0 {
            exit = code;
        }
    }
    if working.is_empty() {
        return Ok(exit);
    }
    let flag = if all { "--all " } else { "" };
    Err(Failure::new(
        Exit::TimedOut,
        format!(
            "{} still working after {}; the work goes on",
            working.join(", "),
            timeout.map_or_else(|| "the timeout".to_owned(), human)
        ),
        format!("rig wait {flag}{}{}", working.join(" "), ctx.host_flag()),
    ))
}

/// `rig/session/wait`.
pub fn call_wait(
    client: &Client,
    id: &SessionId,
    after: Option<CommandSeq>,
    timeout: Option<Duration>,
    ctx: &Ctx,
) -> Result<WaitResult, Failure> {
    client
        .call::<methods::Wait>(&methods::Wait {
            session_id: id.clone(),
            after_command: after,
            timeout_ms: timeout
                .map(|timeout| u64::try_from(timeout.as_millis()).unwrap_or(u64::MAX)),
        })
        .map_err(|error| Failure::client(error, unknown(ctx)))
}

fn status_word(status: WaitStatus) -> &'static str {
    match status {
        WaitStatus::Answered => "answered",
        WaitStatus::NeedsApproval => "needs_approval",
        WaitStatus::NeedsAnswer => "needs_answer",
        WaitStatus::Failed => "failed",
        WaitStatus::Stopped => "stopped",
        WaitStatus::TimedOut => "timed_out",
    }
}

/// The `next:` after a wait ended with `result`.
pub fn wait_next(ctx: &Ctx, id: &SessionId, result: &WaitResult) -> Option<String> {
    let host = ctx.host_flag();
    let id = &id.0;
    match result.status {
        WaitStatus::Answered => None,
        WaitStatus::NeedsApproval => Some(match &result.ask {
            Some(ask) => format!("rig answer {id} {} --allow{host}", ask.ask.ask_id.0),
            None => format!("rig answer {id} --allow{host}"),
        }),
        WaitStatus::NeedsAnswer => Some(match &result.ask {
            Some(ask) => format!("rig answer {id} {} --text \"...\"{host}", ask.ask.ask_id.0),
            None => format!("rig answer {id} --text \"...\"{host}"),
        }),
        WaitStatus::Failed => Some(format!("rig log {id}{host}")),
        WaitStatus::Stopped => Some(format!("rig send {id} \"...\"{host}")),
        WaitStatus::TimedOut => Some(format!("rig wait {id}{host}")),
    }
}

/// Prints a wait's outcome as `verb` (`wait` or `-p`) and returns its exit
/// code: the answer on stdout; an ask, a failure, a stop or a timeout on
/// stderr with `next:`.
pub fn print_wait(
    ctx: &Ctx,
    id: &SessionId,
    result: &WaitResult,
    timeout: Option<Duration>,
    verb: &str,
) -> u8 {
    let next = wait_next(ctx, id, result);
    let exit = result.status.exit().code();
    if ctx.json {
        let mut object = out::object(
            if verb == "-p" { "result" } else { "wait" },
            json!({
                "sessionId": id.0,
                "status": status_word(result.status),
                "seq": result.seq,
                "commandSeq": result.command_seq,
                "exit": exit,
            }),
        );
        if let Some(map) = object.as_object_mut() {
            if let Some(answer) = &result.answer {
                map.insert("answer".to_owned(), answer.clone().into());
            }
            if let Some(ask) = &result.ask {
                map.insert(
                    "ask".to_owned(),
                    serde_json::to_value(ask).unwrap_or(Value::Null),
                );
            }
            if let Some(error) = &result.error {
                map.insert("error".to_owned(), error.clone().into());
            }
            if let Some(next) = &next {
                map.insert("next".to_owned(), next.clone().into());
            }
        }
        out::line(&object);
        return exit;
    }
    let (stdout, stderr) = wait_text(id, result, timeout, next.as_deref());
    if !stdout.is_empty() || result.status == WaitStatus::Answered {
        out::text(&stdout);
    }
    if !stderr.is_empty() {
        eprint!("{stderr}");
    }
    exit
}

/// A wait's outcome as text: what goes to stdout (the answer alone) and
/// what to stderr (an ask, a failure, a stop or a timeout, each with
/// `next:`).
pub fn wait_text(
    id: &SessionId,
    result: &WaitResult,
    timeout: Option<Duration>,
    next: Option<&str>,
) -> (String, String) {
    let next_line = next.map_or_else(String::new, |next| format!("next: {next}\n"));
    match result.status {
        WaitStatus::Answered => (result.answer.clone().unwrap_or_default(), String::new()),
        WaitStatus::NeedsApproval | WaitStatus::NeedsAnswer => {
            let ask = result.ask.as_ref().map_or_else(
                || "needs an answer\n".to_owned(),
                |ask| describe_ask(id, ask),
            );
            (String::new(), format!("{ask}{next_line}"))
        }
        WaitStatus::Failed => {
            let why = result.error.as_deref().unwrap_or("no reason given");
            (
                String::new(),
                format!("rig: session {} failed: {why}\n{next_line}", id.0),
            )
        }
        WaitStatus::Stopped => (
            String::new(),
            format!("rig: session {} was stopped\n{next_line}", id.0),
        ),
        WaitStatus::TimedOut => (
            String::new(),
            format!(
                "rig: session {} is still working after {}; the work goes on\n{next_line}",
                id.0,
                timeout.map_or_else(|| "the timeout".to_owned(), human)
            ),
        ),
    }
}

fn human(duration: Duration) -> String {
    let secs = duration.as_secs();
    if secs >= 3600 && secs.is_multiple_of(3600) {
        format!("{}h", secs / 3600)
    } else if secs >= 60 && secs.is_multiple_of(60) {
        format!("{}m", secs / 60)
    } else {
        format!("{secs}s")
    }
}

/// An ask as text: what it asks, who asks, and its options.
pub fn describe_ask(id: &SessionId, ask: &AskInfo) -> String {
    let a = &ask.ask;
    let what = match a.kind {
        rig_protocol::event::AskKind::Approval => "needs approval",
        rig_protocol::event::AskKind::Question => "needs an answer",
    };
    let mut text = format!(
        "{what}: {}  [ask {}, agent {}, session {}]\n",
        a.title, a.ask_id.0, a.agent_name, id.0
    );
    if !a.body.trim().is_empty() {
        text.push_str(a.body.trim_end());
        text.push('\n');
    }
    if !a.options.is_empty() {
        let options: Vec<String> = a
            .options
            .iter()
            .map(|option| format!("{} ({})", option.option_id, option.label))
            .collect();
        text.push_str(&format!("options: {}\n", options.join(", ")));
    }
    text
}

pub fn answer(args: &Args) -> Result<u8, Failure> {
    let ctx = Ctx::new(args)?;
    let session = args.session("answer")?;
    let chosen: Vec<&str> = ["allow", "deny", "option", "text"]
        .into_iter()
        .filter(|name| args.has(name))
        .collect();
    let [how] = chosen.as_slice() else {
        return Err(Failure::usage(
            "rig answer needs exactly one of --allow, --deny, --option ID, --text TEXT",
            "answer",
        ));
    };
    let client = ctx.connect()?;
    let id = resolve(&client, session);
    let read = client
        .call::<methods::Read>(&methods::Read {
            session_id: id.clone(),
        })
        .map_err(|error| Failure::client(error, unknown(&ctx)))?;
    let host = ctx.host_flag();
    let wait_again = format!("rig wait {}{host}", id.0);
    let ask = match args.words.get(1) {
        Some(ask) => read
            .open_asks
            .iter()
            .find(|open| open.ask.ask_id.0 == *ask)
            .ok_or_else(|| {
                Failure::new(
                    Exit::NotAllowed,
                    format!(
                        "ask {ask} is not open in session {}: answered, expired or unknown",
                        id.0
                    ),
                    wait_again.clone(),
                )
            })?,
        None => match read.open_asks.as_slice() {
            [only] => only,
            [] => {
                return Err(Failure::new(
                    Exit::NotAllowed,
                    format!("session {} has no open ask", id.0),
                    wait_again,
                ));
            }
            [first, ..] => {
                let ids: Vec<&str> = read
                    .open_asks
                    .iter()
                    .map(|a| a.ask.ask_id.0.as_str())
                    .collect();
                return Err(Failure::usage(
                    format!(
                        "session {} has {} open asks ({}); name one",
                        id.0,
                        ids.len(),
                        ids.join(", ")
                    ),
                    "answer",
                )
                .next_is(&format!(
                    "rig answer {} {} --{how}{host}",
                    id.0, first.ask.ask_id.0
                )));
            }
        },
    };
    let (option_id, text) = match *how {
        "allow" | "deny" => {
            let wanted = if *how == "allow" { "allow" } else { "reject" };
            let option = ask
                .ask
                .options
                .iter()
                .find(|option| option.kind == format!("{wanted}_once"))
                .or_else(|| {
                    ask.ask
                        .options
                        .iter()
                        .find(|option| option.kind.starts_with(wanted))
                })
                .ok_or_else(|| {
                    Failure::usage(
                        format!("ask {} has no {how} option", ask.ask.ask_id.0),
                        "answer",
                    )
                    .next_is(&format!("rig wait {}{host}", id.0))
                })?;
            (Some(option.option_id.clone()), None)
        }
        "option" => (args.value("option").map(str::to_owned), None),
        _ => (None, args.value("text").map(str::to_owned)),
    };
    let ask_id: AskId = ask.ask.ask_id.clone();
    client
        .call::<methods::Answer>(&methods::Answer {
            session_id: id.clone(),
            ask_id: ask_id.clone(),
            option_id: option_id.clone(),
            text,
            key: key(args),
        })
        .map_err(|error| Failure::client(error, format!("rig wait {}{host}", id.0)))?;
    let next = format!("rig wait {}{host}", id.0);
    if ctx.json {
        out::line(&out::object(
            "answered",
            json!({
                "sessionId": id.0,
                "askId": ask_id.0,
                "optionId": option_id,
                "resolved": true,
                "next": next,
            }),
        ));
    } else {
        out::text(&format!("answered {}", ask_id.0));
    }
    Ok(0)
}

pub fn log(args: &Args) -> Result<u8, Failure> {
    let ctx = Ctx::new(args)?;
    let session = args.session("log")?;
    let after = Seq(args.number("after", "log")?.unwrap_or(0));
    let client = ctx.connect()?;
    let id = resolve(&client, session);
    if args.has("follow") {
        // Where the journal ends now: a stop before it ended a run the
        // session has since come back from, unless it is stopped now.
        let read = client
            .call::<methods::Read>(&methods::Read {
                session_id: id.clone(),
            })
            .map_err(|error| Failure::client(error, unknown(&ctx)))?;
        let stopped = read.info.state == State::Stopped;
        if stopped && after >= read.last_seq {
            return Ok(0);
        }
        let events = client
            .subscribe(&id, after)
            .map_err(|error| Failure::client(error, unknown(&ctx)))?;
        for record in events {
            let record = record.map_err(|error| {
                Failure::client(error, format!("rig log {} --follow --after N", id.0))
            })?;
            print_event(&ctx, &record);
            let ends = if stopped {
                record.seq >= read.last_seq
            } else {
                record.kind == "session.stopped" && record.seq > read.last_seq
            };
            if ends {
                break;
            }
        }
        return Ok(0);
    }
    let mut after = after;
    loop {
        let page = client
            .call::<methods::ReadEvents>(&methods::ReadEvents {
                session_id: id.clone(),
                after_seq: after,
                limit: Some(500),
            })
            .map_err(|error| Failure::client(error, unknown(&ctx)))?;
        let Some(last) = page.events.last() else {
            break;
        };
        after = last.seq;
        for record in &page.events {
            print_event(&ctx, record);
        }
    }
    Ok(0)
}

/// One event as `rig log` prints it: its JSON line, or its transcript text.
pub fn print_event(ctx: &Ctx, record: &EventRecord) {
    if ctx.json {
        let mut value = serde_json::to_value(record).unwrap_or(Value::Null);
        if let Some(map) = value.as_object_mut() {
            map.insert("rig".to_owned(), 1.into());
        }
        out::line(&value);
    } else if let Some(text) = transcript(record) {
        out::text(&text);
    }
}

/// An event's line in the text transcript; `None` for the kinds it skips.
pub fn transcript(record: &EventRecord) -> Option<String> {
    let data = &record.data;
    let field = |name: &str| data.get(name).and_then(Value::as_str).unwrap_or_default();
    Some(match record.kind.as_str() {
        "message" => match field("role") {
            "user" => format!("> {}", field("text")),
            _ => field("text").to_owned(),
        },
        "call.started" => {
            let args = data.get("args").map(Value::to_string).unwrap_or_default();
            let args: String = args.chars().take(120).collect();
            format!("· {} {args}", field("tool"))
        }
        "call.ended" if data.get("ok").and_then(Value::as_bool) == Some(false) => {
            format!("  failed: {}", field("error"))
        }
        "ask.opened" => format!(
            "? {} [ask {}, agent {}]",
            field("title"),
            field("askId"),
            field("agentName")
        ),
        "ask.resolved" => format!("  answered by {}", field("by")),
        "turn.ended" if field("status") != "answered" => {
            format!("-- turn {}: {}", field("status"), field("error"))
        }
        "notice" => format!("! {}", field("text")),
        "child.started" => format!("-- helper {} started", field("sessionId")),
        "child.ended" => format!("-- helper {} {}", field("sessionId"), field("status")),
        "worker.exited" => "-- worker exited".to_owned(),
        "session.faulted" => format!("-- faulted: {}", field("reason")),
        "session.stopped" => "-- stopped".to_owned(),
        "budget.exhausted" => "-- budget spent".to_owned(),
        _ => return None,
    })
}

pub fn agents(args: &Args) -> Result<u8, Failure> {
    let ctx = Ctx::new(args)?;
    let client = ctx.connect()?;
    let sessions = client
        .call::<methods::List>(&methods::List {
            tree: args.has("tree"),
            all: args.has("all"),
        })
        .map_err(|error| Failure::client(error, "rig server status"))?;
    if ctx.json {
        out::line(&out::object(
            "agents",
            json!({ "sessions": serde_json::to_value(&sessions).unwrap_or(Value::Null) }),
        ));
        return Ok(0);
    }
    if sessions.is_empty() {
        out::text("no sessions; start one with rig start --cwd . \"...\"");
        return Ok(0);
    }
    let mut text = String::new();
    if args.has("tree") {
        let roots = sessions.iter().filter(|info| {
            info.parent
                .as_ref()
                .is_none_or(|parent| !sessions.iter().any(|s| &s.session_id == parent))
        });
        for root in roots {
            tree_lines(&sessions, root, 0, &mut text);
        }
    } else {
        for info in &sessions {
            text.push_str(&agent_line(info, 0));
        }
    }
    out::text(&text);
    Ok(0)
}

fn tree_lines(sessions: &[SessionInfo], info: &SessionInfo, depth: usize, text: &mut String) {
    text.push_str(&agent_line(info, depth));
    if depth > 8 {
        return;
    }
    for child in sessions
        .iter()
        .filter(|s| s.parent.as_ref() == Some(&info.session_id))
    {
        tree_lines(sessions, child, depth + 1, text);
    }
}

fn agent_line(info: &SessionInfo, depth: usize) -> String {
    format!(
        "{}{}  {}  {}  {}\n",
        "  ".repeat(depth),
        info.session_id.0,
        state_word(info.state),
        info.name.as_deref().unwrap_or("-"),
        info.cwd.display()
    )
}

pub fn state_word(state: State) -> &'static str {
    match state {
        State::Idle => "idle",
        State::Running => "running",
        State::NeedsApproval => "needs_approval",
        State::NeedsAnswer => "needs_answer",
        State::Stopping => "stopping",
        State::Stopped => "stopped",
        State::Faulted => "faulted",
    }
}

pub fn stop(args: &Args) -> Result<u8, Failure> {
    let ctx = Ctx::new(args)?;
    let session = args.session("stop")?;
    let scope = if args.has("turn") {
        StopScope::Turn
    } else {
        StopScope::Session
    };
    let client = ctx.connect()?;
    let id = resolve(&client, session);
    let stopping = client
        .call::<methods::Stop>(&methods::Stop {
            session_id: id.clone(),
            scope,
            key: key(args),
        })
        .map_err(|error| Failure::client(error, unknown(&ctx)))?;
    let ids: Vec<&str> = stopping.stopping.iter().map(|s| s.0.as_str()).collect();
    if scope == StopScope::Session {
        let (sent, got) = mpsc::channel();
        let target = ctx.target.clone();
        let waiting = stopping.stopping.clone();
        std::thread::spawn(move || {
            sent.send(wait_stopped(&target, &waiting)).ok();
        });
        match got.recv_timeout(STOP_TIMEOUT) {
            Ok(Ok(())) => {}
            Ok(Err(failure)) => return Err(failure),
            Err(_) => {
                return Err(Failure::new(
                    Exit::TimedOut,
                    format!(
                        "session {} had not stopped after {}",
                        id.0,
                        human(STOP_TIMEOUT)
                    ),
                    format!("rig agents --tree{}", ctx.host_flag()),
                ));
            }
        }
    }
    if ctx.json {
        out::line(&out::object(
            "stopped",
            json!({
                "sessionId": id.0,
                "scope": if scope == StopScope::Turn { "turn" } else { "session" },
                "stopping": ids,
                "status": "stopped",
            }),
        ));
    } else {
        out::text(&format!("stopped {}", ids.join(" ")));
    }
    Ok(0)
}

/// Waits until every session in `ids` has journaled `session.stopped`.
fn wait_stopped(target: &Target, ids: &[SessionId]) -> Result<(), Failure> {
    let client = Client::connect(target, "rig-cli")
        .map_err(|error| Failure::client(error, "rig server status"))?;
    for id in ids {
        let read = client
            .call::<methods::Read>(&methods::Read {
                session_id: id.clone(),
            })
            .map_err(|error| Failure::client(error, "rig agents --all"))?;
        if read.info.state == State::Stopped {
            continue;
        }
        let events = client
            .subscribe(id, read.last_seq)
            .map_err(|error| Failure::client(error, "rig agents --all"))?;
        for record in events {
            let record = record.map_err(|error| Failure::client(error, "rig agents --all"))?;
            if record.kind == "session.stopped" {
                break;
            }
        }
    }
    Ok(())
}
