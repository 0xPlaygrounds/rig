//! What a harness is to the bridge: a program started once per session that
//! takes prompts and reports what its model says. Each harness module
//! implements [`Harness`] for one program's protocol.

use std::fmt;
use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex, PoisonError};
use std::time::Duration;

use rig_harness::rig_core::completion::{Effort, FinishReason, ToolDefinition, Usage};
use rig_harness::rig_tools::process::Process;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::{mpsc, oneshot, watch};

use crate::mcp::Entry;

/// Every coding-agent vendor bridged by this plugin.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub(crate) enum Vendor {
    ClaudeCode,
    GrokBuild,
    Antigravity,
    Codex,
    Muse,
    Copilot,
}

impl Vendor {
    /// Stable catalog vendor name used in `<vendor>/<model>`.
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-code",
            Self::GrokBuild => "grok-build",
            Self::Antigravity => "antigravity",
            Self::Codex => "codex",
            Self::Muse => "muse",
            Self::Copilot => "copilot-cli",
        }
    }

    /// Human-readable display label for `/bridge` output.
    pub(crate) const fn label(self) -> &'static str {
        match self {
            Self::ClaudeCode => "Claude Code",
            Self::GrokBuild => "Grok Build",
            Self::Antigravity => "Antigravity",
            Self::Codex => "Codex",
            Self::Muse => "Meta AI Muse",
            Self::Copilot => "GitHub Copilot",
        }
    }
}

impl fmt::Display for Vendor {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

/// Live single-inference context window metrics reported by a harness.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub(crate) struct ContextSnapshot {
    /// Tokens currently occupying the model's context window.
    pub used: Option<u64>,
    /// Authoritative context window limit in tokens.
    pub window: Option<u32>,
}

/// One piece of a prompt.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Part {
    /// Text.
    Text(String),
    /// An image, base-64 encoded.
    Image {
        /// Its MIME type.
        media_type: String,
        /// Its bytes in base 64.
        data: String,
    },
}

/// One prompt sent to a running harness, with its requested reasoning effort.
#[derive(Debug, Clone, PartialEq)]
pub(crate) struct Prompt {
    pub parts: Vec<Part>,
    pub effort: Option<Effort>,
}

/// A user message for the turn a harness is answering, sent while a tool
/// call of that turn is open.
#[derive(Debug)]
pub(crate) struct Steer {
    pub parts: Vec<Part>,
    /// Told once the program holds the message for its running turn, or
    /// why it does not.
    pub taken: oneshot::Sender<Result<(), String>>,
}

/// Parses a lower-case reasoning effort wire word into [`Effort`].
pub(crate) fn parse_effort(word: &str) -> Option<Effort> {
    match word {
        "minimal" => Some(Effort::Minimal),
        "low" => Some(Effort::Low),
        "medium" => Some(Effort::Medium),
        "high" => Some(Effort::High),
        "xhigh" => Some(Effort::XHigh),
        "max" => Some(Effort::Max),
        _ => None,
    }
}

/// Maps `effort` to a harness wire string, clamping `Minimal` to `"low"` and,
/// when `allow_max` is false, `Max` to `"xhigh"`.
pub(crate) const fn effort_wire(effort: Effort, allow_max: bool) -> &'static str {
    match effort {
        Effort::Minimal | Effort::Low => "low",
        Effort::Medium => "medium",
        Effort::High => "high",
        Effort::XHigh => "xhigh",
        Effort::Max if allow_max => "max",
        Effort::Max => "xhigh",
        _ => "high",
    }
}

/// What a harness reports while it answers a prompt.
#[derive(Debug, Clone, PartialEq)]
pub(crate) enum Event {
    /// A fragment of the model's answer.
    Text(String),
    /// A fragment of the model's reasoning.
    Reasoning(String),
    /// An inference began. A harness that says so ends each with
    /// [`Event::Usage`].
    Began,
    /// The token counters of one inference.
    Usage(Usage),
    /// Context window metrics reported at runtime by a harness.
    Context(ContextSnapshot),
    /// The program's own name for the session it saves, with which a
    /// later process [goes on](Launch::resume) from it.
    Saved(String),
    /// The prompt is answered with its finish reason, or failed.
    Done(Result<FinishReason, String>),
    /// The turn was stopped: the program has the prompt and what it had
    /// answered of it, and takes the next.
    Stopped,
}

/// What a session is started with.
pub(crate) struct Launch<'a> {
    /// The harness's program, with the agent's process rules applied and no
    /// arguments yet.
    pub command: Command,
    /// The session's private directory: the program's working directory,
    /// where the files it is pointed at go, and where what it saves of the
    /// conversation is, if the program lets that be chosen. Empty unless
    /// the session goes on from a saved one.
    pub dir: &'a Path,
    /// The session a process before this one [saved](Event::Saved) in
    /// `dir`, to go on from: the program then has the conversation, and is
    /// sent only what comes next. `None` starts a new session.
    pub resume: Option<&'a str>,
    /// The system prompt.
    pub system: &'a str,
    /// The model's id, as the harness names it.
    pub model: &'a str,
    /// The reasoning effort asked for, if any.
    pub effort: Option<Effort>,
    /// The MCP server entry that serves Rig's tools.
    pub mcp: &'a Entry,
    /// Rig's tools, as the server lists them.
    pub tools: &'a [Arc<ToolDefinition>],
}

/// A started harness. Dropping it kills the program and everything the
/// program started.
pub(crate) struct Running {
    /// Takes each prompt. The harness answers them in order.
    pub prompts: mpsc::UnboundedSender<Prompt>,
    /// Takes a user message for the turn in progress, which the program
    /// gives its model as one. `None` when the program has no way to.
    pub steers: Option<mpsc::UnboundedSender<Steer>>,
    /// Asks the program to end a turn, its own way: that of the prompt with
    /// this number, counting from 1 in the order the prompts were sent,
    /// and sent before the stop is. The program keeps what it has of the
    /// turn, which ends with [`Event::Stopped`]. A turn that is over is
    /// left alone, and so is every other turn.
    pub stops: mpsc::UnboundedSender<u64>,
    /// What the harness reports. It closes when the program is gone.
    pub events: mpsc::UnboundedReceiver<Event>,
    /// What keeps the program alive: dropping it kills the program.
    pub _alive: Box<dyn Send>,
}

/// A coding agent's program, driven as a model.
pub(crate) trait Harness: Send + Sync + 'static {
    /// Starts the program for one session, stripped to inference: none of
    /// its own tools, instructions, memory or extensions, and Rig's tools
    /// as its only MCP server. Runs inside the tokio runtime.
    fn start(&self, launch: Launch<'_>) -> Result<Running, String>;

    /// What the model must be told about this program, after Rig's system
    /// prompt: the names it gives Rig's tools, and what else it says that
    /// Rig's prompt does not mean.
    fn note(&self, project: Option<&Path>) -> String;

    /// Whether the program offers its model a tool that is added while it
    /// runs, once it is told the list changed and has listed the tools
    /// again. One that fixes its tools as it starts says no.
    fn relists(&self) -> bool {
        true
    }

    /// Whether a session the program saved offers its model, when a new
    /// process goes on from it, the tools that process is given. One whose
    /// saved session keeps the tools it was made with says no: nothing
    /// goes on from it under other tools.
    fn retools_saved(&self) -> bool {
        true
    }

    /// Whether the running program can apply a different reasoning effort on
    /// the next prompt without restarting the session.
    fn changes_effort(&self) -> bool {
        false
    }
}

/// The note on a program that shows the tool Rig calls `read` as `shown`
/// and, when `scratch` is given, tells the model of a working directory of
/// its own in place of the project's, `scratch`'s when known.
pub(crate) fn note(shown: &str, how: &str, scratch: Option<Option<&Path>>) -> String {
    let mut note = format!(
        "Your tools come from an MCP server named `rig`, so their names carry its prefix: \
         the tool these instructions and the tools' descriptions call `read` is `{shown}` \
         here{how}, and the others likewise."
    );
    if let Some(project) = scratch {
        note.push_str(
            " Ignore any working directory this program reports to you: that is an empty \
             scratch directory, not the project. Never pass that scratch path to Rig's tools; ",
        );
        match project {
            Some(project) => note.push_str(&format!(
                "your actual working directory is `{}`, and relative paths resolve there.",
                project.display()
            )),
            None => note.push_str("use relative paths or paths given in these instructions."),
        }
    }
    note
}

/// Drops `TMPDIR` from `command` when it points inside `cwd`, avoiding
/// startup failures in CLIs (such as `muse`) that forbid a temp root inside
/// their working directory.
pub(crate) fn clear_nested_tmpdir(command: &mut Command, cwd: &Path) {
    if let Some(tmpdir) = std::env::var_os("TMPDIR")
        && Path::new(&tmpdir).starts_with(cwd)
    {
        command.env_remove("TMPDIR");
    }
}

/// Reads a non-empty path from environment variable `var`.
pub(crate) fn env_path(var: &str) -> Option<std::path::PathBuf> {
    std::env::var_os(var)
        .filter(|value| !value.is_empty())
        .map(std::path::PathBuf::from)
}

/// Makes `to` a symlink to `from` so renewed credentials stay shared with the
/// user's own sessions. A link a session before this one left is replaced.
pub(crate) fn link(from: &Path, to: &Path) -> std::io::Result<()> {
    let _ = std::fs::remove_file(to);
    #[cfg(unix)]
    return std::os::unix::fs::symlink(from, to);
    #[cfg(not(unix))]
    return Err(std::io::Error::other("this platform has no links"));
}

/// The pipes of a process started with all three piped.
pub(crate) struct Pipes {
    pub stdin: tokio::process::ChildStdin,
    pub stdout: tokio::process::ChildStdout,
    pub stderr: tokio::process::ChildStderr,
}

/// Environment variables set on short-lived discovery and usage probes to
/// suppress auto-updaters and offline-stalling version-manager shims.
pub(crate) const PROBE_ENV: &[(&str, &str)] = &[
    ("MISE_OFFLINE", "1"),
    ("MUSE_NO_AUTO_UPDATE", "1"),
    ("GROK_DISABLE_AUTOUPDATER", "1"),
    ("AGY_CLI_DISABLE_AUTO_UPDATE", "1"),
];

/// Starts `command` with its three pipes taken for tokio.
pub(crate) fn spawn(mut command: Command) -> Result<(Process, Pipes), String> {
    let program = command.get_program().to_string_lossy().into_owned();
    command
        .env("MISE_OFFLINE", "1")
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped());
    let mut process = Process::start_with_stdio(command)
        .map_err(|why| format!("could not start `{program}`: {why}"))?;
    let pipes = (
        process.take_stdin(),
        process.take_stdout(),
        process.take_stderr(),
    );
    let (Some(stdin), Some(stdout), Some(stderr)) = pipes else {
        return Err(format!("`{program}` has no pipes"));
    };
    let pipes = tokio::process::ChildStdin::from_std(stdin).and_then(|stdin| {
        Ok(Pipes {
            stdin,
            stdout: tokio::process::ChildStdout::from_std(stdout)?,
            stderr: tokio::process::ChildStderr::from_std(stderr)?,
        })
    });
    let pipes = pipes.map_err(|why| format!("`{program}`'s pipes: {why}"))?;
    Ok((process, pipes))
}

/// How long a harness has to finish its startup handshake (`initialize`,
/// session creation, model selection, and where applicable MCP readiness).
pub(crate) const HANDSHAKE_TIMEOUT: Duration = Duration::from_secs(30);

/// How much of a program's stderr is kept to explain its exit.
const KEPT: usize = 2048;

/// Grace period for the stderr reader to finish draining after stdout closes
/// on an early child exit, so startup errors report stderr instead of EOF.
const STDERR_DRAIN: Duration = Duration::from_millis(100);

/// The end of what a program wrote to stderr so far.
#[derive(Clone)]
pub(crate) struct Stderr {
    tail: Arc<Mutex<String>>,
    done: watch::Receiver<bool>,
}

impl Stderr {
    /// Reads `stderr` to its end on the tokio runtime, keeping its last
    /// lines, so the program never blocks on a full pipe.
    pub(crate) fn follow(stderr: tokio::process::ChildStderr) -> Self {
        let tail = Arc::new(Mutex::new(String::new()));
        let (done_tx, done) = watch::channel(false);
        let shared = tail.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let mut tail = shared.lock().unwrap_or_else(PoisonError::into_inner);
                tail.push_str(&line);
                tail.push('\n');
                let cut = tail.ceil_char_boundary(tail.len().saturating_sub(KEPT));
                tail.drain(..cut);
            }
            let _ = done_tx.send(true);
        });
        Self { tail, done }
    }

    /// `what`, followed by what the program last said, if anything. Waits
    /// briefly for the stderr pipe to drain when the child just exited.
    pub(crate) async fn explain(&self, what: &str) -> String {
        let mut done = self.done.clone();
        let _ = tokio::time::timeout(STDERR_DRAIN, done.wait_for(|finished| *finished)).await;
        let tail = self.tail.lock().unwrap_or_else(PoisonError::into_inner);
        match tail.trim() {
            "" => what.to_owned(),
            said => format!("{what}: {said}"),
        }
    }
}
