//! What a harness is to the bridge: a program started once per session that
//! takes prompts and reports what its model says. Each harness module
//! implements [`Harness`] for one program's protocol.

use std::path::Path;
use std::process::Command;
use std::sync::{Arc, Mutex, PoisonError};

use rig_harness::rig_core::completion::{Effort, ToolDefinition, Usage};
use rig_harness::rig_tools::process::Process;
use tokio::io::{AsyncBufReadExt, BufReader};
use tokio::sync::mpsc;

use crate::mcp::Entry;

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
    /// The prompt is answered, or failed for the reason given.
    Done(Result<(), String>),
}

/// What a session is started with.
pub(crate) struct Launch<'a> {
    /// The harness's program, with the agent's process rules applied and no
    /// arguments yet.
    pub command: Command,
    /// The session's private, empty directory: the program's working
    /// directory, and where the files it is pointed at go.
    pub dir: &'a Path,
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
    pub prompts: mpsc::UnboundedSender<Vec<Part>>,
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
    fn note(&self) -> String;
}

/// The note on a program that shows the tool Rig calls `read` as `shown`
/// and, when `scratch`, tells the model of a working directory of its own.
pub(crate) fn note(shown: &str, how: &str, scratch: bool) -> String {
    let mut note = format!(
        "Your tools come from an MCP server named `rig`, so their names carry its prefix: \
         the tool these instructions and the tools' descriptions call `read` is `{shown}` \
         here{how}, and the others likewise."
    );
    if scratch {
        note.push_str(
            " The working directory this program reports to you is a scratch directory of \
             its own, not the project: paths are as these instructions and the tools give them.",
        );
    }
    note
}

/// The pipes of a process started with all three piped.
pub(crate) struct Pipes {
    pub stdin: tokio::process::ChildStdin,
    pub stdout: tokio::process::ChildStdout,
    pub stderr: tokio::process::ChildStderr,
}

/// Starts `command` with its three pipes taken for tokio.
pub(crate) fn spawn(mut command: Command) -> Result<(Process, Pipes), String> {
    let program = command.get_program().to_string_lossy().into_owned();
    command
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

/// How much of a program's stderr is kept to explain its exit.
const KEPT: usize = 2048;

/// The end of what a program wrote to stderr so far.
#[derive(Clone, Default)]
pub(crate) struct Stderr(Arc<Mutex<String>>);

impl Stderr {
    /// Reads `stderr` to its end on the tokio runtime, keeping its last
    /// lines, so the program never blocks on a full pipe.
    pub(crate) fn follow(stderr: tokio::process::ChildStderr) -> Self {
        let kept = Self::default();
        let tail = kept.0.clone();
        tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                let mut tail = tail.lock().unwrap_or_else(PoisonError::into_inner);
                tail.push_str(&line);
                tail.push('\n');
                let over = tail.len().saturating_sub(KEPT);
                let cut = (over..=tail.len())
                    .find(|at| tail.is_char_boundary(*at))
                    .unwrap_or(0);
                tail.drain(..cut);
            }
        });
        kept
    }

    /// `what`, followed by what the program last said, if anything.
    pub(crate) fn explain(&self, what: &str) -> String {
        let tail = self.0.lock().unwrap_or_else(PoisonError::into_inner);
        match tail.trim() {
            "" => what.to_owned(),
            said => format!("{what}: {said}"),
        }
    }
}
