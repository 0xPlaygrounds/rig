//! The `shell` tool.

use std::collections::VecDeque;
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use rig_core::completion::ToolAnnotations;
use rig_core::tool::{
    ExitStatus, KeptEnd, OutputCut, Tool, ToolContext, ToolExecutionError, args_schema,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::process::{KillOnDrop, Launcher, OutputLine, Process, Spawn};
use crate::workspace::{Local, Workspace};
use crate::{Live, OutputLimit, Spill, blocking};

const DEFAULT_TIMEOUT: u64 = 120;
/// How long output pipes may stay open after the command exits before the
/// rest of its process group is killed, and how long reading goes on after
/// that kill.
const PIPE_GRACE: Duration = Duration::from_millis(100);
/// How often the command's state is checked while no output arrives.
const POLL: Duration = Duration::from_millis(20);
/// Output past which the command is killed. While it runs, only what the
/// [`OutputLimit`] keeps is held in memory.
const OUTPUT_LIMIT: u64 = 10 * 1024 * 1024;

/// Runs a command with `sh -c` in its own session and process group, in its
/// [`Workspace`], as its [`Launcher`] starts processes. Publishes how the
/// command ended ([`ExitStatus`]) beside its output, and how the output was
/// cut ([`OutputCut`]) when it was, and reports the output
/// as it comes to its [`Live`], if it has one.
#[derive(Clone, Debug)]
pub struct Shell {
    /// Where the command runs.
    pub workspace: Arc<dyn Workspace>,
    /// How the command is started: a wrapper, a directory, its variables.
    pub launcher: Launcher,
    /// Where each piece of output is reported as it comes, before the
    /// command ends.
    pub live: Option<Live>,
    /// Where output that is cut is kept whole, named in the cut output.
    pub spill: Option<Spill>,
    /// How much of the output the result keeps; `None` keeps all of it, up
    /// to the 10 MB the command is killed at, for a host that cuts output
    /// itself.
    pub cut: Option<OutputLimit>,
    /// Whether a `cwd` that resolves outside the launcher's directory is
    /// refused. What the command itself does is not confined: it may still
    /// `cd` anywhere.
    pub confine: bool,
    /// The most seconds a call may ask for in `timeout_secs`; a longer
    /// timeout is cut to it.
    pub max_timeout: u64,
}

impl Default for Shell {
    /// Started on this machine as this process starts it, nothing spilled
    /// or reported live, output cut as [`OutputLimit::default`] cuts it, timeouts
    /// of at most ten minutes.
    fn default() -> Self {
        Self {
            workspace: Arc::new(Local),
            launcher: Launcher::default(),
            live: None,
            spill: None,
            cut: Some(OutputLimit::default()),
            confine: false,
            max_timeout: 600,
        }
    }
}

/// Arguments of [`Shell`].
#[derive(Clone, Debug, PartialEq, Eq, Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
pub struct ShellArgs {
    /// The command line.
    pub command: String,
    /// Seconds before the command is killed.
    pub timeout_secs: Option<u64>,
    /// The directory to run it in, relative to the working directory;
    /// default the working directory.
    pub cwd: Option<String>,
}

impl ShellArgs {
    /// Run `command`, with the default timeout, in the working directory.
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            timeout_secs: None,
            cwd: None,
        }
    }

    /// The same arguments, killing the command after `secs` seconds.
    pub fn timeout_secs(mut self, secs: u64) -> Self {
        self.timeout_secs = Some(secs);
        self
    }

    /// The same arguments, running the command in `dir`.
    pub fn in_dir(mut self, dir: impl Into<String>) -> Self {
        self.cwd = Some(dir.into());
        self
    }
}

/// What a command printed and how it ended.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ShellRun {
    /// Its output, stdout and stderr interleaved, as the [`OutputLimit`] keeps it,
    /// ending in a line saying how it ended.
    pub output: String,
    /// How it ended.
    pub status: ExitStatus,
}

impl Shell {
    /// When to pick this tool, for the system prompt.
    pub const RULES: &'static [&'static str] = &[
        "Use `shell` to build, test, run programs and use git. Each call starts a \
         fresh `sh` in the working directory: a `cd` or a variable does not carry \
         over to the next call.",
        "Do not start programs that wait for input or never end, such as editors, \
         pagers or servers in the foreground.",
    ];

    /// Runs `args` as the tool does, waiting on a thread of its own, and
    /// returns the output and how the command ended. Dropping the future
    /// kills the command's process group at once.
    pub async fn run(&self, args: ShellArgs) -> Result<ShellRun, ToolExecutionError> {
        self.execute(args).await.map(|(ran, _)| ran)
    }

    /// [`run`](Self::run), and how its output was cut.
    async fn execute(
        &self,
        args: ShellArgs,
    ) -> Result<(ShellRun, Option<OutputCut>), ToolExecutionError> {
        let timeout = Duration::from_secs(
            args.timeout_secs
                .unwrap_or(DEFAULT_TIMEOUT.min(self.max_timeout))
                .clamp(1, self.max_timeout.max(1)),
        );
        let process = self.start(&args)?;
        let _stop = KillOnDrop(process.group());
        let shell = self.clone();
        blocking(move || shell.finish(process, timeout)).await
    }
}

impl Tool for Shell {
    const NAME: &'static str = "shell";
    type Args = ShellArgs;
    type Output = String;
    type Error = ToolExecutionError;

    fn description(&self) -> String {
        let kept = match self.cut {
            Some(cut) => format!(
                "keeping the last {} lines or {} KB",
                cut.lines,
                cut.bytes / 1024
            ),
            None => "all of it".to_owned(),
        };
        format!(
            "Run a shell command with `sh -c` in the current directory, or in cwd. stdin is \
             empty. Returns the exit code and stdout and stderr interleaved, {kept}. The \
             command is killed after timeout_secs (default {}, at most {}), or once it \
             printed more than {} MB.",
            DEFAULT_TIMEOUT.min(self.max_timeout).max(1),
            self.max_timeout.max(1),
            OUTPUT_LIMIT / 1024 / 1024
        )
    }

    fn parameters(&self) -> serde_json::Value {
        args_schema::<ShellArgs>()
    }

    fn annotations(&self) -> ToolAnnotations {
        ToolAnnotations::default().with_title("Shell")
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: ShellArgs,
    ) -> Result<String, ToolExecutionError> {
        let (ran, cut) = self.execute(args).await?;
        context.insert_result(ran.status)?;
        if let Some(cut) = cut {
            context.insert_result(cut)?;
        }
        Ok(ran.output)
    }
}

impl Shell {
    /// Starts the command `args` names in the workspace, as this shell's
    /// launcher starts processes, its stdout and stderr in one pipe.
    fn start(&self, args: &ShellArgs) -> Result<Process, ToolExecutionError> {
        let base = self.launcher.cwd.as_deref();
        let dir = match &args.cwd {
            Some(dir) => {
                if self.confine {
                    self.workspace.place(base, true, dir)?;
                }
                Some(base.unwrap_or(Path::new("")).join(dir))
            }
            None => base.map(Path::to_path_buf),
        };
        let how = Spawn {
            stdin: None,
            merge_stderr: true,
        };
        let line = [OsString::from("-c"), OsString::from(&args.command)];
        self.workspace
            .spawn(&self.launcher, dir.as_deref(), OsStr::new("sh"), &line, how)
            .map_err(|error| ToolExecutionError::other(format!("could not run sh: {error}")))
    }

    /// Waits for `process` to end, at most `timeout`, keeping the end of
    /// its output as the cut says.
    fn finish(
        &self,
        mut process: Process,
        timeout: Duration,
    ) -> Result<(ShellRun, Option<OutputCut>), ToolExecutionError> {
        let mut tail = Tail {
            spill: self.spill.clone(),
            cut: self.cut,
            live: self.live.clone(),
            ..Tail::default()
        };
        let deadline = Instant::now() + timeout;
        let end = loop {
            match process.try_wait() {
                Ok(Some(status)) => break End::Exited(status),
                Ok(None) if Instant::now() >= deadline => break End::TimedOut,
                Ok(None) if tail.written > OUTPUT_LIMIT => break End::TooLong,
                Ok(None) => match process.lines().recv_timeout(POLL) {
                    Ok(line) => tail.push(line.text()),
                    Err(RecvTimeoutError::Timeout) => {}
                    // The command closed its output but still runs.
                    Err(RecvTimeoutError::Disconnected) => thread::sleep(POLL),
                },
                Err(error) => {
                    return Err(ToolExecutionError::other(format!(
                        "waiting for the command failed: {error}"
                    )));
                }
            }
        };
        // After a normal exit, something the command left running in the
        // background may still hold the output pipe open; then the rest of
        // its group is killed, at most `PIPE_GRACE` after the exit. A
        // process that left the group, such as one started with `setsid`,
        // survives the kill and may hold the pipe for good, so reading
        // stops a moment later with what came.
        if !matches!(end, End::Exited(_)) || !tail.read_rest(process.lines()) {
            process.kill().ok();
            tail.read_rest(process.lines());
        }
        let (mut output, cut) = tail.text();
        let status = match end {
            End::Exited(status) => match status.code() {
                Some(code) => {
                    output.push_str(&format!("[exit code {code}]"));
                    ExitStatus::Exited(code)
                }
                None => {
                    output.push_str("[killed by a signal]");
                    ExitStatus::Signaled
                }
            },
            End::TimedOut => {
                output.push_str(&format!(
                    "[timed out after {}s and killed]",
                    timeout.as_secs()
                ));
                ExitStatus::TimedOut
            }
            End::TooLong => {
                output.push_str(&format!(
                    "[killed after printing more than {} MB]",
                    OUTPUT_LIMIT / 1024 / 1024
                ));
                ExitStatus::Killed
            }
        };
        Ok((ShellRun { output, status }, cut))
    }
}

/// How each line [`Shell`] ends a command's output with begins, one per
/// way the command can end, as `Shell::finish` writes them.
const STATUS_STARTS: [&str; 4] = [
    "[exit code ",
    "[killed by a signal]",
    "[timed out after ",
    "[killed after printing more than ",
];

/// Splits a `shell` call's output ([`ShellRun::output`]) into what the
/// command printed, each line with its `\n`, and the line after it saying
/// how the command ended; `text` whole and `None` when it does not end in
/// such a line. A command's own output never ends the text, so a line it
/// printed that reads like one is kept as printed.
///
/// ```
/// use rig_tools::split_status;
///
/// assert_eq!(split_status("built\n[exit code 0]"), ("built\n", Some("[exit code 0]")));
/// assert_eq!(split_status("no status\n"), ("no status\n", None));
/// ```
pub fn split_status(text: &str) -> (&str, Option<&str>) {
    let start = text.rfind('\n').map_or(0, |newline| newline + 1);
    match text.split_at_checked(start) {
        Some((printed, last))
            if last.ends_with(']') && STATUS_STARTS.iter().any(|at| last.starts_with(at)) =>
        {
            (printed, Some(last))
        }
        _ => (text, None),
    }
}

/// How the wait for the command ended.
enum End {
    /// It exited.
    Exited(std::process::ExitStatus),
    /// It ran past its timeout.
    TimedOut,
    /// It printed more than [`OUTPUT_LIMIT`].
    TooLong,
}

/// The end of a command's output.
#[derive(Default)]
struct Tail {
    /// Every byte received.
    written: u64,
    /// The last bytes.
    bytes: VecDeque<u8>,
    /// How much is kept; `None` keeps everything.
    cut: Option<OutputLimit>,
    /// Whether earlier bytes were dropped.
    dropped: bool,
    /// Whole lines dropped before `bytes`.
    cut_lines: usize,
    /// Where the whole output goes once some is cut.
    spill: Option<Spill>,
    /// The file it goes to, from the first cut on, and its path.
    spilled: Option<(PathBuf, File)>,
    /// Where each piece is reported as it comes.
    live: Option<Live>,
}

impl Tail {
    /// Appends `piece`, keeping only the last bytes the cut keeps, and
    /// reports it live.
    fn push(&mut self, piece: &str) {
        if let Some(live) = &self.live {
            live.line(piece);
        }
        let chunk = piece.as_bytes();
        self.written += chunk.len() as u64;
        if let Some((_, file)) = &mut self.spilled
            && file.write_all(chunk).is_err()
        {
            self.spilled = None;
        }
        self.bytes.extend(chunk);
        let kept = self.cut.map_or(usize::MAX, |cut| cut.bytes);
        let excess = self.bytes.len().saturating_sub(kept);
        if excess > 0 {
            self.spill();
            self.cut_lines += self.bytes.drain(..excess).filter(|&b| b == b'\n').count();
            self.dropped = true;
        }
    }

    /// Receives output until the pipe's end, for at most [`PIPE_GRACE`].
    /// Returns whether the end came: every process holding the pipe exited
    /// or closed it.
    fn read_rest(&mut self, lines: &Receiver<OutputLine>) -> bool {
        let deadline = Instant::now() + PIPE_GRACE;
        loop {
            match lines.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(line) => self.push(line.text()),
                Err(RecvTimeoutError::Disconnected) => return true,
                Err(RecvTimeoutError::Timeout) => return false,
            }
        }
    }

    /// Starts keeping the whole output, from the bytes kept so far, which
    /// are all of it before the first cut. Tried once.
    fn spill(&mut self) {
        let Some(spill) = self.spill.take() else {
            return;
        };
        let (front, back) = self.bytes.as_slices();
        self.spilled = spill.create("shell").ok().filter(|(_, file)| {
            let mut file = file;
            file.write_all(front)
                .and_then(|()| file.write_all(back))
                .is_ok()
        });
    }

    /// The kept output, as the cut keeps it, after a line saying how many
    /// earlier lines were cut and which file has them all, and that cut as
    /// a fact.
    fn text(mut self) -> (String, Option<OutputCut>) {
        let (front, back) = self.bytes.as_slices();
        let output = String::from_utf8_lossy(&[front, back].concat()).into_owned();
        let mut cut = self.cut_lines;
        // After a cut, the first line kept is only the end of a line.
        let output = match output.split_once('\n') {
            Some((_, rest)) if self.dropped => {
                cut += 1;
                rest
            }
            _ => output.as_str(),
        };
        let (kept, dropped) = match self.cut {
            Some(limits) => limits.tail(output),
            None => (output, 0),
        };
        cut += dropped;
        let mut text = String::new();
        let mut fact = None;
        if cut > 0 {
            self.spill();
            match &self.spilled {
                Some((path, _)) => text.push_str(&format!(
                    "[{cut} earlier lines cut; all of the output is in {}: read or search it]\n",
                    path.display()
                )),
                None => text.push_str(&format!("[{cut} earlier lines cut]\n")),
            }
            let kept_lines = kept.lines().count();
            fact = Some(OutputCut {
                kept: KeptEnd::End,
                kept_lines,
                total_lines: kept_lines + cut,
                spill: self
                    .spilled
                    .as_ref()
                    .map(|(path, _)| path.to_string_lossy().into_owned()),
            });
        }
        for line in kept.lines() {
            text.push_str(line);
            text.push('\n');
        }
        (text, fact)
    }
}

#[cfg(test)]
mod tests;
