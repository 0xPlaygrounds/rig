//! The `shell` tool.

use std::collections::VecDeque;
use std::ffi::OsStr;
use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, RecvTimeoutError};
use std::thread;
use std::time::{Duration, Instant};

use rig_core::completion::ToolAnnotations;
use rig_core::tool::{ExitStatus, Tool, ToolContext, ToolExecutionError, args_schema};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::fs::place;
use crate::process::{KillOnDrop, Launcher, OutputLine, Process, Spawn};
use crate::{Cut, Spill, blocking};

const DEFAULT_TIMEOUT: u64 = 120;
const MAX_TIMEOUT: u64 = 600;
/// How long output pipes may stay open after the command exits before the
/// rest of its process group is killed, and how long reading goes on after
/// that kill.
const PIPE_GRACE: Duration = Duration::from_millis(100);
/// How often the command's state is checked while no output arrives.
const POLL: Duration = Duration::from_millis(20);
/// Output past which the command is killed. While it runs, only what the
/// [`Cut`] keeps is held in memory.
const OUTPUT_LIMIT: u64 = 10 * 1024 * 1024;

/// Runs a command with `sh -c` in its own session and process group, as its
/// [`Launcher`] starts processes. Publishes how the command ended
/// ([`ExitStatus`]) beside its output.
#[derive(Clone, Debug)]
pub struct Shell {
    /// How the command is started: a wrapper, a directory, its variables.
    pub launcher: Launcher,
    /// Where output that is cut is kept whole, named in the cut output.
    pub spill: Option<Spill>,
    /// How much of the output the result keeps; `None` keeps all of it, up
    /// to the 10 MB the command is killed at, for a host that cuts output
    /// itself.
    pub cut: Option<Cut>,
    /// Whether a `cwd` that resolves outside the launcher's directory is
    /// refused. What the command itself does is not confined: it may still
    /// `cd` anywhere.
    pub confine: bool,
}

impl Default for Shell {
    /// Started as this process starts it, nothing spilled, output cut as
    /// [`Cut::default`] cuts it.
    fn default() -> Self {
        Self {
            launcher: Launcher::default(),
            spill: None,
            cut: Some(Cut::default()),
            confine: false,
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
    /// Its output, stdout and stderr interleaved, as the [`Cut`] keeps it,
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
        let timeout = Duration::from_secs(
            args.timeout_secs
                .unwrap_or(DEFAULT_TIMEOUT)
                .clamp(1, MAX_TIMEOUT),
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
             command is killed after timeout_secs (default {DEFAULT_TIMEOUT}, at most \
             {MAX_TIMEOUT}), or once it printed more than {} MB.",
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
        let ran = self.run(args).await?;
        context.insert_result(ran.status)?;
        Ok(ran.output)
    }
}

impl Shell {
    /// Starts the command `args` names, as this shell's launcher starts
    /// processes, its stdout and stderr in one pipe.
    fn start(&self, args: &ShellArgs) -> Result<Process, ToolExecutionError> {
        let launcher = match &args.cwd {
            Some(dir) => {
                if self.confine {
                    place(self.launcher.cwd.as_deref(), true, dir)?;
                }
                let base = self.launcher.cwd.as_deref().unwrap_or(Path::new(""));
                self.launcher.clone().in_dir(base.join(dir))
            }
            None => self.launcher.clone(),
        };
        let how = Spawn {
            stdin: None,
            merge_stderr: true,
        };
        launcher
            .spawn("sh", [OsStr::new("-c"), OsStr::new(&args.command)], how)
            .map_err(|error| ToolExecutionError::other(format!("could not run sh: {error}")))
    }

    /// Waits for `process` to end, at most `timeout`, keeping the end of
    /// its output as the cut says.
    fn finish(
        &self,
        mut process: Process,
        timeout: Duration,
    ) -> Result<ShellRun, ToolExecutionError> {
        let mut tail = Tail {
            spill: self.spill.clone(),
            cut: self.cut,
            ..Tail::default()
        };
        let deadline = Instant::now() + timeout;
        let end = loop {
            match process.try_wait() {
                Ok(Some(status)) => break End::Exited(status),
                Ok(None) if Instant::now() >= deadline => break End::TimedOut,
                Ok(None) if tail.written > OUTPUT_LIMIT => break End::TooLong,
                Ok(None) => match process.lines().recv_timeout(POLL) {
                    Ok(line) => tail.push(line.text().as_bytes()),
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
        let mut output = tail.text();
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
        Ok(ShellRun { output, status })
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
    cut: Option<Cut>,
    /// Whether earlier bytes were dropped.
    dropped: bool,
    /// Whole lines dropped before `bytes`.
    cut_lines: usize,
    /// Where the whole output goes once some is cut.
    spill: Option<Spill>,
    /// The file it goes to, from the first cut on, and its path.
    spilled: Option<(PathBuf, File)>,
}

impl Tail {
    /// Appends `chunk`, keeping only the last bytes the cut keeps.
    fn push(&mut self, chunk: &[u8]) {
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
                Ok(line) => self.push(line.text().as_bytes()),
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
    /// earlier lines were cut and which file has them all.
    fn text(mut self) -> String {
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
        if cut > 0 {
            self.spill();
            match &self.spilled {
                Some((path, _)) => text.push_str(&format!(
                    "[{cut} earlier lines cut; all of the output is in {}: read or search it]\n",
                    path.display()
                )),
                None => text.push_str(&format!("[{cut} earlier lines cut]\n")),
            }
        }
        for line in kept.lines() {
            text.push_str(line);
            text.push('\n');
        }
        text
    }
}

#[cfg(test)]
mod tests;
