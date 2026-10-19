//! The `shell` tool.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::time::{Duration, Instant};

use rig_core::completion::ToolAnnotations;
use rig_core::redact;
use rig_core::tool::{
    ExitStatus, KeptEnd, OutputCut, Tool, ToolContext, ToolExecutionError, args_schema,
};
use schemars::JsonSchema;
use serde::Deserialize;

use crate::process::{Ended, KillOnDrop, Launcher, OutputLine, Process, Spawn};
use crate::workspace::{Local, Workspace};
use crate::{Live, OutputLimit, Spill, blocking, clip};

const DEFAULT_TIMEOUT: u64 = 120;
/// Output past which the command is killed. While it runs, only what the
/// [`OutputLimit`] keeps, and each stream's unfinished line, is held in
/// memory.
const OUTPUT_LIMIT: u64 = 10 * 1024 * 1024;

/// Runs a command with `sh -c` in its own session and process group, in its
/// [`Workspace`], as its [`Launcher`] starts processes. Its stdout and
/// stderr come back interleaved line by line, as each line ends; output
/// past the [`OutputLimit`] keeps its first and last lines and the errors
/// and failures from between. Publishes how the
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
    /// How long a call with the default timeout may print nothing before
    /// it is stopped, such as a loop that never ends; one that names its
    /// `timeout_secs` may be quiet as long as that.
    pub quiet: Duration,
}

impl Default for Shell {
    /// Started on this machine as this process starts it, nothing spilled
    /// or reported live, output cut as [`OutputLimit::default`] cuts it, timeouts
    /// of at most ten minutes, stopped after a minute of printing nothing
    /// unless the call names its timeout.
    fn default() -> Self {
        Self {
            workspace: Arc::new(Local),
            launcher: Launcher::default(),
            live: None,
            spill: None,
            cut: Some(OutputLimit::default()),
            confine: false,
            max_timeout: 600,
            quiet: Duration::from_secs(60),
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
    /// Its output, stdout and stderr interleaved line by line in the order
    /// the lines came, as the [`OutputLimit`] keeps it, ending in a line
    /// saying how it ended.
    pub output: String,
    /// How it ended.
    pub status: ExitStatus,
}

impl Shell {
    /// Runs `args` as the tool does, waiting on a thread of its own, and
    /// returns the output and how the command ended. Dropping the future
    /// stops the command's process group at once: `SIGTERM`, then
    /// `SIGKILL` after [`STOP_GRACE`](crate::process::STOP_GRACE).
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
        let quiet = match args.timeout_secs {
            None => Some(self.quiet).filter(|quiet| *quiet < timeout),
            Some(_) => None,
        };
        let process = self.start(&args)?;
        let _stop = KillOnDrop(process.group());
        let shell = self.clone();
        blocking(move || shell.finish(process, timeout, quiet)).await
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
                "; past {} lines or {} KB, its first and last lines and the lines with errors \
                 or failures from between",
                cut.lines,
                cut.bytes / 1024
            ),
            None => String::new(),
        };
        format!(
            "Run a shell command with `sh -c` in the current directory, or in cwd. stdin is \
             empty. Returns the exit code and stdout and stderr interleaved line by line{kept}. \
             The command is killed after timeout_secs (default {}, at most {}), or once it \
             printed more than {} MB; with the default, also once it printed nothing for {}s, \
             so name timeout_secs for a build or wait that is quiet for long; a quick command \
             needs none. Use it to build, test, run \
             programs and use git. Each call starts a fresh `sh`: a `cd` or a variable does not \
             carry over to the next call. Do not start programs that wait for input or never end, such as editors, \
             pagers or servers in the foreground. Do not assume a git repository: a failed \
             `git` command ends a `&&` chain. Check formatting only of files you changed.",
            DEFAULT_TIMEOUT.min(self.max_timeout).max(1),
            self.max_timeout.max(1),
            OUTPUT_LIMIT / 1024 / 1024,
            self.quiet.as_secs()
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
    /// launcher starts processes, its stdout and stderr in pipes of their
    /// own.
    fn start(&self, args: &ShellArgs) -> Result<Process, ToolExecutionError> {
        let base = self.launcher.cwd.as_deref();
        let dir = match &args.cwd {
            Some(dir) => {
                if self.confine {
                    self.workspace.place(base, true, dir)?;
                }
                Some(
                    self.workspace
                        .join(base.unwrap_or(Path::new("")), Path::new(dir)),
                )
            }
            None => base.map(Path::to_path_buf),
        };
        let how = Spawn {
            stdin: None,
            merge_stderr: false,
        };
        let line = [OsString::from("-c"), OsString::from(&args.command)];
        self.workspace
            .spawn(&self.launcher, dir.as_deref(), OsStr::new("sh"), &line, how)
            .map_err(|error| ToolExecutionError::other(format!("could not run sh: {error}")))
    }

    /// Waits for `process` to end, at most `timeout`, keeping its output as
    /// the cut says.
    fn finish(
        &self,
        mut process: Process,
        timeout: Duration,
        quiet: Option<Duration>,
    ) -> Result<(ShellRun, Option<OutputCut>), ToolExecutionError> {
        let mut kept = Kept {
            spill: self.spill.clone(),
            cut: self.cut,
            live: self.live.clone(),
            ..Kept::default()
        };
        let deadline = Instant::now().checked_add(timeout);
        let end = process
            .run(deadline, quiet, |line| {
                kept.push(&line);
                kept.written <= OUTPUT_LIMIT
            })
            .map_err(|error| {
                ToolExecutionError::other(format!("waiting for the command failed: {error}"))
            })?;
        let silent = kept.written == 0;
        let (mut output, cut) = kept.text();
        let status = match end {
            Ended::Exited(status) => match status.code() {
                Some(code) => {
                    output.push_str(&format!("[exit code {code}]"));
                    ExitStatus::Exited(code)
                }
                None => {
                    output.push_str(&killed(status));
                    ExitStatus::Signaled
                }
            },
            Ended::TimedOut => {
                let secs = timeout.as_secs();
                output.push_str(&format!(
                    "[timed out after {secs}s and stopped: {}",
                    stopped(&process)
                ));
                if silent {
                    output.push_str(SILENT);
                }
                if secs < self.max_timeout {
                    output.push_str(&format!(
                        ". Pass timeout_secs up to {} to wait longer",
                        self.max_timeout
                    ));
                }
                output.push(']');
                ExitStatus::TimedOut
            }
            Ended::Quiet => {
                output.push_str(&format!(
                    "[printed nothing for {}s, so it was stopped: {}",
                    quiet.unwrap_or_default().as_secs(),
                    stopped(&process)
                ));
                if silent {
                    output.push_str(SILENT);
                }
                output.push_str(
                    ". A command that is stuck, such as one waiting for input or looping \
                     forever, needs fixing, not a longer wait; only one known to be quiet for \
                     long, such as a build, needs timeout_secs]",
                );
                ExitStatus::TimedOut
            }
            Ended::Stopped => {
                output.push_str(&format!(
                    "[killed after printing more than {} MB: {}]",
                    OUTPUT_LIMIT / 1024 / 1024,
                    stopped(&process)
                ));
                ExitStatus::Killed
            }
        };
        Ok((ShellRun { output, status }, cut))
    }
}

/// What a status line adds when the command printed nothing at all.
const SILENT: &str = ". It printed nothing: a program whose output is not a terminal may hold \
                      it back until it ends, so make it print as it goes (such as \
                      `python3 -u` or `stdbuf -oL`) to see how far it got";

/// What this tool did to stop `process`, which it [stopped](Process::stop),
/// and, when that was a transport, what it means for the command there.
fn stopped(process: &Process) -> String {
    let grace = crate::process::STOP_GRACE.as_secs();
    let how = if !cfg!(unix) {
        "it was killed".to_owned()
    } else if process.forced() {
        format!("it got SIGTERM, then SIGKILL {grace}s later as it still ran")
    } else {
        "it got SIGTERM and ended".to_owned()
    };
    match process.remote() {
        Some(host) => format!(
            "{how}. That was the transport on this machine; whether the command on {host} \
             stopped too is the transport's doing, so check there before running it again"
        ),
        None => how,
    }
}

/// How each line [`Shell`] ends a command's output with begins, one per
/// way the command can end, as `Shell::finish` writes them.
const STATUS_STARTS: [&str; 5] = [
    "[exit code ",
    "[killed by ",
    "[timed out after ",
    "[printed nothing for ",
    "[killed after printing more than ",
];

/// The status line of a command that ended by a signal this tool did not
/// send (its timeout, quiet limit and output limit say so themselves):
/// which signal, what likely sent it, and what to do.
fn killed(status: std::process::ExitStatus) -> String {
    #[cfg(unix)]
    if let Some(signal) = std::os::unix::process::ExitStatusExt::signal(&status) {
        let (name, why) = sent(signal);
        return format!(
            "[killed by {name} (signal {signal}), which this tool did not send: {why}. It was \
             the process this call started here, `sh` or the last program it ran; when that \
             program was a client of another machine, such as a remote shell, the command \
             there may still be running or may have finished, so check there before running \
             it again]"
        );
    }
    #[cfg(not(unix))]
    let _ = status;
    "[killed by a signal]".to_owned()
}

/// The name of `signal` and what likely sent it.
#[cfg(unix)]
fn sent(signal: i32) -> (&'static str, &'static str) {
    match signal {
        libc::SIGHUP => ("SIGHUP", "its terminal or session went away"),
        libc::SIGINT => ("SIGINT", "something on this machine interrupted it"),
        libc::SIGQUIT => ("SIGQUIT", "something on this machine asked it to quit"),
        libc::SIGILL => ("SIGILL", "it crashed"),
        libc::SIGTRAP => ("SIGTRAP", "it crashed, or a debugger stopped it"),
        libc::SIGABRT => ("SIGABRT", "it aborted itself, as on a failed assertion"),
        libc::SIGBUS => ("SIGBUS", "it crashed"),
        libc::SIGFPE => ("SIGFPE", "it crashed on an arithmetic error"),
        libc::SIGKILL => (
            "SIGKILL",
            "something on this machine killed it, such as the out-of-memory killer or a `kill`",
        ),
        libc::SIGSEGV => ("SIGSEGV", "it crashed"),
        libc::SIGPIPE => ("SIGPIPE", "it wrote to a pipe that nothing read any more"),
        libc::SIGTERM => ("SIGTERM", "something on this machine asked it to stop"),
        _ => ("a signal", "something on this machine sent it"),
    }
}

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

/// The longest line [`Kept`] waits for the end of; a longer one is broken
/// into lines of this length.
const MAX_HELD: usize = 1024 * 1024;
/// How many lines after a line that [`matters`] are kept with it, such as
/// an error's place in the code or a failed assertion's values.
const CONTEXT: usize = 3;
/// The longest a line kept from the cut middle is shown.
const MAX_PICKED: usize = 300;

/// Whether `line` from the cut middle of an output is kept: an error, a
/// failure or a panic, as compilers, test runners and scripts print them.
/// Warnings are not, so a flood of them never crowds out an error.
fn matters(line: &str) -> bool {
    let start = line.trim_start();
    start.starts_with("error")
        || start.starts_with("Error")
        || [
            "FAILED",
            "FAIL:",
            "failed",
            "panicked",
            "Traceback",
            "Error:",
        ]
        .iter()
        .any(|word| line.contains(word))
}

/// What [`Shell`] keeps of a command's output while it runs, line by line:
/// all of it while it fits the [`OutputLimit`]; past it, the first lines
/// (half of the limit), the lines that [`matter`](matters) from the
/// middle with a few after each (at most a quarter) and the last lines
/// (the rest). The whole output goes to the spill file from the first cut
/// on.
#[derive(Default)]
struct Kept {
    /// Every byte received.
    written: u64,
    /// How much is kept; `None` keeps everything.
    cut: Option<OutputLimit>,
    /// The start of each stream's line still being written: stdout's,
    /// then stderr's.
    held: [String; 2],
    /// How many lines came.
    lines: usize,
    /// The first lines, by number, once the output was cut.
    head: Vec<(usize, String)>,
    /// The bytes of `head`.
    head_bytes: usize,
    /// Lines that matter from the middle, by number.
    picked: Vec<(usize, String)>,
    /// The bytes of `picked`.
    picked_bytes: usize,
    /// How many lines after the last one that matters are kept too.
    context: usize,
    /// The last lines, by number: all of them before the first cut.
    tail: VecDeque<(usize, String)>,
    /// The bytes of `tail`.
    tail_bytes: usize,
    /// Whether some output was cut, and `head` is final.
    cutting: bool,
    /// Where the whole output goes once some is cut.
    spill: Option<Spill>,
    /// The file it goes to, from the first cut on, and its path.
    spilled: Option<(PathBuf, File)>,
    /// Where each line is reported as it comes.
    live: Option<Live>,
    /// Whether the output is inside a private key block, whose lines are
    /// left out until its end line.
    in_key: bool,
}

impl Kept {
    /// Takes `piece` of the output: a whole line goes in at once, one
    /// still being written waits for its end, so a line of one stream is
    /// never split by the other's.
    fn push(&mut self, piece: &OutputLine) {
        self.written += piece.text().len() as u64;
        let held = match piece {
            OutputLine::Out(_) => 0,
            OutputLine::Err(_) => 1,
        };
        let Some(line) = self.held.get_mut(held) else {
            return;
        };
        line.push_str(piece.text());
        if line.ends_with('\n') || line.len() >= MAX_HELD {
            let line = std::mem::take(line);
            self.line(line);
        }
    }

    /// Takes the lines both streams were still writing, once the output
    /// ended.
    fn flush(&mut self) {
        for held in 0..self.held.len() {
            if let Some(line) = self.held.get_mut(held).map(std::mem::take)
                && !line.is_empty()
            {
                self.line(line);
            }
        }
    }

    /// Takes the next whole `line`, its secrets redacted, reporting it
    /// live and spilling it, and cuts what no longer fits: no key it
    /// printed is shown, spilled or returned.
    fn line(&mut self, line: String) {
        let mut line = self.redact(&line).into_owned();
        if line.is_empty() {
            return;
        }
        if !line.ends_with('\n') {
            line.push('\n');
        }
        if let Some(live) = &self.live {
            live.line(&line);
        }
        if let Some((_, file)) = &mut self.spilled
            && file.write_all(line.as_bytes()).is_err()
        {
            self.spilled = None;
        }
        self.tail_bytes += line.len();
        self.tail.push_back((self.lines, line));
        self.lines += 1;
        let Some(limit) = self.cut else {
            return;
        };
        if !self.cutting {
            if self.tail.len() <= limit.lines && self.tail_bytes <= limit.bytes {
                return;
            }
            self.cutting = true;
            self.spill();
            while self.head.len() < limit.lines / 2
                && let Some((at, line)) = self.tail.pop_front()
            {
                if self.head_bytes + line.len() > limit.bytes / 2 {
                    self.tail.push_front((at, line));
                    break;
                }
                self.head_bytes += line.len();
                self.tail_bytes -= line.len();
                self.head.push((at, line));
            }
        }
        let (lines, bytes) = (limit.lines / 4, limit.bytes / 4);
        // The last lines take what the first and the picked ones leave.
        while self.tail.len()
            > limit
                .lines
                .saturating_sub(self.head.len() + self.picked.len())
            || self.tail_bytes
                > limit
                    .bytes
                    .saturating_sub(self.head_bytes + self.picked_bytes)
        {
            let Some((at, line)) = self.tail.pop_front() else {
                break;
            };
            self.tail_bytes -= line.len();
            if matters(&line) {
                self.context = CONTEXT + 1;
            }
            if self.context > 0 {
                self.context -= 1;
                let mut line = clip(line.trim_end_matches('\n'), MAX_PICKED).to_owned();
                line.push('\n');
                if self.picked.len() < lines && self.picked_bytes + line.len() <= bytes {
                    self.picked_bytes += line.len();
                    self.picked.push((at, line));
                }
            }
        }
    }

    /// `piece`, a line, with its secrets redacted ([`redact::text`]). A
    /// private key block spans lines: its first is redacted to its end, and
    /// the lines after it, through its end line, are left out.
    fn redact<'a>(&mut self, piece: &'a str) -> Cow<'a, str> {
        if self.in_key {
            self.in_key = !piece.contains("-----END ");
            return Cow::Borrowed("");
        }
        let mut redacted = redact::text(piece);
        if let Cow::Owned(line) = &mut redacted {
            self.in_key = redact::find(piece).any(|secret| {
                secret.kind == redact::Kind::PrivateKey
                    && piece
                        .get(secret.range)
                        .is_some_and(|key| !key.contains("-----END "))
            });
            // A key redacted to the line's end took its newline.
            if piece.ends_with('\n') && !line.ends_with('\n') {
                line.push('\n');
            }
        }
        redacted
    }

    /// Starts keeping the whole output, from the lines kept so far, which
    /// are all of it before the first cut. Tried once.
    fn spill(&mut self) {
        let Some(spill) = self.spill.take() else {
            return;
        };
        self.spilled = spill.create("shell").ok().filter(|(_, file)| {
            let mut file = file;
            self.tail
                .iter()
                .try_for_each(|(_, line)| file.write_all(line.as_bytes()))
                .is_ok()
        });
    }

    /// The kept output: after a cut, a line saying how much was cut and
    /// how to read it in the file that has it all, then the kept lines with
    /// a `[… n lines cut (lines a-b) …]` line where lines are missing,
    /// numbered as in that file; and that cut as a fact.
    fn text(mut self) -> (String, Option<OutputCut>) {
        self.flush();
        let kept_lines = self.head.len() + self.picked.len() + self.tail.len();
        let cut = self.lines - kept_lines;
        let mut text = String::new();
        let mut fact = None;
        if cut > 0 {
            match &self.spilled {
                Some((path, _)) => {
                    // The first line cut, from 1, and how many follow it.
                    let first = self.head.len() + 1;
                    let gap = self
                        .picked
                        .first()
                        .or(self.tail.front())
                        .map_or(cut, |(at, _)| at + 1 - first);
                    text.push_str(&format!(
                        "[{cut} of {} lines cut, but for lines with errors or failures; all of the \
                         output is in {}: `read` it there with offset {first} and limit {} for \
                         the first lines cut, rather than running the command again]\n",
                        self.lines,
                        path.display(),
                        gap.min(crate::MAX_LINES),
                    ));
                }
                None => text.push_str(&format!(
                    "[{cut} of {} lines cut, but for lines with errors or failures]\n",
                    self.lines
                )),
            }
            fact = Some(OutputCut {
                kept: KeptEnd::Ends,
                kept_lines,
                total_lines: self.lines,
                spill: self
                    .spilled
                    .as_ref()
                    .map(|(path, _)| path.to_string_lossy().into_owned()),
            });
        }
        let mut next = 0;
        for (at, line) in self.head.iter().chain(&self.picked).chain(&self.tail) {
            if *at > next {
                text.push_str(&format!(
                    "[… {} lines cut (lines {}-{at}) …]\n",
                    at - next,
                    next + 1
                ));
            }
            text.push_str(line);
            next = at + 1;
        }
        (text, fact)
    }
}

#[cfg(test)]
mod tests;
