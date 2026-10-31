//! The `shell` tool.

use std::borrow::Cow;
use std::collections::VecDeque;
use std::ffi::{OsStr, OsString};
use std::fs::File;
use std::io::Write as _;
use std::ops::Range;
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
use crate::{Excerpt, Live, MAX_LINE_CHARS, OutputLimit, Source, Spill, blocking, shell_path};

/// Output past which the command is killed. While it runs, only what the
/// [`OutputLimit`] keeps, and each stream's unfinished line, is held in
/// memory.
const OUTPUT_LIMIT: u64 = 10 * 1024 * 1024;

/// Runs a command with `sh -c` in its own session and process group, in its
/// [`Workspace`], as its [`Launcher`] starts processes. Its stdout and
/// stderr come back interleaved line by line, as each line ends, without
/// terminal escape sequences such as colours; output past the
/// [`OutputLimit`] keeps its first and last lines and the first and last
/// errors and failures from between. Publishes how the
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
}

impl Default for Shell {
    /// Started on this machine as this process starts it, nothing spilled
    /// or reported live, output cut as [`OutputLimit::default`] cuts it.
    fn default() -> Self {
        Self {
            workspace: Arc::new(Local),
            launcher: Launcher::default(),
            live: None,
            spill: None,
            cut: Some(OutputLimit::default()),
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
    /// Seconds before the command is stopped; none waits until it ends.
    pub timeout_secs: Option<u64>,
    /// The directory to run it in, relative to the working directory;
    /// default the working directory.
    pub cwd: Option<String>,
}

impl ShellArgs {
    /// Run `command` until it ends, in the working directory.
    pub fn new(command: impl Into<String>) -> Self {
        Self {
            command: command.into(),
            timeout_secs: None,
            cwd: None,
        }
    }

    /// The same arguments, stopping the command after `secs` seconds.
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
        let timeout = args
            .timeout_secs
            .map(|secs| Duration::from_secs(secs.max(1)));
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
                "; past {} lines or {} KB, its first and last lines and the lines with errors \
                 or failures from between",
                cut.lines,
                cut.bytes / 1024
            ),
            None => String::new(),
        };
        format!(
            "Run a shell command with `sh -c` in the current directory, or in cwd, and wait \
             until it ends, however long it takes; set timeout_secs only for a command that may \
             hang. Returns the exit code and stdout and stderr interleaved line by line{kept}. Each call starts a fresh `sh` with an \
             empty stdin. Do not run what never ends, such as a server, in the foreground, and \
             do not assume a git repository."
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
        timeout: Option<Duration>,
    ) -> Result<(ShellRun, Option<OutputCut>), ToolExecutionError> {
        let mut kept = Kept {
            spill: self.spill.clone(),
            cut: self.cut,
            live: self.live.clone(),
            local: self.workspace.host().is_none(),
            ..Kept::default()
        };
        let deadline = timeout.and_then(|timeout| Instant::now().checked_add(timeout));
        let end = process
            .run(deadline, |line| {
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
                    output.push_str(&format!("{EXIT_PREFIX}{code}]"));
                    ExitStatus::Exited(code)
                }
                None => {
                    output.push_str(&killed(status, process.remote()));
                    ExitStatus::Signaled
                }
            },
            Ended::TimedOut => {
                let secs = timeout.unwrap_or_default().as_secs();
                output.push_str(&format!(
                    "{TIMEOUT_PREFIX}{secs}s and stopped{}",
                    there(&process)
                ));
                if silent {
                    output.push_str(SILENT);
                }
                output.push(']');
                ExitStatus::TimedOut
            }
            Ended::Stopped => {
                output.push_str(&format!(
                    "{LIMIT_PREFIX}{} MB{}]",
                    OUTPUT_LIMIT / 1024 / 1024,
                    there(&process)
                ));
                ExitStatus::Killed
            }
        };
        Ok((ShellRun { output, status }, cut))
    }
}

/// What a status line adds when the command printed nothing at all.
const SILENT: &str = ". It printed nothing. If it should have, it may hold its output back when \
                      that is not a terminal (`python3 -u` or `stdbuf -oL` make it print as it \
                      goes); if it needs longer, give a larger `timeout_secs`";

/// Where `process` was stopped besides this machine: the machine it ran
/// the command on, whose end of the transport stops the command when this
/// end is stopped (see [`remote`](crate::remote)).
fn there(process: &Process) -> String {
    match process.remote() {
        Some(host) => format!(", on {host} too"),
        None => String::new(),
    }
}

/// How the status line of a command that exited begins.
const EXIT_PREFIX: &str = "[exit code ";
/// How the status line of a command killed by a signal begins.
const KILLED_PREFIX: &str = "[killed by ";
/// How the status line of a command stopped at its timeout begins.
const TIMEOUT_PREFIX: &str = "[timed out after ";
/// How the status line of a command stopped at the output limit begins.
const LIMIT_PREFIX: &str = "[stopped after printing more than ";

/// How each line [`Shell`] ends a command's output with begins, one per
/// way the command can end.
const STATUS_STARTS: [&str; 4] = [EXIT_PREFIX, KILLED_PREFIX, TIMEOUT_PREFIX, LIMIT_PREFIX];

/// The status line of a command that ended by a signal this tool did not
/// send (its timeout and output limit say so themselves): which signal,
/// what likely sent it, and what to do. With `remote`, the process was the
/// transport to that machine, and its end there stopped the command.
fn killed(status: std::process::ExitStatus, remote: Option<&str>) -> String {
    #[cfg(unix)]
    if let Some(signal) = std::os::unix::process::ExitStatusExt::signal(&status) {
        let (name, why) = sent(signal);
        if let Some(host) = remote {
            return format!(
                "{KILLED_PREFIX}{name} (signal {signal}), which this tool did not send: {why}. It \
                 was the connection to {host}, and the command there was stopped with it]"
            );
        }
        return format!(
            "{KILLED_PREFIX}{name} (signal {signal}), which this tool did not send: {why}. It was \
             the process this call started here, `sh` or the last program it ran; when that \
             program was a client of another machine, such as a remote shell, the command \
             there may still be running or may have finished, so check there before running \
             it again]"
        );
    }
    #[cfg(not(unix))]
    let _ = (status, remote);
    format!("{KILLED_PREFIX}a signal]")
}

/// The name of `signal` and what likely sent it.
#[cfg(unix)]
pub(crate) fn sent(signal: i32) -> (&'static str, &'static str) {
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

/// Words that make a line [`matter`](matters) wherever they are in it.
const FAILURES: [&str; 8] = [
    "FAILED",
    "FAIL:",
    "failed",
    "panicked",
    "Traceback",
    "Error:",
    "error:",
    "fatal:",
];

/// The word that keeps `line` when it comes from the cut middle of an
/// output, and where it is: an error, a failure or a panic, as compilers,
/// test runners, scripts and git print them. Warnings are not kept, so a
/// flood of them never crowds out an error.
fn matters(line: &str) -> Option<Failure> {
    let start = line.len() - line.trim_start().len();
    let rest = line.get(start..).unwrap_or_default();
    if let Some(word) = ["error", "Error"]
        .into_iter()
        .find(|word| rest.starts_with(word))
    {
        return Some(Failure {
            word,
            span: start..start + word.len(),
        });
    }
    FAILURES
        .into_iter()
        .filter_map(|word| {
            let at = line.find(word)?;
            Some(Failure {
                word,
                span: at..at + word.len(),
            })
        })
        .min_by_key(|failure| failure.span.start)
}

/// A word that makes a line [`matter`](matters), and where it is.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Failure {
    word: &'static str,
    span: Range<usize>,
}

/// `line` without the terminal escape sequences in it, such as the colours
/// a program prints even when its output is no terminal, or a link's or a
/// window title's markup: the text between them stays.
fn plain(line: &str) -> Cow<'_, str> {
    if !line.contains('\x1b') {
        return Cow::Borrowed(line);
    }
    let mut text = String::with_capacity(line.len());
    let mut chars = line.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            text.push(c);
            continue;
        }
        match chars.next() {
            // A control sequence: parameters, then one final character.
            Some('[') => {
                let _ = chars.by_ref().find(|c| ('\x40'..='\x7e').contains(c));
            }
            // A string, such as a link or a title, up to BEL or `ESC \`.
            Some(']' | 'P' | 'X' | '^' | '_') => {
                while let Some(c) = chars.next() {
                    if c == '\x07' || (c == '\x1b' && chars.next_if_eq(&'\\').is_some()) {
                        break;
                    }
                }
            }
            // Intermediate characters, then one final character.
            Some(' '..='/') => {
                let _ = chars.by_ref().find(|c| !(' '..='/').contains(c));
            }
            _ => {}
        }
    }
    Cow::Owned(text)
}

/// A line of the output [`Kept`] holds whole, or cut short once the
/// output is cut.
struct Line {
    /// Its number, from 0.
    at: usize,
    /// Its text, or the start of it kept, without its line end.
    body: String,
    /// What marks its cut, once it was cut short.
    mark: String,
    /// Its length, without its line end.
    bytes: usize,
    /// The word that makes it [`matter`](matters), when the output may be
    /// cut and it does.
    matters: Option<Failure>,
}

impl Line {
    /// The line as kept, ending in `\n`.
    fn text(&self) -> String {
        format!("{}{}\n", self.body, self.mark)
    }

    /// The bytes of [`text`](Self::text).
    fn len(&self) -> usize {
        self.body.len() + self.mark.len() + 1
    }

    /// Whether it was cut short.
    fn long(&self) -> bool {
        !self.mark.is_empty()
    }

    /// Cuts the line at `chars` characters, marked, when it is longer: a
    /// line cut before the word that makes it matter says where that word
    /// is, with the command that shows it in the file `spill` (quoted for
    /// `sh`) that has the whole output, when there is one.
    fn shorten(&mut self, chars: usize, spill: Option<&str>) {
        let span = self.matters.as_ref().map(|failure| failure.span.clone());
        let cut = Excerpt::within(&self.body, self.bytes, span, chars);
        if !cut.is_cut() {
            return;
        }
        let what = match &self.matters {
            Some(failure) => format!("`{}`", failure.word),
            None => String::new(),
        };
        let source = spill.map(|path| Source {
            path,
            line: self.at + 1,
            before: 0,
        });
        let (kept, mark) = (cut.kept().len(), cut.mark(&what, source));
        self.body.truncate(kept);
        self.mark = mark;
    }
}

/// What [`Shell`] keeps of a command's output while it runs, line by line:
/// all of it while it fits the [`OutputLimit`]; past it, the first lines
/// (half of the limit), the lines that [`matter`](matters) from the
/// middle with a few after each (at most a quarter: the first of them in
/// one eighth, the last in the other) and the last lines (the rest), each
/// cut at [`MAX_LINE_CHARS`]. The whole output goes to the spill file from
/// the first cut on.
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
    /// The first lines, once the output was cut.
    head: Vec<Line>,
    /// The bytes of `head`.
    head_bytes: usize,
    /// The first lines that matter from the middle.
    picked: Vec<Line>,
    /// The bytes of `picked`.
    picked_bytes: usize,
    /// The last lines that matter from the middle once `picked` is full,
    /// with the lines after them.
    later: VecDeque<Line>,
    /// The bytes of `later`.
    later_bytes: usize,
    /// Whether some lines that matter, or lines after them, were cut.
    missed: bool,
    /// How many lines after the last one that matters are kept too.
    context: usize,
    /// The last lines: all of them before the first cut.
    tail: VecDeque<Line>,
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
    /// Whether the command runs on this machine, where the spill file is.
    local: bool,
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

    /// Takes the next whole `line`, its terminal escape sequences dropped
    /// ([`plain`]) and its secrets redacted, reporting it live and spilling
    /// it, and cuts what no longer fits: no key it printed is shown,
    /// spilled or returned.
    fn line(&mut self, line: String) {
        let line = plain(&line);
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
        let body = line.strip_suffix('\n').unwrap_or(&line);
        let mut line = Line {
            at: self.lines,
            matters: self.cut.and_then(|_| matters(body)),
            bytes: body.trim_end_matches('\r').len(),
            body: body.to_owned(),
            mark: String::new(),
        };
        let spill = self.spill_path();
        if self.cutting {
            line.shorten(MAX_LINE_CHARS, spill.as_deref());
        }
        self.tail_bytes += line.len();
        self.tail.push_back(line);
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
            let spill = self.spill_path();
            for line in &mut self.tail {
                line.shorten(MAX_LINE_CHARS, spill.as_deref());
            }
            self.tail_bytes = self.tail.iter().map(Line::len).sum();
            while self.head.len() < limit.lines / 2
                && let Some(line) = self.tail.pop_front()
            {
                if self.head_bytes + line.len() > limit.bytes / 2 {
                    self.tail.push_front(line);
                    break;
                }
                self.head_bytes += line.len();
                self.tail_bytes -= line.len();
                self.head.push(line);
            }
        }
        // The last lines take what the first and the picked ones leave.
        let spill = self.spill_path();
        while self.tail.len()
            > limit
                .lines
                .saturating_sub(self.head.len() + self.picked.len() + self.later.len())
            || self.tail_bytes
                > limit
                    .bytes
                    .saturating_sub(self.head_bytes + self.picked_bytes + self.later_bytes)
        {
            let Some(mut line) = self.tail.pop_front() else {
                break;
            };
            self.tail_bytes -= line.len();
            if line.matters.is_some() {
                self.context = CONTEXT + 1;
            }
            if self.context > 0 {
                self.context -= 1;
                line.shorten(MAX_PICKED, spill.as_deref());
                self.pick(limit, line);
            }
        }
    }

    /// Keeps `line` from the middle: in `picked` while it has room, then
    /// in `later`, which drops its oldest lines that matter, with the lines
    /// after them, to make room for new ones.
    fn pick(&mut self, limit: OutputLimit, line: Line) {
        let (lines, bytes) = (limit.lines / 8, limit.bytes / 8);
        if self.later.is_empty()
            && self.picked.len() < lines
            && self.picked_bytes + line.len() <= bytes
        {
            self.picked_bytes += line.len();
            self.picked.push(line);
            return;
        }
        self.later_bytes += line.len();
        self.later.push_back(line);
        while self.later.len() > lines || self.later_bytes > bytes {
            self.missed = true;
            while let Some(line) = self.later.pop_front() {
                self.later_bytes -= line.len();
                if self.later.front().is_none_or(|line| line.matters.is_some()) {
                    break;
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
                .try_for_each(|line| file.write_all(line.text().as_bytes()))
                .is_ok()
        });
    }

    /// The file that has the whole output, quoted for `sh`, when a command
    /// run in `shell` would find it: it is written on this machine, so only
    /// when the commands run here too.
    fn spill_path(&self) -> Option<String> {
        let (path, _) = self.spilled.as_ref().filter(|_| self.local)?;
        shell_path(&path.to_string_lossy())
    }

    /// The kept output: after a cut, a line saying how much was cut and
    /// how to read it in the file that has it all, then the kept lines with
    /// a `[… n lines cut (lines a-b) …]` line where lines are missing,
    /// numbered as in that file; and that cut as a fact.
    fn text(mut self) -> (String, Option<OutputCut>) {
        self.flush();
        let kept_lines = self.head.len() + self.picked.len() + self.later.len() + self.tail.len();
        let errors = match self.missed {
            true => "the first and last lines with errors or failures",
            false => "lines with errors or failures",
        };
        let cut = self.lines - kept_lines;
        let kept: Vec<&Line> = (self.head.iter().chain(&self.picked))
            .chain(&self.later)
            .chain(&self.tail)
            .collect();
        let long: Vec<usize> = kept
            .iter()
            .filter(|line| line.long())
            .map(|line| line.at)
            .collect();
        let mut text = String::new();
        let mut fact = None;
        if cut > 0 || !long.is_empty() {
            let mut what = Vec::new();
            if cut > 0 {
                what.push(format!(
                    "{cut} of {} lines cut, but for {errors}",
                    self.lines
                ));
            }
            if !long.is_empty() {
                what.push(format!(
                    "{} long {} cut short",
                    long.len(),
                    if long.len() == 1 { "line" } else { "lines" }
                ));
            }
            let what = what.join(", and ");
            match &self.spilled {
                Some((path, _)) => {
                    // The first line cut, from 1, and how many follow it.
                    let mut next = 0;
                    let (first, gap) = kept
                        .iter()
                        .find_map(|line| {
                            let gap = (line.at > next).then(|| (next + 1, line.at - next));
                            next = line.at + 1;
                            gap
                        })
                        .unwrap_or((next + 1, cut));
                    let read = match long.first() {
                        Some(at) if cut == 0 => format!(
                            "with offset {} and limit 1 for the first long line, whose note says \
                             how to read on in it",
                            at + 1
                        ),
                        _ => format!(
                            "with offset {first} and limit {} for the first lines cut",
                            gap.min(crate::MAX_LINES)
                        ),
                    };
                    text.push_str(&format!(
                        "[{what}; all of the output is in {}: `read` it there {read}, rather \
                         than running the command again]\n",
                        path.display(),
                    ));
                }
                None => text.push_str(&format!("[{what}]\n")),
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
        for line in kept {
            if line.at > next {
                text.push_str(&format!(
                    "[… {} lines cut (lines {}-{}) …]\n",
                    line.at - next,
                    next + 1,
                    line.at
                ));
            }
            text.push_str(&line.text());
            next = line.at + 1;
        }
        (text, fact)
    }
}

#[cfg(test)]
mod tests;
