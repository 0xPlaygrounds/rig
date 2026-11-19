//! `rig build`: regenerate the agent project, check that every plugin uses
//! the agent's Bevy version, compile with cargo's output on stderr (or only
//! in the log, for a headless start: [`BuildOutput`]), and stage the new
//! binary for the next start. Before it is staged, a build
//! describes what its plugins add ([`described`]): a build whose plugins
//! clash, or that does not describe itself, is rejected. Everything a build
//! prints also goes to [`Home::build_log`], and a failure names its reason
//! and first compiler errors ([`BuildFailure`]).
//!
//! Every process it starts comes from the launch ([`Launch`]), so none
//! inherits a secret.

use std::collections::BTreeSet;
use std::fmt;
use std::fs::File;
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant, UNIX_EPOCH};

use rig_harness_protocol::config::toml_edit::Item;
use rig_harness_protocol::fs;
use rig_harness_protocol::launch::{Launch, Purpose};
use rig_harness_protocol::{AgentDescription, BUILD_SAYS, Home, RunDir, config, first_errors};

use super::project::{self, Manifest, PACKAGE, RigSource, workspace_dependency};
use super::{BEVY_VERSION, COMMIT, LONG_VERSION, Result, locks};

/// Whether a build that is already known gets staged again.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Staging {
    /// Stage every build, so a build rolled back for crashing is retried.
    Always,
    /// Stage only a build that is not already the good one, staged for any
    /// launcher, or rejected for crashing at startup. At startup that keeps
    /// a rolled-back build from being retried until something changes.
    OnlyNew,
    /// Stage nothing: `rig plugin check --build` only checks that the
    /// agent builds.
    Never,
}

/// Where a build's output goes besides [`Home::build_log`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BuildOutput {
    /// To stderr as it comes, with cargo's progress bar on a terminal.
    Live,
    /// To the log only, for a headless start whose stderr a script reads:
    /// one line says so once cargo starts real work, so a long build does
    /// not look stuck, and a build with nothing to do says nothing.
    LogOnly,
}

/// How long cargo may wait for another cargo's lock before a
/// [`BuildOutput::LogOnly`] build says it builds: a wait this short is
/// not a build.
const BLOCKED_QUIET: Duration = Duration::from_secs(3);

/// Lines of a failed build's output a [`BuildFailure`] keeps, from its
/// first error on.
const ERROR_LINES: usize = 40;

/// Which step of a build failed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FailedAt {
    /// Reading what to build: `settings.toml` or `RIG_SOURCE` is wrong, so
    /// no build of these settings exists until the user fixes them.
    Settings,
    /// Building it, or the new build's start.
    Build,
}

/// Why a build failed: the step, the reason, the first compiler errors,
/// and the log with the whole output. Its [`Display`](fmt::Display) is one line, for
/// a terminal that showed cargo's output already; [`Self::details`] adds
/// the errors.
#[derive(Debug)]
pub struct BuildFailure {
    /// The step that failed.
    pub at: FailedAt,
    /// What failed, such as `cargo could not build the agent (exit
    /// status: 101)` or a `settings.toml` mistake.
    pub reason: String,
    /// The output from its first error on, at most [`ERROR_LINES`] lines;
    /// empty when no line starts with `error`.
    pub errors: Vec<String>,
    /// The build's whole output: [`Home::build_log`], or [`safe_log`].
    pub log: PathBuf,
    /// Whether the reason says the cause ([`Explained`]).
    explained: bool,
}

impl BuildFailure {
    /// The reason, then the first errors.
    pub fn details(&self) -> String {
        std::iter::once(&self.reason)
            .chain(&self.errors)
            .map(String::as_str)
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The [`BUILD_SAYS`] line for a failure whose cause rig says: its reason on
    /// one line, without the closing period, as a notice goes on after it,
    /// then the first error it kept.
    fn explanation(&self) -> Option<String> {
        self.explained.then(|| {
            let mut line = BUILD_SAYS.to_owned();
            for (index, part) in self.reason.lines().map(str::trim).enumerate() {
                if index > 0 {
                    line.push_str(if line.ends_with(':') { " " } else { "; " });
                }
                line.push_str(part);
            }
            line.truncate(line.trim_end_matches('.').len());
            if let Some(first) = self.errors.first() {
                line.push_str(": ");
                line.push_str(first.trim());
            }
            line
        })
    }

    /// What to tell about it after a build whose output went as `output`
    /// went: its one line after output shown live, else its details and
    /// the log, which nothing showed yet.
    pub fn told(&self, output: BuildOutput) -> String {
        match output {
            BuildOutput::Live => self.to_string(),
            BuildOutput::LogOnly => {
                format!("{}\n(whole output: {})", self.details(), self.log.display())
            }
        }
    }
}

impl fmt::Display for BuildFailure {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.reason)?;
        // The first error line, when the reason is not one already.
        if let Some(first) = self.errors.first()
            && !self.reason.contains(first.as_str())
        {
            write!(f, ": {first}")?;
        }
        write!(f, " (whole output: {})", self.log.display())
    }
}

impl std::error::Error for BuildFailure {}

/// A failed build rig says the cause of in its own words. When sure, cargo's
/// errors, in the build log, are not repeated after it; a likely cause keeps
/// them.
#[derive(Debug)]
struct Explained {
    cause: String,
    sure: bool,
}

impl Explained {
    fn sure(cause: String) -> Self {
        let sure = true;
        Self { cause, sure }
    }
}

impl fmt::Display for Explained {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.cause)
    }
}

impl std::error::Error for Explained {}

/// Writes the agent project with the plugins `settings.toml` turns on,
/// checks its Bevy version, builds it, and copies the binary to `staged` as
/// `staging` says, after putting back lock files a stopped safe build kept
/// ([`put_back`]). The build's output goes to [`Home::build_log`], and to
/// stderr as `output` says.
pub fn compile(
    launch: &Launch,
    staged: &Path,
    staging: Staging,
    output: BuildOutput,
) -> std::result::Result<(), BuildFailure> {
    let home = &launch.home();
    let mut log = BuildLog::create(home.build_log(), output, Box::new(std::io::stderr()));
    put_back_logged(home, &mut log);
    let wanted = (|| -> Result<_> {
        let plugins = config::chosen(home)?;
        local_crates(home, &plugins)?;
        Ok((plugins, RigSource::detect(launch.source.as_deref())?))
    })();
    let (plugins, source) = match wanted {
        Ok(wanted) => wanted,
        Err(failure) => return Err(log.fail(FailedAt::Settings, &*failure)),
    };
    let built = compile_logged(launch, &plugins, &source, &mut log)
        .and_then(|()| stage(launch, staged, staging));
    match built {
        Ok(()) => {
            log.record("build succeeded");
            Ok(())
        }
        Err(failure) => Err(log.fail(FailedAt::Build, &*failure)),
    }
}

/// Writes the agent project with `plugins` from `source`, checks its Bevy
/// version and builds it into [`artifact`], logging to `log`.
fn compile_logged(
    launch: &Launch,
    plugins: &config::Plugins,
    source: &RigSource,
    log: &mut BuildLog,
) -> Result<()> {
    // A build that cannot be made, not a settings mistake: the previous
    // build still starts where a failed build lets it.
    source.check(COMMIT)?;
    let home = &launch.home();
    let generate = || project::generate(home, plugins, source);
    if let Some(warning) = super::plugin::skill::set_up(home, generate)? {
        log.say(&warning);
    }
    let rig_crates = project::rig_crates(plugins, source)?;
    let rig_version = project::rig_version(source);
    let copies = check_bevy(launch, plugins, &rig_crates, &rig_version, log)?;
    // `/reload` shows the latest of these lines and of cargo's.
    log.say("Compiling the agent…");
    let mut command = cargo(launch);
    command.args(["build", "--package", PACKAGE]);
    log.show_progress(launch, &mut command);
    let status = log.run(command)?;
    if status.success() {
        return Ok(());
    }
    let failed = format!("cargo could not build the agent ({status})");
    Err(match likely_bevy_cause(launch, plugins, &copies, log) {
        Some(advice) => Explained {
            cause: format!("Likely cause: {advice} {failed}"),
            sure: false,
        }
        .into(),
        None => failed.into(),
    })
}

/// The build `rig --safe` starts: rig's default plugins alone.
pub fn safe(home: &Home) -> PathBuf {
    home.bin().join("safe")
}

/// What [`safe`] was built by and from: the launcher's version and the
/// source of rig's crates.
fn safe_version(home: &Home) -> PathBuf {
    home.bin().join("safe.version")
}

/// The whole output of the last safe build, kept apart from
/// [`Home::build_log`] so the log of a failed build of the user's plugins
/// stays.
pub fn safe_log(home: &Home) -> PathBuf {
    home.root().join("safe-build.log")
}

/// The stamp [`safe_version`] holds for a safe build by this launcher from
/// `source`.
fn safe_stamp(source: &RigSource) -> String {
    let source = match source {
        RigSource::Local(checkout) => format!("path {}", checkout.display()),
        RigSource::Registry => "crates.io".to_owned(),
    };
    format!("{LONG_VERSION}\n{source}\n")
}

/// Whether a safe start can run [`safe`] as it is, with no cargo: only
/// from crates.io, where the launcher's version pins every crate, when the
/// safe build has this launcher's [`safe_stamp`] and writes a description
/// report (an empty file that runs as a shell script writes none). A
/// checkout's files may have changed since, so a safe build from one is
/// never reused this way.
fn safe_is_current(launch: &Launch, source: &RigSource) -> bool {
    let home = &launch.home();
    matches!(source, RigSource::Registry)
        && fs::read_to_string(safe_version(home)).is_ok_and(|kept| kept == safe_stamp(source))
        && describe(launch, &safe(home), None).is_ok_and(|report| !report.trim().is_empty())
}

/// Makes [`safe`] the build of rig's default plugins, ignoring `[plugins]`
/// in `settings.toml`, unless [`safe_is_current`]; from a checkout, cargo
/// rebuilds only what changed. It builds in the agent project, keeping the
/// project's lock files for the next build of the user's plugins
/// ([`keep_locks`], [`put_back`]), and logs to [`safe_log`] and to stderr as
/// `output` says.
pub fn compile_safe(launch: &Launch, output: BuildOutput) -> std::result::Result<(), BuildFailure> {
    let home = &launch.home();
    let source = RigSource::detect(launch.source.as_deref());
    if source
        .as_ref()
        .is_ok_and(|source| safe_is_current(launch, source))
    {
        return Ok(());
    }
    let mut log = BuildLog::create(safe_log(home), output, Box::new(std::io::stderr()));
    let source = match source {
        Ok(source) => source,
        Err(failure) => return Err(log.fail(FailedAt::Settings, &*failure)),
    };
    log.say("Building rig's default plugins for safe mode…");
    // Not built at all when the user's lock files cannot be kept.
    if let Err(failure) = keep_locks(home) {
        let failure: Box<dyn std::error::Error> =
            format!("the project's lock files could not be kept: {failure}").into();
        return Err(log.fail(FailedAt::Build, &*failure));
    }
    let built = (|| -> Result<()> {
        compile_logged(launch, &project::defaults(), &source, &mut log)?;
        let artifact = artifact(home);
        described(launch, &artifact)
            .map_err(|why| format!("the safe build does not start: {why}"))?;
        let temporary = home.bin().join("safe.tmp");
        fs::create_dir_all(home.bin())?;
        copy(&artifact, &temporary)?;
        fs::rename(&temporary, safe(home))?;
        fs::write(safe_version(home), safe_stamp(&source))?;
        Ok(())
    })();
    put_back_logged(home, &mut log);
    match built {
        Ok(()) => {
            log.record("build succeeded");
            Ok(())
        }
        Err(failure) => Err(log.fail(FailedAt::Build, &*failure)),
    }
}

/// The agent project's lock files a safe build changes: its `Cargo.lock`
/// and the checkout lock that one was seeded from.
const LOCKS: [&str; 2] = ["Cargo.lock", "Cargo.lock.seed"];

/// Where [`keep_locks`] keeps the project's lock files while a safe build
/// runs. It is there only while one runs, or after one was stopped. It
/// sits beside `project/`, not in it, so keeping the locks of a new home
/// makes no project, and the build still sees a new home.
fn kept_locks(home: &Home) -> PathBuf {
    home.root().join("kept-locks")
}

/// A used [`kept_locks`], renamed before it is removed, so that a removal
/// cut short leaves no partial backup to read.
fn retired_locks(home: &Home) -> PathBuf {
    home.root().join("kept-locks.old")
}

/// Copies the project's [`LOCKS`] that exist to [`kept_locks`], putting
/// back first what a stopped safe build left there. The copy is made
/// aside and renamed into place whole, and a used one is renamed away
/// before it is removed, so [`kept_locks`] always holds a complete set.
fn keep_locks(home: &Home) -> Result<()> {
    put_back(home)?;
    let keeping = home.root().join("kept-locks.tmp");
    fs::optional(fs::remove_dir_all(&keeping))?;
    fs::create_dir_all(&keeping)?;
    for name in LOCKS {
        let path = home.project().join(name);
        if fs::optional(fs::metadata(&path))?.is_some() {
            fs::copy(&path, keeping.join(name))?;
        }
    }
    fs::rename(&keeping, kept_locks(home))?;
    Ok(())
}

/// Puts the lock files in [`kept_locks`] back as they were, removing one
/// that was not there, then retires the kept directory; nothing when there
/// is none. Every build calls it first, so the user's locks come back after
/// a safe build that a signal stopped. When any of them cannot be put back,
/// the seed is removed, so the next build reseeds from the checkout rather
/// than trust a lock the safe build left.
fn put_back(home: &Home) -> Result<()> {
    let retired = retired_locks(home);
    fs::optional(fs::remove_dir_all(&retired))?;
    let kept = kept_locks(home);
    if fs::optional(fs::metadata(&kept))?.is_none() {
        return Ok(());
    }
    for name in LOCKS {
        let path = home.project().join(name);
        let restored = match fs::optional(fs::metadata(kept.join(name))) {
            Ok(Some(_)) => fs::copy(kept.join(name), &path).map(drop),
            Ok(None) => fs::optional(fs::remove_file(&path)).map(drop),
            Err(why) => Err(why),
        };
        if let Err(why) = restored {
            fs::optional(fs::remove_file(home.project().join("Cargo.lock.seed"))).ok();
            return Err(why.into());
        }
    }
    fs::rename(&kept, &retired)?;
    fs::remove_dir_all(&retired)?;
    Ok(())
}

/// [`put_back`], warning on stderr and in `log` when it fails: the next
/// build of the user's plugins tries again, and may resolve anew.
fn put_back_logged(home: &Home, log: &mut BuildLog) {
    if let Err(why) = put_back(home) {
        let warning = format!(
            "warning: the project's lock files from before the safe build could not be put \
             back from {}: {why}",
            kept_locks(home).display()
        );
        eprintln!("{warning}");
        log.record(&warning);
    }
}

/// Checks that every `[[plugins.extra]]` crate that is on and comes from a
/// `path` has a `Cargo.toml` that reads, and says why when one does not.
fn local_crates(home: &Home, plugins: &config::Plugins) -> Result<()> {
    for extra in plugins.extras_on() {
        if let config::Source::Path(path) = &extra.source
            && let Err(why) = Manifest::read(path)
        {
            return Err(format!(
                "{} [[plugins.extra]] `{}`: {why}; fix the crate or its `path`, \
                 or take the table out",
                home.settings().display(),
                extra.plugin,
            )
            .into());
        }
    }
    Ok(())
}

/// `rig build`, holding the root's build lock. It always stages, so it also
/// retries a build that was rolled back. Run by an agent's `/reload`
/// (`RIG_SESSION` set), it stages for that agent's launcher alone. The error
/// of a failure rig explained is its [`BUILD_SAYS`] line, after the log's path.
pub fn build(launch: &Launch) -> Result<()> {
    let home = &launch.home();
    let _lock = locks::hold_build_lock(home, BuildOutput::Live)?;
    let staged = match &launch.carry.session {
        Some(session) => home.staged_for(session),
        None => home.staged(),
    };
    compile(launch, &staged, Staging::Always, BuildOutput::Live).map_err(|failure| {
        match failure.explanation() {
            Some(line) => {
                eprintln!("whole output: {}", failure.log.display());
                line.into()
            }
            None => failure.into(),
        }
    })
}

/// `rig plugin check --build`, holding the root's build lock: builds the
/// agent like [`build`] but stages nothing, and returns its report. Fails,
/// as a start would refuse, when a table of the home's settings does not
/// read as the plugin that reads it expects.
pub fn check(launch: &Launch) -> Result<AgentDescription> {
    let home = &launch.home();
    let _lock = locks::hold_build_lock(home, BuildOutput::Live)?;
    compile(launch, &home.staged(), Staging::Never, BuildOutput::Live)?;
    let binary = artifact(home);
    let report = described(launch, &binary)?;
    describe(launch, &binary, Some(&home.settings()))?;
    Ok(report)
}

/// The binary cargo builds the agent into.
pub fn artifact(home: &Home) -> PathBuf {
    home.target()
        .join("debug")
        .join(format!("{PACKAGE}{}", std::env::consts::EXE_SUFFIX))
}

/// cargo in the agent project, with stdout discarded and stderr piped, for
/// [`BuildLog::run`] to pass on. Without colors, so the log and the first
/// errors read as plain text.
fn cargo(launch: &Launch) -> Command {
    let mut command = launch.build_child(Purpose::Cargo);
    command
        // The project's own `.cargo/config.toml` names the target directory.
        .env_remove("CARGO_TARGET_DIR")
        .env("CARGO_TERM_COLOR", "never")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    command
}

/// Keeps cargo's progress bar on a terminal, which cargo no longer sees
/// through the pipe; whether it did. A caller's own settings win.
fn show_progress(launch: &Launch, command: &mut Command) -> bool {
    if std::io::stderr().is_terminal()
        && std::env::var_os("CARGO_TERM_PROGRESS_WHEN").is_none()
        && let Some(width) = terminal_width(launch)
    {
        command
            .env("CARGO_TERM_PROGRESS_WHEN", "always")
            .env("CARGO_TERM_PROGRESS_WIDTH", width.to_string());
        return true;
    }
    false
}

/// The terminal's width in columns: `COLUMNS`, else `stty size`.
fn terminal_width(launch: &Launch) -> Option<u16> {
    if let Some(columns) = std::env::var("COLUMNS")
        .ok()
        .and_then(|columns| columns.parse().ok())
    {
        return Some(columns);
    }
    let tty = fs::open(File::options().read(true), "/dev/tty").ok()?;
    let output = launch
        .build_child(Purpose::Stty)
        .arg("size")
        .stdin(tty)
        .stderr(Stdio::null())
        .output()
        .ok()?;
    String::from_utf8_lossy(&output.stdout)
        .split_whitespace()
        .nth(1)?
        .parse()
        .ok()
}

/// [`Home::build_log`] while a build runs: every line it shows, without
/// cargo's progress bar.
struct BuildLog {
    path: PathBuf,
    /// `None` when the file could not be written; the build goes on.
    file: Option<File>,
    lines: Vec<String>,
    output: BuildOutput,
    /// Where it shows what `output` shows: stderr, but in tests.
    shown: Box<dyn Write>,
    /// Whether a [`BuildOutput::LogOnly`] build said it works.
    announced: bool,
    /// Since when cargo waits for another cargo's lock, without saying so
    /// yet.
    blocked_since: Option<Instant>,
    /// Whether what it shows is a terminal that draws cargo's progress
    /// bar, whose redrawn lines it erases.
    erase: bool,
    /// Whether the last byte passed on was a `\r`, which a `\n` may end.
    after_cr: bool,
    /// What the last `cargo tree` said besides its progress, which
    /// [`Self::end_segment`] keeps back.
    held: Vec<u8>,
}

impl BuildLog {
    /// Starts the log at `path` over, showing on `shown` what `output` says.
    fn create(path: PathBuf, output: BuildOutput, shown: Box<dyn Write>) -> Self {
        let file = (path.parent().map_or(Ok(()), fs::create_dir_all))
            .and_then(|()| {
                fs::open(
                    File::options().write(true).create(true).truncate(true),
                    &path,
                )
            })
            .inspect_err(|failure| eprintln!("warning: could not keep the build log: {failure}"))
            .ok();
        Self {
            path,
            file,
            lines: Vec::new(),
            output,
            shown,
            announced: false,
            blocked_since: None,
            erase: false,
            after_cr: false,
            held: Vec::new(),
        }
    }

    /// Shows `line` when the output is live, and logs it.
    fn say(&mut self, line: &str) {
        if self.output == BuildOutput::Live {
            writeln!(self.shown, "{line}").ok();
        }
        self.record(line);
    }

    /// Logs `line`.
    fn record(&mut self, line: &str) {
        if let Some(file) = &mut self.file
            && writeln!(file, "{line}").is_err()
        {
            self.file = None;
        }
        self.lines.push(line.to_owned());
    }

    /// Has `command`, cargo, draw its progress bar when the output is live
    /// on a terminal. A Windows console may not take the erase sequence,
    /// which cargo itself writes on Unix only.
    fn show_progress(&mut self, launch: &Launch, command: &mut Command) {
        self.erase =
            cfg!(unix) && self.output == BuildOutput::Live && show_progress(launch, command);
    }

    /// Runs `command`, cargo, whose stderr is piped, in a process group of
    /// its own ([`Build`]): passes its stderr on as it comes and logs each
    /// of its lines.
    fn run(&mut self, command: Command) -> Result<ExitStatus> {
        let mut build = Build::spawn(command)?;
        self.pass_on(&mut build.child, false);
        Ok(build.child.wait()?)
    }

    /// Passes `child`'s stderr on when the output is live, and logs its
    /// lines. cargo redraws its progress bar after `\r`, so `\r` ends a
    /// line too; the bar is not logged. Through a pipe cargo does not
    /// erase the bar's line before a shorter one, which would keep the
    /// bar's tail on screen, so on a terminal each `\r` but a `\r\n` erases
    /// it. With `hold` it takes whole lines, keeping back all but progress
    /// ([`Self::end_segment`]).
    fn pass_on(&mut self, child: &mut Child, hold: bool) {
        // A wait of an earlier command is over.
        self.blocked_since = None;
        let Some(stderr) = child.stderr.take() else {
            return;
        };
        // Read on a thread, so a wait for cargo's lock is announced once it
        // passes its time even while cargo prints nothing.
        let chunks = chunks(stderr);
        let mut segment: Vec<u8> = Vec::new();
        loop {
            let chunk = match self.blocked_since {
                Some(since) => {
                    let left = (since + BLOCKED_QUIET).saturating_duration_since(Instant::now());
                    match chunks.recv_timeout(left) {
                        Ok(chunk) => chunk,
                        Err(mpsc::RecvTimeoutError::Timeout) => {
                            self.say_building();
                            continue;
                        }
                        Err(mpsc::RecvTimeoutError::Disconnected) => break,
                    }
                }
                None => match chunks.recv() {
                    Ok(chunk) => chunk,
                    Err(_) => break,
                },
            };
            if self.output == BuildOutput::Live && !hold {
                self.show(&chunk);
            }
            for &byte in &chunk {
                if byte == b'\r' || byte == b'\n' {
                    self.end_segment(&segment, &[byte], hold);
                    segment.clear();
                } else {
                    segment.push(byte);
                }
            }
        }
        self.end_segment(&segment, &[], hold);
    }

    /// Logs `line`, which `end` ended. With `hold`, only cargo's progress is
    /// shown and logged, so a long first resolve does not look stuck. The
    /// rest is held: a warning comes again from `cargo build`, and rig may
    /// explain an error.
    fn end_segment(&mut self, line: &[u8], end: &[u8], hold: bool) {
        if hold && !is_progress(line) {
            self.held.extend_from_slice(line);
            self.held.extend_from_slice(end);
            return;
        }
        if hold && self.output == BuildOutput::Live {
            self.show(line);
            self.show(end);
        }
        self.record_segment(line);
    }

    /// Logs what [`Self::end_segment`] held, for a `cargo tree` that failed,
    /// and returns it.
    fn log_held(&mut self) -> Vec<u8> {
        let held = std::mem::take(&mut self.held);
        for line in held.split(|&byte| byte == b'\n') {
            self.record_segment(line);
        }
        held
    }

    /// Shows `chunk`, erasing after each `\r` that no `\n` follows when
    /// [`Self::erase`], as soon as the byte after it shows.
    fn show(&mut self, chunk: &[u8]) {
        let mut start = 0;
        for (at, &byte) in chunk.iter().enumerate() {
            if std::mem::take(&mut self.after_cr) && byte != b'\n' {
                let before = chunk.get(start..at).unwrap_or_default();
                self.shown.write_all(before).ok();
                self.shown.write_all(b"\x1b[K").ok();
                start = at;
            }
            self.after_cr = self.erase && byte == b'\r';
        }
        self.shown
            .write_all(chunk.get(start..).unwrap_or_default())
            .ok();
        self.shown.flush().ok();
    }

    fn record_segment(&mut self, segment: &[u8]) {
        let line = String::from_utf8_lossy(segment);
        let line = line.trim_end();
        if line.trim().is_empty() || line.trim_start().starts_with("Building [") {
            return;
        }
        self.announce(line);
        self.record(line);
    }

    /// Says once, for a [`BuildOutput::LogOnly`] build, that cargo works
    /// or waits, as its `line` shows: it fetches or compiles, or it has
    /// waited for another cargo's lock for [`BLOCKED_QUIET`].
    fn announce(&mut self, line: &str) {
        let working = ["Updating ", "Downloading ", "Compiling "];
        let line = line.trim_start();
        if working.iter().any(|word| line.starts_with(word)) {
            self.say_building();
        } else if line.starts_with("Blocking ") && !self.announced {
            self.blocked_since.get_or_insert_with(Instant::now);
        } else {
            // Any other line means the wait is over.
            self.blocked_since = None;
        }
    }

    /// Says, for a [`BuildOutput::LogOnly`] build, that the agent is being
    /// built, unless it said so already.
    fn say_building(&mut self) {
        self.blocked_since = None;
        if self.output == BuildOutput::LogOnly && !self.announced {
            self.announced = true;
            let log = self.path.display();
            writeln!(self.shown, "rig: building the agent (output in {log})…").ok();
        }
    }

    /// Ends the log with `failure` and makes it the [`BuildFailure`],
    /// with cargo's first errors unless rig [`Explained`] it.
    fn fail(mut self, at: FailedAt, failure: &(dyn std::error::Error + 'static)) -> BuildFailure {
        let reason = failure.to_string();
        let explained = failure.downcast_ref::<Explained>();
        let errors = match explained.is_some_and(|explained| explained.sure) {
            true => Vec::new(),
            false => first_errors(self.lines.iter().map(String::as_str), ERROR_LINES)
                .into_iter()
                .map(str::to_owned)
                .collect(),
        };
        self.record(&format!("build failed: {reason}"));
        BuildFailure {
            at,
            reason,
            errors,
            log: self.path,
            explained: explained.is_some(),
        }
    }
}

/// Copies the built binary to `staged`, keeping its modification time so
/// the copy carries its [`stamp`], once it described itself: a build that
/// does not is rejected, and the build fails. A build `rig build` staged
/// for any launcher that differs from this one is older, so it goes. A
/// good or staged build whose stamp cannot be read fails the build; a
/// rejected stamp that cannot be read is warned of and counts as none.
pub fn stage(launch: &Launch, staged: &Path, staging: Staging) -> Result<()> {
    let home = &launch.home();
    let artifact = artifact(home);
    let built = stamp(&artifact)?;
    if let Err(why) = described(launch, &artifact) {
        if staging != Staging::Never {
            fs::write(home.rejected(), &built)?;
        }
        return Err(format!("the new build does not start: {why}").into());
    }
    if staging == Staging::Never {
        return Ok(());
    }
    let shared = home.staged();
    let shared_stamp = stamp_if_any(&shared)?;
    if shared_stamp.as_ref().is_some_and(|stamp| *stamp != built) {
        fs::remove_file(&shared)?;
    }
    let built = Some(built);
    if staging == Staging::OnlyNew
        && (stamp_if_any(&home.good())? == built
            || shared_stamp == built
            || rejected(home).unwrap_or_else(|why| {
                // Only a hint against retrying: the build runs as a trial, rolled back if it crashes.
                eprintln!("warning: a build that crashed at startup may be tried again: {why}");
                None
            }) == built)
    {
        return Ok(());
    }
    let temporary = home.bin().join("staged.tmp");
    fs::create_dir_all(home.bin())?;
    copy(&artifact, &temporary)?;
    fs::rename(&temporary, staged)?;
    prune_described(home);
    Ok(())
}

/// Copies the build `from` to `to`, keeping its modification time so the
/// copy carries its [`stamp`].
fn copy(from: &Path, to: &Path) -> std::io::Result<()> {
    fs::copy(from, to)?;
    let modified = fs::metadata(from)?
        .modified()
        .map_err(fs::named("look at", from))?;
    let copied = fs::open(File::options().write(true), to)?;
    copied
        .set_modified(modified)
        .map_err(fs::named("set the time of", to))
}

/// Links the build `from` at `to`, a path with no file, else copies it
/// ([`copy`]), such as on another file system. Either way `to` carries
/// its [`stamp`] and stays when `from` is replaced.
pub fn hold(from: &Path, to: &Path) -> std::io::Result<()> {
    fs::hard_link(from, to).or_else(|_| copy(from, to))
}

/// Removes the reports of builds no file in [`Home::bin`], nor any build a
/// launcher holds ([`locks::builds`]), is a copy of.
fn prune_described(home: &Home) {
    let held = fs::read_dir(home.runs())
        .into_iter()
        .flatten()
        .flatten()
        .map(|run| locks::builds(&RunDir::at(run.path())));
    let stamps: BTreeSet<String> = std::iter::once(home.bin())
        .chain(held)
        .flat_map(|directory| fs::read_dir(directory).into_iter().flatten().flatten())
        .filter_map(|entry| stamp(&entry.path()).ok())
        .map(|stamp| home.described(&stamp))
        .filter_map(|report| Some(report.file_name()?.to_owned()))
        .map(|name| name.to_string_lossy().into_owned())
        .collect();
    let reports = home.described("-");
    let Some(directory) = reports.parent() else {
        return;
    };
    for entry in fs::read_dir(directory).into_iter().flatten().flatten() {
        if !stamps.contains(&*entry.file_name().to_string_lossy()) {
            fs::remove_file(entry.path()).ok();
        }
    }
}

/// How long a build may take to describe itself before it is killed.
const DESCRIBE_TIMEOUT: Duration = Duration::from_secs(10);

/// What the plugins of the build `binary` add to the command line: the
/// report kept for its stamp ([`Home::described`]), else the one it gives
/// when asked, which is kept. A build from before reports, which ignores
/// the request, has an empty one. Fails when the build exits with a
/// failure, such as two plugins adding one command, or takes over
/// [`DESCRIBE_TIMEOUT`].
pub fn described(launch: &Launch, binary: &Path) -> Result<AgentDescription> {
    let home = &launch.home();
    if let Some(report) = good_report(home, binary)? {
        return Ok(report);
    }
    let kept = home.described(&stamp(binary)?);
    let text = describe(launch, binary, None)?;
    let parent = kept.parent().ok_or("no directory for build reports")?;
    fs::create_dir_all(parent)?;
    // Written whole, then renamed: another launcher may read it meanwhile.
    let written = parent.join(format!(".{}.tmp", std::process::id()));
    fs::write(&written, &text)?;
    fs::rename(&written, &kept)?;
    Ok(AgentDescription::parse(&text))
}

/// The report kept for the build `binary`; `None` when there is no such
/// build or no report for it.
///
/// # Errors
///
/// When the build's stamp or its report cannot be read, naming the file.
pub fn good_report(home: &Home, binary: &Path) -> std::io::Result<Option<AgentDescription>> {
    let Some(stamp) = stamp_if_any(binary)? else {
        return Ok(None);
    };
    let text = fs::optional(fs::read_to_string(home.described(&stamp)))?;
    Ok(text.map(|text| AgentDescription::parse(&text)))
}

/// Runs `binary` with [`env::DESCRIBE`] and returns what it wrote, empty
/// when it wrote nothing. It runs with stdin closed, in a scratch
/// `RIG_HOME` of its own that is its working directory, without the
/// session, run or secrets, so a build that ignores the request cannot
/// reach the user's sessions; it is killed after [`DESCRIBE_TIMEOUT`].
/// With `settings` ([`env::DESCRIBE_SETTINGS`]) it reads that settings
/// file, and fails when a start would refuse it.
fn describe(launch: &Launch, binary: &Path, settings: Option<&Path>) -> Result<String> {
    let run = launch.home().run(std::process::id());
    let scratch = run.path().join("describe");
    fs::remove_dir_all(&scratch).ok();
    let scratch_home = scratch.join("home");
    fs::create_dir_all(&scratch_home)?;
    let report = scratch.join("report");
    let command = launch.describing(binary, &scratch_home, &report, settings);
    let result = wait_described(command, &report, DESCRIBE_TIMEOUT);
    fs::remove_dir_all(&scratch).ok();
    // A launcher with no run of its own, `rig build`, leaves none.
    fs::remove_dir(run.path()).ok();
    result
}

/// What `pipe` gives, in chunks read on a thread of its own until it ends
/// or the receiver is gone. A thread that cannot start drops the pipe, so
/// the child's writes to it fail and the receiver ends at once.
fn chunks(mut pipe: impl Read + Send + 'static) -> mpsc::Receiver<Vec<u8>> {
    let (sender, chunks) = mpsc::channel();
    let reading = std::thread::Builder::new().spawn(move || {
        let mut buffer = [0_u8; 8192];
        while let Ok(read @ 1..) = pipe.read(&mut buffer)
            && sender
                .send(buffer.get(..read).unwrap_or_default().to_vec())
                .is_ok()
        {}
    });
    reading.ok();
    chunks
}

/// Runs a describing build as a [`Build`] and waits for it, at most
/// `timeout`, and returns its `report`, or what it said on stderr when it
/// failed, on one line and without Rust's backtrace hint. Its group is
/// killed once it exits or the time is up, so nothing it started outlives
/// it or holds its stderr.
fn wait_described(mut command: Command, report: &Path, timeout: Duration) -> Result<String> {
    command.stderr(Stdio::piped());
    let mut build = Build::spawn(command)?;
    let child = &mut build.child;
    let said = child.stderr.take().map(chunks);
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break Some(status);
        }
        if Instant::now() > deadline {
            child.kill().ok();
            break None;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    #[cfg(unix)]
    stop::kill();
    child.wait().ok();
    let Some(status) = status else {
        return Err(format!(
            "it did not describe itself within {} s",
            timeout.as_secs_f32()
        )
        .into());
    };
    if status.success() {
        return Ok(fs::optional(fs::read_to_string(report))?.unwrap_or_default());
    }
    let mut errors = Vec::new();
    while let Some(left) = deadline.checked_duration_since(Instant::now())
        && let Some(Ok(chunk)) = said.as_ref().map(|said| said.recv_timeout(left))
    {
        errors.extend(chunk);
    }
    // One line, since `/reload` shows the first: a panic's message comes
    // on the line after its location.
    let said = String::from_utf8_lossy(&errors)
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with("note: run with `RUST_BACKTRACE"))
        .collect::<Vec<_>>()
        .join(" ");
    Err(if said.is_empty() {
        format!("describing it failed ({status})").into()
    } else {
        said.into()
    })
}

/// Records `trial`, a build that stopped before it was ready, as rejected,
/// so a start does not stage it again until something changes.
pub fn reject(home: &Home, trial: &Path) -> Result<()> {
    fs::write(home.rejected(), stamp(trial)?)?;
    Ok(())
}

/// Whether the last build is the one [`reject`] recorded, is not the good
/// build, and neither `staged` nor the build staged for any launcher
/// replaces it, so a start runs the previous build instead.
///
/// # Errors
///
/// When one of those builds' stamps cannot be read, naming the file.
pub fn passed_over(home: &Home, staged: &Path) -> std::io::Result<bool> {
    let Some(built) = stamp_if_any(&artifact(home))? else {
        return Ok(false);
    };
    Ok(stamp_if_any(staged)?.is_none()
        && stamp_if_any(&home.staged())?.is_none()
        && stamp_if_any(&home.good())?.is_some_and(|good| good != built)
        && rejected(home)?.is_some_and(|rejected| rejected == built))
}

/// The stamp [`reject`] recorded; `None` when none was.
fn rejected(home: &Home) -> std::io::Result<Option<String>> {
    fs::optional(fs::read_to_string(home.rejected()))
}

/// The [`stamp`] of `binary`; `None` when there is no file.
///
/// # Errors
///
/// When `binary` is there but cannot be looked at, naming it.
pub fn stamp_if_any(binary: &Path) -> std::io::Result<Option<String>> {
    fs::optional(stamp(binary))
}

/// What tells builds apart: the binary's modification time and size.
///
/// # Errors
///
/// When `binary` cannot be looked at, naming it.
pub fn stamp(binary: &Path) -> std::io::Result<String> {
    let metadata = fs::metadata(binary)?;
    let modified = metadata
        .modified()
        .map_err(fs::named("look at", binary))?
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_nanos())
        .unwrap_or_default();
    Ok(format!("{modified} {}\n", metadata.len()))
}

/// Resolves the project's graph and fails, in plain words, when a plugin
/// pulls in a `bevy_app` or `bevy_ecs` other than [`BEVY_VERSION`] or rig
/// crates other than the agent's. It runs before anything compiles, as
/// rustc's errors for a foreign Bevy would say nothing of the cause. Returns
/// the other versions of Bevy crates the graph holds beside the agent's, as
/// `name@version`: a plugin may keep those private, so they matter only to
/// [`likely_bevy_cause`].
fn check_bevy(
    launch: &Launch,
    plugins: &config::Plugins,
    rig_crates: &BTreeSet<String>,
    rig_version: &str,
    log: &mut BuildLog,
) -> Result<BTreeSet<String>> {
    let lines = tree(launch, &[], log).map_err(|failure| {
        let held = log.log_held();
        let local = plugins.extras_on().filter_map(|extra| match &extra.source {
            config::Source::Path(path) => Some(path.as_path()),
            _ => None,
        });
        let lock = fs::read_to_string(launch.home().project().join("Cargo.lock"));
        let why = unresolved(failure, &log.lines, local, &lock.unwrap_or_default());
        if log.output == BuildOutput::Live && !why.is::<Explained>() {
            log.show(&held);
        }
        why
    })?;
    check_rig(&lines, rig_crates, rig_version)?;
    let packages: Vec<(&str, &str)> = lines.iter().filter_map(|line| package(line)).collect();
    let foreign = || {
        packages
            .iter()
            .filter(|(_, version)| *version != BEVY_VERSION)
    };
    if let Some((name, version)) =
        foreign().find(|(name, _)| ["bevy_app", "bevy_ecs"].contains(name))
    {
        // Every package that depends on it, directly or not. cargo cannot
        // say when two sources have that version.
        let inverted = ["--invert", &format!("{name}@{version}")];
        let dependents = tree(launch, &inverted, log).unwrap_or_default();
        let culprit = plugins.extras_on().find(|extra| {
            (dependents.iter())
                .any(|line| package(line).is_some_and(|(name, _)| name == extra.krate))
        });
        return Err(Explained::sure(foreign_bevy_advice(culprit, version)).into());
    }
    Ok(foreign()
        .filter(|(name, _)| name.starts_with("bevy") && packages.contains(&(*name, BEVY_VERSION)))
        .map(|(name, version)| format!("{name}@{version}"))
        .collect())
}

/// What to do about a Bevy crate at the foreign `version`: addressed to
/// `plugin`, which depends on it, and in general terms without one.
fn foreign_bevy_advice(plugin: Option<&config::Extra>, version: &str) -> String {
    match plugin {
        Some(plugin) => foreign_bevy(&plugin.krate, &plugin.plugin, version),
        None => format!(
            "the agent project pulls in Bevy {version}, but this rig agent is built on Bevy \
             {BEVY_VERSION}. Every Bevy crate must take {BEVY_VERSION}, as \"{}\" does.",
            super::bevy_range()
        ),
    }
}

/// The advice for a failed build whose log shows the compiler rejecting code
/// (`error[E...]`) of a crate that depends on one of `copies`, the
/// [`check_bevy`] result, addressed to that crate's plugin. `None` for any
/// other failure, such as one in another crate or of the linker, and when
/// cargo cannot say who depends on them.
fn likely_bevy_cause(
    launch: &Launch,
    plugins: &config::Plugins,
    copies: &BTreeSet<String>,
    log: &mut BuildLog,
) -> Option<String> {
    let failed: Vec<String> = (log.lines.iter())
        .filter_map(|line| quoted(line, "error: could not compile `"))
        .filter(|name| *name != PACKAGE)
        .map(str::to_owned)
        .collect();
    let rejected = log.lines.iter().any(|line| line.starts_with("error[E"));
    if copies.is_empty() || failed.is_empty() || !rejected {
        return None;
    }
    // Without `--no-dedupe` a later copy's list leaves out what an earlier
    // one showed.
    let mut args = vec!["--no-dedupe"];
    for copy in copies {
        args.extend(["--invert", copy]);
    }
    let inverted = tree(launch, &args, log).ok()?;
    inverted.split(|line| line.is_empty()).find_map(|listed| {
        let (copy, dependents) = listed.split_first()?;
        let names: Vec<&str> = (dependents.iter())
            .filter_map(|line| Some(package(line)?.0))
            .collect();
        let culprit = failed.iter().find(|name| names.contains(&name.as_str()))?;
        let plugin = (plugins.extras_on())
            .filter(|extra| names.contains(&extra.krate.as_str()))
            .min_by_key(|extra| extra.krate != *culprit);
        Some(foreign_bevy_advice(plugin, package(copy)?.1))
    })
}

/// The error for cargo's `failure` to resolve the agent project, after the
/// output `lines`, with the plugins in `local` and the Cargo.lock `lock`:
/// [`pinned_bevy`] when cargo's first error is a [`bevy_conflict`], listing
/// it and the [`bevy_pins`], else `failure`.
fn unresolved<'a>(
    failure: Box<dyn std::error::Error>,
    lines: &[String],
    local: impl IntoIterator<Item = &'a Path>,
    lock: &str,
) -> Box<dyn std::error::Error> {
    let bevy = bevy_crates(lock);
    let Some(conflict) = bevy_conflict(lines, &bevy) else {
        return failure;
    };
    let mut pins = vec![conflict];
    pins.extend(bevy_pins(local, &bevy));
    Explained::sure(pinned_bevy(&failure, &pins)).into()
}

/// What cargo's first error among `lines` shows asking for a version of one
/// of `bevy` that is not [`BEVY_VERSION`], and who asks. That takes a version
/// conflict and nothing else: cargo writes any other reason, such as a
/// missing feature, ahead of its conflict line. The side at fault is the
/// selected package when that is a foreign version, else the requirement
/// cargo could not meet, unless it allows the agent's version. `None` for
/// any other error.
fn bevy_conflict(lines: &[String], bevy: &BTreeSet<String>) -> Option<String> {
    let mut lines = (lines.iter())
        .map(String::as_str)
        .filter(|line| !line.trim().is_empty())
        .skip_while(|line| !line.starts_with("error"));
    let name = quoted(lines.next()?, "error: failed to select a version for `")?;
    if !bevy.contains(name) {
        return None;
    }
    let by = quoted(lines.next()?, "required by package `")?;
    let allowed = lines.find(|line| line.starts_with("versions that meet the requirements `"))?;
    if lines.next()? != "all possible versions conflict with previously selected packages" {
        return None;
    }
    let selected = lines.find_map(|line| {
        let selected = quoted(line, "previously selected package `")?;
        (package(selected)?.0 == name).then_some(selected)
    })?;
    if package(selected)?.1 != BEVY_VERSION {
        let why = lines.next()?.trim().trim_start_matches("... ");
        return Some(format!("{selected}, {why}"));
    }
    // A lock narrows the list to its version, which then proves nothing.
    let (wanted_part, versions) = allowed.rsplit_once(" are: ")?;
    if wanted_part.ends_with(')') || versions.split(", ").any(|version| version == BEVY_VERSION) {
        return None;
    }
    let wanted = quoted(allowed, "requirements `")?;
    Some(format!("{name} {wanted}, required by {by}"))
}

/// What `line` has between `start`, which ends with a backtick, and the
/// next backtick.
fn quoted<'a>(line: &'a str, start: &str) -> Option<&'a str> {
    line.split_once(start)?.1.split('`').next()
}

/// The error for cargo's `failure` to resolve a graph in which plugins ask
/// for a Bevy other than [`BEVY_VERSION`], as `pins` lists.
/// It opens with the requirement to write.
fn pinned_bevy(failure: &dyn fmt::Display, pins: &[String]) -> String {
    format!(
        "a plugin must write its Bevy requirements as \"{range}\", such as `bevy_ecs = \
         {{ version = \"{range}\", default-features = false }}`, or be taken out with \
         `rig plugin remove <type>`. These ask for a Bevy that is not this agent's \
         {BEVY_VERSION}:\n  {}\n{failure}.",
        pins.join("\n  "),
        range = super::bevy_range(),
    )
}

/// The error for plugin `plugin` of crate `krate` built on Bevy `version`,
/// not [`BEVY_VERSION`]. It opens with the requirement to write.
fn foreign_bevy(krate: &str, plugin: &str, version: &str) -> String {
    format!(
        "plugin `{krate}` must write its bevy dependencies as \"{range}\" (with \
         default-features = false), the agent's Bevy, or be taken out with \
         `rig plugin remove {plugin}`. It uses Bevy {version}, and this rig agent is built on \
         Bevy {BEVY_VERSION}.",
        range = super::bevy_range(),
    )
}

/// Each requirement of one of Bevy's own crates (`bevy`, a crate named in
/// the [`bevy_crates`] of the agent's lock) that the crates in `crates`
/// build with and [`BEVY_VERSION`] does not meet, as `<its Cargo.toml>:<line>:
/// [<its table>] <the line>`: of their dependencies and build dependencies,
/// for every target, a renamed one by its `package`, and an inherited one
/// (`workspace = true`) where its workspace writes it. Requirements are read
/// in the forms `=V`, `^V`, `~V` and `V`; another form, or a manifest that
/// does not read, is not listed.
fn bevy_pins<'a>(
    crates: impl IntoIterator<Item = &'a Path>,
    bevy: &BTreeSet<String>,
) -> Vec<String> {
    let mut pins = Vec::new();
    for directory in crates {
        let Ok(manifest) = Manifest::read(directory) else {
            continue;
        };
        for (table, dependencies) in
            manifest.dependency_tables(&["dependencies", "build-dependencies"])
        {
            for (key, item) in dependencies.iter() {
                let pin = if item.get("workspace").and_then(Item::as_bool) == Some(true) {
                    workspace_dependency(&manifest, directory, key).and_then(|(workspace, item)| {
                        workspace.refused("workspace.dependencies", key, &item, bevy)
                    })
                } else {
                    manifest.refused(&table, key, item, bevy)
                };
                pins.extend(pin);
            }
        }
    }
    pins
}

/// Bevy's own crates, which [`bevy_pins`] reads: those a plugin names
/// ([`NAMED_BEVY`]), and each `bevy_*` crate the Cargo.lock `lock` has at
/// the version it has `bevy_ecs` at, whichever that is, as a home upgraded
/// from an older rig keeps its old lock until a resolution succeeds.
fn bevy_crates(lock: &str) -> BTreeSet<String> {
    let lock = lock.parse::<config::toml_edit::DocumentMut>().ok();
    let packages: Vec<(&str, &str)> = lock
        .as_ref()
        .and_then(|lock| lock.get("package"))
        .and_then(Item::as_array_of_tables)
        .into_iter()
        .flat_map(|packages| packages.iter())
        .filter_map(|package| {
            let name = package.get("name").and_then(Item::as_str)?;
            Some((name, package.get("version").and_then(Item::as_str)?))
        })
        .collect();
    let bevy = packages.iter().find(|(name, _)| *name == "bevy_ecs");
    let locked = (packages.iter())
        .filter(|(name, version)| {
            name.starts_with("bevy_") && bevy.is_some_and(|(_, bevy)| bevy == version)
        })
        .map(|(name, _)| *name);
    NAMED_BEVY
        .into_iter()
        .chain(locked)
        .map(str::to_owned)
        .collect()
}

/// The Bevy crates a plugin names: `bevy`, and those `rig plugin new`
/// writes or its comments name.
const NAMED_BEVY: [&str; 4] = ["bevy", "bevy_app", "bevy_ecs", "bevy_reflect"];

impl project::Manifest {
    /// The requirement `item` of `key` in `table` writes, as
    /// `<file>:<line>: [<table>] <the line>`, when it is of one of `bevy`
    /// (by its `package`, else `key`) and [`BEVY_VERSION`] does not meet
    /// it; `None` otherwise, or in a form not read.
    fn refused(
        &self,
        table: &str,
        key: &str,
        item: &Item,
        bevy: &BTreeSet<String>,
    ) -> Option<String> {
        let package = item.get("package").and_then(Item::as_str).unwrap_or(key);
        if !bevy.contains(package) {
            return None;
        }
        let (version, table) = match item.as_str() {
            Some(_) => (item, table.to_owned()),
            None if item.is_table() => (item.get("version")?, format!("{table}.{key}")),
            None => (item.get("version")?, table.to_owned()),
        };
        if takes_bevy(version.as_str()?)? {
            return None;
        }
        let at = version.span()?.start;
        let text = self.document.raw();
        let line = text.get(..at)?.matches('\n').count();
        let written = text.lines().nth(line)?.trim();
        Some(format!(
            "{}:{}: [{table}] {written}",
            self.file.display(),
            line + 1
        ))
    }
}

/// Whether the requirement `wanted` takes [`BEVY_VERSION`], cargo's way
/// for a `0.x` version: `=V` that version, `^V`, `~V` and `V` that minor at
/// `V` or above; `None` for another form.
fn takes_bevy(wanted: &str) -> Option<bool> {
    let numbers = |version: &str| {
        (version.trim().split('.'))
            .map(|number| number.parse::<u64>().ok())
            .collect::<Option<Vec<u64>>>()
    };
    let wanted = wanted.trim();
    let (exact, version) = match wanted.strip_prefix('=') {
        Some(version) => (true, version),
        None => (false, wanted.trim_start_matches(['^', '~'])),
    };
    let (wanted, have) = (numbers(version)?, numbers(BEVY_VERSION)?);
    let prefix = |length: usize| have.get(..length.min(have.len())) == wanted.get(..length);
    Some(if exact {
        prefix(wanted.len())
    } else {
        prefix(wanted.len().min(2)) && wanted <= have
    })
}

/// Fails, in plain words, when a plugin pulls in a second copy of one of
/// the `rig_crates` the agent uses, such as rig-harness from crates.io
/// beside the checkout's: its types would not be the agent's, which are at
/// `rig_version`.
fn check_rig(packages: &[String], rig_crates: &BTreeSet<String>, rig_version: &str) -> Result<()> {
    for name in rig_crates {
        let mut copies: Vec<&str> = packages
            .iter()
            .map(|line| line.trim_end_matches(" (*)"))
            .filter(|line| package(line).is_some_and(|(package, _)| package == name.as_str()))
            .collect();
        copies.sort_unstable();
        copies.dedup();
        if let [first, second, ..] = copies.as_slice() {
            return Err(format!(
                "the agent project has two copies of `{name}`: {first} and {second}. A plugin \
                 must use the agent's rig crates: depend on `{name} = \"{rig_version}\"` (a \
                 version, not a path or git source); with RIG_SOURCE the agent project patches \
                 that version to the checkout. `rig plugin new` writes it so."
            )
            .into());
        }
    }
    Ok(())
}

/// The lines of `cargo tree` over the agent project with `args`, one
/// package per line, as `name vVERSION (source)`. `log` passes on cargo's
/// progress and holds the rest ([`BuildLog::end_segment`]).
fn tree(launch: &Launch, args: &[&str], log: &mut BuildLog) -> Result<Vec<String>> {
    let mut command = cargo(launch);
    command
        .args(["tree", "--prefix", "none", "--format", "{p}"])
        .args(args)
        .stdout(Stdio::piped());
    log.show_progress(launch, &mut command);
    let mut build = Build::spawn(command)?;
    let child = &mut build.child;
    // Read on its own thread, so neither pipe fills while the other is read.
    let out = child.stdout.take().map(chunks);
    log.pass_on(child, true);
    // What is written next would not cover a bar left on the line.
    if log.erase {
        log.show(b"\x1b[K");
    }
    let stdout: Vec<u8> = out.into_iter().flatten().flatten().collect();
    let status = child.wait()?;
    if !status.success() {
        return Err("cargo could not resolve the agent project's dependencies".into());
    }
    log.held.clear();
    Ok(String::from_utf8_lossy(&stdout)
        .lines()
        .map(str::to_owned)
        .collect())
}

/// Whether `line` is one of cargo's status lines, a capitalized word ending
/// at column 12, such as `    Updating crates.io index`.
fn is_progress(line: &[u8]) -> bool {
    line.get(12) == Some(&b' ')
        && line.get(..12).is_some_and(|status| {
            let word = status.trim_ascii_start();
            word.first().is_some_and(u8::is_ascii_uppercase)
                && word.iter().all(u8::is_ascii_alphabetic)
        })
}

/// The name and version of a `cargo tree` line.
fn package(line: &str) -> Option<(&str, &str)> {
    let mut words = line.split_whitespace();
    let name = words.next()?;
    let version = words.next()?.strip_prefix('v')?;
    Some((name, version))
}

/// cargo or a describing build, running in a process group of its own (or
/// the launcher's, when the launcher leads a session of its own), which is
/// killed should the launcher be told to end (`SIGTERM`, `SIGINT`, `SIGHUP`)
/// meanwhile, so nothing it started outlives it. Elsewhere than unix, the
/// child alone.
struct Build {
    child: Child,
    /// The actions the build's signal handler replaced, put back when it
    /// drops.
    #[cfg(unix)]
    replaced: Vec<(libc::c_int, libc::sigaction)>,
}

impl Build {
    /// Starts `command`; an error names its program.
    fn spawn(mut command: Command) -> std::io::Result<Self> {
        let program = PathBuf::from(command.get_program());
        #[cfg(unix)]
        let build = stop::spawn(&mut command);
        #[cfg(not(unix))]
        let build = command.spawn().map(|child| Self { child });
        build.map_err(fs::named("start", &program))
    }
}

impl Drop for Build {
    fn drop(&mut self) {
        #[cfg(unix)]
        stop::forget(&self.replaced);
    }
}

/// The signals that end the launcher, while a [`Build`] runs, first kill
/// its group. A signal the launcher was started ignoring, as under `nohup`,
/// stays ignored.
#[cfg(unix)]
mod stop {
    use std::os::unix::process::CommandExt;
    use std::process::Command;
    use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

    use super::Build;
    use crate::run::stop::unix::{listen, restore};

    /// The build's process group; 0 while none runs.
    static GROUP: AtomicI32 = AtomicI32::new(0);

    /// Whether that group is the launcher's own.
    static OWN: AtomicBool = AtomicBool::new(false);

    /// cargo's own id, killed outright when the group is the launcher's.
    static CARGO: AtomicI32 = AtomicI32::new(0);

    /// Starts `command` as the leader of a new process group and makes it
    /// the group the ending signals kill. A launcher that leads a session of
    /// its own, as an agent's `/reload` starts it, keeps cargo in its own
    /// group instead, which the agent kills as a whole. The signals wait
    /// while it starts, so one that comes meanwhile finds the group. The
    /// actions it replaced go with the build; when it does not start they
    /// are not put back.
    pub(super) fn spawn(command: &mut Command) -> std::io::Result<Build> {
        // SAFETY: plain calls without arguments that can fail.
        let (leads, own) = unsafe { (libc::getsid(0) == libc::getpid(), libc::getpgrp()) };
        if !leads {
            command.process_group(0);
        }
        // SAFETY: plain calls on zeroed signal sets this function owns;
        // `on_signal` only calls async-signal-safe functions. The closure
        // runs in the forked child before `exec` and only calls
        // `pthread_sigmask`, which is async-signal-safe. The child gets the
        // default action for every caught signal (exec does).
        let mut ending: libc::sigset_t = unsafe { std::mem::zeroed() };
        let mut mask: libc::sigset_t = unsafe { std::mem::zeroed() };
        let replaced = unsafe { listen(on_signal, 0) };
        unsafe {
            libc::sigemptyset(&raw mut ending);
            for (signal, _) in &replaced {
                libc::sigaddset(&raw mut ending, *signal);
            }
            libc::pthread_sigmask(libc::SIG_BLOCK, &raw const ending, &raw mut mask);
            // std keeps the parent's mask in the child: cargo gets the one
            // from before.
            command.pre_exec(move || {
                libc::pthread_sigmask(libc::SIG_SETMASK, &raw const mask, std::ptr::null_mut());
                Ok(())
            });
        }
        let spawned = command.spawn();
        if let Ok(child) = &spawned {
            let group = if leads {
                own
            } else {
                i32::try_from(child.id()).unwrap_or(0)
            };
            OWN.store(leads, Ordering::SeqCst);
            CARGO.store(i32::try_from(child.id()).unwrap_or(0), Ordering::SeqCst);
            GROUP.store(group, Ordering::SeqCst);
        }
        // SAFETY: restores the mask saved above.
        unsafe {
            libc::pthread_sigmask(libc::SIG_SETMASK, &raw const mask, std::ptr::null_mut());
        }
        spawned.map(|child| Build { child, replaced })
    }

    /// The build ended: the ending signals act as they did before it,
    /// `replaced` put back.
    pub(super) fn forget(replaced: &[(libc::c_int, libc::sigaction)]) {
        GROUP.store(0, Ordering::SeqCst);
        CARGO.store(0, Ordering::SeqCst);
        restore(replaced);
    }

    /// Kills the build's group, if one runs and is not the launcher's own.
    pub(super) fn kill() {
        let group = GROUP.load(Ordering::SeqCst);
        if group > 0 && !OWN.load(Ordering::SeqCst) {
            // SAFETY: `kill` takes plain integers; a negative pid names the
            // build's process group.
            unsafe {
                libc::kill(-group, libc::SIGKILL);
            }
        }
    }

    /// Kills the build's group, then ends the launcher as `signal` would
    /// have. When the group is the launcher's own, it sends `signal` to all
    /// of it and kills cargo outright, so the launcher too ends as that
    /// signal says once this returns, and a cargo that ignores the signal
    /// does not outlive it. Only async-signal-safe calls.
    extern "C" fn on_signal(signal: libc::c_int) {
        let (group, cargo) = (GROUP.load(Ordering::SeqCst), CARGO.load(Ordering::SeqCst));
        // SAFETY: `signal`, `kill` and `raise` are async-signal-safe and take
        // plain integers.
        unsafe {
            libc::signal(signal, libc::SIG_DFL);
            if OWN.load(Ordering::SeqCst) && group > 0 {
                // cargo first: the group's signal may end the launcher at
                // once, through another of its threads.
                if cargo > 0 {
                    libc::kill(cargo, libc::SIGKILL);
                }
                libc::kill(-group, signal);
            } else {
                kill();
                libc::raise(signal);
            }
        }
    }
}

#[cfg(all(test, unix))]
mod tests;
