//! `rig build`: regenerate the agent project, check that every plugin uses
//! the agent's Bevy version, compile with cargo's output on stderr (or only
//! in the log, for a headless start: [`BuildOutput`]), and stage the new
//! binary for the next start. Before it is staged, a build
//! describes what its plugins add ([`described`]): a build whose plugins
//! clash, or that does not describe itself, is rejected. Everything a build
//! prints also goes to [`Home::build_log`], and a failure names its reason
//! and first compiler errors ([`BuildFailure`]).
//!
//! No process the launcher starts but the agent inherits a secret
//! ([`child`]).

use std::collections::BTreeSet;
use std::ffi::OsStr;
use std::fmt;
use std::fs::{self, File};
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant, UNIX_EPOCH};

use rig_harness_protocol::{
    AgentDescription, Home, SessionId, env, first_errors, secret_variables,
};

use super::plugin_list::PluginList;
use super::project::{self, PACKAGE, RigSource};
use super::{BEVY_VERSION, Result, locks};

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

/// Why a build failed: the reason, the first compiler errors, and the log
/// with the whole output. Its [`Display`](fmt::Display) is one line, for
/// a terminal that showed cargo's output already; [`Self::details`] adds
/// the errors.
#[derive(Debug)]
pub struct BuildFailure {
    /// What failed, such as `cargo could not build the agent (exit
    /// status: 101)` or a `plugins.toml` mistake.
    pub reason: String,
    /// The output from its first error on, at most [`ERROR_LINES`] lines;
    /// empty when no line starts with `error`.
    pub errors: Vec<String>,
    /// [`Home::build_log`].
    pub log: PathBuf,
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

/// Reads `plugins.toml`, writes the agent project, checks its Bevy version,
/// builds it, and copies the binary to `staged` as `staging` says. The
/// build's output goes to [`Home::build_log`], and to stderr as `output`
/// says.
pub fn compile(
    home: &Home,
    staged: &Path,
    staging: Staging,
    output: BuildOutput,
) -> std::result::Result<(), BuildFailure> {
    let mut log = BuildLog::create(home, output, Box::new(std::io::stderr()));
    match compile_logged(home, staged, staging, &mut log) {
        Ok(()) => {
            log.record("build succeeded");
            Ok(())
        }
        Err(failure) => Err(log.fail(failure.to_string())),
    }
}

fn compile_logged(home: &Home, staged: &Path, staging: Staging, log: &mut BuildLog) -> Result<()> {
    let list = PluginList::load(&home.plugins_toml())?;
    let source = RigSource::detect()?;
    project::generate(home, &list, &source)?;
    // `/reload` shows the latest of these lines and of cargo's.
    log.say("Compiling the agent…");
    let mut command = cargo(home);
    command.args(["build", "--package", PACKAGE]);
    // A Windows console may not take the erase sequence, which cargo
    // itself writes on Unix only.
    log.erase = cfg!(unix) && log.output == BuildOutput::Live && show_progress(&mut command);
    let status = log.run(command)?;
    // The check reads the graph the build resolved into `Cargo.lock`, so a
    // lock it passed passes again; after a failed build it runs anyway, to
    // say in plain words when a foreign Bevy or rig crate is why.
    let lock = fs::read(home.project().join("Cargo.lock")).ok();
    let checked = home.project().join("Cargo.lock.checked");
    if !status.success() || lock.is_none() || lock != fs::read(&checked).ok() {
        log.say("Resolving dependencies…");
        let rig_crates = project::rig_crates(&list, &source);
        check_bevy(
            home,
            &list,
            &rig_crates,
            &project::rig_version(&source),
            log,
        )?;
        if let Some(lock) = lock {
            fs::write(&checked, lock).ok();
        }
    }
    if !status.success() {
        return Err(format!("cargo could not build the agent ({status})").into());
    }
    stage(home, staged, staging)
}

/// `rig build`, holding the root's build lock. It always stages, so it also
/// retries a build that was rolled back. Run by an agent's `/reload`
/// (`RIG_SESSION` set), it stages for that agent's launcher alone.
pub fn build(home: &Home) -> Result<()> {
    let _lock = locks::hold_build_lock(home, BuildOutput::Live)?;
    let staged = match SessionId::from_env()? {
        Some(session) => home.staged_for(&session),
        None => home.staged(),
    };
    Ok(compile(home, &staged, Staging::Always, BuildOutput::Live)?)
}

/// `rig plugin check --build`, holding the root's build lock: builds the
/// agent like [`build`] but stages nothing, and returns its report.
pub fn check(home: &Home) -> Result<AgentDescription> {
    let _lock = locks::hold_build_lock(home, BuildOutput::Live)?;
    compile(home, &home.staged(), Staging::Never, BuildOutput::Live)?;
    described(home, &artifact(home))
}

/// The binary cargo builds the agent into.
fn artifact(home: &Home) -> PathBuf {
    home.target()
        .join("debug")
        .join(format!("{PACKAGE}{}", std::env::consts::EXE_SUFFIX))
}

/// A process the launcher starts that is not the agent, such as cargo or
/// `stty`: without any secret ([`secret_variables`]), which a user who
/// exported one for a headless run gave the agent alone.
pub fn child(program: impl AsRef<OsStr>) -> Command {
    let mut command = Command::new(program);
    for secret in secret_variables() {
        command.env_remove(secret);
    }
    command
}

/// cargo in the agent project, with stdout discarded and stderr piped, for
/// [`BuildLog::run`] to pass on. Without colors, so the log and the first
/// errors read as plain text.
fn cargo(home: &Home) -> Command {
    let mut command = child("cargo");
    command
        .current_dir(home.project())
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
fn show_progress(command: &mut Command) -> bool {
    if std::io::stderr().is_terminal()
        && std::env::var_os("CARGO_TERM_PROGRESS_WHEN").is_none()
        && let Some(width) = terminal_width()
    {
        command
            .env("CARGO_TERM_PROGRESS_WHEN", "always")
            .env("CARGO_TERM_PROGRESS_WIDTH", width.to_string());
        return true;
    }
    false
}

/// The terminal's width in columns: `COLUMNS`, else `stty size`.
fn terminal_width() -> Option<u16> {
    if let Some(columns) = std::env::var("COLUMNS")
        .ok()
        .and_then(|columns| columns.parse().ok())
    {
        return Some(columns);
    }
    let tty = File::open("/dev/tty").ok()?;
    let output = child("stty")
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
}

impl BuildLog {
    /// Starts the log over, showing on `shown` what `output` says.
    fn create(home: &Home, output: BuildOutput, shown: Box<dyn Write>) -> Self {
        let path = home.build_log();
        let file = fs::create_dir_all(home.root())
            .and_then(|()| File::create(&path))
            .inspect_err(|failure| {
                eprintln!("warning: could not write {}: {failure}", path.display());
            })
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

    /// Runs `command`, cargo, whose stderr is piped, in a process group of
    /// its own ([`Build`]): passes its stderr on as it comes and logs each
    /// of its lines.
    fn run(&mut self, command: Command) -> Result<ExitStatus> {
        let mut build = Build::spawn(command)?;
        self.pass_on(&mut build.0);
        Ok(build.0.wait()?)
    }

    /// Passes `child`'s stderr on when the output is live, and logs its
    /// lines. cargo redraws its progress bar after `\r`, so `\r` ends a
    /// line too; the bar is not logged. Through a pipe cargo does not
    /// erase the bar's line before a shorter one, which would keep the
    /// bar's tail on screen, so on a terminal each `\r` but a `\r\n` erases
    /// it.
    fn pass_on(&mut self, child: &mut Child) {
        let Some(mut stderr) = child.stderr.take() else {
            return;
        };
        // Read on a thread, so a wait for cargo's lock is announced once it
        // passes its time even while cargo prints nothing.
        let (sender, chunks) = mpsc::channel();
        // A thread that cannot start drops the pipe, which fails the build.
        let reading = std::thread::Builder::new();
        reading
            .spawn(move || {
                let mut buffer = [0_u8; 8192];
                while let Ok(read @ 1..) = stderr.read(&mut buffer) {
                    if sender
                        .send(buffer.get(..read).unwrap_or_default().to_vec())
                        .is_err()
                    {
                        return;
                    }
                }
            })
            .ok();
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
            if self.output == BuildOutput::Live {
                self.show(&chunk);
            }
            for &byte in &chunk {
                if byte == b'\r' || byte == b'\n' {
                    self.record_segment(&segment);
                    segment.clear();
                } else {
                    segment.push(byte);
                }
            }
        }
        self.record_segment(&segment);
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

    /// Ends the log with `reason` and makes it the [`BuildFailure`].
    fn fail(mut self, reason: String) -> BuildFailure {
        let errors = first_errors(self.lines.iter().map(String::as_str), ERROR_LINES)
            .into_iter()
            .map(str::to_owned)
            .collect();
        self.record(&format!("build failed: {reason}"));
        BuildFailure {
            reason,
            errors,
            log: self.path,
        }
    }
}

/// Copies the built binary to `staged`, keeping its modification time so
/// the copy carries its [`stamp`], once it described itself: a build that
/// does not is rejected, and the build fails. A build `rig build` staged
/// for any launcher that differs from this one is older, so it goes.
fn stage(home: &Home, staged: &Path, staging: Staging) -> Result<()> {
    let artifact = artifact(home);
    let built = stamp(&artifact)?;
    if let Err(why) = described(home, &artifact) {
        if staging != Staging::Never {
            fs::write(home.rejected(), &built)?;
        }
        return Err(format!("the new build does not start: {why}").into());
    }
    if staging == Staging::Never {
        return Ok(());
    }
    let shared = home.staged();
    if stamp(&shared).is_ok_and(|stamp| stamp != built) {
        fs::remove_file(&shared)?;
    }
    let known = [home.good(), shared]
        .iter()
        .any(|binary| stamp(binary).is_ok_and(|stamp| stamp == built))
        || fs::read_to_string(home.rejected()).is_ok_and(|rejected| rejected == built);
    if staging == Staging::OnlyNew && known {
        return Ok(());
    }
    let temporary = home.bin().join("staged.tmp");
    fs::create_dir_all(home.bin())?;
    fs::copy(&artifact, &temporary)?;
    File::options()
        .write(true)
        .open(&temporary)?
        .set_modified(fs::metadata(&artifact)?.modified()?)?;
    fs::rename(&temporary, staged)?;
    prune_described(home);
    Ok(())
}

/// Removes the reports of builds no file in [`Home::bin`] is a copy of.
fn prune_described(home: &Home) {
    let stamps: BTreeSet<String> = fs::read_dir(home.bin())
        .into_iter()
        .flatten()
        .flatten()
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
pub fn described(home: &Home, binary: &Path) -> Result<AgentDescription> {
    if let Some(report) = good_report(home, binary) {
        return Ok(report);
    }
    let kept = home.described(&stamp(binary)?);
    let text = describe(home, binary)?;
    let parent = kept.parent().ok_or("no directory for build reports")?;
    fs::create_dir_all(parent)?;
    // Written whole, then renamed: another launcher may read it meanwhile.
    let written = parent.join(format!(".{}.tmp", std::process::id()));
    fs::write(&written, &text)?;
    fs::rename(&written, &kept)?;
    Ok(AgentDescription::parse(&text))
}

/// The report kept for the build `binary`, if any.
pub fn good_report(home: &Home, binary: &Path) -> Option<AgentDescription> {
    let text = fs::read_to_string(home.described(&stamp(binary).ok()?)).ok()?;
    Some(AgentDescription::parse(&text))
}

/// Runs `binary` with [`env::DESCRIBE`] and returns what it wrote, empty
/// when it wrote nothing. It runs with stdin closed, in a scratch
/// `RIG_HOME` of its own that is its working directory, without the
/// session, run or secrets, so a build that ignores the request cannot
/// reach the user's sessions; it is killed after [`DESCRIBE_TIMEOUT`].
fn describe(home: &Home, binary: &Path) -> Result<String> {
    let run = home.run(std::process::id());
    let scratch = run.path().join("describe");
    fs::remove_dir_all(&scratch).ok();
    let scratch_home = scratch.join("home");
    fs::create_dir_all(&scratch_home)?;
    let report = scratch.join("report");
    let mut command = child(binary);
    command
        .env(env::DESCRIBE, &report)
        .env(env::HOME, &scratch_home)
        .current_dir(&scratch_home)
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    for name in env::AGENT_ONLY.iter().chain(&[env::LAUNCHER]) {
        command.env_remove(name);
    }
    let result = wait_described(command.spawn()?, &report, DESCRIBE_TIMEOUT);
    fs::remove_dir_all(&scratch).ok();
    // A launcher with no run of its own, `rig build`, leaves none.
    fs::remove_dir(run.path()).ok();
    result
}

/// Waits for a describing build, at most `timeout`, and returns its
/// `report`, or what it said on stderr when it failed.
fn wait_described(mut child: Child, report: &Path, timeout: Duration) -> Result<String> {
    let errors = child.stderr.take();
    let errors = std::thread::spawn(move || {
        let mut text = String::new();
        if let Some(mut errors) = errors {
            errors.read_to_string(&mut text).ok();
        }
        text
    });
    let deadline = Instant::now() + timeout;
    let status = loop {
        if let Some(status) = child.try_wait()? {
            break status;
        }
        if Instant::now() > deadline {
            child.kill().ok();
            child.wait().ok();
            return Err(format!(
                "it did not describe itself within {} s",
                timeout.as_secs_f32()
            )
            .into());
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let errors = errors.join().unwrap_or_default();
    if !status.success() {
        let said = errors.trim();
        return Err(if said.is_empty() {
            format!("describing it failed ({status})").into()
        } else {
            said.to_owned().into()
        });
    }
    Ok(fs::read_to_string(report).unwrap_or_default())
}

/// Records `trial`, a build that stopped before it was ready, as rejected,
/// so a start does not stage it again until something changes.
pub fn reject(home: &Home, trial: &Path) -> Result<()> {
    fs::write(home.rejected(), stamp(trial)?)?;
    Ok(())
}

/// What tells builds apart: the binary's modification time and size.
fn stamp(binary: &Path) -> Result<String> {
    let metadata = fs::metadata(binary)?;
    let modified = metadata
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_nanos())
        .unwrap_or_default();
    Ok(format!("{modified} {}\n", metadata.len()))
}

/// Reads the project's resolved graph and fails, in plain words,
/// when a plugin pulls in a Bevy other than [`BEVY_VERSION`] or rig crates
/// other than the agent's.
fn check_bevy(
    home: &Home,
    list: &PluginList,
    rig_crates: &BTreeSet<String>,
    rig_version: &str,
    log: &mut BuildLog,
) -> Result<()> {
    let packages = tree(home, &[], log)?;
    check_rig(&packages, rig_crates, rig_version)?;
    let Some((bevy, version)) = packages.iter().find_map(|line| {
        let (name, version) = package(line)?;
        (["bevy_app", "bevy_ecs"].contains(&name) && version != BEVY_VERSION)
            .then_some((name, version))
    }) else {
        return Ok(());
    };
    // Every package that depends on the foreign Bevy, directly or not.
    let dependents = tree(home, &["--invert", &format!("{bevy}@{version}")], log)?;
    let culprit = list
        .plugins
        .iter()
        .map(|plugin| &plugin.package)
        .find(|plugin| {
            dependents
                .iter()
                .any(|line| package(line).is_some_and(|(name, _)| name == plugin.name))
        });
    Err(match culprit {
        Some(plugin) => format!(
            "plugin `{name}` uses Bevy {version}, but this rig agent is built on Bevy \
             {BEVY_VERSION}. A Bevy plugin only works with the exact Bevy version of its app. \
             Change {name}'s bevy dependencies to `={BEVY_VERSION}` (with default-features = \
             false), or remove it from plugins.toml.",
            name = plugin.name
        ),
        None => format!(
            "the agent project pulls in Bevy {version}, but this rig agent is built on Bevy \
             {BEVY_VERSION}. Every Bevy crate must use exactly {BEVY_VERSION}."
        ),
    }
    .into())
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
/// package per line, as `name vVERSION (source)`. Its stderr goes through
/// `log`.
fn tree(home: &Home, args: &[&str], log: &mut BuildLog) -> Result<Vec<String>> {
    let mut command = cargo(home);
    command
        .args(["tree", "--prefix", "none", "--format", "{p}"])
        .args(args)
        .stdout(Stdio::piped());
    let mut build = Build::spawn(command)?;
    let child = &mut build.0;
    // Read on its own thread, so neither pipe fills while the other is read.
    let out = child.stdout.take();
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        if let Some(mut out) = out {
            out.read_to_string(&mut text).ok();
        }
        text
    });
    log.pass_on(child);
    let stdout = reader
        .join()
        .map_err(|_| "reading cargo tree's output failed")?;
    let status = child.wait()?;
    if !status.success() {
        return Err("cargo could not resolve the agent project's dependencies".into());
    }
    Ok(stdout.lines().map(str::to_owned).collect())
}

/// The name and version of a `cargo tree` line.
fn package(line: &str) -> Option<(&str, &str)> {
    let mut words = line.split_whitespace();
    let name = words.next()?;
    let version = words.next()?.strip_prefix('v')?;
    Some((name, version))
}

/// cargo, running in a process group of its own (or the launcher's, when
/// the launcher leads a session of its own), which is killed should the
/// launcher be told to end (`SIGTERM`, `SIGINT`, `SIGHUP`) meanwhile, so no
/// cargo or rustc outlives it. Elsewhere than unix, cargo alone.
struct Build(Child);

impl Build {
    fn spawn(mut command: Command) -> std::io::Result<Self> {
        #[cfg(unix)]
        return stop::spawn(&mut command).map(Self);
        #[cfg(not(unix))]
        command.spawn().map(Self)
    }
}

impl Drop for Build {
    fn drop(&mut self) {
        #[cfg(unix)]
        stop::forget();
    }
}

/// The signals that end the launcher, while a [`Build`] runs, first kill
/// its group. A signal the launcher was started ignoring, as under `nohup`,
/// stays ignored.
#[cfg(unix)]
mod stop {
    use std::os::unix::process::CommandExt;
    use std::process::{Child, Command};
    use std::sync::Mutex;
    use std::sync::atomic::{AtomicBool, AtomicI32, Ordering};

    /// The signals a terminal or a supervisor ends the launcher with.
    const ENDING: [libc::c_int; 3] = [libc::SIGTERM, libc::SIGINT, libc::SIGHUP];

    /// The build's process group; 0 while none runs.
    static GROUP: AtomicI32 = AtomicI32::new(0);

    /// Whether that group is the launcher's own.
    static OWN: AtomicBool = AtomicBool::new(false);

    /// The actions [`spawn`] replaced, which [`forget`] puts back.
    static BEFORE: Mutex<Vec<(libc::c_int, libc::sigaction)>> = Mutex::new(Vec::new());

    /// Starts `command` as the leader of a new process group and makes it
    /// the group the ending signals kill. A launcher that leads a session of
    /// its own, as an agent's `/reload` starts it, keeps cargo in its own
    /// group instead, which the agent kills as a whole. The signals wait
    /// while it starts, so one that comes meanwhile finds the group.
    pub(super) fn spawn(command: &mut Command) -> std::io::Result<Child> {
        // SAFETY: plain calls without arguments that can fail.
        let (leads, own) = unsafe { (libc::getsid(0) == libc::getpid(), libc::getpgrp()) };
        if !leads {
            command.process_group(0);
        }
        let mut replaced = Vec::new();
        // SAFETY: plain calls on zeroed signal sets and actions this
        // function owns; the closure runs in the forked child before `exec`
        // and only calls `pthread_sigmask`, which is async-signal-safe. The
        // child gets the default action for every caught signal (exec does).
        let mut ending: libc::sigset_t = unsafe { std::mem::zeroed() };
        let mut mask: libc::sigset_t = unsafe { std::mem::zeroed() };
        unsafe {
            libc::sigemptyset(&raw mut ending);
            for signal in ENDING {
                let mut before: libc::sigaction = std::mem::zeroed();
                libc::sigaction(signal, std::ptr::null(), &raw mut before);
                if before.sa_sigaction == libc::SIG_IGN {
                    continue;
                }
                let mut action: libc::sigaction = std::mem::zeroed();
                action.sa_sigaction = on_signal as *const () as libc::sighandler_t;
                libc::sigemptyset(&raw mut action.sa_mask);
                libc::sigaction(signal, &raw const action, std::ptr::null_mut());
                libc::sigaddset(&raw mut ending, signal);
                replaced.push((signal, before));
            }
            libc::pthread_sigmask(libc::SIG_BLOCK, &raw const ending, &raw mut mask);
            // std keeps the parent's mask in the child: cargo gets the one
            // from before.
            command.pre_exec(move || {
                libc::pthread_sigmask(libc::SIG_SETMASK, &raw const mask, std::ptr::null_mut());
                Ok(())
            });
        }
        *BEFORE
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner) = replaced;
        let spawned = command.spawn();
        if let Ok(child) = &spawned {
            let group = if leads {
                own
            } else {
                i32::try_from(child.id()).unwrap_or(0)
            };
            OWN.store(leads, Ordering::SeqCst);
            GROUP.store(group, Ordering::SeqCst);
        }
        // SAFETY: restores the mask saved above.
        unsafe {
            libc::pthread_sigmask(libc::SIG_SETMASK, &raw const mask, std::ptr::null_mut());
        }
        spawned
    }

    /// The build ended: the ending signals act as they did before it.
    pub(super) fn forget() {
        GROUP.store(0, Ordering::SeqCst);
        let replaced = std::mem::take(
            &mut *BEFORE
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner),
        );
        for (signal, before) in replaced {
            // SAFETY: puts back the action `spawn` read for this signal.
            unsafe {
                libc::sigaction(signal, &raw const before, std::ptr::null_mut());
            }
        }
    }

    /// Kills the build's group, if one runs.
    pub(super) fn kill() {
        let group = GROUP.load(Ordering::SeqCst);
        if group > 0 {
            // SAFETY: `kill` takes plain integers; a negative pid names the
            // build's process group.
            unsafe {
                libc::kill(-group, libc::SIGKILL);
            }
        }
    }

    /// Kills the build's group, then ends the launcher as `signal` would
    /// have; when the group is the launcher's own, sends `signal` to all of
    /// it, so the launcher too ends as that signal says once this returns.
    /// Only async-signal-safe calls.
    extern "C" fn on_signal(signal: libc::c_int) {
        let group = GROUP.load(Ordering::SeqCst);
        // SAFETY: `signal`, `kill` and `raise` are async-signal-safe and take
        // plain integers.
        unsafe {
            libc::signal(signal, libc::SIG_DFL);
            if OWN.load(Ordering::SeqCst) && group > 0 {
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
