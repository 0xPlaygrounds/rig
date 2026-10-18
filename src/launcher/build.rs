//! `rig build`: regenerate the agent project, check that every plugin uses
//! the agent's Bevy version, compile with cargo's output on stderr, and
//! stage the new binary for the next start. Before it is staged, a build
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
use std::time::{Duration, Instant, UNIX_EPOCH};

use rig_harness_protocol::{
    AgentDescription, Home, SessionId, chatty, env, first_errors, secret_variables,
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
/// build's output goes to stderr and to [`Home::build_log`].
pub fn compile(
    home: &Home,
    staged: &Path,
    staging: Staging,
) -> std::result::Result<(), BuildFailure> {
    let mut log = BuildLog::create(home);
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
    show_progress(&mut command);
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
    let _lock = locks::hold_build_lock(home)?;
    let staged = match SessionId::from_env()? {
        Some(session) => home.staged_for(&session),
        None => home.staged(),
    };
    Ok(compile(home, &staged, Staging::Always)?)
}

/// `rig plugin check --build`, holding the root's build lock: builds the
/// agent like [`build`] but stages nothing, and returns its report.
pub fn check(home: &Home) -> Result<AgentDescription> {
    let _lock = locks::hold_build_lock(home)?;
    compile(home, &home.staged(), Staging::Never)?;
    described(home, &artifact(home))
}

/// The binary cargo builds the agent into.
pub fn artifact(home: &Home) -> PathBuf {
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
/// through the pipe. A caller's own settings win.
fn show_progress(command: &mut Command) {
    if std::io::stderr().is_terminal()
        && std::env::var_os("CARGO_TERM_PROGRESS_WHEN").is_none()
        && let Some(width) = terminal_width()
    {
        command
            .env("CARGO_TERM_PROGRESS_WHEN", "always")
            .env("CARGO_TERM_PROGRESS_WIDTH", width.to_string());
    }
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
}

impl BuildLog {
    /// Starts the log over.
    fn create(home: &Home) -> Self {
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
        }
    }

    /// Shows `line` on a chatty stderr ([`chatty`]) and logs it.
    fn say(&mut self, line: &str) {
        if chatty() {
            eprintln!("{line}");
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

    /// Runs `command`, whose stderr is piped: passes its stderr on as it
    /// comes and logs each of its lines.
    fn run(&mut self, mut command: Command) -> Result<ExitStatus> {
        let mut child = command.spawn()?;
        self.pass_on(&mut child);
        Ok(child.wait()?)
    }

    /// Passes `child`'s stderr on to ours when it is chatty ([`chatty`])
    /// and logs its lines. cargo
    /// redraws its progress bar after `\r`, so `\r` ends a line too; the
    /// bar is not logged.
    fn pass_on(&mut self, child: &mut Child) {
        let Some(mut stderr) = child.stderr.take() else {
            return;
        };
        let shown = chatty();
        let mut out = std::io::stderr();
        let mut buffer = [0_u8; 8192];
        let mut segment: Vec<u8> = Vec::new();
        loop {
            let read = match stderr.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(read) => read,
            };
            let chunk = buffer.get(..read).unwrap_or_default();
            if shown {
                out.write_all(chunk).ok();
                out.flush().ok();
            }
            for &byte in chunk {
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

    fn record_segment(&mut self, segment: &[u8]) {
        let line = String::from_utf8_lossy(segment);
        let line = line.trim_end();
        if line.trim().is_empty() || line.trim_start().starts_with("Building [") {
            return;
        }
        self.record(line);
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
pub fn stamp(binary: &Path) -> Result<String> {
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
    let mut child = cargo(home)
        .args(["tree", "--prefix", "none", "--format", "{p}"])
        .args(args)
        .stdout(Stdio::piped())
        .spawn()?;
    // Read on its own thread, so neither pipe fills while the other is read.
    let out = child.stdout.take();
    let reader = std::thread::spawn(move || {
        let mut text = String::new();
        if let Some(mut out) = out {
            out.read_to_string(&mut text).ok();
        }
        text
    });
    log.pass_on(&mut child);
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

#[cfg(all(test, unix))]
mod tests;
