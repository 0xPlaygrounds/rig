//! `rig build`: regenerate the agent project, check that every plugin uses
//! the agent's Bevy version, compile with cargo's output on stderr, and
//! stage the new binary for the next start. Everything a build prints also
//! goes to [`Home::build_log`], and a failure names its reason and first
//! compiler errors ([`BuildFailure`]).

use std::fmt;
use std::fs::{self, File};
use std::io::{IsTerminal, Read, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, ExitStatus, Stdio};
use std::time::UNIX_EPOCH;

use rig::harness_protocol::{Home, SessionId};

use super::config::Config;
use super::project::{self, PACKAGE, RigSource};
use super::{BEVY_VERSION, Result, home};

/// Whether a build that is already known gets staged again.
#[derive(Clone, Copy, PartialEq, Eq)]
pub enum Staging {
    /// Stage every build, so a build rolled back for crashing is retried.
    Always,
    /// Stage only a build that is not already the good one, staged for any
    /// launcher, or rejected for crashing at startup. At startup that keeps
    /// a rolled-back build from being retried until something changes.
    OnlyNew,
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
        let mut text = self.reason.clone();
        for line in &self.errors {
            text.push('\n');
            text.push_str(line);
        }
        text
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
    let config = Config::load(&home.config())?;
    project::generate(home, &config, &RigSource::detect()?)?;
    // `/reload` shows these lines, and cargo's, until cargo's counter
    // appears.
    log.say("Resolving dependencies…");
    check_bevy(home, &config, log)?;
    log.say("Compiling the agent…");
    let mut command = cargo(home);
    command.args(["build", "--package", PACKAGE]);
    show_progress(&mut command);
    let status = log.run(command)?;
    if !status.success() {
        return Err(format!("cargo could not build the agent ({status})").into());
    }
    stage(home, staged, staging)
}

/// `rig build`, holding the root's build lock. It always stages, so it also
/// retries a build that was rolled back. Run by an agent's `/reload`
/// (`RIG_SESSION` set), it stages for that agent's launcher alone.
pub fn build(home: &Home) -> Result<()> {
    let _lock = home::lock(home)?;
    let staged = match SessionId::from_env()? {
        Some(session) => home.staged_for(&session),
        None => home.staged(),
    };
    Ok(compile(home, &staged, Staging::Always)?)
}

/// cargo in the agent project, with stdout discarded and stderr piped, for
/// [`BuildLog::run`] to pass on.
fn cargo(home: &Home) -> Command {
    let mut command = Command::new("cargo");
    command
        .current_dir(home.project())
        // The project's own `.cargo/config.toml` names the target directory.
        .env_remove("CARGO_TARGET_DIR")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::piped());
    command
}

/// Keeps cargo's colors and progress bar on a terminal, which cargo no
/// longer sees through the pipe. A caller's own settings win: `/reload`
/// sets them for its view.
fn show_progress(command: &mut Command) {
    if !std::io::stderr().is_terminal() {
        return;
    }
    if std::env::var_os("CARGO_TERM_COLOR").is_none() {
        command.env("CARGO_TERM_COLOR", "always");
    }
    if std::env::var_os("CARGO_TERM_PROGRESS_WHEN").is_none()
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
    let output = Command::new("stty")
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
/// cargo's progress bar and colors.
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

    /// Shows `line` on stderr and logs it.
    fn say(&mut self, line: &str) {
        eprintln!("{line}");
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

    /// Passes `child`'s stderr on to ours and logs its lines. cargo
    /// redraws its progress bar after `\r`, so `\r` ends a line too; the
    /// bar is not logged.
    fn pass_on(&mut self, child: &mut Child) {
        let Some(mut stderr) = child.stderr.take() else {
            return;
        };
        let mut out = std::io::stderr();
        let mut buffer = [0_u8; 8192];
        let mut segment: Vec<u8> = Vec::new();
        loop {
            let read = match stderr.read(&mut buffer) {
                Ok(0) | Err(_) => break,
                Ok(read) => read,
            };
            let chunk = buffer.get(..read).unwrap_or_default();
            out.write_all(chunk).ok();
            out.flush().ok();
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
        let line = without_ansi(&String::from_utf8_lossy(segment));
        let line = line.trim_end();
        if line.trim().is_empty() || line.trim_start().starts_with("Building [") {
            return;
        }
        self.record(line);
    }

    /// Ends the log with `reason` and makes it the [`BuildFailure`].
    fn fail(mut self, reason: String) -> BuildFailure {
        let errors: Vec<String> = match self.lines.iter().position(|line| line.starts_with("error"))
        {
            Some(first) => self
                .lines
                .iter()
                .skip(first)
                .take(ERROR_LINES)
                .cloned()
                .collect(),
            None => Vec::new(),
        };
        self.record(&format!("build failed: {reason}"));
        BuildFailure {
            reason,
            errors,
            log: self.path,
        }
    }
}

/// `text` without its ANSI escape sequences.
fn without_ansi(text: &str) -> String {
    let mut plain = String::with_capacity(text.len());
    let mut chars = text.chars();
    while let Some(c) = chars.next() {
        if c != '\x1b' {
            plain.push(c);
            continue;
        }
        // `ESC [ … final`, where the final byte is in `@`..=`~`; any other
        // escape is two characters.
        if chars.next() == Some('[') {
            for c in chars.by_ref() {
                if ('@'..='~').contains(&c) {
                    break;
                }
            }
        }
    }
    plain
}

/// Copies the built binary to `staged`, keeping its modification time so
/// the copy carries its [`stamp`]. A build `rig build` staged for any
/// launcher that differs from this one is older, so it goes.
fn stage(home: &Home, staged: &Path, staging: Staging) -> Result<()> {
    let artifact = home
        .target()
        .join("debug")
        .join(format!("{PACKAGE}{}", std::env::consts::EXE_SUFFIX));
    let built = stamp(&artifact)?;
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
    Ok(())
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

/// Resolves the project (writing `Cargo.lock`) and fails, in plain words,
/// when a plugin pulls in a Bevy other than [`BEVY_VERSION`].
fn check_bevy(home: &Home, config: &Config, log: &mut BuildLog) -> Result<()> {
    let packages = tree(home, &[], log)?;
    let Some((bevy, version)) = packages.iter().find_map(|line| {
        let (name, version) = package(line)?;
        (["bevy_app", "bevy_ecs"].contains(&name) && version != BEVY_VERSION)
            .then_some((name, version))
    }) else {
        return Ok(());
    };
    // Every package that depends on the foreign Bevy, directly or not.
    let dependents = tree(home, &["--invert", &format!("{bevy}@{version}")], log)?;
    let culprit = config
        .plugins
        .iter()
        .filter_map(|plugin| plugin.package.as_ref())
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
