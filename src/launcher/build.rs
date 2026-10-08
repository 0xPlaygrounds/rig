//! `rig build`: regenerate the agent project, check that every plugin uses
//! the agent's Bevy version, compile with cargo's output on stderr, and
//! stage the new binary for the next start.

use std::fs::{self, File};
use std::path::Path;
use std::process::{Command, Stdio};
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

/// Reads `plugins.toml`, writes the agent project, checks its Bevy version,
/// builds it, and copies the binary to `staged` as `staging` says.
pub fn compile(home: &Home, staged: &Path, staging: Staging) -> Result<()> {
    let config = Config::load(&home.config())?;
    project::generate(home, &config, &RigSource::detect()?)?;
    // `/reload` shows these lines, and cargo's, until cargo's counter
    // appears.
    eprintln!("Resolving dependencies…");
    check_bevy(home, &config)?;
    eprintln!("Compiling the agent…");
    let status = cargo(home).args(["build", "--package", PACKAGE]).status()?;
    if !status.success() {
        return Err("building the agent failed".into());
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
    compile(home, &staged, Staging::Always)
}

/// cargo in the agent project, with stdout discarded and stderr passed
/// through.
fn cargo(home: &Home) -> Command {
    let mut command = Command::new("cargo");
    command
        .current_dir(home.project())
        // The project's own `.cargo/config.toml` names the target directory.
        .env_remove("CARGO_TARGET_DIR")
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::inherit());
    command
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
fn check_bevy(home: &Home, config: &Config) -> Result<()> {
    let packages = tree(home, &[])?;
    let Some((bevy, version)) = packages.iter().find_map(|line| {
        let (name, version) = package(line)?;
        (["bevy_app", "bevy_ecs"].contains(&name) && version != BEVY_VERSION)
            .then_some((name, version))
    }) else {
        return Ok(());
    };
    // Every package that depends on the foreign Bevy, directly or not.
    let dependents = tree(home, &["--invert", &format!("{bevy}@{version}")])?;
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
/// package per line, as `name vVERSION (source)`.
fn tree(home: &Home, args: &[&str]) -> Result<Vec<String>> {
    let output = cargo(home)
        .args(["tree", "--prefix", "none", "--format", "{p}"])
        .args(args)
        .stdout(Stdio::piped())
        .output()?;
    if !output.status.success() {
        return Err("cargo could not resolve the agent project's dependencies".into());
    }
    Ok(String::from_utf8_lossy(&output.stdout)
        .lines()
        .map(str::to_owned)
        .collect())
}

/// The name and version of a `cargo tree` line.
fn package(line: &str) -> Option<(&str, &str)> {
    let mut words = line.split_whitespace();
    let name = words.next()?;
    let version = words.next()?.strip_prefix('v')?;
    Some((name, version))
}
