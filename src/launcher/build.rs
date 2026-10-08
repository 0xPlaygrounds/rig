//! `rig build`: regenerate the agent project, check that every plugin uses
//! the agent's Bevy version, compile with cargo's output on stderr, and
//! stage the new binary for the next start.

use std::collections::BTreeSet;
use std::fs;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::UNIX_EPOCH;

use super::config::Config;
use super::home::Home;
use super::project::{self, PACKAGE, RigSource};
use super::{BEVY_VERSION, Result};

/// The agent project, generated.
pub struct Project {
    config: Config,
    /// Whether generation changed a file.
    pub changed: bool,
    /// Whether `rig-code` comes from a local checkout.
    pub local: bool,
}

/// Reads `plugins.toml` and writes the agent project.
pub fn prepare(home: &Home) -> Result<Project> {
    let config = Config::load(&home.config())?;
    let source = RigSource::detect()?;
    let changed = project::generate(home, &config, &source)?;
    Ok(Project {
        config,
        changed,
        local: matches!(source, RigSource::Local(_)),
    })
}

/// Checks the Rust and Bevy versions, builds, and copies the binary to
/// `staged`. Unless `again`, a build already staged once is not staged
/// again: at startup, that keeps a build rolled back for crashing from
/// being retried until something changes.
pub fn compile(home: &Home, project: &Project, staged: &Path, again: bool) -> Result<()> {
    // `/reload` shows these lines, and cargo's, until cargo's counter
    // appears.
    eprintln!("Resolving dependencies…");
    check_bevy(home, &project.config)?;
    eprintln!("Compiling the agent…");
    let status = cargo(home).args(["build", "--package", PACKAGE]).status()?;
    if !status.success() {
        return Err("building the agent failed".into());
    }
    stage(home, staged, again)
}

/// `rig build`, holding the root's build lock. It always stages, so it also
/// retries a build that was rolled back. Run by an agent's `/reload`
/// (`RIG_SESSION` set), it stages for that agent's launcher alone.
pub fn build(home: &Home) -> Result<()> {
    let _lock = home.lock()?;
    let staged = match std::env::var("RIG_SESSION") {
        Ok(session) if !session.is_empty() => home.staged_for(&session),
        _ => home.bin("staged"),
    };
    compile(home, &prepare(home)?, &staged, true)
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

/// Copies the built binary to `staged`. `built` records the build last
/// staged; unless `again`, that build is not staged a second time.
fn stage(home: &Home, staged: &Path, again: bool) -> Result<()> {
    let artifact = home
        .target()
        .join("debug")
        .join(format!("{PACKAGE}{}", std::env::consts::EXE_SUFFIX));
    let metadata = fs::metadata(&artifact)?;
    let modified = metadata
        .modified()?
        .duration_since(UNIX_EPOCH)
        .map(|since| since.as_nanos())
        .unwrap_or_default();
    let stamp = format!("{modified} {}\n", metadata.len());
    let record = home.bin("built");
    if !again && fs::read_to_string(&record).is_ok_and(|built| built == stamp) {
        return Ok(());
    }
    let staging = home.bin("staged.tmp");
    fs::create_dir_all(home.root().join("bin"))?;
    fs::copy(&artifact, &staging)?;
    fs::rename(&staging, staged)?;
    fs::write(record, stamp)?;
    Ok(())
}

/// One `[[package]]` of `Cargo.lock`.
struct Locked {
    name: String,
    version: String,
    /// Each dependency's name and, when the name alone is ambiguous, its
    /// version.
    dependencies: Vec<(String, Option<String>)>,
}

/// Resolves the project (writing `Cargo.lock`) and fails, in plain words,
/// when a plugin pulls in a Bevy other than [`BEVY_VERSION`].
fn check_bevy(home: &Home, config: &Config) -> Result<()> {
    let status = cargo(home)
        .args(["metadata", "--format-version", "1"])
        .status()?;
    if !status.success() {
        return Err("cargo could not resolve the agent project's dependencies".into());
    }
    let lock = locked(&fs::read_to_string(home.project().join("Cargo.lock"))?);
    let Some(foreign) = lock.iter().position(|package| {
        ["bevy_app", "bevy_ecs"].contains(&package.name.as_str()) && package.version != BEVY_VERSION
    }) else {
        return Ok(());
    };
    let version = lock
        .get(foreign)
        .map(|package| package.version.as_str())
        .unwrap_or_default();
    let culprit = config
        .plugins
        .iter()
        .filter_map(|plugin| plugin.package.as_ref())
        .find(|package| reaches(&lock, &package.name, foreign));
    Err(match culprit {
        Some(package) => format!(
            "plugin `{name}` uses Bevy {version}, but this rig agent is built on Bevy \
             {BEVY_VERSION}. A Bevy plugin only works with the exact Bevy version of its app. \
             Change {name}'s bevy dependencies to `={BEVY_VERSION}` (with default-features = \
             false), or remove it from plugins.toml.",
            name = package.name
        ),
        None => format!(
            "the agent project pulls in Bevy {version}, but this rig agent is built on Bevy \
             {BEVY_VERSION}. Every Bevy crate must use exactly {BEVY_VERSION}."
        ),
    }
    .into())
}

fn locked(lock: &str) -> Vec<Locked> {
    let mut packages: Vec<Locked> = Vec::new();
    let mut in_dependencies = false;
    // Whether the lines belong to the last `[[package]]`, rather than to
    // another table such as `[[patch.unused]]` or `[metadata]`.
    let mut in_package = false;
    for line in lock.lines().map(str::trim) {
        if line.starts_with('[') && !in_dependencies {
            in_package = line == "[[package]]";
            if in_package {
                packages.push(Locked {
                    name: String::new(),
                    version: String::new(),
                    dependencies: Vec::new(),
                });
            }
            continue;
        }
        let Some(package) = packages.last_mut().filter(|_| in_package) else {
            continue;
        };
        if in_dependencies {
            if line == "]" {
                in_dependencies = false;
            } else {
                let entry = line.trim_end_matches(',').trim_matches('"');
                let mut words = entry.split_whitespace();
                if let Some(name) = words.next() {
                    package
                        .dependencies
                        .push((name.to_owned(), words.next().map(str::to_owned)));
                }
            }
        } else if let Some(name) = line.strip_prefix("name = ") {
            package.name = name.trim_matches('"').to_owned();
        } else if let Some(version) = line.strip_prefix("version = ") {
            package.version = version.trim_matches('"').to_owned();
        } else if line == "dependencies = [" {
            in_dependencies = true;
        }
    }
    packages
}

/// Whether package `target` is among the dependencies of package `name`.
fn reaches(lock: &[Locked], name: &str, target: usize) -> bool {
    let mut seen = BTreeSet::new();
    let mut stack: Vec<usize> = (0..lock.len())
        .filter(|&index| lock.get(index).is_some_and(|package| package.name == name))
        .collect();
    while let Some(index) = stack.pop() {
        if index == target {
            return true;
        }
        if !seen.insert(index) {
            continue;
        }
        let Some(package) = lock.get(index) else {
            continue;
        };
        for (dependency, version) in &package.dependencies {
            stack.extend((0..lock.len()).filter(|&candidate| {
                lock.get(candidate).is_some_and(|locked| {
                    &locked.name == dependency
                        && version
                            .as_ref()
                            .is_none_or(|version| &locked.version == version)
                })
            }));
        }
    }
    false
}
