//! `rig build`: generate the agent project, check that it holds one Bevy
//! version, build it, and stage a new binary as the candidate for the next
//! start.

use std::io::IsTerminal;
use std::path::Path;
use std::process::{Command, Stdio};
use std::time::SystemTime;

use crate::Failure;
use crate::dirs::{Dirs, executable};
use crate::project::{self, BEVY_VERSION, PACKAGE};

/// What a build did with its binary.
#[derive(PartialEq)]
pub(crate) enum Staged {
    /// It is the new candidate, or the current binary is already as new.
    Ready,
    /// It is the build that crashed during startup before, so it was not
    /// staged again.
    Rejected,
}

/// Builds the agent and stages it when it is newer than the current
/// binary. Cargo's output, with its progress bar, goes to this process's
/// stderr.
pub(crate) fn build(dirs: &Dirs, jobs: Option<u32>) -> Result<Staged, Failure> {
    let plugins = crate::plugins::load(&dirs.config)?;
    project::generate(dirs, &plugins, jobs)?;
    let project = dirs.project();
    check_bevy(&project)?;
    check_rig_code(&project)?;
    let built = dirs
        .cache
        .join("target")
        .join("debug")
        .join(executable(PACKAGE));
    if !built.exists() {
        eprintln!("rig: building the agent (the first build takes a few minutes)");
    }
    let mut command = cargo(&project);
    command.arg("build");
    if !std::io::stderr().is_terminal() {
        // Cargo draws its progress bar only on a terminal unless told to;
        // `/reload` reads it from the build log.
        command
            .env("CARGO_TERM_PROGRESS_WHEN", "always")
            .env("CARGO_TERM_PROGRESS_WIDTH", "100");
    }
    let status = command
        .status()
        .map_err(|error| Failure::io("cannot run cargo", error))?;
    if !status.success() {
        return Err(Failure::build("the agent build failed"));
    }
    stage(dirs, &built)
}

/// Copies `built` to the candidate slot when it is newer than the current
/// binary and is not the build that was rolled back.
fn stage(dirs: &Dirs, built: &Path) -> Result<Staged, Failure> {
    let modified = |path: &Path| path.metadata().and_then(|meta| meta.modified()).ok();
    let built_at = modified(built)
        .ok_or_else(|| Failure::build(format!("cargo built no binary at {}", built.display())))?;
    if modified(&dirs.current_bin()).is_some_and(|current: SystemTime| current >= built_at) {
        return Ok(Staged::Ready);
    }
    let stamp = built_at
        .duration_since(SystemTime::UNIX_EPOCH)
        .map(|elapsed| elapsed.as_nanos().to_string())
        .unwrap_or_default();
    if std::fs::read_to_string(dirs.rejected_stamp()).is_ok_and(|rejected| rejected == stamp) {
        return Ok(Staged::Rejected);
    }
    let candidate = dirs.candidate_bin();
    let partial = candidate.with_extension("partial");
    let failed = |error| Failure::io(format!("cannot stage {}", candidate.display()), error);
    if let Some(parent) = candidate.parent() {
        std::fs::create_dir_all(parent).map_err(failed)?;
    }
    std::fs::copy(built, &partial).map_err(failed)?;
    std::fs::write(dirs.staged_stamp(), &stamp).map_err(failed)?;
    std::fs::rename(&partial, &candidate).map_err(failed)?;
    eprintln!("rig: staged the new build");
    Ok(Staged::Ready)
}

/// `cargo` run in the project directory, so its `.cargo/config.toml`
/// applies, with any outside target directory setting removed.
fn cargo(project: &Path) -> Command {
    let mut command = Command::new("cargo");
    command
        .current_dir(project)
        .env_remove("CARGO_TARGET_DIR")
        .env_remove("CARGO_BUILD_TARGET_DIR");
    command
}

/// Fails, in plain words, when the agent project would contain a Bevy
/// version other than [`BEVY_VERSION`], naming the crates that bring it.
fn check_bevy(project: &Path) -> Result<(), Failure> {
    let output = cargo(project)
        .args([
            "tree", "-q", "-e", "normal", "-i", "bevy_ecs", "--depth", "0",
        ])
        .stdin(Stdio::null())
        .output()
        .map_err(|error| Failure::io("cannot run cargo tree", error))?;
    let stdout = String::from_utf8_lossy(&output.stdout);
    let stderr = String::from_utf8_lossy(&output.stderr);
    // One version prints `bevy_ecs v<version>`; several make the name
    // ambiguous and cargo lists each as `bevy_ecs@<version>`.
    let versions: Vec<String> = if output.status.success() {
        stdout
            .lines()
            .filter_map(|line| line.trim().strip_prefix("bevy_ecs v"))
            .map(|version| {
                version
                    .split_whitespace()
                    .next()
                    .unwrap_or(version)
                    .to_owned()
            })
            .collect()
    } else {
        stderr
            .lines()
            .filter_map(|line| line.trim().strip_prefix("bevy_ecs@"))
            .map(str::to_owned)
            .collect()
    };
    if versions.is_empty() {
        return Err(Failure::config(format!(
            "cannot resolve the agent project's dependencies:\n{}",
            stderr.trim()
        )));
    }
    let mut problems = Vec::new();
    for version in versions.iter().filter(|version| *version != BEVY_VERSION) {
        let crates = bringers(project, version);
        let who = if crates.is_empty() {
            "A plugin crate".to_owned()
        } else {
            format!("Plugin crate {}", crates.join(", "))
        };
        problems.push(format!(
            "{who} uses Bevy {version}, but rig-code uses Bevy {BEVY_VERSION}. One app cannot \
             mix two Bevy versions: the plugin's types would not be Bevy types to the agent. \
             Update its Bevy dependency to `={BEVY_VERSION}` (or use `rig_code::bevy`), or \
             remove it from plugins.toml."
        ));
    }
    if problems.is_empty() {
        return Ok(());
    }
    Err(Failure::config(problems.join("\n")))
}

/// Fails, in plain words, when a plugin brings a second copy of rig-code:
/// its tools and commands would register into types the agent never reads.
fn check_rig_code(project: &Path) -> Result<(), Failure> {
    let output = cargo(project)
        .args([
            "tree", "-q", "-e", "normal", "-i", "rig-code", "--depth", "0",
        ])
        .stdin(Stdio::null())
        .output()
        .map_err(|error| Failure::io("cannot run cargo tree", error))?;
    let stderr = String::from_utf8_lossy(&output.stderr);
    if output.status.success() || !stderr.contains("ambiguous") {
        return Ok(());
    }
    let copies: Vec<&str> = stderr
        .lines()
        .map(str::trim)
        .filter(|line| line.contains("rig-code@"))
        .collect();
    Err(Failure::config(format!(
        "The agent project holds {} copies of rig-code:\n  {}\nA plugin depends on rig-code \
         from another source than the agent, so its tools and commands would register into a \
         copy the agent never reads. Make the plugin depend on rig-code by version \
         (`rig-code = \"{}\"`; with a local checkout rig points that at the checkout), or remove \
         it from plugins.toml.",
        copies.len(),
        copies.join("\n  "),
        crate::project::VERSION
    )))
}

/// The direct dependencies of the agent project that pull in
/// `bevy_ecs@version`, as `` `name` `` strings.
fn bringers(project: &Path, version: &str) -> Vec<String> {
    let Ok(output) = cargo(project)
        .args(["tree", "-q", "-e", "normal", "--prefix", "depth", "-i"])
        .arg(format!("bevy_ecs@{version}"))
        .stdin(Stdio::null())
        .output()
    else {
        return Vec::new();
    };
    // In the inverted tree a line one level above `rig-code-app` is a
    // crate the project depends on directly.
    let lines: Vec<(usize, String)> = String::from_utf8_lossy(&output.stdout)
        .lines()
        .filter_map(|line| {
            let digits = line.find(|c: char| !c.is_ascii_digit())?;
            let (depth, rest) = line.split_at(digits);
            let name = rest.split_whitespace().next()?;
            Some((depth.parse().ok()?, name.to_owned()))
        })
        .collect();
    let mut names: Vec<String> = lines
        .windows(2)
        .filter_map(|pair| match pair {
            [(depth, name), (next_depth, next)] if next == PACKAGE && *next_depth == depth + 1 => {
                Some(format!("`{name}`"))
            }
            _ => None,
        })
        .collect();
    names.sort();
    names.dedup();
    names
}
