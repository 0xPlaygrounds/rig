//! `rig build`: regenerate the agent project, check that it has one Bevy
//! version, and build it with cargo. Cargo's output goes to stderr, where
//! the terminal or the agent's `/reload` reads it.

use std::{
    env,
    ffi::OsString,
    fs,
    process::{Command, Stdio},
};

use super::{
    Error, Result,
    config::{self, Config},
    paths::Paths,
    project::{self, BEVY_VERSION},
};

/// Generate and build the agent project. `jobs` overrides
/// `CARGO_BUILD_JOBS` and `[build] jobs`.
pub fn build(paths: &Paths, jobs: Option<u32>) -> Result<()> {
    let config = config::load(&paths.plugins())?;
    project::generate(paths, &config)?;
    check_bevy(paths, &config)?;
    let manifest = paths.project.join("Cargo.toml");
    let mut cargo = Command::new(cargo());
    cargo
        .arg("build")
        .arg("--manifest-path")
        .arg(&manifest)
        .arg("--target-dir")
        .arg(paths.project.join("target"))
        .stdin(Stdio::null())
        .stdout(Stdio::null());
    let jobs = jobs.or_else(|| {
        env::var("CARGO_BUILD_JOBS")
            .ok()
            .and_then(|jobs| jobs.parse().ok())
            .or(config.jobs)
    });
    if let Some(jobs) = jobs {
        cargo.arg("-j").arg(jobs.to_string());
    }
    let status = cargo.status()?;
    if status.success() {
        Ok(())
    } else {
        Err(Error(format!("building the agent failed ({status})")))
    }
}

fn cargo() -> OsString {
    env::var_os("CARGO").unwrap_or_else(|| "cargo".into())
}

/// Resolve the project, then find any `bevy_ecs` that is not
/// [`BEVY_VERSION`] and name the plugin that pulls it in.
fn check_bevy(paths: &Paths, config: &Config) -> Result<()> {
    let status = Command::new(cargo())
        .args(["metadata", "--format-version", "1", "--manifest-path"])
        .arg(paths.project.join("Cargo.toml"))
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .status()?;
    if !status.success() {
        return Err(Error(format!(
            "cargo cannot resolve the agent project; check {}",
            paths.plugins().display()
        )));
    }
    let lock = fs::read_to_string(paths.project.join("Cargo.lock"))?;
    let packages = packages(&lock);
    let foreign = |package: &Package| package.name == "bevy_ecs" && package.version != BEVY_VERSION;
    if !packages.iter().any(foreign) {
        return Ok(());
    }
    for plugin in &config.plugins {
        let name = plugin.package();
        let Some(start) = packages.iter().position(|package| package.name == name) else {
            continue;
        };
        let mut seen = vec![false; packages.len()];
        if let Some(path) = trail(&packages, start, &foreign, &mut seen) {
            let bevy = path
                .last()
                .and_then(|index| packages.get(*index))
                .map_or("another version", |package| package.version.as_str());
            let through = path
                .get(1)
                .and_then(|index| packages.get(*index))
                .map(|package| format!(" (through `{} {}`)", package.name, package.version))
                .unwrap_or_default();
            return Err(Error(format!(
                "Plugin `{}` depends on Bevy {bevy}{through}, but the agent is built on Bevy \
                 {BEVY_VERSION}. Every plugin must use bevy = \"={BEVY_VERSION}\". Update the \
                 plugin or remove it from {}.",
                plugin.name,
                paths.plugins().display()
            )));
        }
    }
    Err(Error(format!(
        "The agent project pulls in a second Bevy version, but the agent is built on Bevy \
         {BEVY_VERSION}. Every plugin must use bevy = \"={BEVY_VERSION}\"; check {}.",
        paths.plugins().display()
    )))
}

/// One `[[package]]` of a `Cargo.lock`.
struct Package {
    name: String,
    version: String,
    /// `name` or `name version`, as the lock writes them.
    dependencies: Vec<String>,
}

/// The packages of a `Cargo.lock`.
fn packages(lock: &str) -> Vec<Package> {
    let mut packages: Vec<Package> = Vec::new();
    let mut in_dependencies = false;
    for line in lock.lines().map(str::trim) {
        if line == "[[package]]" {
            packages.push(Package {
                name: String::new(),
                version: String::new(),
                dependencies: Vec::new(),
            });
            in_dependencies = false;
            continue;
        }
        let Some(package) = packages.last_mut() else {
            continue;
        };
        if in_dependencies {
            if line.starts_with(']') {
                in_dependencies = false;
            } else {
                package
                    .dependencies
                    .push(line.trim_end_matches(',').trim_matches('"').to_owned());
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

/// The package indices from `from` to the first package matching `target`,
/// depth first.
fn trail(
    packages: &[Package],
    from: usize,
    target: &dyn Fn(&Package) -> bool,
    seen: &mut [bool],
) -> Option<Vec<usize>> {
    let package = packages.get(from)?;
    if target(package) {
        return Some(vec![from]);
    }
    if std::mem::replace(seen.get_mut(from)?, true) {
        return None;
    }
    for dependency in &package.dependencies {
        let mut words = dependency.split(' ');
        let name = words.next().unwrap_or_default();
        let version = words.next();
        let found = packages.iter().position(|candidate| {
            candidate.name == name && version.is_none_or(|version| candidate.version == version)
        });
        if let Some(mut path) = found.and_then(|next| trail(packages, next, target, seen)) {
            path.insert(0, from);
            return Some(path);
        }
    }
    None
}
