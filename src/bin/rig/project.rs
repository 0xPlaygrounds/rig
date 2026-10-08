//! The generated agent project: `Cargo.toml`, `src/main.rs`,
//! `.cargo/config.toml` and a stamp, all derived from the plugin list.
//! Files are written only when their content changes, so an unchanged list
//! does not trigger a rebuild.

use std::collections::BTreeSet;
use std::fmt::Write as _;
use std::path::{Path, PathBuf};

use crate::Failure;
use crate::dirs::{Dirs, absolute, variable};
use crate::plugins::{Package, Plugin};

/// The launcher's version, which is also the rig-code version it builds.
pub(crate) const VERSION: &str = env!("CARGO_PKG_VERSION");
/// The one Bevy version an agent may contain.
pub(crate) const BEVY_VERSION: &str = "0.20.0-rc.2";
/// The generated package, and so its binary.
pub(crate) const PACKAGE: &str = "rig-code-app";

/// Where the agent project gets rig-code.
enum CodeSource {
    /// A local checkout of the crate.
    Path(PathBuf),
    /// The crates.io release matching the launcher.
    Release,
}

impl CodeSource {
    /// `RIG_CODE_SOURCE` (a repository root or the crate itself), else the
    /// repository the launcher was built from when it is still there, else
    /// crates.io.
    fn resolve() -> Result<Self, Failure> {
        if let Some(source) = variable("RIG_CODE_SOURCE") {
            let nested = source.join("crates").join("rig-code");
            let crate_dir = if nested.join("Cargo.toml").is_file() {
                nested
            } else if source.join("Cargo.toml").is_file() {
                source.clone()
            } else {
                return Err(Failure::config(format!(
                    "RIG_CODE_SOURCE={} holds neither crates/rig-code nor a Cargo.toml",
                    source.display()
                )));
            };
            return Ok(Self::Path(absolute(&crate_dir)?));
        }
        let built_from = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("crates")
            .join("rig-code");
        if built_from.join("Cargo.toml").is_file() {
            return Ok(Self::Path(built_from));
        }
        Ok(Self::Release)
    }

    fn dependency(&self) -> String {
        match self {
            Self::Path(path) => format!("{{ path = {} }}", quote(&path.display().to_string())),
            Self::Release => quote(&format!("={VERSION}")),
        }
    }

    /// The repository holding a local checkout, when it has a lockfile.
    fn repository(&self) -> Option<&Path> {
        match self {
            Self::Path(path) => path
                .parent()
                .and_then(Path::parent)
                .filter(|root| root.join("Cargo.lock").is_file()),
            Self::Release => None,
        }
    }

    fn describe(&self) -> String {
        match self {
            Self::Path(path) => format!("path {}", path.display()),
            Self::Release => format!("crates.io {VERSION}"),
        }
    }
}

/// Writes the agent project for `plugins`. `jobs` replaces the remembered
/// `-j` when given.
pub(crate) fn generate(dirs: &Dirs, plugins: &[Plugin], jobs: Option<u32>) -> Result<(), Failure> {
    let project = dirs.project();
    let source = CodeSource::resolve()?;
    let stamp = format!("rig {VERSION}\nrig-code {}\n", source.describe());
    let stamp_file = project.join(".rig-stamp");
    let stale = std::fs::read_to_string(&stamp_file).ok();
    // A different launcher or source rewrites every file, so nothing of the
    // old generation survives.
    let force = stale.as_deref() != Some(stamp.as_str());
    if let (true, Some(stale)) = (force, &stale) {
        eprintln!(
            "rig: regenerating the agent project ({} -> {})",
            stale.lines().next().unwrap_or_default(),
            stamp.lines().next().unwrap_or_default()
        );
    }
    let config_file = project.join(".cargo").join("config.toml");
    let jobs = jobs.or_else(|| {
        std::fs::read_to_string(&config_file)
            .ok()
            .as_deref()
            .and_then(remembered_jobs)
    });
    write(
        &project.join("Cargo.toml"),
        &manifest(&source, plugins)?,
        force,
    )?;
    write(
        &project.join("src").join("main.rs"),
        &main_rs(plugins),
        force,
    )?;
    write(&config_file, &cargo_config(dirs, jobs), force)?;
    // A local checkout is built with its own tested dependency versions and
    // toolchain rather than the newest ones and the user's default.
    let toolchain = project.join("rust-toolchain.toml");
    match source.repository() {
        Some(repository) => {
            let lock = project.join("Cargo.lock");
            if force || !lock.exists() {
                std::fs::copy(repository.join("Cargo.lock"), &lock).map_err(|error| {
                    Failure::io(format!("cannot write {}", lock.display()), error)
                })?;
            }
            // Only the channel: the checkout's extra targets and components
            // are not needed to build the agent.
            if let Some(channel) = std::fs::read_to_string(repository.join("rust-toolchain.toml"))
                .ok()
                .as_deref()
                .and_then(toolchain_channel)
            {
                write(
                    &toolchain,
                    &format!("[toolchain]\nchannel = {}\n", quote(&channel)),
                    force,
                )?;
            }
        }
        None => {
            if toolchain.exists() {
                std::fs::remove_file(&toolchain).map_err(|error| {
                    Failure::io(format!("cannot remove {}", toolchain.display()), error)
                })?;
            }
        }
    }
    write(&stamp_file, &stamp, force)
}

/// Writes `content` to `path` unless it already holds exactly that.
fn write(path: &Path, content: &str, force: bool) -> Result<(), Failure> {
    if !force && std::fs::read_to_string(path).is_ok_and(|old| old == content) {
        return Ok(());
    }
    let failed = |error| Failure::io(format!("cannot write {}", path.display()), error);
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(failed)?;
    }
    std::fs::write(path, content).map_err(failed)
}

fn manifest(source: &CodeSource, plugins: &[Plugin]) -> Result<String, Failure> {
    let features: BTreeSet<&str> = plugins
        .iter()
        .flat_map(|plugin| plugin.bevy_features.iter().map(String::as_str))
        .collect();
    let features: Vec<String> = features.into_iter().map(quote).collect();
    let mut packages: Vec<&Package> = Vec::new();
    for package in plugins.iter().filter_map(|plugin| plugin.package.as_ref()) {
        if ["rig-code", "bevy", PACKAGE].contains(&package.name.as_str()) {
            return Err(Failure::config(format!(
                "plugins.toml: the crate name `{}` is taken by the agent project",
                package.name
            )));
        }
        match packages.iter().find(|kept| kept.name == package.name) {
            Some(kept) if *kept == package => {}
            Some(_) => {
                return Err(Failure::config(format!(
                    "plugins.toml: crate `{}` is listed with two different sources",
                    package.name
                )));
            }
            None => packages.push(package),
        }
    }
    let mut text = format!(
        "# Generated by rig {VERSION} from plugins.toml. Edits are overwritten.\n\
         [package]\n\
         name = \"{PACKAGE}\"\n\
         version = \"0.0.0\"\n\
         edition = \"2024\"\n\
         publish = false\n\
         \n\
         # Not part of any enclosing workspace.\n\
         [workspace]\n\
         \n\
         [dependencies]\n\
         rig-code = {}\n\
         bevy = {{ version = \"={BEVY_VERSION}\", default-features = false, features = [{}] }}\n",
        source.dependency(),
        features.join(", ")
    );
    for package in packages {
        let fields: Vec<String> = package
            .source
            .iter()
            .map(|(key, value)| format!("{key} = {}", quote(value)))
            .collect();
        // Writing into a String cannot fail.
        let _ = writeln!(
            text,
            "{} = {{ {} }}",
            quote(&package.name),
            fields.join(", ")
        );
    }
    text.push_str("\n[profile.dev]\ndebug = false\n");
    if let CodeSource::Path(path) = source {
        // A plugin that asks for rig-code by version gets this checkout too,
        // so the app holds one rig-code and sees the plugin's tools.
        let _ = write!(
            text,
            "\n[patch.crates-io]\nrig-code = {{ path = {} }}\n",
            quote(&path.display().to_string())
        );
    }
    Ok(text)
}

fn main_rs(plugins: &[Plugin]) -> String {
    let mut text = format!(
        "// Generated by rig {VERSION} from plugins.toml. Edits are overwritten.\n\
         fn main() -> rig_code::bevy::app::AppExit {{\n    rig_code::run(|app| {{\n"
    );
    for plugin in plugins {
        let _ = writeln!(
            text,
            "        app.add_plugins(<{} as Default>::default());",
            plugin.type_path
        );
    }
    if plugins.is_empty() {
        text.push_str("        let _ = app;\n");
    }
    text.push_str("    })\n}\n");
    text
}

fn cargo_config(dirs: &Dirs, jobs: Option<u32>) -> String {
    let target = dirs.cache.join("target").display().to_string();
    let mut text = format!(
        "# Generated by rig {VERSION}. `rig -j N` sets the jobs.\n[build]\ntarget-dir = {}\n",
        quote(&target)
    );
    if let Some(jobs) = jobs {
        let _ = writeln!(text, "jobs = {jobs}");
    }
    text
}

/// The `jobs = N` of a config this module wrote.
fn remembered_jobs(config: &str) -> Option<u32> {
    config.lines().find_map(|line| {
        line.strip_prefix("jobs = ")
            .and_then(|jobs| jobs.trim().parse().ok())
    })
}

/// The `channel = "..."` of a `rust-toolchain.toml`.
fn toolchain_channel(text: &str) -> Option<String> {
    text.lines().find_map(|line| {
        let (key, value) = line.split_once('=')?;
        (key.trim() == "channel").then(|| value.trim().trim_matches('"').to_owned())
    })
}

/// `text` as a TOML basic string.
fn quote(text: &str) -> String {
    let mut quoted = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => quoted.push_str("\\\""),
            '\\' => quoted.push_str("\\\\"),
            '\n' => quoted.push_str("\\n"),
            '\t' => quoted.push_str("\\t"),
            c if c.is_control() => {
                let _ = write!(quoted, "\\u{:04X}", u32::from(c));
            }
            c => quoted.push(c),
        }
    }
    quoted.push('"');
    quoted
}
