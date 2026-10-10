//! The generated agent project: `Cargo.toml`, `src/main.rs` and
//! `.cargo/config.toml` under `$RIG_HOME/project`. Each file is written only
//! when its contents change, so cargo's fingerprints stay valid, and each
//! names the launcher version, so a new launcher rebuilds the agent.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command;

use rig::harness_protocol::Home;

use super::config::{Config, Source, parse_string};
use super::{BEVY_VERSION, Result, VERSION};

/// The generated package's name, and so its binary's.
pub const PACKAGE: &str = "rig-harness-agent";

/// The lock this launcher was built with: the workspace's in a checkout,
/// the `rig` crate's own when installed from crates.io.
const LOCK: &str = include_str!(concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.lock"));

/// Where the agent project gets the `rig-harness` crate from.
pub enum RigSource {
    /// A rig checkout, by path.
    Local(PathBuf),
    /// crates.io, at this launcher's exact version.
    Registry,
}

impl RigSource {
    /// `RIG_SOURCE` when set; else the checkout this launcher was installed
    /// from (`cargo install --path`), when it is still there; else
    /// crates.io. A checkout in cargo's own cache (`cargo install --git`)
    /// does not count: cargo may delete it at any time.
    pub fn detect() -> Result<Self> {
        if let Some(checkout) = std::env::var_os("RIG_SOURCE").filter(|path| !path.is_empty()) {
            let checkout = std::path::absolute(PathBuf::from(checkout))?;
            if !is_checkout(&checkout) {
                return Err(format!(
                    "RIG_SOURCE={} is not a rig checkout: it has no crates/rig-harness/Cargo.toml",
                    checkout.display()
                )
                .into());
            }
            return Ok(Self::Local(checkout));
        }
        let installed_from = Path::new(env!("CARGO_MANIFEST_DIR"));
        let cargo_home = std::env::var_os("CARGO_HOME")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
            .or_else(|| std::env::home_dir().map(|home| home.join(".cargo")));
        let in_cargo_cache = cargo_home.is_some_and(|cargo| installed_from.starts_with(cargo));
        Ok(if is_checkout(installed_from) && !in_cargo_cache {
            Self::Local(installed_from.to_path_buf())
        } else {
            Self::Registry
        })
    }
}

fn is_checkout(path: &Path) -> bool {
    path.join("crates/rig-harness/Cargo.toml").is_file()
}

/// The rig crates a plugin may depend on, which the agent and its plugins
/// must share: one copy of each in the agent's dependency graph.
pub const RIG_CRATES: [&str; 5] = ["rig", "rig-core", "rig-ecs", "rig-tools", "rig-harness"];

/// The `[patch.crates-io]` table that builds [`RIG_CRATES`] from
/// `checkout`, so a plugin that names their crates.io release (as
/// `rig plugin new` writes it) uses the agent's own.
pub fn rig_patch(checkout: &Path) -> String {
    let mut text = String::from("[patch.crates-io]\n");
    for name in RIG_CRATES {
        let path = match name {
            "rig" => checkout.to_path_buf(),
            name => checkout.join("crates").join(name),
        };
        text.push_str(&format!("{name} = {{ path = {} }}\n", quoted_path(&path)));
    }
    text
}

/// The version of the rig crates the agent is built from: the checkout's
/// workspace version, else this launcher's.
pub fn rig_version(source: &RigSource) -> String {
    let RigSource::Local(checkout) = source else {
        return VERSION.to_owned();
    };
    fs::read_to_string(checkout.join("Cargo.toml"))
        .ok()
        .and_then(|manifest| manifest_string(&manifest, "workspace.package", "version"))
        .unwrap_or_else(|| VERSION.to_owned())
}

/// The string `key` of the table `[table]` of a Cargo manifest, such as the
/// `name` of `[package]`, when it is written as `key = "value"`.
pub fn manifest_string(manifest: &str, table: &str, key: &str) -> Option<String> {
    let mut current = String::new();
    for line in manifest.lines() {
        let line = line.trim();
        if let Some(header) = line
            .strip_prefix('[')
            .and_then(|line| line.strip_suffix(']'))
        {
            current = header.trim().to_owned();
            continue;
        }
        if current != table {
            continue;
        }
        let Some((name, value)) = line.split_once('=') else {
            continue;
        };
        if name.trim() == key {
            return parse_string(value.trim()).map(|(value, _)| value);
        }
    }
    None
}

/// Writes the agent project for `config`.
pub fn generate(home: &Home, config: &Config, source: &RigSource) -> Result<()> {
    let project = home.project();
    write_if_changed(&project.join("Cargo.toml"), &manifest(home, config, source))?;
    write_if_changed(&project.join("src/main.rs"), &main_rs(home, config, source))?;
    write_if_changed(&project.join(".cargo/config.toml"), &cargo_config(home))?;
    // A lock seeds the versions CI tested: the checkout's, or the one
    // packaged with this launcher. The packaged one covers the `rig` crate's
    // own dependencies only; cargo resolves rig-harness's and Bevy's fresh.
    let lock = project.join("Cargo.lock");
    if !lock.exists() {
        match source {
            RigSource::Local(checkout) => {
                fs::copy(checkout.join("Cargo.lock"), lock)?;
            }
            RigSource::Registry => fs::write(lock, LOCK)?,
        }
    }
    Ok(())
}

fn manifest(home: &Home, config: &Config, source: &RigSource) -> String {
    let features: BTreeSet<&str> = config
        .plugins
        .iter()
        .flat_map(|plugin| plugin.bevy_features.iter().map(String::as_str))
        .collect();
    let features: Vec<String> = features.into_iter().map(quoted).collect();
    let mut text = format!(
        "# Generated by rig {VERSION} from {}. `rig build` and /reload rewrite this file.\n\
         [package]\n\
         name = \"{PACKAGE}\"\n\
         version = \"0.0.0\"\n\
         edition = \"2024\"\n\
         publish = false\n\
         \n\
         # Its own workspace, so a RIG_HOME inside another workspace still builds.\n\
         [workspace]\n\
         \n\
         [dependencies]\n",
        home.config().display()
    );
    match source {
        RigSource::Local(checkout) => text.push_str(&format!(
            "rig-harness = {{ path = {} }}\n",
            quoted_path(&checkout.join("crates/rig-harness"))
        )),
        RigSource::Registry => {
            text.push_str(&format!("rig-harness = {{ version = \"={VERSION}\" }}\n"))
        }
    }
    // Only a plugin asking for Bevy features makes the project depend on the
    // `bevy` crate, which carries them; rig-harness itself uses the `bevy_*`
    // crates, which compile faster.
    if !features.is_empty() {
        text.push_str(&format!(
            "bevy = {{ version = \"={BEVY_VERSION}\", default-features = false, features = [{}] }}\n",
            features.join(", ")
        ));
    }
    let mut listed = BTreeSet::new();
    for package in config
        .plugins
        .iter()
        .filter_map(|plugin| plugin.package.as_ref())
    {
        // A crate with several plugins is one dependency.
        if !listed.insert(package.name.as_str()) {
            continue;
        }
        let source = match &package.source {
            Source::Path(path) => format!("path = {}", quoted_path(path)),
            Source::Git { url, branch, rev } => {
                let mut source = format!("git = {}", quoted(url));
                if let Some(branch) = branch {
                    source.push_str(&format!(", branch = {}", quoted(branch)));
                }
                if let Some(rev) = rev {
                    source.push_str(&format!(", rev = {}", quoted(rev)));
                }
                source
            }
            Source::Version(version) => format!("version = {}", quoted(version)),
        };
        text.push_str(&format!("{} = {{ {source} }}\n", package.name));
    }
    if let RigSource::Local(checkout) = source {
        // A plugin naming the crates.io releases builds against the checkout.
        text.push('\n');
        text.push_str(&rig_patch(checkout));
    }
    text.push_str(
        "\n[profile.dev]\ndebug = \"line-tables-only\"\n\
         \n[profile.dev.package.\"*\"]\nopt-level = 1\n",
    );
    text
}

/// The agent: the core, the [`revision`] it is built from, then each
/// plugin in the list's order, through `rig_harness::load`, which skips a
/// plugin an earlier one added and records where each comes from.
fn main_rs(home: &Home, config: &Config, source: &RigSource) -> String {
    let mut text = format!(
        "// Generated by rig {VERSION} from {}. /reload rewrites this file.\n\
         fn main() -> rig_harness::AppExit {{\n    \
             let mut app = rig_harness::App::new();\n    \
             // A failing system, observer or command from a plugin is logged.\n    \
             app.set_error_handler(rig_harness::error::warn);\n    \
             // Bevy's base and the headless loop come first, so the agent runs on\n    \
             // their clock and a windowing plugin listed below can replace the loop.\n    \
             app.add_plugins((rig_harness::HeadlessPlugins, rig_harness::RigHarnessPlugins));\n    \
             app.insert_resource(rig_harness::Build::running({:?}));\n",
        home.config().display(),
        revision(source),
    );
    let harness = match source {
        RigSource::Local(checkout) => {
            format!("path {}", checkout.join("crates/rig-harness").display())
        }
        RigSource::Registry => format!("version {VERSION}"),
    };
    for plugin in &config.plugins {
        let (krate, from) = match &plugin.package {
            Some(package) => (package.name.as_str(), package.source.to_string()),
            None if plugin.type_path.starts_with("bevy::") => {
                ("bevy", format!("version {BEVY_VERSION}"))
            }
            None => ("rig-harness", harness.clone()),
        };
        text.push_str(&format!(
            "    rig_harness::load::<{}>(&mut app, {krate:?}, {from:?});\n",
            plugin.type_path
        ));
    }
    text.push_str("    app.run()\n}\n");
    text
}

/// The commit of the rig checkout the agent is built from, else the
/// crates.io version.
fn revision(source: &RigSource) -> String {
    let RigSource::Local(checkout) = source else {
        return VERSION.to_owned();
    };
    Command::new("git")
        .arg("-C")
        .arg(checkout)
        .args(["rev-parse", "--short=12", "HEAD"])
        .output()
        .ok()
        .filter(|output| output.status.success())
        .and_then(|output| String::from_utf8(output.stdout).ok())
        .map(|revision| revision.trim().to_owned())
        .filter(|revision| !revision.is_empty())
        .unwrap_or_else(|| rig_version(source))
}

/// cargo's settings: the target directory. `CARGO_BUILD_JOBS` sets the
/// jobs, as for any cargo build.
fn cargo_config(home: &Home) -> String {
    format!(
        "# Generated by rig {VERSION}.\n[build]\ntarget-dir = {}\n",
        quoted_path(&home.target())
    )
}

/// Writes `contents` to `path` unless it already holds them.
pub fn write_if_changed(path: &Path, contents: &str) -> Result<()> {
    if fs::read_to_string(path).is_ok_and(|current| current == contents) {
        return Ok(());
    }
    if let Some(parent) = path.parent() {
        fs::create_dir_all(parent)?;
    }
    fs::write(path, contents)?;
    Ok(())
}

pub fn quoted_path(path: &Path) -> String {
    quoted(&path.to_string_lossy())
}

/// A TOML basic string.
pub fn quoted(text: &str) -> String {
    let mut quoted = String::from("\"");
    for c in text.chars() {
        match c {
            '"' => quoted.push_str("\\\""),
            '\\' => quoted.push_str("\\\\"),
            c if c.is_control() => quoted.push_str(&format!("\\u{:04X}", u32::from(c))),
            c => quoted.push(c),
        }
    }
    quoted.push('"');
    quoted
}
