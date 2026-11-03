//! The generated agent project: `Cargo.toml`, `src/main.rs` and
//! `.cargo/config.toml` under `$RIG_HOME/project`. It holds every one of
//! rig's own plugins that builds on this target and the home's extra
//! crates, so turning a plugin on or off needs no build: the agent adds
//! those its settings turn on when it starts ([`rig_harness::load_chosen`]).
//! Each file is written only when its contents change, so cargo's
//! fingerprints stay valid, and each names the launcher version, so a new
//! launcher rebuilds the agent.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use rig_harness_protocol::config::toml_edit::{DocumentMut, Item};
use rig_harness_protocol::config::{BUILT_IN, Extra, Source};
use rig_harness_protocol::{Home, env};

use super::{Result, VERSION};

/// The generated package's name, and so its binary's.
pub const PACKAGE: &str = "rig-harness-agent";

/// The rig checkout this launcher is built in, when it is built from one:
/// `crates/rig-cli` is two folders below its root.
pub(super) const CHECKOUT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

/// The crates of rig's own plugins that build on this target.
pub fn built_in_crates() -> BTreeSet<&'static str> {
    (BUILT_IN.iter())
        .filter(|built_in| built_in.available())
        .map(|built_in| built_in.krate)
        .collect()
}

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
        if let Some(checkout) = std::env::var_os(env::SOURCE).filter(|path| !path.is_empty()) {
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
        let installed_from = std::path::absolute(CHECKOUT)?;
        let cargo_home = std::env::var_os("CARGO_HOME")
            .filter(|path| !path.is_empty())
            .map(PathBuf::from)
            .or_else(|| std::env::home_dir().map(|home| home.join(".cargo")));
        let in_cargo_cache = cargo_home.is_some_and(|cargo| installed_from.starts_with(cargo));
        Ok(if is_checkout(&installed_from) && !in_cargo_cache {
            Self::Local(installed_from)
        } else {
            Self::Registry
        })
    }
}

fn is_checkout(path: &Path) -> bool {
    path.join("crates/rig-harness/Cargo.toml").is_file()
}

/// The rig crates every agent has: rig-harness and the rig crates it
/// depends on. A plugin may depend on each, and the agent and its plugins
/// must share them: one copy of each in the agent's dependency graph.
pub const CORE_CRATES: [&str; 7] = [
    "rig-core",
    "rig-ecs",
    "rig-harness",
    "rig-harness-protocol",
    "rig-http",
    "rig-memory",
    "rig-tools",
];

/// The directory of the rig crate `name` in `checkout`: the checkout itself
/// for the `rig` facade, else `crates/<name>` or `plugins/<name>`.
pub fn crate_dir(checkout: &Path, name: &str) -> Option<PathBuf> {
    if name == "rig" {
        return Some(checkout.to_path_buf());
    }
    ["crates", "plugins"]
        .into_iter()
        .map(|folder| checkout.join(folder).join(name))
        .find(|directory| directory.join("Cargo.toml").is_file())
}

/// The rig crates the agent uses: [`CORE_CRATES`], rig's plugin crates,
/// the rig crates an extra crate by path depends on, and what each of
/// those that is a plugin crate depends on in turn. Without a checkout,
/// [`CORE_CRATES`] and rig's plugin crates.
pub fn rig_crates(extras: &[Extra], source: &RigSource) -> BTreeSet<String> {
    let mut used: BTreeSet<String> = CORE_CRATES.map(str::to_owned).into();
    let RigSource::Local(checkout) = source else {
        used.extend(built_in_crates().into_iter().map(str::to_owned));
        return used;
    };
    let mut pending: Vec<String> = built_in_crates().into_iter().map(str::to_owned).collect();
    for extra in extras {
        if let Source::Path(directory) = &extra.source {
            pending.extend(dependencies(directory));
        }
    }
    while let Some(name) = pending.pop() {
        let Some(directory) = crate_dir(checkout, &name) else {
            continue;
        };
        if used.insert(name) && directory.starts_with(checkout.join("plugins")) {
            pending.extend(dependencies(&directory));
        }
    }
    used
}

/// The manifest of the crate in `directory`, if it reads.
pub fn read_manifest(directory: &Path) -> Option<DocumentMut> {
    fs::read_to_string(directory.join("Cargo.toml"))
        .ok()?
        .parse()
        .ok()
}

/// The names of the dependencies the crate in `directory` always has: the
/// keys of its `[dependencies]` tables, its target-specific ones included,
/// without the optional ones.
fn dependencies(directory: &Path) -> Vec<String> {
    let Some(manifest) = read_manifest(directory) else {
        return Vec::new();
    };
    let targets = (manifest.get("target").and_then(Item::as_table_like))
        .into_iter()
        .flat_map(|targets| targets.iter().map(|(_, target)| target.get("dependencies")));
    let tables = [manifest.get("dependencies")]
        .into_iter()
        .chain(targets)
        .flatten()
        .filter_map(Item::as_table_like);
    let required = |item: &Item| {
        let optional = item.get("optional").and_then(Item::as_bool);
        optional != Some(true)
    };
    tables
        .flat_map(|table| {
            (table.iter())
                .filter(|(_, item)| required(item))
                .map(|(name, _)| name.to_owned())
                .collect::<Vec<_>>()
        })
        .collect()
}

/// What is wrong with `extra` when it is a crate of a rig checkout listed
/// by path while the agent's rig crates come from elsewhere (`source`): it
/// would bring a second copy of each rig crate it uses.
pub fn foreign_rig_crate(extra: &Extra, source: &RigSource) -> Option<String> {
    let Source::Path(directory) = &extra.source else {
        return None;
    };
    // Through symlinks and `..`, so an alias of the directory is the same. A
    // directory that is not there is no crate of a checkout; cargo says so.
    let resolve = |path: &Path| fs::canonicalize(path).unwrap_or_else(|_| path.to_path_buf());
    let directory = fs::canonicalize(directory).ok()?;
    let checkout = directory
        .parent()
        .filter(|folder| folder.ends_with("plugins") || folder.ends_with("crates"))?
        .parent()
        .filter(|checkout| is_checkout(checkout))?;
    let agent = match source {
        RigSource::Local(own) if resolve(own) == checkout => return None,
        RigSource::Local(own) => format!("the rig checkout {}", own.display()),
        RigSource::Registry => format!("crates.io at rig {VERSION}"),
    };
    let name = &extra.krate;
    let checkout = checkout.display();
    let entry = if built_in_crates().contains(name.as_str()) {
        "Its plugins are rig's own, which every agent has: name them in `[plugins] enable` \
         instead, or build"
    } else {
        "rig ships no crate of that name; build"
    };
    Some(format!(
        "`{name}` is listed by path from the rig checkout {checkout}, but the agent's rig \
         crates come from {agent}, so it would bring a second copy of them. {entry} the agent \
         from that checkout with RIG_SOURCE={checkout}."
    ))
}

/// The `[patch.crates-io]` table that builds the rig crates `used` from
/// `checkout`, so a plugin that names their crates.io release (as `rig
/// plugin new` writes it) uses the agent's own. It names only crates the
/// agent uses, since cargo warns about a patch nothing uses.
pub fn rig_patch(checkout: &Path, used: &BTreeSet<String>) -> String {
    let mut text = String::from("[patch.crates-io]\n");
    for name in used {
        if let Some(directory) = crate_dir(checkout, name) {
            text.push_str(&format!(
                "{name} = {{ path = {} }}\n",
                quoted_path(&directory)
            ));
        }
    }
    text
}

/// The version of the rig crates the agent is built from: the checkout's
/// workspace version, else this launcher's.
pub fn rig_version(source: &RigSource) -> String {
    let RigSource::Local(checkout) = source else {
        return VERSION.to_owned();
    };
    let manifest = read_manifest(checkout);
    let version = manifest.as_ref().and_then(|manifest| {
        let workspace = manifest.get("workspace")?.get("package")?;
        workspace.get("version")?.as_str().map(str::to_owned)
    });
    version.unwrap_or_else(|| VERSION.to_owned())
}

/// Writes the agent project with rig's plugins and `extras`.
pub fn generate(home: &Home, extras: &[Extra], source: &RigSource) -> Result<()> {
    let project = home.project();
    write_if_changed(
        &project.join("Cargo.toml"),
        &manifest(home, extras, source)?,
    )?;
    write_if_changed(&project.join("src/main.rs"), &main_rs(home, extras, source))?;
    write_if_changed(&project.join(".cargo/config.toml"), &cargo_config(home))?;
    // A checkout's lock seeds the versions CI tested.
    let lock = project.join("Cargo.lock");
    if let RigSource::Local(checkout) = source
        && !lock.exists()
    {
        fs::copy(checkout.join("Cargo.lock"), lock)?;
    }
    Ok(())
}

fn manifest(home: &Home, extras: &[Extra], source: &RigSource) -> Result<String> {
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
        home.settings().display()
    );
    for name in ["rig-harness"].into_iter().chain(built_in_crates()) {
        text.push_str(&format!("{name} = {{ {} }}\n", rig_source(source, name)?));
    }
    let mut listed = BTreeSet::new();
    for extra in extras {
        // A crate with several plugins is one dependency.
        if !listed.insert(extra.krate.as_str()) {
            continue;
        }
        let from = match &extra.source {
            Source::Path(path) => {
                if let Some(problem) = foreign_rig_crate(extra, source) {
                    return Err(problem.into());
                }
                format!("path = {}", quoted_path(path))
            }
            Source::Git { url, branch, rev } => {
                let mut from = format!("git = {}", quoted(url));
                if let Some(branch) = branch {
                    from.push_str(&format!(", branch = {}", quoted(branch)));
                }
                if let Some(rev) = rev {
                    from.push_str(&format!(", rev = {}", quoted(rev)));
                }
                from
            }
            Source::Version(version) => format!("version = {}", quoted(version)),
        };
        text.push_str(&format!("{} = {{ {from} }}\n", extra.krate));
    }
    if let RigSource::Local(checkout) = source {
        // A plugin naming the crates.io releases builds against the checkout.
        text.push('\n');
        text.push_str(&rig_patch(checkout, &rig_crates(extras, source)));
    }
    text.push_str(&profile(extras, source));
    Ok(text)
}

/// The agent's build profile: dependencies at full speed, built once, and
/// the crates `/reload` rebuilds (rig's own from a checkout, and extras by
/// path or git) at opt-level 1, so a rebuild stays quick.
fn profile(extras: &[Extra], source: &RigSource) -> String {
    let mut text = String::from(
        "\n[profile.dev]\ndebug = \"line-tables-only\"\n\
         \n[profile.dev.package.\"*\"]\nopt-level = 3\ndebug-assertions = false\n\
         overflow-checks = false\n",
    );
    let mut edited: BTreeSet<String> = match source {
        RigSource::Local(_) => rig_crates(extras, source),
        RigSource::Registry => BTreeSet::new(),
    };
    edited.extend(
        (extras.iter())
            .filter(|extra| matches!(extra.source, Source::Path(_) | Source::Git { .. }))
            .map(|extra| extra.krate.clone()),
    );
    for name in edited {
        text.push_str(&format!("\n[profile.dev.package.{name}]\nopt-level = 1\n"));
    }
    text
}

/// The dependency source of the rig crate `name`: its directory in the
/// checkout, else this launcher's exact version on crates.io.
fn rig_source(source: &RigSource, name: &str) -> Result<String> {
    Ok(match source {
        RigSource::Local(checkout) => {
            let directory = crate_dir(checkout, name).ok_or_else(|| {
                format!(
                    "the rig checkout {} has no crates/{name} or plugins/{name}, which this \
                     launcher builds the agent with; build it with the launcher of that checkout",
                    checkout.display()
                )
            })?;
            format!("path = {}", quoted_path(&directory))
        }
        RigSource::Registry => format!("version = \"={VERSION}\""),
    })
}

/// Where the rig crate `name` comes from, as the agent's `PluginSource`
/// of a plugin names it: `path <directory>` or `version <version>`.
fn rig_origin(source: &RigSource, name: &str) -> String {
    match source {
        RigSource::Local(checkout) => crate_dir(checkout, name)
            .map(|directory| format!("path {}", directory.display()))
            .unwrap_or_default(),
        RigSource::Registry => format!("version {VERSION}"),
    }
}

/// The agent: the harness, then every plugin it is built with, of which
/// `rig_harness::load_chosen` adds those the settings turn on, in their
/// order; `rig_harness::run` runs it, or describes what its plugins add
/// when the launcher asks.
fn main_rs(home: &Home, extras: &[Extra], source: &RigSource) -> String {
    let mut text = format!(
        "// Generated by rig {VERSION} from {}. /reload rewrites this file.\n\
         fn main() -> rig_harness::AppExit {{\n    \
             let mut app = rig_harness::App::new();\n    \
             // A failing system, observer or command from a plugin is logged.\n    \
             app.set_error_handler(rig_harness::error::warn);\n    \
             // Bevy's base and the headless loop come first, so the agent runs on\n    \
             // their clock and a windowing plugin added later can replace the loop.\n    \
             app.add_plugins((rig_harness::HeadlessPlugins, rig_harness::RigHarnessPlugins));\n    \
             rig_harness::load_chosen(\n        \
                 &mut app,\n        \
                 &[\n",
        home.settings().display(),
    );
    let built_in = (BUILT_IN.iter())
        .filter(|built_in| built_in.available())
        .map(|built_in| {
            let from = rig_origin(source, built_in.krate);
            (built_in.plugin, built_in.krate, from)
        });
    let extra = extras.iter().map(|extra| {
        (
            extra.plugin.as_str(),
            extra.krate.as_str(),
            extra.source.to_string(),
        )
    });
    for (plugin, krate, from) in built_in.chain(extra) {
        text.push_str(&format!(
            "            rig_harness::Compiled::new::<{plugin}>({plugin:?}, {krate:?}, {from:?}),\n"
        ));
    }
    text.push_str(
        "        ],\n    );\n    \
         // A debug build warns at start of agent-loop systems left unordered.\n    \
         #[cfg(debug_assertions)]\n    \
         rig_harness::check_stages_at_start(&mut app);\n",
    );
    // Runs the app, or describes it when the launcher asks.
    text.push_str("    rig_harness::run(app)\n}\n");
    text
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

/// `path` as a TOML basic string.
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

#[cfg(test)]
mod tests;
