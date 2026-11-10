//! The generated agent project: `Cargo.toml`, `src/main.rs` and
//! `.cargo/config.toml` under `$RIG_HOME/project`. It holds exactly the
//! plugins `RIG_HOME/settings.toml` turns on, added as one `PluginGroup`,
//! so turning a plugin on or off rewrites it and the next start rebuilds
//! the agent: only this crate compiles again, and a plugin's own crate the
//! first time it is on. Each file is written
//! only when its contents change, so cargo's fingerprints stay valid, and
//! each names the launcher version, so a new launcher rebuilds the agent.

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use rig_harness_protocol::config::toml_edit::{DocumentMut, Item};
use rig_harness_protocol::config::{self, BUILT_IN, Extra, Plugins, Source, built_in};
use rig_harness_protocol::{Home, env};

use super::{Result, VERSION};

/// The terminal view, whose plugin alone needs rig-basics' `tui` feature.
const TUI: &str = "rig_tui::TuiPlugin";

/// The generated package's name, and so its binary's.
pub const PACKAGE: &str = "rig-harness-agent";

/// The rig checkout this launcher is built in, when it is built from one:
/// `crates/rig-cli` is two folders below its root.
pub(super) const CHECKOUT: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/../..");

/// The crates of rig's own plugins that are on.
fn built_in_crates(plugins: &Plugins) -> BTreeSet<&'static str> {
    (plugins.on.iter())
        .filter_map(|plugin| built_in(plugin))
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
    /// crates.io. A checkout in cargo's own cache (`cargo install
    /// --git`) does not count: cargo may delete it at any time.
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

    /// Fails when this is crates.io and the launcher, whose [`LONG_VERSION`](super::LONG_VERSION)
    /// is `long`, was built from a commit: its generated agent needs that
    /// commit's crates, which crates.io's release may not be.
    pub fn check(&self, long: &str) -> Result<()> {
        if matches!(self, Self::Local(_)) || long == VERSION {
            return Ok(());
        }
        Err(format!(
            "rig {long} was built from a rig checkout that is gone or in cargo's cache \
             (`cargo install --git`), so its agent cannot build from crates.io's rig {VERSION}: \
             run `cargo install --path crates/rig-cli` in a checkout you keep, or set \
             RIG_SOURCE to one"
        )
        .into())
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

/// The rig crates the agent uses: [`CORE_CRATES`], the crates of rig's
/// plugins that are on, the rig crates an extra crate on by path depends
/// on, and what each of those that is a plugin crate depends on in turn.
/// Without a checkout, [`CORE_CRATES`] and the crates of rig's plugins on.
/// Fails when one of those crates' `Cargo.toml` cannot be read.
pub fn rig_crates(plugins: &Plugins, source: &RigSource) -> Result<BTreeSet<String>> {
    let mut used: BTreeSet<String> = CORE_CRATES.map(str::to_owned).into();
    let on = built_in_crates(plugins).into_iter().map(str::to_owned);
    let RigSource::Local(checkout) = source else {
        used.extend(on);
        return Ok(used);
    };
    let mut pending: Vec<String> = on.collect();
    for extra in plugins.extras_on() {
        if let Source::Path(directory) = &extra.source {
            pending.extend(dependencies(directory)?);
        }
    }
    while let Some(name) = pending.pop() {
        let Some(directory) = crate_dir(checkout, &name) else {
            continue;
        };
        if used.insert(name) && directory.starts_with(checkout.join("plugins")) {
            pending.extend(dependencies(&directory)?);
        }
    }
    Ok(used)
}

/// The manifest of the crate in `directory`. The error names the file and
/// says why: the operating system's error, or the TOML error.
pub fn read_manifest(directory: &Path) -> Result<DocumentMut> {
    let path = directory.join("Cargo.toml");
    let text = fs::read_to_string(&path)
        .map_err(|error| format!("cannot read {}: {error}", path.display()))?;
    Ok((text.parse()).map_err(|error| format!("{} is not TOML: {error}", path.display()))?)
}

/// The names of the dependencies the crate in `directory` always has: the
/// keys of its `[dependencies]` tables, its target-specific ones included,
/// without the optional ones.
fn dependencies(directory: &Path) -> Result<Vec<String>> {
    let manifest = read_manifest(directory)?;
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
    Ok(tables
        .flat_map(|table| {
            (table.iter())
                .filter(|(_, item)| required(item))
                .map(|(name, _)| name.to_owned())
                .collect::<Vec<_>>()
        })
        .collect())
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
    let entry = if BUILT_IN.iter().any(|built_in| built_in.krate == name) {
        "Its plugins are rig's own: name them in `[plugins] enable` instead, or build"
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
    let manifest = read_manifest(checkout).ok();
    let version = manifest.as_ref().and_then(|manifest| {
        let workspace = manifest.get("workspace")?.get("package")?;
        workspace.get("version")?.as_str().map(str::to_owned)
    });
    version.unwrap_or_else(|| VERSION.to_owned())
}

/// Whether the build `binary` is what building now would make, as far as
/// the launcher can tell without cargo: the agent project generated from
/// the settings and source of now is the one on disk, and `binary` is
/// newer than its files and than every source file the last agent build
/// read ([`dep_info`]), a checkout's and a path plugin's included. False
/// when that cannot be read.
pub fn is_current(home: &Home, binary: &Path) -> bool {
    let current = || -> Result<bool> {
        let plugins = config::chosen(home)?;
        let source = RigSource::detect()?;
        let built = fs::metadata(binary)?.modified()?;
        let not_newer =
            |path: &Path| -> Result<bool> { Ok(fs::metadata(path)?.modified()? <= built) };
        let mut same = true;
        for (file, text) in [
            ("Cargo.toml", manifest(home, &plugins, &source)?),
            ("src/main.rs", main_rs(home, &plugins, &source)),
        ] {
            let path = home.project().join(file);
            same = same
                && fs::read_to_string(&path).is_ok_and(|written| written == text)
                && not_newer(&path)?;
        }
        for path in sources(&fs::read_to_string(dep_info(home))?) {
            same = same && not_newer(&path)?;
        }
        Ok(same)
    };
    current().unwrap_or(false)
}

/// cargo's dep-info file of the agent binary: every source file outside
/// the registry that its last build read.
pub fn dep_info(home: &Home) -> PathBuf {
    home.target().join("debug").join(format!("{PACKAGE}.d"))
}

/// The paths a dep-info file lists after its target's `: `; a space in a
/// path is written `\ `.
fn sources(dep_info: &str) -> Vec<PathBuf> {
    let Some((_, listed)) = dep_info
        .lines()
        .next()
        .and_then(|line| line.split_once(": "))
    else {
        return Vec::new();
    };
    let mut paths = Vec::new();
    let mut path = String::new();
    let mut chars = listed.chars();
    while let Some(c) = chars.next() {
        match c {
            '\\' => path.extend(chars.next()),
            ' ' if !path.is_empty() => paths.push(PathBuf::from(std::mem::take(&mut path))),
            ' ' => {}
            c => path.push(c),
        }
    }
    if !path.is_empty() {
        paths.push(PathBuf::from(path));
    }
    paths
}

/// Writes the agent project with the plugins on of `plugins`.
pub fn generate(home: &Home, plugins: &Plugins, source: &RigSource) -> Result<()> {
    let project = home.project();
    write_if_changed(
        &project.join("Cargo.toml"),
        &manifest(home, plugins, source)?,
    )?;
    write_if_changed(
        &project.join("src/main.rs"),
        &main_rs(home, plugins, source),
    )?;
    write_if_changed(&project.join(".cargo/config.toml"), &cargo_config(home))?;
    // A checkout's lock seeds the versions CI tested, again whenever it
    // changes, so a version switch builds offline with that version's lock.
    let lock = project.join("Cargo.lock");
    let seed = project.join("Cargo.lock.seed");
    if let RigSource::Local(checkout) = source {
        let checkouts = fs::read(checkout.join("Cargo.lock"))?;
        if !lock.exists() || fs::read(&seed).ok().as_deref() != Some(checkouts.as_slice()) {
            fs::write(&lock, &checkouts)?;
            fs::write(&seed, checkouts)?;
        }
    }
    Ok(())
}

fn manifest(home: &Home, plugins: &Plugins, source: &RigSource) -> Result<String> {
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
    for name in ["rig-harness"].into_iter().chain(built_in_crates(plugins)) {
        let mut from = rig_source(source, name)?;
        // rig-basics' default `tui` feature builds rig-tui, for the terminal
        // view alone.
        if name == "rig-basics" {
            from.push_str(", default-features = false");
            if plugins.on.iter().any(|plugin| plugin == TUI) {
                from.push_str(", features = [\"tui\"]");
            }
        }
        text.push_str(&format!("{name} = {{ {from} }}\n"));
    }
    let mut listed = BTreeSet::new();
    for extra in plugins.extras_on() {
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
        text.push_str(&rig_patch(checkout, &rig_crates(plugins, source)?));
    }
    text.push_str(&profile(plugins, source)?);
    Ok(text)
}

/// The agent's build profile: dependencies at full speed, built once, and
/// the crates `/reload` rebuilds (rig's own from a checkout, and extras by
/// path or git) at opt-level 1, so a rebuild stays quick.
fn profile(plugins: &Plugins, source: &RigSource) -> Result<String> {
    let mut text = String::from(
        "\n[profile.dev]\ndebug = \"line-tables-only\"\n\
         \n[profile.dev.package.\"*\"]\nopt-level = 3\ndebug-assertions = false\n\
         overflow-checks = false\n",
    );
    let mut edited: BTreeSet<String> = match source {
        RigSource::Local(_) => rig_crates(plugins, source)?,
        RigSource::Registry => BTreeSet::new(),
    };
    edited.extend(
        (plugins.extras_on())
            .filter(|extra| matches!(extra.source, Source::Path(_) | Source::Git { .. }))
            .map(|extra| extra.krate.clone()),
    );
    for name in edited {
        text.push_str(&format!("\n[profile.dev.package.{name}]\nopt-level = 1\n"));
    }
    Ok(text)
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

/// The agent: the harness, then the plugins on, in their order, as the
/// `PluginGroup` `Chosen` of [`rig_harness::Loaded`] plugins;
/// `rig_harness::run` runs it, or describes what its plugins add when the
/// launcher asks.
fn main_rs(home: &Home, plugins: &Plugins, source: &RigSource) -> String {
    let mut text = format!(
        "// Generated by rig {VERSION} from {}. `rig build` and /reload rewrite this file.\n\
         fn main() -> rig_harness::AppExit {{\n    \
             let mut app = rig_harness::App::new();\n    \
             // A failing system, observer or command from a plugin is logged.\n    \
             app.set_error_handler(rig_harness::error::warn);\n    \
             // Bevy's base and the headless loop come first, so the agent runs on\n    \
             // their clock and a windowing plugin added later can replace the loop.\n    \
             app.add_plugins((rig_harness::HeadlessPlugins, rig_harness::RigHarnessPlugins, Chosen));\n    \
             // A debug build warns at start of agent-loop systems left unordered.\n    \
             #[cfg(debug_assertions)]\n    \
             rig_harness::check_stages_at_start(&mut app);\n    \
             // Runs the app, or describes it when the launcher asks.\n    \
             rig_harness::run(app)\n\
         }}\n\
         \n\
         /// The plugins settings.toml turns on, in the order the agent adds them.\n\
         struct Chosen;\n\
         \n\
         impl rig_harness::PluginGroup for Chosen {{\n    \
             fn build(self) -> rig_harness::PluginGroupBuilder {{\n        \
                 rig_harness::PluginGroupBuilder::start::<Self>()\n",
        home.settings().display(),
    );
    for plugin in &plugins.on {
        let (krate, from) = match built_in(plugin) {
            Some(built_in) => (built_in.krate, rig_origin(source, built_in.krate)),
            None => match plugins.extra.iter().find(|extra| extra.plugin == *plugin) {
                Some(extra) => (extra.krate.as_str(), extra.source.to_string()),
                None => continue,
            },
        };
        text.push_str(&format!(
            "            .add(rig_harness::Loaded::<{plugin}>::new({krate:?}, {from:?}))\n"
        ));
    }
    text.push_str("    }\n}\n");
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
