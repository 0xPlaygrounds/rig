//! `rig plugin`: make a plugin crate under `RIG_HOME/plugins` and list it
//! in `plugins.toml` (`new`), add or remove an entry (`add`, `remove`),
//! show the list (`list`), and check it (`check`, with `--build` also
//! building the agent without staging it). Every
//! change is checked before it is written, so nobody edits the file by
//! hand. Plugin crates live outside every workspace, so making one never
//! touches the rig repository or a project.

use std::fs;
use std::path::{Path, PathBuf};

use rig::harness_protocol::Home;

use super::Result;
use super::config::{self, Config, Source};
use super::project::{
    PACKAGE, RIG_CRATES, RigSource, manifest_string, quoted, quoted_path, rig_patch, rig_version,
};

/// The `src/lib.rs` of a new plugin crate, with `ScaffoldPlugin` and
/// `__name__` for the plugin type and the crate name. rig-harness's
/// `tests/scaffold.rs` builds it, so it stays a working plugin.
const SCAFFOLD: &str = include_str!("plugin/scaffold.rs");

/// `rig plugin`'s usage.
pub const USAGE: &str = "\
  rig plugin new <name>  Make a plugin crate in RIG_HOME/plugins/<name> and add
                         it to plugins.toml; /reload or `rig build` builds it.
  rig plugin add <type> [--path <dir> | --git <url> [--branch <b> | --rev <r>]
                 | --version <req>] [--crate <name>] [--bevy-features <a,b>]
                         Add the plugin type <type> to plugins.toml, from the
                         crate at that source (its name read from <dir> for
                         --path), or one of rig-harness's own without a source.
  rig plugin remove <type> [--delete]
                         Remove the plugin type <type> from plugins.toml; its
                         crate stays where it is, unless --delete is given and
                         it is a crate of RIG_HOME/plugins no other entry uses.
  rig plugin list        List the plugins in plugins.toml, in the order they
                         are added.
  rig plugin check [--build]
                         Check plugins.toml and the plugin crates it names by
                         path; with --build, also build the agent with them
                         (in RIG_HOME/target, as /reload does) without
                         staging the build.
";

/// Runs `rig plugin` with `args`, the words after `plugin`.
pub fn run(home: &Home, args: &[&str]) -> Result<()> {
    match args {
        ["new", name] => new(home, name),
        ["add", type_path, options @ ..] => add(home, type_path, options),
        ["remove", type_path] => remove(home, type_path, false),
        ["remove", type_path, "--delete"] | ["remove", "--delete", type_path] => {
            remove(home, type_path, true)
        }
        ["list"] => list(home),
        ["check"] => check(home),
        ["check", "--build"] => check(home).and_then(|()| check_build(home)),
        _ => Err(format!("usage:\n{USAGE}").into()),
    }
}

/// `rig plugin new <name>`: the crate `name` in `RIG_HOME/plugins/<name>`,
/// with one plugin that adds `/<name>`, listed at the end of
/// `plugins.toml`.
fn new(home: &Home, name: &str) -> Result<()> {
    validate_name(name)?;
    let config_path = home.config();
    let config = Config::load(&config_path)?;
    if let Some(listed) = config.plugins.iter().find(|plugin| {
        plugin
            .package
            .as_ref()
            .is_some_and(|package| package.name == name)
    }) {
        return Err(format!(
            "plugins.toml already lists the crate `{name}` (as `{}`)",
            listed.type_path
        )
        .into());
    }
    let directory = home.plugins().join(name);
    if directory.exists() {
        return Err(format!(
            "{} exists already; list it in plugins.toml or pick another name",
            directory.display()
        )
        .into());
    }
    let library = name.replace('-', "_");
    let type_name = type_name(name);
    let source = RigSource::detect()?;
    fs::create_dir_all(directory.join("src"))?;
    fs::write(
        directory.join("Cargo.toml"),
        manifest(name, &rig_version(&source), &source),
    )?;
    fs::write(directory.join("src/lib.rs"), lib_rs(name, &type_name))?;
    fs::create_dir_all(directory.join(".cargo"))?;
    fs::write(
        directory.join(".cargo/config.toml"),
        format!(
            "# The agent's target directory, so building this crate on its own reuses\n\
             # the agent build's dependencies. `rig plugin check --build` builds the\n\
             # agent with this crate, which is the check that counts.\n\
             [build]\ntarget-dir = {}\n",
            quoted_path(&home.target())
        ),
    )?;
    let entry = format!(
        "# Made by `rig plugin new {name}`.\n[[plugin]]\ncrate = {}\npath = {}\nplugin = {}\n",
        quoted(name),
        quoted(&format!("plugins/{name}")),
        quoted(&format!("{library}::{type_name}")),
    );
    if let Err(failure) = config::append(&config_path, &entry, |_| Ok(())) {
        fs::remove_dir_all(&directory).ok();
        return Err(failure);
    }
    println!(
        "Made the plugin crate {} with `{library}::{type_name}`, which adds /{name},\n\
         and listed it in {}.\n\
         `rig plugin check --build` builds the agent with it; /reload in the agent,\n\
         or `rig build`, builds it and applies it.",
        directory.display(),
        config_path.display()
    );
    Ok(())
}

/// `rig plugin add <type> [options]`: an entry for `type_path` at the end
/// of `plugins.toml`, written only when the whole file still parses and,
/// for a crate by path, the crate is there and provides the type's crate.
fn add(home: &Home, type_path: &str, options: &[&str]) -> Result<()> {
    let mut keys: Vec<(&str, String)> = Vec::new();
    let mut name: Option<String> = None;
    let mut features: Option<Vec<String>> = None;
    let mut rest = options.iter();
    while let Some(&option) = rest.next() {
        let value = rest
            .next()
            .ok_or_else(|| format!("`{option}` needs a value\nusage:\n{USAGE}"))?;
        match option {
            "--path" => {
                let directory = std::path::absolute(value)?;
                name = name.or_else(|| manifest_name(&directory));
                keys.push(("path", relative_to(&directory, home.root())));
            }
            "--git" => keys.push(("git", (*value).to_owned())),
            "--branch" => keys.push(("branch", (*value).to_owned())),
            "--rev" => keys.push(("rev", (*value).to_owned())),
            "--version" => keys.push(("version", (*value).to_owned())),
            "--crate" => name = Some((*value).to_owned()),
            "--bevy-features" => {
                features = Some(
                    value
                        .split(',')
                        .map(str::trim)
                        .filter(|feature| !feature.is_empty())
                        .map(str::to_owned)
                        .collect(),
                );
            }
            _ => return Err(format!("unknown option `{option}`\nusage:\n{USAGE}").into()),
        }
    }
    let mut entry = String::from("# Added by `rig plugin add`.\n[[plugin]]\n");
    if let Some(name) = &name {
        entry.push_str(&format!("crate = {}\n", quoted(name)));
    }
    for (key, value) in &keys {
        entry.push_str(&format!("{key} = {}\n", quoted(value)));
    }
    entry.push_str(&format!("plugin = {}\n", quoted(type_path)));
    if let Some(features) = &features {
        let list: Vec<String> = features.iter().map(|feature| quoted(feature)).collect();
        entry.push_str(&format!("bevy_features = [{}]\n", list.join(", ")));
    }
    let config_path = home.config();
    config::append(&config_path, &entry, |config| {
        match config.plugins.last().and_then(check_plugin) {
            Some(problem) => Err(problem.into()),
            None => Ok(()),
        }
    })?;
    println!(
        "Added `{type_path}` to {}.\n/reload in the agent, or `rig build`, builds it.",
        config_path.display()
    );
    Ok(())
}

/// `rig plugin remove <type> [--delete]`: the entry of `type_path`, with the
/// comment lines right above it, taken out of `plugins.toml`. With
/// `delete`, its crate goes too when it is a crate of `RIG_HOME/plugins`
/// ([`deletable`]) that no remaining entry uses.
fn remove(home: &Home, type_path: &str, delete: bool) -> Result<()> {
    let config_path = home.config();
    let directory = Config::load(&config_path)?
        .plugins
        .into_iter()
        .find(|plugin| plugin.type_path == type_path)
        .and_then(|plugin| plugin.package)
        .and_then(|package| match package.source {
            Source::Path(directory) => Some(directory),
            Source::Git { .. } | Source::Version(_) => None,
        });
    let remaining = config::remove(&config_path, type_path)?;
    let crate_note = match directory {
        None => String::new(),
        Some(directory) => {
            let used = remaining.plugins.iter().any(|plugin| {
                plugin
                    .package
                    .as_ref()
                    .is_some_and(|package| package.source == Source::Path(directory.clone()))
            });
            if !delete {
                format!("; its crate {} is left in place", directory.display())
            } else if used {
                format!(
                    "; its crate {} is left in place: another entry uses it",
                    directory.display()
                )
            } else if let Some(directory) = deletable(&home.plugins(), &directory) {
                fs::remove_dir_all(&directory)?;
                format!(" and deleted its crate {}", directory.display())
            } else {
                format!(
                    "; its crate {} is left in place: --delete only deletes a crate of {}",
                    directory.display(),
                    home.plugins().display()
                )
            }
        }
    };
    println!(
        "Removed `{type_path}` from {}{crate_note}.\n\
         /reload in the agent, or `rig build`, applies it.",
        config_path.display()
    );
    Ok(())
}

/// `directory`, resolved, when it is a crate directly in `plugins`
/// (`RIG_HOME/plugins/<name>`, where `rig plugin new` makes them), the
/// only crates `rig plugin remove --delete` deletes.
fn deletable(plugins: &Path, directory: &Path) -> Option<PathBuf> {
    let plugins = fs::canonicalize(plugins).ok()?;
    let directory = fs::canonicalize(directory).ok()?;
    (directory.parent() == Some(plugins.as_path()) && directory.join("Cargo.toml").is_file())
        .then_some(directory)
}

/// The package name in `directory`'s `Cargo.toml`, if it can be read.
fn manifest_name(directory: &Path) -> Option<String> {
    let manifest = fs::read_to_string(directory.join("Cargo.toml")).ok()?;
    manifest_string(&manifest, "package", "name")
}

/// `directory` as a `path` entry: relative to `RIG_HOME` (where
/// `plugins.toml` is) when it is under it, so the entry moves with it;
/// absolute otherwise.
fn relative_to(directory: &Path, root: &Path) -> String {
    let relative: PathBuf = directory
        .strip_prefix(root)
        .map_or_else(|_| directory.to_path_buf(), Path::to_path_buf);
    relative.to_string_lossy().into_owned()
}

/// A name `rig plugin new` takes: a crate name in lowercase that is not
/// the agent's own or one of its crates.
fn validate_name(name: &str) -> Result<()> {
    let valid = name.starts_with(|c: char| c.is_ascii_lowercase())
        && name
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '_');
    if !valid {
        return Err(format!(
            "`{name}` is not a plugin name: use lowercase letters, digits, `-` and `_`, \
             starting with a letter"
        )
        .into());
    }
    if RIG_CRATES.contains(&name) || name == PACKAGE || name.starts_with("bevy") {
        return Err(format!("`{name}` is a crate of the agent itself; pick another name").into());
    }
    Ok(())
}

/// The plugin type of the crate `name`: `agent-viz` makes `AgentVizPlugin`.
fn type_name(name: &str) -> String {
    let mut type_name: String = name
        .split(['-', '_'])
        .map(|word| {
            let mut chars = word.chars();
            chars.next().map_or_else(String::new, |first| {
                first.to_ascii_uppercase().to_string() + chars.as_str()
            })
        })
        .collect();
    if !type_name.ends_with("Plugin") {
        type_name.push_str("Plugin");
    }
    type_name
}

/// The plugin crate's `Cargo.toml`. It depends on the crates.io release of
/// rig-harness; the agent project's `[patch]` builds it from `RIG_SOURCE`
/// when that is set, and so does this crate's own when built on its own.
fn manifest(name: &str, rig_version: &str, source: &RigSource) -> String {
    let mut text = format!(
        "[package]\n\
         name = {}\n\
         version = \"0.1.0\"\n\
         edition = \"2024\"\n\
         publish = false\n\
         \n\
         # Its own workspace: the agent builds it as a dependency, from RIG_HOME.\n\
         [workspace]\n\
         \n\
         [dependencies]\n\
         # Everything a plugin uses: Bevy's app and ECS (`rig_harness::prelude`), the\n\
         # agent runtime (`rig_harness::rig_ecs`) and the terminal view\n\
         # (`rig_harness::tui`, with `rig_harness::tui::ratatui`). The agent builds every\n\
         # plugin with its own rig crates. A Bevy crate, such as `bevy` for a window,\n\
         # must be at exactly ={}.\n\
         rig-harness = \"{rig_version}\"\n\
         # Bevy's derives (`Component`, `Resource`, `Message`, `SystemSet`) name the\n\
         # crate they expand to from this file, so it is a direct dependency.\n\
         bevy_ecs = {{ version = \"={}\", default-features = false }}\n\
         # A tool's arguments and parameters.\n\
         serde = {{ version = \"1\", features = [\"derive\"] }}\n\
         serde_json = \"1\"\n",
        quoted(name),
        super::BEVY_VERSION,
        super::BEVY_VERSION,
    );
    if let RigSource::Local(checkout) = source {
        text.push_str(
            "\n# The agent is built from this rig checkout (RIG_SOURCE); this crate on its\n\
             # own is too. The agent project has the same table.\n",
        );
        text.push_str(&rig_patch(checkout));
    }
    text
}

/// The plugin crate's `src/lib.rs`, from [`SCAFFOLD`]: a plugin that adds
/// `/<name>`, with comments that name the other extension points.
fn lib_rs(name: &str, type_name: &str) -> String {
    SCAFFOLD
        .replace("ScaffoldPlugin", type_name)
        .replace("__name__", name)
}

/// `rig plugin list`: each plugin, with the crate it comes from.
fn list(home: &Home) -> Result<()> {
    let config = Config::load(&home.config())?;
    println!("{}:", home.config().display());
    for (number, plugin) in (1..).zip(&config.plugins) {
        let from = match &plugin.package {
            None => "rig-harness".to_owned(),
            Some(package) => format!("{}, {}", package.name, package.source),
        };
        print!("{number:>3}. {} ({from})", plugin.type_path);
        if !plugin.bevy_features.is_empty() {
            print!(", Bevy features: {}", plugin.bevy_features.join(", "));
        }
        println!();
    }
    Ok(())
}

/// `rig plugin check`: parses `plugins.toml` and checks each plugin crate
/// listed by path: that it is there, has the package name its entry gives,
/// and its library is the crate the plugin type names. Whether the code
/// builds is for [`check_build`] (`--build`) or `rig build` to say.
fn check(home: &Home) -> Result<()> {
    let config = Config::load(&home.config())?;
    let problems: Vec<String> = config.plugins.iter().filter_map(check_plugin).collect();
    if problems.is_empty() {
        println!(
            "{}: {} plugins, all valid; `rig plugin check --build` builds the agent with them.",
            home.config().display(),
            config.plugins.len()
        );
        return Ok(());
    }
    Err(format!("{}:\n  {}", home.config().display(), problems.join("\n  ")).into())
}

/// `rig plugin check --build`: builds the agent project with the listed
/// plugins, as `rig build` does but without staging the build, so cargo's
/// errors come with no restart and the build `/reload` then makes reuses
/// it.
fn check_build(home: &Home) -> Result<()> {
    super::build::check(home)?;
    println!(
        "The agent builds with these plugins. Call the `reload` tool, or type /reload, to \
         restart on them."
    );
    Ok(())
}

/// What is wrong with a plugin listed by path, if anything.
fn check_plugin(plugin: &config::Plugin) -> Option<String> {
    let package = plugin.package.as_ref()?;
    let Source::Path(directory) = &package.source else {
        return None;
    };
    let what = format!("`{}` ({})", plugin.type_path, directory.display());
    let manifest_path = directory.join("Cargo.toml");
    let manifest = match fs::read_to_string(&manifest_path) {
        Ok(manifest) => manifest,
        Err(failure) => {
            return Some(format!(
                "{what}: cannot read {}: {failure}",
                manifest_path.display()
            ));
        }
    };
    let Some(name) = manifest_string(&manifest, "package", "name") else {
        return Some(format!(
            "{what}: {} has no [package] name",
            manifest_path.display()
        ));
    };
    if name != package.name {
        return Some(format!(
            "{what}: the entry says `crate = \"{}\"`, but the package is named `{name}`",
            package.name
        ));
    }
    let library =
        manifest_string(&manifest, "lib", "name").unwrap_or_else(|| name.replace('-', "_"));
    let root = plugin.type_path.split("::").next().unwrap_or_default();
    if root != library {
        return Some(format!(
            "{what}: the plugin type starts with `{root}`, but the crate's library is `{library}`"
        ));
    }
    if !has_lib(directory, &manifest) {
        return Some(format!("{what}: the crate has no library (src/lib.rs)"));
    }
    None
}

/// Whether the crate in `directory` has a library target.
fn has_lib(directory: &Path, manifest: &str) -> bool {
    let path = manifest_string(manifest, "lib", "path").unwrap_or_else(|| "src/lib.rs".to_owned());
    directory.join(path).is_file()
}

#[cfg(test)]
mod tests;
