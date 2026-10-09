//! `rig plugin`: make a plugin crate under `RIG_HOME/plugins` and list it
//! in `plugins.toml` (`new`), show the list (`list`), and check it without
//! a build (`check`). Plugin crates live outside every workspace, so making
//! one never touches the rig repository or a project.

use std::fs;
use std::path::Path;

use rig::harness_protocol::Home;

use super::Result;
use super::config::{self, Config, Source};
use super::project::{
    PACKAGE, RIG_CRATES, RigSource, manifest_string, quoted, quoted_path, rig_patch, rig_version,
};

/// `rig plugin`'s usage.
pub const USAGE: &str = "\
  rig plugin new <name>  Make a plugin crate in RIG_HOME/plugins/<name> and add
                         it to plugins.toml; /reload or `rig build` builds it.
  rig plugin list        List the plugins in plugins.toml, in the order they
                         are added.
  rig plugin check       Check plugins.toml and the plugin crates it names by
                         path, without a build.
";

/// Runs `rig plugin` with `args`, the words after `plugin`.
pub fn run(home: &Home, args: &[&str]) -> Result<()> {
    match args {
        ["new", name] => new(home, name),
        ["list"] => list(home),
        ["check"] => check(home),
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
            "# The agent's target directory, so checking this crate on its own reuses\n\
             # the agent build's dependencies.\n[build]\ntarget-dir = {}\n",
            quoted_path(&home.target())
        ),
    )?;
    let entry = format!(
        "# Made by `rig plugin new {name}`.\n[[plugin]]\ncrate = {}\npath = {}\nplugin = {}\n",
        quoted(name),
        quoted(&format!("plugins/{name}")),
        quoted(&format!("{library}::{type_name}")),
    );
    if let Err(failure) = config::append(&config_path, &entry) {
        fs::remove_dir_all(&directory).ok();
        return Err(failure);
    }
    println!(
        "Made the plugin crate {} with `{library}::{type_name}`, which adds /{name},\n\
         and listed it in {}.\n\
         /reload in the agent, or `rig build`, builds it.",
        directory.display(),
        config_path.display()
    );
    Ok(())
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
         # A tool's arguments and parameters.\n\
         serde = {{ version = \"1\", features = [\"derive\"] }}\n\
         serde_json = \"1\"\n",
        quoted(name),
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

/// The plugin crate's `src/lib.rs`: a plugin that adds `/<name>`.
fn lib_rs(name: &str, type_name: &str) -> String {
    format!(
        "//! The `{name}` plugin of the rig agent. rig-harness's `plugin_guide` docs\n\
         //! (its PLUGINS.md) show each kind of extension: tools, slash commands,\n\
         //! terminal panels and windows.\n\
         \n\
         use rig_harness::prelude::*;\n\
         \n\
         /// Listed in plugins.toml; the agent adds it with `Default`.\n\
         #[derive(Default)]\n\
         pub struct {type_name};\n\
         \n\
         impl Plugin for {type_name} {{\n    \
             fn build(&self, app: &mut App) {{\n        \
                 app.add_command(\"{name}\", \"Say that {name} is loaded\", hello);\n    \
             }}\n\
         }}\n\
         \n\
         /// `/{name}`: a notice for the agent it was typed for.\n\
         fn hello(In(args): In<CommandArgs>, mut notices: MessageWriter<Notice>) {{\n    \
             notices.write(Notice::info(args.agent, \"{name} is loaded.\"));\n\
         }}\n"
    )
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
/// builds is for `rig build` to say.
fn check(home: &Home) -> Result<()> {
    let config = Config::load(&home.config())?;
    let problems: Vec<String> = config.plugins.iter().filter_map(check_plugin).collect();
    if problems.is_empty() {
        println!(
            "{}: {} plugins, all valid; `rig build` builds them.",
            home.config().display(),
            config.plugins.len()
        );
        return Ok(());
    }
    Err(format!("{}:\n  {}", home.config().display(), problems.join("\n  ")).into())
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
