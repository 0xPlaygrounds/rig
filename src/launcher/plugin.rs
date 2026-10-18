//! `rig plugin`: make a plugin crate under `RIG_HOME/plugins` and list it
//! in `plugins.toml` (`new`), add or remove an entry (`add`, `remove`),
//! show the entries (`list`), turn one off by commenting it out and back on
//! (`disable`, `enable`), and check the list (`check`, with `--build` also
//! building the agent without staging it and checking how its plugins'
//! agent-loop systems are ordered). Every change is checked before it is
//! written, so a command never leaves a broken file. Plugin crates live
//! outside every workspace, so making one never touches the rig repository
//! or a project.

use std::fs;
use std::path::{Path, PathBuf};

use rig_harness_protocol::Home;

use super::Result;
use super::plugin_list::{self, PluginList, Source};
use super::project::{
    CORE_CRATES, PACKAGE, RigSource, crate_dir, manifest_string, quoted, quoted_path, rig_patch,
    rig_version,
};

/// The `src/lib.rs` of a new plugin crate, with `ScaffoldPlugin` and
/// `__name__` for the plugin type and the crate name.
/// rig-harness-test-support's `tests/removal.rs` builds it, so it stays a
/// working plugin.
const SCAFFOLD: &str = include_str!("plugin/scaffold.rs");

/// `rig plugin`'s usage.
pub const USAGE: &str =
    "  rig plugin new <name>  Make a plugin crate in RIG_HOME/plugins/<name> and add
                         it to plugins.toml; /reload or `rig build` builds it.
  rig plugin add <type> [--path <dir> | --git <url> [--branch <b> | --rev <r>]
                 | --version <req>] [--crate <name>]
                         Add the plugin type <type> to plugins.toml, from the
                         crate at that source (its name read from <dir> for
                         --path), or with --crate alone from one of rig's own
                         plugin crates, such as rig-inspect. A type listed
                         disabled gets its entry enabled instead.
  rig plugin remove <type>
                         Remove the plugin type <type> from plugins.toml; its
                         crate stays where it is.
  rig plugin list        List every plugin in plugins.toml, enabled (`on`) or
                         disabled (`off`, its entry commented out), with its
                         crate and the comment above its entry.
  rig plugin enable <type>
  rig plugin disable <type>
                         Uncomment or comment out the entry of <type>,
                         keeping every other line; /reload or `rig build`
                         applies it.
  rig plugin check [--build]
                         Check plugins.toml and the plugin crates it names by
                         path; with --build, also build the agent with them
                         (in RIG_HOME/target, as /reload does) without
                         staging the build, and name the plugins whose
                         systems in one agent-loop stage use the same data
                         with no order between them.
";

/// Runs `rig plugin` with `args`, the words after `plugin`.
pub fn run(home: &Home, args: &[&str]) -> Result<()> {
    match args {
        ["new", name] => new(home, name),
        ["add", type_path, options @ ..] => add(home, type_path, options),
        ["remove", type_path] => remove(home, type_path),
        ["list"] => list(home),
        ["enable", type_path] => enable(home, type_path, true),
        ["disable", type_path] => enable(home, type_path, false),
        ["check"] => check(home),
        ["check", "--build"] => check(home).and_then(|()| check_build(home)),
        _ => Err(format!("usage:\n{USAGE}").into()),
    }
}

/// `rig plugin new <name>`: the crate `name` in `RIG_HOME/plugins/<name>`,
/// with one plugin that adds `/<name>`, listed at the end of
/// `plugins.toml`.
fn new(home: &Home, name: &str) -> Result<()> {
    let source = RigSource::detect()?;
    validate_name(name, &source)?;
    let plugins_toml = home.plugins_toml();
    let list = PluginList::load(&plugins_toml)?;
    if let Some(listed) = list
        .plugins
        .iter()
        .find(|plugin| plugin.package.name == name)
    {
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
    if let Err(failure) = plugin_list::append(&plugins_toml, &entry, |_| Ok(())) {
        fs::remove_dir_all(&directory).ok();
        return Err(failure);
    }
    println!(
        "Made the plugin crate {} with `{library}::{type_name}`, which adds /{name},\n\
         and listed it in {}.\n\
         `rig plugin check --build` builds the agent with it; /reload in the agent,\n\
         or `rig build`, builds it and applies it. Its src/lib.rs lists what else a\n\
         plugin adds, such as a top-level `rig {name}` command of its own.",
        directory.display(),
        plugins_toml.display()
    );
    Ok(())
}

/// `rig plugin add <type> [options]`: an entry for `type_path` at the end
/// of `plugins.toml`, written only when the whole file still parses and,
/// for a crate by path, the crate is there and provides the type's crate.
fn add(home: &Home, type_path: &str, options: &[&str]) -> Result<()> {
    let plugins_toml = home.plugins_toml();
    let disabled = plugin_list::listed(&plugins_toml)?
        .iter()
        .any(|listed| listed.entry.type_path == type_path && !listed.enabled);
    if disabled && plugin_list::set_enabled(&plugins_toml, type_path, true)? {
        println!(
            "`{type_path}` was listed disabled in {}; enabled its entry.\n\
             /reload in the agent, or `rig build`, builds it.",
            plugins_toml.display()
        );
        return Ok(());
    }
    let mut keys: Vec<(&str, String)> = Vec::new();
    let mut name: Option<String> = None;
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
    plugin_list::append(&plugins_toml, &entry, |list| {
        match list.plugins.last().and_then(check_plugin) {
            Some(problem) => Err(problem.into()),
            None => Ok(()),
        }
    })?;
    println!(
        "Added `{type_path}` to {}.\n/reload in the agent, or `rig build`, builds it.",
        plugins_toml.display()
    );
    Ok(())
}

/// `rig plugin remove <type>`: the entry of `type_path`, with the comment
/// lines right above it, taken out of `plugins.toml`. Its crate stays.
fn remove(home: &Home, type_path: &str) -> Result<()> {
    let plugins_toml = home.plugins_toml();
    let directory = PluginList::load(&plugins_toml)?
        .plugins
        .into_iter()
        .find(|plugin| plugin.type_path == type_path)
        .and_then(|plugin| match plugin.package.source {
            Source::Path(directory) => Some(directory),
            Source::Git { .. } | Source::Version(_) | Source::Rig => None,
        });
    plugin_list::remove(&plugins_toml, type_path)?;
    let crate_note = directory
        .map(|directory| format!("; its crate {} is left in place", directory.display()))
        .unwrap_or_default();
    println!(
        "Removed `{type_path}` from {}{crate_note}.\n\
         /reload in the agent, or `rig build`, applies it.",
        plugins_toml.display()
    );
    Ok(())
}

/// `rig plugin list`: every entry of `plugins.toml`, one line each: `on`
/// or `off`, the type, its crate, and the comment above its entry, with a
/// warning on a disabled essential plugin. `/plugins` reads these lines.
fn list(home: &Home) -> Result<()> {
    let listed = plugin_list::listed(&home.plugins_toml())?;
    let width =
        |of: fn(&plugin_list::Listed) -> usize| listed.iter().map(of).max().unwrap_or_default();
    let types = width(|listed| listed.entry.type_path.len());
    let crates = width(|listed| listed.entry.package.name.len());
    for listed in &listed {
        let entry = &listed.entry;
        let mut line = format!(
            "{:<3} {:<types$}  {:<crates$}  {}",
            if listed.enabled { "on" } else { "off" },
            entry.type_path,
            entry.package.name,
            listed.description
        );
        if !listed.enabled
            && let Some(what) = rig_harness_protocol::essential(&entry.type_path)
        {
            line.push_str(&format!(" (warning: disabled, which takes away {what})"));
        }
        println!("{}", line.trim_end());
    }
    Ok(())
}

/// `rig plugin enable <type>` and `rig plugin disable <type>`: uncomments
/// or comments out the entry of `type_path`.
fn enable(home: &Home, type_path: &str, enabled: bool) -> Result<()> {
    let plugins_toml = home.plugins_toml();
    PluginList::load(&plugins_toml)?;
    let changed = plugin_list::set_enabled(&plugins_toml, type_path, enabled)?;
    let state = if enabled { "enabled" } else { "disabled" };
    if !changed {
        println!("`{type_path}` is {state} already.");
        return Ok(());
    }
    println!(
        "{} `{type_path}` in {}.\n/reload in the agent, or `rig build`, applies it.",
        if enabled { "Enabled" } else { "Disabled" },
        plugins_toml.display()
    );
    if !enabled && let Some(what) = rig_harness_protocol::essential(type_path) {
        println!(
            "Warning: this takes away {what}; `rig plugin enable {type_path}` brings it back."
        );
    }
    Ok(())
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
/// the agent's own, one of its core crates or a crate of the rig checkout
/// `source` names.
fn validate_name(name: &str, source: &RigSource) -> Result<()> {
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
    let rig_crate = match source {
        RigSource::Local(checkout) => crate_dir(checkout, name).is_some(),
        RigSource::Registry => false,
    };
    if rig_crate || CORE_CRATES.contains(&name) || name == PACKAGE || name.starts_with("bevy") {
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
         # Everything a plugin uses: Bevy's app and ECS (`rig_harness::prelude`) and the\n\
         # kernel (`rig_harness::rig_ecs`). Another plugin's types come from its\n\
         # crate, at the same version: the terminal view's panels from rig-tui (with\n\
         # `rig_tui::ratatui`), what agents do from rig-basics. Add rig-basics with\n\
         # `default-features = false` unless this plugin draws in the terminal view:\n\
         # its default `tui` feature builds rig-tui. The agent builds every plugin\n\
         # with its own rig crates. A Bevy crate, such as `bevy` for a window, must\n\
         # be at exactly ={}.\n\
         rig-harness = \"{rig_version}\"\n\
         # Bevy's derives (`Component`, `Resource`, `Message`, `SystemSet`, `Reflect`)\n\
         # name the crate they expand to from this file, so those are direct dependencies.\n\
         bevy_ecs = {{ version = \"={}\", default-features = false }}\n\
         bevy_reflect = {{ version = \"={}\", default-features = false }}\n\
         # A tool's arguments, and its parameters derived from them.\n\
         schemars = \"1\"\n\
         serde = {{ version = \"1\", features = [\"derive\"] }}\n\
         serde_json = \"1\"\n",
        quoted(name),
        super::BEVY_VERSION,
        super::BEVY_VERSION,
        super::BEVY_VERSION,
    );
    if let RigSource::Local(checkout) = source {
        text.push_str(
            "\n# The agent is built from this rig checkout (RIG_SOURCE); this crate on its\n\
             # own is too. The agent project patches these and the plugin crates it uses.\n",
        );
        text.push_str(&rig_patch(checkout, &CORE_CRATES.map(str::to_owned).into()));
        text.push_str(
            "# Another rig crate this crate depends on, such as rig-tui, needs a line\n\
             # here too when this crate is built on its own.\n",
        );
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

/// `rig plugin check`: parses `plugins.toml` and checks each plugin: that
/// a crate listed by path is there, has the package name its entry gives,
/// and its library is the crate the plugin type names. Whether the code
/// builds is for [`check_build`] (`--build`) or `rig build` to say.
fn check(home: &Home) -> Result<()> {
    let list = PluginList::load(&home.plugins_toml())?;
    let problems: Vec<String> = list.plugins.iter().filter_map(check_plugin).collect();
    if problems.is_empty() {
        println!(
            "{}: {} plugins, all valid; `rig plugin check --build` builds the agent with them \
             and checks how their agent-loop systems are ordered.",
            home.plugins_toml().display(),
            list.plugins.len()
        );
        return Ok(());
    }
    Err(format!(
        "{}:\n  {}",
        home.plugins_toml().display(),
        problems.join("\n  ")
    )
    .into())
}

/// `rig plugin check --build`: builds the agent project with the listed
/// plugins, as `rig build` does but without staging the build, so cargo's
/// errors come with no restart and the build `/reload` then makes reuses
/// it. Fails, naming the plugins, when the build reports systems of one
/// agent-loop stage that use the same data unordered, one writing it:
/// which runs first would be chance.
fn check_build(home: &Home) -> Result<()> {
    let report = super::build::check(home)?;
    if !report.conflicts.is_empty() {
        return Err(format!(
            "the agent builds, but these plugins' agent-loop systems use the same data with \
             no order between them, so which runs first is chance; order one against the \
             other (`.before`/`.after` their systems or sets), or leave one out:\n  {}",
            report.conflicts.join("\n  ")
        )
        .into());
    }
    println!(
        "The agent builds with these plugins. Call the `reload` tool, or type /reload, to \
         restart on them."
    );
    Ok(())
}

/// What is wrong with a listed plugin, if anything: one listed by path
/// that is not where or what its entry says.
fn check_plugin(plugin: &plugin_list::PluginEntry) -> Option<String> {
    let package = &plugin.package;
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
