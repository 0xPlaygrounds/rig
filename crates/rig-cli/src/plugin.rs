//! `rig plugin`: turn a plugin on or off in `RIG_HOME/settings.toml`
//! (`enable`, `disable`), show every plugin and whether it is on
//! (`list`), make a plugin crate under `RIG_HOME/plugins` and add it to the
//! home's `[[plugins.extra]]` (`new`), and check the settings (`check`,
//! with `--build` also building the agent without staging it and checking
//! how its plugins' agent-loop systems are ordered). Another crate's plugin
//! is a `[[plugins.extra]]` table the user writes. Every change keeps the
//! file's comments and is checked before it is written, so a command never
//! leaves a broken file. Plugin crates live outside every workspace, so
//! making one never touches the rig repository or a project.

use std::fs;
use std::path::Path;
use std::process::ExitCode;

use rig_harness_protocol::config::toml_edit::{ArrayOfTables, Item, Table, value};
use rig_harness_protocol::config::{
    self, BUILT_IN, Extra, Source, built_in, essential, set_on, table,
};
use rig_harness_protocol::{Home, REFUSED_EXIT_CODE};

use super::Result;
use super::project::{
    CORE_CRATES, PACKAGE, RigSource, crate_dir, foreign_rig_crate, quoted, quoted_path,
    read_manifest, rig_patch, rig_version,
};

/// The `src/lib.rs` of a new plugin crate, with `ScaffoldPlugin` and
/// `__name__` for the plugin type and the crate name.
/// rig-harness-test-support's `tests/removal.rs` builds it, so it stays a
/// working plugin.
const SCAFFOLD: &str = include_str!("plugin/scaffold.rs");

/// `rig plugin`'s usage.
pub const USAGE: &str = "  rig plugin list        Every plugin, rig's own and the extra crates of
                         RIG_HOME/settings.toml, `on` or `off`, with its crate
                         and what it adds: rig's defaults, changed by [plugins]
                         in RIG_HOME/settings.toml.
  rig plugin enable <type>
  rig plugin disable <type>
                         Turn a plugin on or off in RIG_HOME/settings.toml
                         ([plugins] enable and disable), keeping its comments.
                         The agent holds only the plugins on, so the next
                         start, or /reload, rebuilds it: only the generated
                         crate compiles again, and a plugin's own crate the
                         first time it is on.
  rig plugin new <name>  Make a plugin crate in RIG_HOME/plugins/<name> and add
                         it to [[plugins.extra]]; the next start, or /reload,
                         builds it.
  rig plugin check [--build]
                         Check the settings and the extra crates they name by
                         path; with --build, also build the agent with them
                         (in RIG_HOME/target, as /reload does) without
                         staging the build, and name the plugins whose
                         systems in one agent-loop stage use the same data
                         with no order between them.
  Another crate's plugin is a [[plugins.extra]] table of RIG_HOME/settings.toml:
  `crate`, `plugin` and one of `path`, `git` (with `branch` or `rev`) or
  `version`. The next start, or /reload, builds it.
";

/// Runs `rig plugin` with `args`, the words after `plugin`.
pub fn run(home: &Home, args: &[&str]) -> Result<ExitCode> {
    if args.iter().any(|arg| matches!(*arg, "--help" | "-h")) {
        print!("{USAGE}");
        return Ok(ExitCode::SUCCESS);
    }
    match args {
        ["new", name] => new(home, name),
        ["list"] => list(home),
        ["enable", type_path] => enable(home, type_path, true),
        ["disable", type_path] => enable(home, type_path, false),
        ["check"] => check(home),
        ["check", "--build"] => check(home).and_then(|()| check_build(home)),
        _ => {
            eprint!("usage:\n{USAGE}");
            return Ok(ExitCode::from(REFUSED_EXIT_CODE));
        }
    }?;
    Ok(ExitCode::SUCCESS)
}

/// `rig plugin new <name>`: the crate `name` in `RIG_HOME/plugins/<name>`,
/// with one plugin that adds `/<name>`, added to the home's
/// `[[plugins.extra]]`.
fn new(home: &Home, name: &str) -> Result<()> {
    let source = RigSource::detect()?;
    validate_name(name, &source)?;
    if let Some(extra) = (config::chosen(home)?.extra.iter()).find(|extra| extra.krate == name) {
        return Err(format!(
            "{} already has the crate `{name}` (as `{}`)",
            home.settings().display(),
            extra.plugin
        )
        .into());
    }
    let directory = home.plugins().join(name);
    if directory.exists() {
        return Err(format!(
            "{} exists already; add its [[plugins.extra]] table or pick another name",
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
    let keys = [
        ("crate", name.to_owned()),
        ("path", format!("plugins/{name}")),
        ("plugin", format!("{library}::{type_name}")),
    ];
    let comment = format!("Made by `rig plugin new {name}`.");
    if let Err(failure) = add_extra(home, &keys, &comment, &source) {
        fs::remove_dir_all(&directory).ok();
        return Err(failure);
    }
    println!(
        "Made the plugin crate {} with `{library}::{type_name}`, which adds /{name},\n\
         and added it to {}.\n\
         `rig plugin check --build` builds the agent with it; /reload in the agent,\n\
         or the next start, builds it and runs it. Its src/lib.rs lists what else a\n\
         plugin adds, such as a top-level `rig {name}` command of its own.",
        directory.display(),
        home.settings().display()
    );
    Ok(())
}

/// Adds the `[[plugins.extra]]` table of `keys`, with `comment` above it,
/// to the home's settings, once the result reads and the new plugin passes
/// [`check_plugin`].
fn add_extra(
    home: &Home,
    keys: &[(&str, String)],
    comment: &str,
    source: &RigSource,
) -> Result<()> {
    config::edit(&home.settings(), |file| -> Result<_> {
        let mut extra = Table::new();
        for (key, text) in keys {
            extra.insert(key, value(text.as_str()));
        }
        extra.decor_mut().set_prefix(format!("\n# {comment}\n"));
        let plugins = table(&mut file.document, "plugins").ok_or("[plugins] is not a table")?;
        let tables = plugins
            .entry("extra")
            .or_insert_with(|| Item::ArrayOfTables(ArrayOfTables::new()))
            .as_array_of_tables_mut()
            .ok_or("[plugins] extra is not a list of [[plugins.extra]] tables")?;
        tables.push(extra);
        let chosen = config::plugins(file)?;
        if let Some(problem) = chosen
            .extra
            .last()
            .and_then(|extra| check_plugin(extra, source))
        {
            return Err(format!("{}: {problem}", file.path.display()).into());
        }
        Ok((true, ()))
    })?;
    Ok(())
}

/// `rig plugin list`: every plugin of rig's and of the home's extra crates,
/// one line each: `on` or `off`, the type, its crate and what it adds, with
/// a note when the settings changed rig's default, and a warning on an
/// essential plugin turned off. `/plugins` reads these lines.
fn list(home: &Home) -> Result<()> {
    let chosen = config::chosen(home)?;
    let mut rows: Vec<(bool, String, String, String)> = Vec::new();
    for built_in in BUILT_IN {
        let on = chosen.on.iter().any(|plugin| plugin == built_in.plugin);
        let mut about = built_in.about.to_owned();
        if !built_in.available() {
            about.push_str(&format!(
                " (builds on {} only)",
                built_in.targets.join(" and ")
            ));
        } else if on != built_in.on {
            about.push_str(" (set by RIG_HOME/settings.toml)");
        }
        if !on && let Some(what) = essential(built_in.plugin) {
            about.push_str(&format!(" (warning: off, which takes away {what})"));
        }
        rows.push((
            on,
            built_in.plugin.to_owned(),
            built_in.krate.to_owned(),
            about,
        ));
    }
    for extra in &chosen.extra {
        let on = chosen.on.contains(&extra.plugin);
        let about = format!("An extra crate, from {}.", extra.source);
        rows.push((on, extra.plugin.clone(), extra.krate.clone(), about));
    }
    let width = |of: fn(&(bool, String, String, String)) -> usize| {
        rows.iter().map(of).max().unwrap_or_default()
    };
    let types = width(|row| row.1.len());
    let crates = width(|row| row.2.len());
    for (on, plugin, krate, about) in &rows {
        let state = if *on { "on" } else { "off" };
        println!(
            "{}",
            format!("{state:<3} {plugin:<types$}  {krate:<crates$}  {about}").trim_end()
        );
    }
    Ok(())
}

/// `rig plugin enable <type>` and `rig plugin disable <type>`: names
/// `type_path` in `[plugins] enable` or `disable` of the home's settings,
/// or takes it out of the other list when that gives rig's default; a
/// name in this list stays.
fn enable(home: &Home, type_path: &str, on: bool) -> Result<()> {
    let path = home.settings();
    let (changed, ()) = config::edit(&path, |file| -> Result<_> {
        let before = config::plugins(file)?;
        let by_default = match built_in(type_path) {
            Some(built_in) if on && !built_in.available() => {
                return Err(config::Error::Unavailable {
                    plugin: type_path.to_owned(),
                    targets: built_in.targets,
                }
                .into());
            }
            Some(built_in) => built_in.on,
            None if before.extra.iter().any(|extra| extra.plugin == type_path) => true,
            None => {
                return Err(format!(
                    "no plugin `{type_path}`; `rig plugin list` shows them, and a \
                     [[plugins.extra]] table adds a crate's"
                )
                .into());
            }
        };
        let changed = set_on(&mut file.document, type_path, on, by_default);
        config::plugins(file)?;
        Ok((changed, ()))
    })?;
    let state = if on { "on" } else { "off" };
    if !changed {
        println!("`{type_path}` is {state} already in {}.", path.display());
    } else {
        println!(
            "Turned `{type_path}` {state} in {}.\n\
             The next start, or /reload in the agent, rebuilds the agent with it.",
            path.display()
        );
    }
    if !on && let Some(what) = essential(type_path) {
        println!(
            "Warning: this takes away {what}; `rig plugin enable {type_path}` brings it back."
        );
    }
    Ok(())
}

/// The string at `keys` in `directory`'s `Cargo.toml`, such as its
/// `package.name`, if it can be read.
fn manifest_string(directory: &Path, keys: &[&str]) -> Option<String> {
    let manifest = read_manifest(directory)?;
    let mut item = manifest.as_item();
    for key in keys {
        item = item.get(key)?;
    }
    item.as_str().map(str::to_owned)
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

/// `rig plugin check`: reads the home's settings and checks each extra
/// plugin on: that a crate listed by path is
/// there, has the package name its table gives, and its library is the
/// crate the plugin type names. Whether the code builds is for
/// [`check_build`] (`--build`) or `rig build` to say.
fn check(home: &Home) -> Result<()> {
    let chosen = config::chosen(home)?;
    let source = RigSource::detect()?;
    let problems: Vec<String> = (chosen.extras_on())
        .filter_map(|extra| check_plugin(extra, &source))
        .collect();
    if problems.is_empty() {
        println!(
            "{}: {} plugins on, all valid; `rig plugin check --build` builds the agent with \
             them and checks how their agent-loop systems are ordered.",
            home.settings().display(),
            chosen.on.len()
        );
        return Ok(());
    }
    Err(format!(
        "{}:\n  {}",
        home.settings().display(),
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
    println!("The agent builds with these plugins. Type /reload to restart on them.");
    Ok(())
}

/// What is wrong with an extra plugin, if anything: one by path that is
/// not where or what its table says, or a crate of a rig checkout other
/// than the one the agent's rig crates come from (`source`).
fn check_plugin(extra: &Extra, source: &RigSource) -> Option<String> {
    let Source::Path(directory) = &extra.source else {
        return None;
    };
    let what = format!("`{}` ({})", extra.plugin, directory.display());
    let manifest_path = directory.join("Cargo.toml");
    if read_manifest(directory).is_none() {
        return Some(format!(
            "{what}: cannot read {} as TOML",
            manifest_path.display()
        ));
    }
    let Some(name) = manifest_string(directory, &["package", "name"]) else {
        return Some(format!(
            "{what}: {} has no [package] name",
            manifest_path.display()
        ));
    };
    if name != extra.krate {
        return Some(format!(
            "{what}: the table says `crate = \"{}\"`, but the package is named `{name}`",
            extra.krate
        ));
    }
    let library =
        manifest_string(directory, &["lib", "name"]).unwrap_or_else(|| name.replace('-', "_"));
    let root = extra.plugin.split("::").next().unwrap_or_default();
    if root != library {
        return Some(format!(
            "{what}: the plugin type starts with `{root}`, but the crate's library is `{library}`"
        ));
    }
    let lib =
        manifest_string(directory, &["lib", "path"]).unwrap_or_else(|| "src/lib.rs".to_owned());
    if !directory.join(lib).is_file() {
        return Some(format!("{what}: the crate has no library (src/lib.rs)"));
    }
    foreign_rig_crate(extra, source)
}

#[cfg(test)]
mod tests;
