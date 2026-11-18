//! `rig plugin`: turn a plugin, or a branch of the plugin tree, on or off
//! in `RIG_HOME/settings.toml` (`enable`, `disable`), show the tree of
//! every plugin and whether it is on (`list`), make a plugin crate under
//! `RIG_HOME/plugins` and add it to the home's `[[plugins.extra]]` (`new`),
//! add or take out another crate's plugin there (`add`, `remove`), and
//! check the settings (`check`, with `--build` also building the agent
//! without staging it and checking how its plugins' agent-loop systems are
//! ordered). Every change keeps the
//! file's comments and is checked before it is written, so a command never
//! leaves a broken file. Plugin crates live outside every workspace, so
//! making one never touches the rig repository or a project.

use std::path::Path;
use std::process::ExitCode;

use rig_harness_protocol::config::toml_edit::{ArrayOfTables, DocumentMut, Item, Table, value};
use rig_harness_protocol::config::{
    self, BUILT_IN, Extra, Plugins, Source, area, built_in, essential, holds, set_on, table,
};
use rig_harness_protocol::fs;
use rig_harness_protocol::launch::Launch;
use rig_harness_protocol::{Home, REFUSED_EXIT_CODE};

use super::Result;
use super::project::{
    CORE_CRATES, Manifest, PACKAGE, RigSource, crate_dir, foreign_rig_crate, quoted, quoted_path,
    rig_patch, rig_version,
};

/// The `src/lib.rs` of a new plugin crate, with `ScaffoldPlugin` and
/// `__name__` for the plugin type and the crate name.
/// rig-harness-test-support's `tests/removal.rs` builds it, so it stays a
/// working plugin.
const SCAFFOLD: &str = include_str!("plugin/scaffold.rs");

/// `rig plugin`'s usage.
pub const USAGE: &str =
    "  rig plugin list        Every plugin, `on` or `off`, and what it adds, as a
                         tree: each crate, the areas in it (such as
                         rig-basics/sessions) and their plugins. rig's own
                         come first, with their defaults, which [plugins] in
                         RIG_HOME/settings.toml changes, then yours, the extra
                         crates of RIG_HOME/settings.toml, with where each
                         comes from.
  rig plugin enable <type or branch>
  rig plugin disable <type or branch>
                         Turn a plugin on or off in RIG_HOME/settings.toml
                         ([plugins] enable and disable), keeping its comments.
                         A branch, a crate or one of its areas, goes in
                         [plugins] off: its plugins stay off whatever their
                         own choice, which `enable` of the branch brings back.
                         `disable` also takes out a name nothing provides.
                         The agent holds only the plugins on, so the next
                         start, or /reload, rebuilds it: only the generated
                         crate compiles again, and a plugin's own crate the
                         first time it is on.
  rig plugin new <name>  Make a plugin crate in RIG_HOME/plugins/<name> and add
                         it to [[plugins.extra]]; the next start, or /reload,
                         builds it.
  rig plugin add <type> (--path <dir> | --git <url> [--branch <b> | --rev <r>]
                         | --version <req>) [--crate <name>]
                         Add another crate's plugin, such as
                         `rig plugin add my_save::SavePlugin --path ./save`,
                         as a [[plugins.extra]] table of RIG_HOME/settings.toml,
                         keeping its comments; the next start, or /reload,
                         builds it. A relative --path is the working
                         directory's, written resolved. --crate is the package
                         name; it defaults to the package at --path, else the
                         type's first segment (`my_save`), so a git or
                         crates.io package named with `-` needs it.
  rig plugin remove <type>
                         Take that table out, with the comment lines right
                         above it, and the type out of [plugins] enable and
                         disable. The crate's files stay.
  rig plugin check [--build]
                         Check the settings and the extra crates they name by
                         path; with --build, also build the agent with them
                         (in RIG_HOME/target, as /reload does) without
                         staging the build, and name the plugins whose
                         systems in one agent-loop stage use the same data
                         with no order between them.
";

/// Runs `rig plugin` with `args`, the words after `plugin`.
pub fn run(launch: &Launch, args: &[&str]) -> Result<ExitCode> {
    let home = &launch.home();
    let source = launch.source.as_deref();
    if args.iter().any(|arg| matches!(*arg, "--help" | "-h")) {
        print!("{USAGE}");
        return Ok(ExitCode::SUCCESS);
    }
    for warning in super::legacy::warnings(home) {
        eprintln!("warning: {warning}");
    }
    match args {
        ["new", name] => new(home, name, source),
        ["add", type_path, flags @ ..] => add(home, type_path, flags, source),
        ["remove", type_path] => remove(home, type_path),
        ["list"] => list(home),
        ["enable", type_path] => enable(home, type_path, true),
        ["disable", type_path] => enable(home, type_path, false),
        ["check"] => check(home, source),
        ["check", "--build"] => check(home, source).and_then(|()| check_build(launch)),
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
fn new(home: &Home, name: &str, source: Option<&Path>) -> Result<()> {
    let source = RigSource::detect(source)?;
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
            "{} exists already; `rig plugin add <type> --path {}` adds it, or pick \
             another name",
            directory.display(),
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

/// `rig plugin add <type> <flags>`: adds the `[[plugins.extra]]` table of
/// the plugin `type_path` from another crate to the home's settings. The
/// flags give its source: `--path <dir>` (relative to the working
/// directory, written resolved), or `--git <url>` with at most one of `--branch` and `--rev`,
/// or `--version <requirement>`. `--crate` names the package; without it
/// the package is the one `--path` holds, else the type path's first
/// segment.
///
/// # Errors
///
/// A flag that is unknown, without a value or given twice, not exactly one
/// source, `--branch` or `--rev` without `--git` or both, a `--path` that
/// does not resolve, and whatever [`add_extra`] refuses.
fn add(home: &Home, type_path: &str, flags: &[&str], source: Option<&Path>) -> Result<()> {
    let mut given: Vec<(&str, &str)> = Vec::new();
    let mut words = flags.iter();
    while let Some(&flag) = words.next() {
        let key = match flag {
            "--path" | "--git" | "--branch" | "--rev" | "--version" | "--crate" => {
                flag.trim_start_matches('-')
            }
            other => {
                return Err(format!(
                    "`rig plugin add` takes no `{other}`; `rig plugin --help` lists its flags"
                )
                .into());
            }
        };
        let text = words.next().copied().unwrap_or_default();
        if text.is_empty() || text.starts_with("--") {
            let not = if text.is_empty() {
                String::new()
            } else {
                format!(", not `{text}`")
            };
            return Err(format!("{flag} needs a value{not}").into());
        }
        if given.iter().any(|(other, _)| *other == key) {
            return Err(format!("{flag} is given more than once").into());
        }
        given.push((key, text));
    }
    let get = |key: &str| {
        given
            .iter()
            .find(|(other, _)| *other == key)
            .map(|(_, text)| *text)
    };
    let mut keys: Vec<(&str, String)> = Vec::new();
    match (get("path"), get("git"), get("version")) {
        (Some(path), None, None) => {
            let directory = fs::canonicalize(path)?;
            let text = directory
                .to_str()
                .ok_or_else(|| format!("{} is not UTF-8", directory.display()))?;
            keys.push(("path", text.to_owned()));
        }
        (None, Some(url), None) => keys.push(("git", url.to_owned())),
        (None, None, Some(version)) => keys.push(("version", version.to_owned())),
        (None, None, None) => {
            return Err(
                "give the crate's source: one of --path <dir>, --git <url> or --version <req>"
                    .into(),
            );
        }
        _ => return Err("give only one of --path, --git and --version".into()),
    }
    match (get("branch"), get("rev")) {
        (Some(_), Some(_)) => return Err("give --branch or --rev, not both".into()),
        (None, None) => {}
        _ if get("git").is_none() => return Err("--branch and --rev only go with --git".into()),
        (branch, rev) => {
            keys.extend(branch.map(|branch| ("branch", branch.to_owned())));
            keys.extend(rev.map(|rev| ("rev", rev.to_owned())));
        }
    }
    let root = type_path.split("::").next().unwrap_or_default();
    let krate = match (get("crate"), keys.first()) {
        (Some(krate), _) => krate.to_owned(),
        (None, Some(("path", directory))) => Manifest::read(Path::new(directory))
            .ok()
            .and_then(|manifest| manifest_string(&manifest, &["package", "name"]))
            .unwrap_or_else(|| root.to_owned()),
        (None, _) => root.to_owned(),
    };
    keys.splice(0..0, [("crate", krate), ("plugin", type_path.to_owned())]);
    let comment = format!("Added by `rig plugin add {type_path}`.");
    let added = add_extra(home, &keys, &comment, &RigSource::detect(source)?)?;
    println!(
        "Added `{type_path}` ({}) to {}.\n\
         `rig plugin check --build` builds the agent with it; /reload in the agent,\n\
         or the next start, builds it and runs it. `rig plugin remove {type_path}`\n\
         takes it out.",
        added.source,
        home.settings().display()
    );
    Ok(())
}

/// `rig plugin remove <type>`: takes the `[[plugins.extra]]` table of
/// `type_path` out of the home's settings, with the comment block touching
/// its header (comments past a blank line stay), the name out of
/// `[plugins] enable` and `disable`, and out of `[plugins] off` each branch
/// of its crate that then holds no plugin. The crate's files stay where
/// they are.
///
/// # Errors
///
/// When no table names `type_path`, or the file does not read without it.
fn remove(home: &Home, type_path: &str) -> Result<()> {
    let path = home.settings();
    let (_, directory) = config::edit(&path, |file| -> Result<_> {
        let mut plugins = file
            .document
            .get_mut("plugins")
            .and_then(Item::as_table_like_mut);
        let tables = plugins
            .as_deref_mut()
            .and_then(|plugins| plugins.get_mut("extra"))
            .and_then(Item::as_array_of_tables_mut);
        let found = tables.and_then(|tables| {
            let at = tables
                .iter()
                .position(|table| table.get("plugin").and_then(Item::as_str) == Some(type_path))?;
            let table = tables.get(at)?;
            let directory = (table.get("path").and_then(Item::as_str)).map(str::to_owned);
            let krate = (table.get("crate").and_then(Item::as_str)).map(str::to_owned);
            let prefix = table.decor().prefix().and_then(|it| it.as_str());
            let kept = kept_comments(prefix.unwrap_or_default());
            let position = table.position();
            tables.remove(at);
            Some((directory, krate, tables.is_empty(), kept, position))
        });
        let Some((directory, krate, empty, kept, position)) = found else {
            return Err(format!(
                "{} has no [[plugins.extra]] table for `{type_path}`; `rig plugin list` shows \
                 the plugins",
                path.display()
            )
            .into());
        };
        if empty && let Some(plugins) = plugins {
            plugins.remove("extra");
        }
        let stale: Vec<String> = (file.document.get("plugins"))
            .and_then(|plugins| plugins.get("off"))
            .and_then(Item::as_array)
            .map(|off| {
                (off.iter().filter_map(|name| name.as_str()))
                    .filter(|name| {
                        krate.as_deref().is_some_and(|krate| {
                            name.split('/').next() == Some(krate) && !known(file, name)
                        })
                    })
                    .map(str::to_owned)
                    .collect()
            })
            .unwrap_or_default();
        let document = &mut file.document;
        for branch in &stale {
            config::forget(document, branch);
        }
        if !kept.is_empty() {
            put_before_next(document, position, &kept);
        }
        // Only a list that names it changes; an empty one of the user's stays.
        let named = |document: &DocumentMut, key: &str| {
            (document.get("plugins").and_then(|plugins| plugins.get(key)))
                .and_then(Item::as_array)
                .is_some_and(|list| list.iter().any(|name| name.as_str() == Some(type_path)))
        };
        if named(document, "enable") {
            set_on(document, type_path, false, false);
        }
        if named(document, "disable") {
            set_on(document, type_path, true, true);
        }
        config::plugins(file)?;
        Ok((true, directory))
    })?;
    println!("Took `{type_path}` out of {}.", path.display());
    if let Some(directory) = directory {
        println!(
            "Its crate stays in {}.",
            home.root().join(directory).display()
        );
    }
    Ok(())
}

/// The part of a removed table's `prefix` that is not its own: everything
/// up to the last blank line, when it holds a comment. The comment block
/// touching the header goes with the table.
fn kept_comments(prefix: &str) -> String {
    let lines: Vec<&str> = prefix.split_inclusive('\n').collect();
    let Some(last) = lines.iter().rposition(|line| line.trim().is_empty()) else {
        return String::new();
    };
    let kept: String = lines.iter().take(last + 1).copied().collect();
    if kept.contains('#') {
        kept
    } else {
        String::new()
    }
}

/// Puts `text` in front of what follows the table at `after` in
/// `document`: the next table's header, else the end of the file. One
/// blank line stays between them.
fn put_before_next(document: &mut DocumentMut, after: Option<isize>, text: &str) {
    let next = after.and_then(|after| next_position(document.as_table(), after));
    let join = |next: &str| {
        let blank = (next.split_inclusive('\n').next())
            .is_some_and(|line| line.ends_with('\n') && line.trim().is_empty());
        if blank {
            format!("{}\n{next}", text.trim_end())
        } else {
            format!("{text}{next}")
        }
    };
    if let Some(next) = next
        && prefix_table_at(document.as_table_mut(), next, &join)
    {
        return;
    }
    let trailing = document.trailing().as_str().unwrap_or_default().to_owned();
    document.set_trailing(join(&trailing));
}

/// The first position after `after` of a table header under `table`.
fn next_position(table: &Table, after: isize) -> Option<isize> {
    let mut next: Option<isize> = None;
    for (_, item) in table.iter() {
        let children: Vec<&Table> = match item {
            Item::Table(child) => vec![child],
            Item::ArrayOfTables(array) => array.iter().collect(),
            _ => Vec::new(),
        };
        for child in children {
            let here = (child.position()).filter(|at| *at > after && !child.is_implicit());
            let below = next_position(child, after);
            next = [next, here, below].into_iter().flatten().min();
        }
    }
    next
}

/// Sets the prefix of the table header at `position` under `table` to
/// `join` of it. Returns whether it found the table.
fn prefix_table_at(table: &mut Table, position: isize, join: &dyn Fn(&str) -> String) -> bool {
    for (_, item) in table.iter_mut() {
        let children: Vec<&mut Table> = match item {
            Item::Table(child) => vec![child],
            Item::ArrayOfTables(array) => array.iter_mut().collect(),
            _ => Vec::new(),
        };
        for child in children {
            if child.position() == Some(position) && !child.is_implicit() {
                let was = (child.decor().prefix().and_then(|it| it.as_str()))
                    .unwrap_or_default()
                    .to_owned();
                child.decor_mut().set_prefix(join(&was));
                return true;
            }
            if prefix_table_at(child, position, join) {
                return true;
            }
        }
    }
    false
}

/// Adds the `[[plugins.extra]]` table of `keys`, with `comment` above it,
/// to the home's settings, once the result reads and the new plugin passes
/// [`check_plugin`]. A plugin type the file has already is refused. An
/// inline `plugins = { .. }` becomes a `[plugins]` table, which can hold the
/// new one. Returns the plugin as the written file reads.
fn add_extra(
    home: &Home,
    keys: &[(&str, String)],
    comment: &str,
    source: &RigSource,
) -> Result<Extra> {
    let plugin = (keys.iter())
        .find(|(key, _)| *key == "plugin")
        .map(|(_, plugin)| plugin.as_str())
        .unwrap_or_default();
    let (_, added) = config::edit(&home.settings(), |file| -> Result<_> {
        if self::extra(file, plugin) {
            return Err(format!(
                "{} has `{plugin}` already; `rig plugin remove {plugin}` takes it out",
                file.path.display()
            )
            .into());
        }
        let mut extra = Table::new();
        for (key, text) in keys {
            extra.insert(key, value(text.as_str()));
        }
        extra.decor_mut().set_prefix(format!("\n# {comment}\n"));
        uninline(file, "plugins")?;
        let plugins = table(&mut file.document, "plugins").ok_or("[plugins] is not a table")?;
        let tables = plugins
            .entry("extra")
            .or_insert_with(|| Item::ArrayOfTables(ArrayOfTables::new()))
            .as_array_of_tables_mut()
            .ok_or("[plugins] extra is not a list of [[plugins.extra]] tables")?;
        tables.push(extra);
        // The file as it will read back, so a table the writer would not
        // write is refused rather than lost.
        let written = config::SettingsFile::parse(file.path.clone(), &file.document.to_string())?;
        let added = (config::plugins(&written)?.extra.into_iter())
            .find(|extra| extra.plugin == plugin)
            .ok_or_else(|| {
                format!(
                    "{}: the [[plugins.extra]] table of `{plugin}` would not be written",
                    file.path.display()
                )
            })?;
        if let Some(problem) = check_plugin(&added, source) {
            return Err(format!("{}: {problem}", file.path.display()).into());
        }
        Ok((true, added))
    })?;
    Ok(added)
}

/// Makes the inline table `key = { .. }` of `file` a `[key]` table with
/// the same keys, the comments above it and the one after it.
///
/// # Errors
///
/// When a comment, or a `#` in a string, is inside the braces: a `[key]`
/// table would not keep it where it was written.
fn uninline(file: &mut config::SettingsFile, key: &str) -> Result<()> {
    let document = &mut file.document;
    let Some(inline) = document.get(key).and_then(Item::as_inline_table) else {
        return Ok(());
    };
    let mut inside = inline.clone();
    inside.decor_mut().clear();
    if inside.to_string().contains('#') {
        return Err(format!(
            "{}: `{key} = {{ .. }}` holds a comment, which a [{key}] table would not keep \
             where it is; write it as a [{key}] table",
            file.path.display()
        )
        .into());
    }
    let mut table = inline.clone().into_table();
    let suffix = inline
        .decor()
        .suffix()
        .and_then(|it| it.as_str())
        .unwrap_or_default();
    table.decor_mut().set_suffix(suffix.to_owned());
    if let Some(mut name) = document.key_mut(key) {
        let above = (name.leaf_decor().prefix())
            .and_then(|it| it.as_str())
            .unwrap_or_default()
            .to_owned();
        table.decor_mut().set_prefix(above);
        name.leaf_decor_mut().set_prefix("");
    }
    document.insert(key, Item::Table(table));
    Ok(())
}

/// `rig plugin list`: prints [`listing`].
fn list(home: &Home) -> Result<()> {
    print!("{}", listing(home)?);
    Ok(())
}

/// Every plugin of rig's, under `rig's plugins:`, then of the home's extra
/// crates, under `your plugins (extras):`, as a tree: each crate, then in
/// it each area of more than one plugin, then the plugins, one line each:
/// `on` or `off`, the type and what it adds. A branch `[plugins] off` turns
/// off says so. A plugin of rig's ends with its default and whether the
/// settings changed it, with a warning on an essential plugin turned off;
/// an extra ends with where its crate comes from.
fn listing(home: &Home) -> Result<String> {
    let chosen = config::chosen(home)?;
    let mut rigs = Vec::new();
    for built_in in BUILT_IN {
        let own = chosen.own.iter().any(|plugin| plugin == built_in.plugin);
        let default = if built_in.on { "on" } else { "off" };
        let mut about = built_in.about.to_owned();
        if !built_in.available() {
            about.push_str(&format!(
                " (default {default}; builds on {} only)",
                built_in.targets.join(" and ")
            ));
        } else if own != built_in.on {
            about.push_str(&format!(
                " (default {default}; set by RIG_HOME/settings.toml)"
            ));
        } else {
            about.push_str(&format!(" (default {default})"));
        }
        rigs.push((built_in.krate, built_in.plugin, about));
    }
    let yours: Vec<_> = (chosen.extra.iter())
        .map(|extra| {
            (
                extra.krate.as_str(),
                extra.plugin.as_str(),
                extra.source.to_string(),
            )
        })
        .collect();
    let mut lines = vec![(String::from("rig's plugins:"), None)];
    lines.extend(tree(&chosen, rigs));
    lines.push((String::new(), None));
    if yours.is_empty() {
        lines.push((
            "your plugins (extras): none; `rig plugin new <name>` makes one, \
             `rig plugin add` adds another crate's."
                .to_owned(),
            None,
        ));
    } else {
        lines.push(("your plugins (extras):".to_owned(), None));
        lines.extend(tree(&chosen, yours));
    }
    let width = (lines.iter())
        .filter(|(_, about)| about.is_some())
        .map(|(head, _)| head.len())
        .max()
        .unwrap_or_default();
    let mut text = String::new();
    for (head, about) in lines {
        let line = match about {
            Some(about) => format!("{head:<width$}  {about}"),
            None => head,
        };
        text.push_str(line.trim_end());
        text.push('\n');
    }
    Ok(text)
}

/// The lines of `plugins`, each its crate, type path and what it adds, as
/// [`listing`] shows them: a heading for each crate where its first plugin
/// is, in it a heading for each area of more than one plugin where its
/// first is, and each plugin under them with its state, its line's head
/// apart from what it adds.
fn tree(chosen: &Plugins, plugins: Vec<(&str, &str, String)>) -> Vec<(String, Option<String>)> {
    let heading = |depth: usize, branch: &str| {
        let mut line = format!("{:indent$}{branch}", "", indent = depth * 2);
        if chosen.off.iter().any(|off| off == branch) {
            line.push_str("  off ([plugins] off)");
        }
        (line, None)
    };
    let mut crates: Vec<&str> = Vec::new();
    for (krate, ..) in &plugins {
        if !crates.contains(krate) {
            crates.push(krate);
        }
    }
    let mut lines = Vec::new();
    for krate in crates {
        lines.push(heading(0, krate));
        let of_crate: Vec<(&str, &str)> = (plugins.iter())
            .filter(|(of, ..)| *of == krate)
            .map(|(_, plugin, about)| (*plugin, about.as_str()))
            .collect();
        let mut areas: Vec<&str> = Vec::new();
        for &(plugin, about) in &of_crate {
            let together: Vec<(&str, &str)> = match area(plugin) {
                Some(of) => (of_crate.iter().copied())
                    .filter(|(it, _)| area(it) == Some(of))
                    .collect(),
                None => Vec::new(),
            };
            match area(plugin) {
                Some(of) if together.len() > 1 => {
                    if !areas.contains(&of) {
                        areas.push(of);
                        lines.push(heading(1, &format!("{krate}/{of}")));
                        lines
                            .extend((together.iter()).map(|(it, about)| row(chosen, 2, it, about)));
                    }
                }
                _ => lines.push(row(chosen, 1, plugin, about)),
            }
        }
    }
    lines
}

/// The line of `plugin` at `depth` in [`tree`], `about` what it adds.
fn row(chosen: &Plugins, depth: usize, plugin: &str, about: &str) -> (String, Option<String>) {
    let on = chosen.on.iter().any(|it| it == plugin);
    let own = chosen.own.iter().any(|it| it == plugin);
    let state = if on { "on" } else { "off" };
    let mut about = about.to_owned();
    if let Some(branch) = chosen.off_with(plugin).filter(|_| own) {
        about.push_str(&format!(" (off with its branch {branch})"));
    }
    if !on && let Some(what) = essential(plugin) {
        about.push_str(&format!(" (warning: off, which takes away {what})"));
    }
    let head = format!("{:indent$}{state:<3} {plugin}", "", indent = depth * 2);
    (head, Some(about))
}

/// `rig plugin enable <name>` and `rig plugin disable <name>`: names the
/// plugin `name` in `[plugins] enable` or `disable` of the home's settings,
/// or takes it out of the other list when that gives rig's default; a
/// name in this list stays. A branch `name` leaves or joins `[plugins]
/// off`. `disable` takes a name that is neither a plugin nor a branch out
/// of every list, and turns off a plugin that does not build here, which
/// is how the settings' refusal of either says to fix it, even while the
/// file holds other mistakes.
fn enable(home: &Home, name: &str, on: bool) -> Result<()> {
    let path = home.settings();
    let (changed, forgot) = config::edit(&path, |file| -> Result<_> {
        if !on && let Err(failure) = config::plugins(file) {
            if !known(file, name) {
                if !config::forget(&mut file.document, name) {
                    return Err(failure.into());
                }
                return Ok((true, true));
            }
            // Turning off the plugin the file cannot run here is its fix.
            if let config::Error::Unavailable { plugin, .. } = &failure
                && plugin == name
            {
                let by_default = built_in(name).is_some_and(|built_in| built_in.on);
                let changed = set_on(&mut file.document, name, false, by_default);
                return Ok((changed, false));
            }
        }
        let changed = config::turn(file, &[(name, on)])?;
        Ok((changed, false))
    })?;
    if forgot {
        println!(
            "Took the unknown plugin or branch `{name}` out of {}.",
            path.display()
        );
        return Ok(());
    }
    let chosen = config::chosen(home)?;
    let branch = chosen.krate(name).is_none();
    let state = if on { "on" } else { "off" };
    let what = if branch { "the branch " } else { "" };
    if !changed {
        let what = if branch { "The branch " } else { "" };
        println!("{what}`{name}` is {state} already in {}.", path.display());
    } else {
        let then = match (branch, on) {
            (false, _) => "",
            (true, false) => {
                ": its plugins stay off whatever their own choice, and enabling it brings \
                 each back as it was"
            }
            (true, true) => ": each of its plugins is as its own choice says",
        };
        println!(
            "Turned {what}`{name}` {state} in {}{then}.\n\
             The next start, or /reload in the agent, rebuilds the agent with it.",
            path.display()
        );
    }
    let plugins: Vec<&str> = if branch {
        held(&chosen, name)
    } else {
        vec![name]
    };
    if on {
        let mut keeping: Vec<&str> = Vec::new();
        for plugin in &plugins {
            if let Some(off) = chosen.off_with(plugin)
                && !keeping.contains(&off)
            {
                keeping.push(off);
            }
        }
        for off in keeping {
            println!(
                "The branch `{off}` is still off, which keeps {} off; \
                 `rig plugin enable {off}` turns it on.",
                if branch { "some of them" } else { "it" }
            );
        }
    } else {
        for what in plugins.iter().filter_map(|plugin| essential(plugin)) {
            println!("Warning: this takes away {what}; `rig plugin enable {name}` brings it back.");
        }
    }
    Ok(())
}

/// The plugins of rig's and the extras of `chosen` that the branch
/// `branch` holds.
fn held<'a>(chosen: &'a Plugins, branch: &str) -> Vec<&'a str> {
    (BUILT_IN.iter())
        .map(|built_in| (built_in.krate, built_in.plugin))
        .chain((chosen.extra.iter()).map(|extra| (extra.krate.as_str(), extra.plugin.as_str())))
        .filter(|(krate, plugin)| holds(branch, krate, plugin))
        .map(|(_, plugin)| plugin)
        .collect()
}

/// Whether a `[[plugins.extra]]` table of `file` names `plugin`, read
/// without the checks the rest of the file may fail.
fn extra(file: &config::SettingsFile, plugin: &str) -> bool {
    (file.document.get("plugins"))
        .and_then(|plugins| plugins.get("extra"))
        .and_then(Item::as_array_of_tables)
        .is_some_and(|tables| {
            (tables.iter()).any(|table| table.get("plugin").and_then(Item::as_str) == Some(plugin))
        })
}

/// Whether `name` is one of rig's plugins, a plugin of a `[[plugins.extra]]`
/// table of `file` or a branch holding one, read without the checks the
/// rest of the file may fail.
fn known(file: &config::SettingsFile, name: &str) -> bool {
    let extras: Vec<(&str, &str)> = (file.document.get("plugins"))
        .and_then(|plugins| plugins.get("extra"))
        .and_then(Item::as_array_of_tables)
        .map(|tables| {
            (tables.iter())
                .map(|table| {
                    let text = |key| table.get(key).and_then(Item::as_str).unwrap_or_default();
                    (text("crate"), text("plugin"))
                })
                .collect()
        })
        .unwrap_or_default();
    (BUILT_IN.iter())
        .map(|built_in| (built_in.krate, built_in.plugin))
        .chain(extras)
        .any(|(krate, plugin)| plugin == name || holds(name, krate, plugin))
}

/// The string at `keys` in `manifest`, such as its `package.name`.
fn manifest_string(manifest: &Manifest, keys: &[&str]) -> Option<String> {
    let mut item = manifest.document.as_item();
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
         # with its own rig crates. A Bevy crate, such as `bevy` for a window, takes\n\
         # \"{bevy}\" too: cargo builds it with the agent's Bevy.\n\
         rig-harness = \"{rig_version}\"\n\
         # Bevy's derives (`Component`, `Resource`, `Message`, `SystemSet`, `Reflect`)\n\
         # name the crate they expand to from this file, so those are direct dependencies.\n\
         bevy_ecs = {{ version = \"{bevy}\", default-features = false }}\n\
         bevy_reflect = {{ version = \"{bevy}\", default-features = false }}\n\
         # A tool's arguments, and its parameters derived from them.\n\
         schemars = \"1\"\n\
         serde = {{ version = \"1\", features = [\"derive\"] }}\n\
         serde_json = \"1\"\n",
        quoted(name),
        bevy = super::bevy_range(),
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
fn check(home: &Home, source: Option<&Path>) -> Result<()> {
    let chosen = config::chosen(home)?;
    let source = RigSource::detect(source)?;
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
fn check_build(launch: &Launch) -> Result<()> {
    let report = super::build::check(launch)?;
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
    let manifest = match Manifest::read(directory) {
        Ok(manifest) => manifest,
        Err(why) => return Some(format!("{what}: {why}")),
    };
    let Some(name) = manifest_string(&manifest, &["package", "name"]) else {
        return Some(format!(
            "{what}: {} has no [package] name",
            directory.join("Cargo.toml").display()
        ));
    };
    if name != extra.krate {
        return Some(format!(
            "{what}: the table says `crate = \"{}\"`, but the package is named `{name}`",
            extra.krate
        ));
    }
    let library =
        manifest_string(&manifest, &["lib", "name"]).unwrap_or_else(|| name.replace('-', "_"));
    let root = extra.plugin.split("::").next().unwrap_or_default();
    if root != library {
        return Some(format!(
            "{what}: the plugin type starts with `{root}`, but the crate's library is `{library}`"
        ));
    }
    let lib =
        manifest_string(&manifest, &["lib", "path"]).unwrap_or_else(|| "src/lib.rs".to_owned());
    if !directory.join(lib).is_file() {
        return Some(format!("{what}: the crate has no library (src/lib.rs)"));
    }
    foreign_rig_crate(extra, source)
}

pub mod skill;

#[cfg(test)]
mod tests;
