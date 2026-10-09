//! `plugins.toml`: the plugin list. The file is a small TOML subset:
//! comments and `[[plugin]]` tables whose keys hold a string or a list of
//! strings.

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use super::Result;
use super::project::PACKAGE;

/// Written when `plugins.toml` does not exist yet.
const TEMPLATE: &str = r#"# The rig agent's plugins, added in this order. `rig build`, or /reload in
# the agent, applies changes. CARGO_BUILD_JOBS sets cargo's -j for them.

# Each [[plugin]] names a type implementing Bevy's Plugin + Default. An entry
# without `crate` comes from rig-harness itself.

# The read, edit, write, shell and search tools.
[[plugin]]
plugin = "rig_harness::builtin::BuiltinToolsPlugin"

# /model, /effort, /help and /quit. (/reload is always there.)
[[plugin]]
plugin = "rig_harness::builtin::BuiltinCommandsPlugin"

# Subagents: the task and message tools.
[[plugin]]
plugin = "rig_harness::builtin::SubagentsPlugin"

# The terminal view. Without it the agent runs headless.
[[plugin]]
plugin = "rig_harness::tui::TuiPlugin"

# A plugin from another crate:
# [[plugin]]
# crate = "rig-hello"               # the package name
# path = "../rig-hello"             # exactly one of: path (relative to this file),
#                                   # git (with optional branch or rev), version
# plugin = "rig_hello::HelloPlugin"
# bevy_features = []                # extra Bevy features the plugin needs
"#;

/// The parsed plugin list.
pub struct Config {
    /// The plugins, in file order.
    pub plugins: Vec<Plugin>,
}

/// One `[[plugin]]` table.
pub struct Plugin {
    /// The plugin type's path, such as `rig_hello::HelloPlugin`.
    pub type_path: String,
    /// The package that provides it, or `None` for rig-harness's own.
    pub package: Option<Package>,
    /// Bevy features the plugin needs.
    pub bevy_features: Vec<String>,
}

/// A plugin package.
pub struct Package {
    /// The package name.
    pub name: String,
    /// Where it comes from.
    pub source: Source,
}

/// Where a plugin package comes from.
#[derive(PartialEq, Eq)]
pub enum Source {
    /// A local directory, made absolute against the directory of
    /// `plugins.toml`.
    Path(PathBuf),
    /// A git repository, optionally at a branch or revision.
    Git {
        /// The repository URL.
        url: String,
        /// The branch.
        branch: Option<String>,
        /// The revision.
        rev: Option<String>,
    },
    /// A crates.io version requirement.
    Version(String),
}

enum Value {
    String(String),
    List(Vec<String>),
}

impl Config {
    /// Reads `path`, writing a commented template first when it does not
    /// exist. Errors name the file and line.
    pub fn load(path: &Path) -> Result<Self> {
        let text = match fs::read_to_string(path) {
            Ok(text) => text,
            Err(failure) if failure.kind() == ErrorKind::NotFound => {
                if let Some(parent) = path.parent() {
                    fs::create_dir_all(parent)?;
                }
                fs::write(path, TEMPLATE)?;
                TEMPLATE.to_owned()
            }
            Err(failure) => return Err(format!("{}: {failure}", path.display()).into()),
        };
        let base = path.parent().unwrap_or(Path::new("."));
        parse(&text, base).map_err(|failure| format!("{}: {failure}", path.display()).into())
    }
}

/// Parses `text`; relative plugin paths are relative to `base`.
fn parse(text: &str, base: &Path) -> Result<Config> {
    let mut tables: Vec<(usize, BTreeMap<String, Value>)> = Vec::new();
    for (number, line) in (1..).zip(text.lines()) {
        let line = without_comment(line).trim();
        if line.is_empty() {
            continue;
        }
        if line == "[[plugin]]" {
            tables.push((number, BTreeMap::new()));
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(format!("line {number}: expected `key = value` or `[[plugin]]`").into());
        };
        let key = key.trim();
        let value = parse_value(value.trim()).ok_or_else(|| {
            format!(
                "line {number}: `{}` is not a string or a list of strings",
                value.trim()
            )
        })?;
        let Some((_, table)) = tables.last_mut() else {
            return Err(format!("line {number}: `{key}` comes before the first [[plugin]]").into());
        };
        if table.insert(key.to_owned(), value).is_some() {
            return Err(format!("line {number}: `{key}` is set twice").into());
        }
    }
    let mut plugins: Vec<Plugin> = Vec::new();
    let mut types = BTreeSet::new();
    for (number, table) in tables {
        let plugin = plugin(table, base)
            .and_then(|plugin| {
                if !types.insert(plugin.type_path.clone()) {
                    return Err(format!("`{}` is listed twice", plugin.type_path).into());
                }
                if let Some(package) = &plugin.package
                    && plugins
                        .iter()
                        .filter_map(|listed| listed.package.as_ref())
                        .any(|listed| {
                            listed.name == package.name && listed.source != package.source
                        })
                {
                    return Err(
                        format!("`{}` is listed before with another source", package.name).into(),
                    );
                }
                Ok(plugin)
            })
            .map_err(|failure| format!("[[plugin]] at line {number}: {failure}"))?;
        plugins.push(plugin);
    }
    Ok(Config { plugins })
}

fn plugin(mut table: BTreeMap<String, Value>, base: &Path) -> Result<Plugin> {
    let type_path = string(&mut table, "plugin")?.ok_or("`plugin` (the type path) is missing")?;
    if !type_path.split("::").all(|segment| {
        segment.chars().next().is_some_and(|c| !c.is_ascii_digit())
            && segment
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
    }) {
        return Err(format!("`{type_path}` is not a Rust type path").into());
    }
    let path = string(&mut table, "path")?;
    let git = string(&mut table, "git")?;
    let version = string(&mut table, "version")?;
    let branch = string(&mut table, "branch")?;
    let rev = string(&mut table, "rev")?;
    let package = match string(&mut table, "crate")? {
        None if [&path, &git, &version, &branch, &rev]
            .iter()
            .any(|key| key.is_some()) =>
        {
            return Err("a plugin from another crate needs `crate`, its package name".into());
        }
        None => {
            // The generated project depends on rig-harness, and on bevy
            // when a plugin asks for Bevy features; any other crate needs
            // its own entry.
            let root = type_path.split("::").next().unwrap_or_default();
            if !["rig_harness", "bevy"].contains(&root) {
                return Err(format!(
                    "`{type_path}` names the crate `{root}`, which is no dependency of the agent: \
                     an entry without `crate` is one of rig-harness's own plugins, under \
                     `rig_harness::`; a plugin from another crate needs `crate` and a source."
                )
                .into());
            }
            None
        }
        Some(name) => {
            if name.is_empty()
                || !name
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                return Err(format!("`{name}` is not a crate name").into());
            }
            if ["rig-harness", PACKAGE, "bevy"].contains(&name.as_str()) {
                return Err(format!(
                    "`{name}` is part of the agent itself; rig-harness's own plugins need no `crate`"
                )
                .into());
            }
            let source = match (path, git, version) {
                (None, Some(url), None) => Source::Git { url, branch, rev },
                _ if branch.is_some() || rev.is_some() => {
                    return Err("`branch` and `rev` only go with `git`".into());
                }
                (Some(path), None, None) => {
                    let path = base.join(path);
                    Source::Path(std::path::absolute(&path).unwrap_or(path))
                }
                (None, None, Some(version)) => Source::Version(version),
                _ => return Err("set exactly one of `path`, `git` and `version`".into()),
            };
            Some(Package { name, source })
        }
    };
    let bevy_features = match table.remove("bevy_features") {
        None => Vec::new(),
        Some(Value::List(features)) => features,
        Some(_) => return Err("`bevy_features` must be a list of strings".into()),
    };
    if let Some(key) = table.keys().next() {
        return Err(format!("unknown key `{key}`").into());
    }
    Ok(Plugin {
        type_path,
        package,
        bevy_features,
    })
}

fn string(table: &mut BTreeMap<String, Value>, key: &str) -> Result<Option<String>> {
    match table.remove(key) {
        None => Ok(None),
        Some(Value::String(value)) => Ok(Some(value)),
        Some(_) => Err(format!("`{key}` must be a string").into()),
    }
}

/// The line up to a `#` that is not inside a string.
fn without_comment(line: &str) -> &str {
    let mut in_string = false;
    let mut escaped = false;
    for (index, c) in line.char_indices() {
        match c {
            _ if escaped => escaped = false,
            '\\' if in_string => escaped = true,
            '"' => in_string = !in_string,
            '#' if !in_string => return line.get(..index).unwrap_or(line),
            _ => {}
        }
    }
    line
}

fn parse_value(text: &str) -> Option<Value> {
    if text.starts_with('"') {
        let (value, rest) = parse_string(text)?;
        return rest.trim().is_empty().then_some(Value::String(value));
    }
    if let Some(mut rest) = text.strip_prefix('[') {
        let mut items = Vec::new();
        loop {
            rest = rest.trim_start();
            if let Some(after) = rest.strip_prefix(']') {
                return after.trim().is_empty().then_some(Value::List(items));
            }
            let (item, after) = parse_string(rest)?;
            items.push(item);
            rest = after.trim_start();
            rest = match rest.strip_prefix(',') {
                Some(after) => after,
                None if rest.starts_with(']') => rest,
                None => return None,
            };
        }
    }
    None
}

/// A basic TOML string at the start of `text`, and the text after it.
fn parse_string(text: &str) -> Option<(String, &str)> {
    let mut chars = text.strip_prefix('"')?.char_indices();
    let mut value = String::new();
    while let Some((index, c)) = chars.next() {
        match c {
            '"' => return Some((value, text.get(index + 2..)?)),
            '\\' => value.push(match chars.next()?.1 {
                'n' => '\n',
                't' => '\t',
                '"' => '"',
                '\\' => '\\',
                _ => return None,
            }),
            c => value.push(c),
        }
    }
    None
}
