//! `rig.toml`: the plugin list and build settings. The file is a small TOML
//! subset: comments, a top-level `jobs = <integer>`, and `[[plugin]]`
//! tables whose keys hold a string, an integer or a list of strings.

use std::collections::BTreeMap;
use std::fs;
use std::io::ErrorKind;
use std::path::Path;

use super::Result;

/// Written when `rig.toml` does not exist yet.
const TEMPLATE: &str = r#"# rig agent settings. `rig build` or /reload in the agent applies changes.

# jobs = 8                        # cargo -j for building the agent

# One table per Bevy plugin crate added to the agent:
# [[plugin]]
# crate = "rig-hello"             # the package name
# path = "/abs/path/to/rig-hello" # exactly one of: path, git (with optional branch or rev), version
# plugin = "rig_hello::HelloPlugin" # a type implementing Plugin + Default
# bevy_features = []              # extra Bevy features the plugin needs
"#;

/// The parsed settings.
pub struct Config {
    /// cargo's `-j` for the agent project.
    pub jobs: Option<u64>,
    /// The plugins, in file order.
    pub plugins: Vec<Plugin>,
}

/// One `[[plugin]]` table.
pub struct Plugin {
    /// The package name.
    pub krate: String,
    /// Where the package comes from.
    pub source: Source,
    /// The plugin type's path, such as `rig_hello::HelloPlugin`.
    pub type_path: String,
    /// Bevy features the plugin needs.
    pub bevy_features: Vec<String>,
}

/// Where a plugin package comes from.
pub enum Source {
    /// A local directory; a relative one is relative to `RIG_HOME`.
    Path(String),
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
    Integer(u64),
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
        parse(&text).map_err(|failure| format!("{}: {failure}", path.display()).into())
    }
}

fn parse(text: &str) -> Result<Config> {
    let mut jobs = None;
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
                "line {number}: `{}` is not a string, an integer or a list of strings",
                value.trim()
            )
        })?;
        match tables.last_mut() {
            Some((_, table)) => {
                if table.insert(key.to_owned(), value).is_some() {
                    return Err(format!("line {number}: `{key}` is set twice").into());
                }
            }
            None => match (key, value) {
                ("jobs", Value::Integer(count)) => jobs = Some(count),
                _ => {
                    return Err(format!(
                        "line {number}: only `jobs = <integer>` may come before the first \
                         [[plugin]]"
                    )
                    .into());
                }
            },
        }
    }
    let plugins = tables
        .into_iter()
        .map(|(number, table)| {
            plugin(table).map_err(|failure| format!("[[plugin]] at line {number}: {failure}"))
        })
        .collect::<std::result::Result<_, _>>()?;
    Ok(Config { jobs, plugins })
}

fn plugin(mut table: BTreeMap<String, Value>) -> Result<Plugin> {
    let krate = string(&mut table, "crate")?.ok_or("`crate` is missing")?;
    if krate.is_empty()
        || !krate
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
    {
        return Err(format!("`{krate}` is not a crate name").into());
    }
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
    let source = match (path, git, version) {
        (None, Some(url), None) => Source::Git { url, branch, rev },
        _ if branch.is_some() || rev.is_some() => {
            return Err("`branch` and `rev` only go with `git`".into());
        }
        (Some(path), None, None) => Source::Path(path),
        (None, None, Some(version)) => Source::Version(version),
        _ => return Err("set exactly one of `path`, `git` and `version`".into()),
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
        krate,
        source,
        type_path,
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
    text.parse().ok().map(Value::Integer)
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
