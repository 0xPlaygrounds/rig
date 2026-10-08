//! `plugins.toml`: the Bevy plugins added to the agent, in order.
//!
//! The file is a small TOML subset: `[[plugin]]` tables of `key = value`
//! lines, where a value is a string or a one-line array of strings, plus
//! comments. Errors name the line.

use std::path::Path;

use crate::Failure;

/// Written when the config directory has no `plugins.toml`.
const DEFAULT: &str = r#"# Plugins added to the agent, in order. Edit, then /reload.
# Each entry: type = Rust path of a Bevy Plugin that implements Default.
# crate = package name; source = exactly one of path / git (+ rev, branch or tag) / version.
# bevy_features = Bevy features the plugin needs, such as ["bevy_winit"].
# Entries without crate and source come from rig-code itself.

[[plugin]]
type = "rig_code::BuiltinTools"

[[plugin]]
type = "rig_code::BuiltinCommands"

[[plugin]]
type = "rig_code::TuiPlugin"
"#;

/// Keys naming where a plugin crate comes from.
const SOURCES: [&str; 3] = ["path", "git", "version"];
/// Keys that pick a commit of a `git` source.
const GIT_REFS: [&str; 3] = ["rev", "branch", "tag"];

/// One `[[plugin]]` entry.
#[derive(Debug)]
pub(crate) struct Plugin {
    /// The Rust path of the plugin type.
    pub(crate) type_path: String,
    /// The package providing it and its Cargo source keys, or `None` when
    /// it comes from rig-code.
    pub(crate) package: Option<Package>,
    /// Bevy features the plugin needs.
    pub(crate) bevy_features: Vec<String>,
}

/// A plugin crate and where Cargo gets it.
#[derive(Debug, PartialEq)]
pub(crate) struct Package {
    pub(crate) name: String,
    /// `path`, `git` (with `rev`, `branch` or `tag`) or `version`, as
    /// written.
    pub(crate) source: Vec<(String, String)>,
}

/// Reads `plugins.toml` from `config`, writing the default list first when
/// it is missing. A type listed twice is kept once, at its first place.
pub(crate) fn load(config: &Path) -> Result<Vec<Plugin>, Failure> {
    let file = config.join("plugins.toml");
    if !file.exists() {
        std::fs::create_dir_all(config)
            .and_then(|()| std::fs::write(&file, DEFAULT))
            .map_err(|error| Failure::io(format!("cannot write {}", file.display()), error))?;
        eprintln!("rig: wrote the default plugin list to {}", file.display());
    }
    let text = std::fs::read_to_string(&file)
        .map_err(|error| Failure::io(format!("cannot read {}", file.display()), error))?;
    let mut plugins: Vec<Plugin> = Vec::new();
    for plugin in parse(&text).map_err(|error| {
        Failure::config(format!("{} line {}: {}", file.display(), error.0, error.1))
    })? {
        if plugins
            .iter()
            .any(|kept| kept.type_path == plugin.type_path)
        {
            eprintln!(
                "rig: {} lists {} twice; using the first entry",
                file.display(),
                plugin.type_path
            );
            continue;
        }
        plugins.push(plugin);
    }
    Ok(plugins)
}

/// A parse error: the 1-based line and what is wrong.
type ParseError = (usize, String);

/// A table being read: its first line and its `key = value` pairs.
type Table = (usize, Vec<(String, Value)>);

#[derive(Debug)]
enum Value {
    String(String),
    Array(Vec<String>),
}

fn parse(text: &str) -> Result<Vec<Plugin>, ParseError> {
    let mut tables: Vec<Table> = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let number = index + 1;
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        if line == "[[plugin]]" {
            tables.push((number, Vec::new()));
            continue;
        }
        if line.starts_with('[') {
            return Err((
                number,
                format!("only [[plugin]] tables are allowed: {line}"),
            ));
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err((number, format!("expected `key = value`: {line}")));
        };
        let Some((_, pairs)) = tables.last_mut() else {
            return Err((number, "a key before the first [[plugin]]".to_owned()));
        };
        let key = key.trim();
        if pairs.iter().any(|(seen, _)| seen == key) {
            return Err((number, format!("`{key}` is set twice")));
        }
        let value = parse_value(value.trim()).map_err(|message| (number, message))?;
        pairs.push((key.to_owned(), value));
    }
    tables.into_iter().map(plugin).collect()
}

/// Checks one table and turns it into a plugin.
fn plugin((line, pairs): Table) -> Result<Plugin, ParseError> {
    let error = |message: String| (line, message);
    let mut type_path = None;
    let mut name = None;
    let mut bevy_features = Vec::new();
    let mut source = Vec::new();
    for (key, value) in pairs {
        match (key.as_str(), value) {
            ("type", Value::String(value)) => type_path = Some(value),
            ("crate", Value::String(value)) => name = Some(value),
            ("bevy_features", Value::Array(values)) => bevy_features = values,
            (key, Value::String(value)) if SOURCES.contains(&key) || GIT_REFS.contains(&key) => {
                source.push((key.to_owned(), value));
            }
            (key, _) => {
                return Err(error(format!(
                    "unknown key or wrong value type: `{key}` (keys: type, crate, path, git, \
                     rev, branch, tag, version, bevy_features)"
                )));
            }
        }
    }
    let type_path = type_path.ok_or_else(|| error("the [[plugin]] has no `type`".to_owned()))?;
    if type_path.is_empty()
        || !type_path
            .split("::")
            .all(|part| !part.is_empty() && part.chars().all(|c| c.is_alphanumeric() || c == '_'))
    {
        return Err(error(format!("`{type_path}` is not a Rust type path")));
    }
    let kinds: Vec<&str> = source
        .iter()
        .map(|(key, _)| key.as_str())
        .filter(|key| SOURCES.contains(key))
        .collect();
    let refs = source
        .iter()
        .filter(|(key, _)| GIT_REFS.contains(&key.as_str()))
        .count();
    let package = match (name, kinds.as_slice()) {
        (None, []) => None,
        (Some(name), [kind]) => {
            if refs > 1 || (refs == 1 && *kind != "git") {
                return Err(error(
                    "use at most one of rev, branch or tag, and only with git".to_owned(),
                ));
            }
            Some(Package { name, source })
        }
        (None, _) => return Err(error(format!("{type_path} has a source but no `crate`"))),
        (Some(name), _) => {
            return Err(error(format!(
                "crate `{name}` needs exactly one of path, git or version"
            )));
        }
    };
    Ok(Plugin {
        type_path,
        package,
        bevy_features,
    })
}

/// `line` without a `#` comment that is outside a string.
fn strip_comment(line: &str) -> &str {
    let mut quote = None;
    let mut escaped = false;
    for (at, c) in line.char_indices() {
        match (quote, c) {
            (Some('"'), '\\') if !escaped => {
                escaped = true;
                continue;
            }
            (Some(open), c) if c == open && !escaped => quote = None,
            (None, '"' | '\'') => quote = Some(c),
            (None, '#') => return line.get(..at).unwrap_or(line),
            _ => {}
        }
        escaped = false;
    }
    line
}

fn parse_value(text: &str) -> Result<Value, String> {
    if let Some(inner) = text.strip_prefix('[') {
        let inner = inner
            .strip_suffix(']')
            .ok_or("an array must close on the same line")?;
        let mut values = Vec::new();
        let mut rest = inner.trim();
        while !rest.is_empty() {
            let (value, after) = parse_string(rest)?;
            values.push(value);
            rest = after.trim_start();
            rest = match rest.strip_prefix(',') {
                Some(after) => after.trim_start(),
                None if rest.is_empty() => rest,
                None => return Err(format!("expected `,` in the array: {rest}")),
            };
        }
        return Ok(Value::Array(values));
    }
    let (value, rest) = parse_string(text)?;
    if !rest.trim().is_empty() {
        return Err(format!("unexpected text after the string: {rest}"));
    }
    Ok(Value::String(value))
}

/// Reads one quoted string at the start of `text`; returns it and the rest.
fn parse_string(text: &str) -> Result<(String, &str), String> {
    if let Some(literal) = text.strip_prefix('\'') {
        let end = literal.find('\'').ok_or("unterminated string")?;
        let (value, rest) = literal.split_at(end);
        return Ok((value.to_owned(), rest.get(1..).unwrap_or_default()));
    }
    let basic = text
        .strip_prefix('"')
        .ok_or_else(|| format!("expected a quoted string: {text}"))?;
    let mut value = String::new();
    let mut chars = basic.char_indices();
    while let Some((at, c)) = chars.next() {
        match c {
            '"' => return Ok((value, basic.get(at + 1..).unwrap_or_default())),
            '\\' => match chars.next().map(|(_, c)| c) {
                Some('"') => value.push('"'),
                Some('\\') => value.push('\\'),
                Some('n') => value.push('\n'),
                Some('t') => value.push('\t'),
                other => return Err(format!("unsupported escape \\{}", other.unwrap_or(' '))),
            },
            c => value.push(c),
        }
    }
    Err("unterminated string".to_owned())
}
