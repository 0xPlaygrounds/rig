//! `plugins.toml`: a hand-parsed TOML subset of `[[plugin]]` tables with
//! string and string-array values.

use std::path::Path;

/// What `plugins.toml` holds on first run.
pub const TEMPLATE: &str = r#"# Plugins of the rig-code agent. Each [[plugin]] table adds one Bevy plugin.
# Run /reload in the agent (or `rig build`) after editing.
#
# [[plugin]]
# crate = "rig-hello"                 # package name
# path = "/home/me/rig-hello"         # or: git = "https://..." with optional branch, tag or rev
#                                     # or: version = "0.1"
# plugin = "rig_hello::HelloPlugin"   # a type implementing Plugin + Default
# bevy_features = []                  # optional features of the pinned bevy crate
"#;

/// One plugin entry.
pub struct Plugin {
    /// The package name.
    pub krate: String,
    /// Where the package comes from.
    pub source: Source,
    /// The plugin type's path, e.g. `rig_hello::HelloPlugin`.
    pub plugin: String,
    /// Features the plugin needs on the pinned `bevy` dependency.
    pub bevy_features: Vec<String>,
}

/// Where a plugin package comes from.
pub enum Source {
    /// A local directory.
    Path(String),
    /// A git repository, with an optional `branch`, `tag` or `rev`.
    Git {
        /// The repository URL.
        url: String,
        /// `("branch" | "tag" | "rev", value)`.
        reference: Option<(&'static str, String)>,
    },
    /// A crates.io version requirement.
    Version(String),
}

/// A parsed value.
enum Value {
    Str(String),
    List(Vec<String>),
}

/// One `[[plugin]]` table as written, before validation.
#[derive(Default)]
struct Table {
    line: usize,
    entries: Vec<(String, Value, usize)>,
}

/// Parse `text`, read from `path`. Errors name the file and line.
pub fn parse(path: &Path, text: &str) -> Result<Vec<Plugin>, String> {
    let at = |line: usize, message: &str| format!("{}:{line}: {message}", path.display());
    let mut tables: Vec<Table> = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let number = index + 1;
        let line = raw.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') {
            if strip_comment(line).trim() != "[[plugin]]" {
                return Err(at(number, "only [[plugin]] tables are allowed"));
            }
            tables.push(Table {
                line: number,
                entries: Vec::new(),
            });
            continue;
        }
        let Some((key, rest)) = line.split_once('=') else {
            return Err(at(number, "expected `key = value`"));
        };
        let Some(table) = tables.last_mut() else {
            return Err(at(number, "a key outside a [[plugin]] table"));
        };
        let key = key.trim().to_owned();
        if table.entries.iter().any(|(seen, _, _)| *seen == key) {
            return Err(at(number, &format!("`{key}` is set twice")));
        }
        let value = parse_value(rest.trim()).map_err(|message| at(number, &message))?;
        table.entries.push((key, value, number));
    }
    tables
        .into_iter()
        .map(|table| plugin(table).map_err(|(line, message)| at(line, &message)))
        .collect()
}

/// A validated plugin from its table, or the line and the problem.
fn plugin(table: Table) -> Result<Plugin, (usize, String)> {
    let mut krate = None;
    let mut plugin = None;
    let mut bevy_features = Vec::new();
    let mut path = None;
    let mut git = None;
    let mut version = None;
    let mut reference = None;
    for (key, value, line) in table.entries {
        let fail = |message: String| (line, message);
        match (key.as_str(), value) {
            ("bevy_features", Value::List(features)) => {
                if let Some(bad) = features.iter().find(|feature| !is_name(feature)) {
                    return Err(fail(format!("`{bad}` is not a feature name")));
                }
                bevy_features = features;
            }
            ("bevy_features", Value::Str(_)) => {
                return Err(fail("`bevy_features` takes a list of strings".to_owned()));
            }
            (_, Value::List(_)) => return Err(fail(format!("`{key}` takes a string"))),
            ("crate", Value::Str(name)) if is_name(&name) => krate = Some(name),
            ("crate", Value::Str(name)) => {
                return Err(fail(format!("`{name}` is not a package name")));
            }
            ("plugin", Value::Str(name)) if is_type_path(&name) => plugin = Some(name),
            ("plugin", Value::Str(name)) => {
                return Err(fail(format!(
                    "`{name}` is not a type path such as my_crate::MyPlugin"
                )));
            }
            ("path", Value::Str(value)) => path = Some(value),
            ("git", Value::Str(value)) => git = Some(value),
            ("version", Value::Str(value)) => version = Some(value),
            ("branch", Value::Str(value)) => reference = Some(("branch", value, line)),
            ("tag", Value::Str(value)) => reference = Some(("tag", value, line)),
            ("rev", Value::Str(value)) => reference = Some(("rev", value, line)),
            _ => return Err(fail(format!("unknown key `{key}`"))),
        }
    }
    let line = table.line;
    let source = match (path, git, version) {
        (Some(path), None, None) => Source::Path(path),
        (None, Some(url), None) => Source::Git {
            url,
            reference: reference.take().map(|(kind, value, _)| (kind, value)),
        },
        (None, None, Some(version)) => Source::Version(version),
        _ => {
            return Err((
                line,
                "a plugin needs exactly one of `path`, `git` or `version`".to_owned(),
            ));
        }
    };
    if let Some((kind, _, line)) = reference {
        return Err((line, format!("`{kind}` needs `git`")));
    }
    Ok(Plugin {
        krate: krate.ok_or((line, "a plugin needs `crate`".to_owned()))?,
        source,
        plugin: plugin.ok_or((line, "a plugin needs `plugin`".to_owned()))?,
        bevy_features,
    })
}

fn parse_value(text: &str) -> Result<Value, String> {
    if let Some(rest) = text.strip_prefix('[') {
        let mut items = Vec::new();
        let mut rest = rest.trim_start();
        loop {
            if let Some(after) = rest.strip_prefix(']') {
                return end(after).map(|()| Value::List(items));
            }
            let (item, after) = string(rest)?;
            items.push(item);
            rest = after.trim_start();
            if let Some(after) = rest.strip_prefix(',') {
                rest = after.trim_start();
            } else if !rest.starts_with(']') {
                return Err("expected `,` or `]` in the list".to_owned());
            }
        }
    }
    let (value, rest) = string(text)?;
    end(rest).map(|()| Value::Str(value))
}

/// Only a comment may follow a value.
fn end(rest: &str) -> Result<(), String> {
    let rest = rest.trim();
    if rest.is_empty() || rest.starts_with('#') {
        Ok(())
    } else {
        Err(format!("unexpected `{rest}` after the value"))
    }
}

/// A basic `"..."` string at the start of `text`, and what follows it.
fn string(text: &str) -> Result<(String, &str), String> {
    let Some(body) = text.strip_prefix('"') else {
        return Err("expected a \"string\"".to_owned());
    };
    let mut value = String::new();
    let mut chars = body.char_indices();
    while let Some((at, c)) = chars.next() {
        match c {
            '"' => return Ok((value, body.get(at + 1..).unwrap_or_default())),
            '\\' => match chars.next().map(|(_, escaped)| escaped) {
                Some('"') => value.push('"'),
                Some('\\') => value.push('\\'),
                Some('n') => value.push('\n'),
                Some('t') => value.push('\t'),
                _ => return Err("unsupported escape in a string".to_owned()),
            },
            c => value.push(c),
        }
    }
    Err("unterminated string".to_owned())
}

/// The part of a line before a `#` comment, for lines without strings.
fn strip_comment(line: &str) -> &str {
    line.split_once('#').map_or(line, |(code, _)| code)
}

fn is_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn is_type_path(path: &str) -> bool {
    path.contains("::")
        && path.split("::").all(|segment| {
            segment.chars().next().is_some_and(|c| !c.is_ascii_digit())
                && segment
                    .chars()
                    .all(|c| c.is_ascii_alphanumeric() || c == '_')
        })
}
