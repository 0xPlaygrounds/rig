//! `plugins.toml`: the plugin list. The file is a small TOML subset:
//! comments and `[[plugin]]` tables whose keys hold a string.

use std::collections::{BTreeMap, BTreeSet};
use std::fmt;
use std::fs;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use super::Result;
use super::project::PACKAGE;

/// Written when `plugins.toml` does not exist yet.
const TEMPLATE: &str = include_str!("plugins.toml");

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
}

/// A plugin package.
pub struct Package {
    /// The package name.
    pub name: String,
    /// Where it comes from.
    pub source: Source,
}

/// Where a plugin package comes from.
#[derive(PartialEq, Eq, Debug)]
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

impl fmt::Display for Source {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Path(path) => write!(f, "path {}", path.display()),
            Self::Git { url, branch, rev } => {
                write!(f, "git {url}")?;
                if let Some(branch) = branch {
                    write!(f, ", branch {branch}")?;
                }
                if let Some(rev) = rev {
                    write!(f, ", rev {rev}")?;
                }
                Ok(())
            }
            Self::Version(version) => write!(f, "version {version}"),
        }
    }
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
        parse_file(path, &text)
    }
}

/// Parses `text` as the plugin list at `path`: relative plugin paths are
/// relative to its directory, and errors name it.
fn parse_file(path: &Path, text: &str) -> Result<Config> {
    let base = path.parent().unwrap_or(Path::new("."));
    parse(text, base).map_err(|failure| format!("{}: {failure}", path.display()).into())
}

/// Adds `entry`, the text of one `[[plugin]]` table, at the end of the
/// plugin list at `path` (made from the template when missing), and
/// checks the result, then with `check`; nothing is written when either
/// fails.
pub fn append(
    path: &Path,
    entry: &str,
    check: impl FnOnce(&Config) -> Result<()>,
) -> Result<Config> {
    Config::load(path)?;
    let mut text = fs::read_to_string(path)?;
    if !text.is_empty() && !text.ends_with('\n') {
        text.push('\n');
    }
    text.push('\n');
    text.push_str(entry);
    let config = parse_file(path, &text)?;
    check(&config).map_err(|failure| format!("{}: {failure}", path.display()))?;
    fs::write(path, text)?;
    Ok(config)
}

/// Takes the `[[plugin]]` table whose `plugin` is `type_path` out of the
/// plugin list at `path`, with the comment lines right above it, and checks
/// the result; nothing is written when it is not valid.
pub fn remove(path: &Path, type_path: &str) -> Result<()> {
    let text =
        fs::read_to_string(path).map_err(|failure| format!("{}: {failure}", path.display()))?;
    let base = path.parent().unwrap_or(Path::new("."));
    let kept = without_table(&text, type_path, base)
        .map_err(|failure| format!("{}: {failure}", path.display()))?;
    parse_file(path, &kept)?;
    fs::write(path, kept)?;
    Ok(())
}

/// `text` without the table of `type_path` and the comment lines right
/// above it.
fn without_table(text: &str, type_path: &str, base: &Path) -> Result<String> {
    let config = parse(text, base)?;
    let Some(index) = config
        .plugins
        .iter()
        .position(|plugin| plugin.type_path == type_path)
    else {
        return Err(format!("`{type_path}` is not listed").into());
    };
    let lines: Vec<&str> = text.lines().collect();
    let headers: Vec<usize> = lines
        .iter()
        .enumerate()
        .filter(|(_, line)| without_comment(line).trim() == "[[plugin]]")
        .map(|(number, _)| number)
        .collect();
    let start = headers
        .get(index)
        .copied()
        .ok_or("the table is not found")?;
    let mut end = headers.get(index + 1).copied().unwrap_or(lines.len());
    // Comments right above the next table are that table's.
    while end > start + 1
        && lines
            .get(end - 1)
            .is_some_and(|line| without_comment(line).trim().is_empty())
    {
        end -= 1;
    }
    let mut from = start;
    while from > 0
        && lines
            .get(from - 1)
            .is_some_and(|line| line.trim_start().starts_with('#'))
    {
        from -= 1;
    }
    let mut kept: Vec<&str> = lines.get(..from).unwrap_or_default().to_vec();
    kept.extend(lines.get(end..).unwrap_or_default());
    // No run of blank lines where the table was.
    if kept.get(from).is_some_and(|line| line.trim().is_empty())
        && from
            .checked_sub(1)
            .and_then(|before| kept.get(before))
            .is_some_and(|line| line.trim().is_empty())
    {
        kept.remove(from);
    }
    let mut kept = kept.join("\n");
    kept.push('\n');
    Ok(kept)
}

/// Parses `text`; relative plugin paths are relative to `base`.
fn parse(text: &str, base: &Path) -> Result<Config> {
    let mut tables: Vec<(usize, BTreeMap<String, String>)> = Vec::new();
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
        let value = parse_value(value.trim())
            .ok_or_else(|| format!("line {number}: `{}` is not a string", value.trim()))?;
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

fn plugin(mut table: BTreeMap<String, String>, base: &Path) -> Result<Plugin> {
    let type_path = table
        .remove("plugin")
        .ok_or("`plugin` (the type path) is missing")?;
    if !type_path.split("::").all(|segment| {
        segment.chars().next().is_some_and(|c| !c.is_ascii_digit())
            && segment
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_')
    }) {
        return Err(format!("`{type_path}` is not a Rust type path").into());
    }
    let path = table.remove("path");
    let git = table.remove("git");
    let version = table.remove("version");
    let branch = table.remove("branch");
    let rev = table.remove("rev");
    let package = match table.remove("crate") {
        None if [&path, &git, &version, &branch, &rev]
            .iter()
            .any(|key| key.is_some()) =>
        {
            return Err("a plugin from another crate needs `crate`, its package name".into());
        }
        None => {
            // The generated project depends on rig-harness; any other
            // crate needs its own entry.
            let root = type_path.split("::").next().unwrap_or_default();
            if root != "rig_harness" {
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
    if let Some(key) = table.keys().next() {
        return Err(format!("unknown key `{key}`").into());
    }
    Ok(Plugin { type_path, package })
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

/// A value that is one basic TOML string.
fn parse_value(text: &str) -> Option<String> {
    let (value, rest) = parse_string(text)?;
    rest.trim().is_empty().then_some(value)
}

/// A basic TOML string at the start of `text`, and the text after it.
pub fn parse_string(text: &str) -> Option<(String, &str)> {
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

#[cfg(test)]
mod tests;
