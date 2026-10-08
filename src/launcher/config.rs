//! The plugin list, `plugins.toml`: valid TOML in a restricted form the
//! launcher reads without a TOML library. It holds comment lines,
//! `[section]` headers and single-line `key = value` pairs.

use std::{fs, path::Path};

use super::{Error, Result};

/// Written when no plugin list exists yet.
const TEMPLATE: &str = r#"# Plugins for the rig coding agent. Run /reload after editing.
#
# One [plugins.<crate>] section per plugin crate. `plugin` is the plugin type
# path; the type must implement Bevy's Plugin and Default. `bevy_features`
# lists extra Bevy features the plugin needs. Every other key is copied into
# the crate's Cargo dependency (path, git, branch, rev, version, features...).
# Plugins must use bevy = "=0.20.0-rc.2".
#
# [plugins.rig-hello]
# plugin = "rig_hello::HelloPlugin"
# path = "/home/me/rig-hello"

[build]
# Parallel cargo jobs; `rig -j N` overrides it.
# jobs = 8
"#;

/// The parsed plugin list.
#[derive(Debug, Default)]
pub struct Config {
    /// `[build] jobs`.
    pub jobs: Option<u32>,
    /// The plugin crates, in file order.
    pub plugins: Vec<PluginCrate>,
}

/// One `[plugins.<crate>]` section.
#[derive(Debug)]
pub struct PluginCrate {
    /// The dependency name: the section name.
    pub name: String,
    /// The plugin type path.
    pub plugin: String,
    /// Extra Bevy features.
    pub bevy_features: Vec<String>,
    /// Every other key, with its value as written.
    pub dependency: Vec<(String, String)>,
}

impl PluginCrate {
    /// The crate's package name: its `package` key, else its dependency name.
    pub fn package(&self) -> String {
        self.dependency
            .iter()
            .find(|(key, _)| key == "package")
            .and_then(|(_, value)| string(value))
            .unwrap_or_else(|| self.name.clone())
    }
}

/// Read the plugin list at `path`, writing the commented template first if
/// it does not exist.
pub fn load(path: &Path) -> Result<Config> {
    if !path.exists() {
        if let Some(dir) = path.parent() {
            fs::create_dir_all(dir)?;
        }
        fs::write(path, TEMPLATE)?;
    }
    let text = fs::read_to_string(path)?;
    parse(&text).map_err(|(line, message)| Error(format!("{}:{line}: {message}", path.display())))
}

/// Parse the restricted TOML form. An error names its line.
fn parse(text: &str) -> std::result::Result<Config, (usize, String)> {
    enum Section {
        None,
        Build,
        Plugin,
    }
    let mut config = Config::default();
    let mut section = Section::None;
    let mut starts = Vec::new();
    for (index, raw) in text.lines().enumerate() {
        let number = index + 1;
        let fail = |message: String| (number, message);
        let line = strip_comment(raw).trim();
        if line.is_empty() {
            continue;
        }
        if let Some(header) = line
            .strip_prefix('[')
            .and_then(|line| line.strip_suffix(']'))
        {
            let header = header.trim();
            section = if header == "build" {
                Section::Build
            } else if let Some(name) = header.strip_prefix("plugins.") {
                let name = string(name).unwrap_or_else(|| name.to_owned());
                if !is_crate_name(&name) {
                    return Err(fail(format!("`{name}` is not a crate name")));
                }
                config.plugins.push(PluginCrate {
                    name,
                    plugin: String::new(),
                    bevy_features: Vec::new(),
                    dependency: Vec::new(),
                });
                starts.push(number);
                Section::Plugin
            } else {
                return Err(fail(format!(
                    "unknown section `[{header}]`; use [build] or [plugins.<crate>]"
                )));
            };
            continue;
        }
        let Some((key, value)) = line.split_once('=') else {
            return Err(fail("expected `key = value`".into()));
        };
        let (key, value) = (key.trim(), value.trim());
        if !is_crate_name(key) {
            return Err(fail(format!("`{key}` is not a plain key")));
        }
        match (&section, config.plugins.last_mut()) {
            (Section::Build, _) if key == "jobs" => {
                let jobs = value.parse().ok().filter(|jobs| *jobs > 0);
                config.jobs =
                    Some(jobs.ok_or_else(|| fail("jobs takes a positive number".into()))?);
            }
            (Section::Build, _) => return Err(fail(format!("unknown key `{key}` in [build]"))),
            (Section::Plugin, Some(plugin)) if key == "plugin" => {
                plugin.plugin =
                    string(value)
                        .filter(|path| is_type_path(path))
                        .ok_or_else(|| {
                            fail("plugin takes a type path such as \"my_crate::MyPlugin\"".into())
                        })?;
            }
            (Section::Plugin, Some(plugin)) if key == "bevy_features" => {
                plugin.bevy_features = strings(value)
                    .ok_or_else(|| fail("bevy_features takes a list of strings".into()))?;
            }
            (Section::Plugin, Some(plugin)) => {
                plugin.dependency.push((key.to_owned(), value.to_owned()));
            }
            _ => return Err(fail("a key outside [build] or [plugins.<crate>]".into())),
        }
    }
    for (plugin, start) in config.plugins.iter().zip(starts) {
        if plugin.plugin.is_empty() {
            return Err((
                start,
                format!("[plugins.{}] has no `plugin` key", plugin.name),
            ));
        }
    }
    Ok(config)
}

/// The line up to a `#` that is not inside a string.
fn strip_comment(line: &str) -> &str {
    let mut quote = None;
    for (index, character) in line.char_indices() {
        match (quote, character) {
            (None, '#') => return line.get(..index).unwrap_or(line),
            (None, '"' | '\'') => quote = Some(character),
            (Some(open), _) if open == character => quote = None,
            _ => {}
        }
    }
    line
}

/// The contents of a quoted string without escapes.
fn string(value: &str) -> Option<String> {
    let value = value.trim();
    let inner = value
        .strip_prefix('"')
        .and_then(|value| value.strip_suffix('"'))
        .or_else(|| {
            value
                .strip_prefix('\'')
                .and_then(|value| value.strip_suffix('\''))
        })?;
    (!inner.contains(['"', '\'', '\\'])).then(|| inner.to_owned())
}

/// A one-line array of strings.
fn strings(value: &str) -> Option<Vec<String>> {
    let inner = value.trim().strip_prefix('[')?.strip_suffix(']')?;
    inner
        .split(',')
        .map(str::trim)
        .filter(|item| !item.is_empty())
        .map(string)
        .collect()
}

fn is_crate_name(name: &str) -> bool {
    !name.is_empty()
        && name
            .chars()
            .all(|character| character.is_ascii_alphanumeric() || matches!(character, '-' | '_'))
}

fn is_type_path(path: &str) -> bool {
    path.split("::").all(|segment| {
        segment
            .chars()
            .next()
            .is_some_and(|first| first.is_ascii_alphabetic() || first == '_')
            && segment
                .chars()
                .all(|character| character.is_ascii_alphanumeric() || character == '_')
    })
}
