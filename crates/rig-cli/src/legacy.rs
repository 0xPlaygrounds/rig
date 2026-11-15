//! The files rig 0.46 kept in `RIG_HOME` that rig no longer reads:
//! `plugins.toml`, the plugin list, and `defaults.json`, the model a new
//! session starts with. While `settings.toml` has no table for what one
//! held, the start and `rig plugin` warn of it with the commands and lines
//! that move it. The files are never changed.

use std::path::Path;

use rig_harness_protocol::config::{SettingsFile, built_in, toml_edit};
use rig_harness_protocol::{Home, fs};

#[cfg(test)]
mod tests;

/// A warning for each 0.46 file in `home` whose settings `settings.toml`
/// has no table for: `plugins.toml` without `[plugins]`, `defaults.json`
/// without `[model]`. A `settings.toml` rig cannot read gives none: the
/// command refuses it with its own error.
pub fn warnings(home: &Home) -> Vec<String> {
    let settings = home.settings();
    let Ok(file) = SettingsFile::read(&settings) else {
        return Vec::new();
    };
    let lacks = |table: &str| !file.document.contains_key(table);
    let mut warnings = Vec::new();
    let plugins = home.root().join("plugins.toml");
    if lacks("plugins")
        && let Some(text) = read(&plugins)
    {
        let mut moves = plugin_moves(&text, &settings);
        if moves.is_empty() {
            moves.push(format!(
                "  Add an empty [plugins] table to {}: the list was rig's defaults.",
                settings.display()
            ));
        }
        warnings.push(warning(
            &plugins,
            "plugin list",
            "plugins",
            &settings,
            &moves,
        ));
    }
    let defaults = home.root().join("defaults.json");
    if lacks("model")
        && let Some(text) = read(&defaults)
    {
        let moves = match model(&text) {
            Some(model) => vec![
                format!("  Add to {}:", settings.display()),
                "  [model]".to_owned(),
                format!("  default = {}", toml_edit::Value::from(model)),
            ],
            None => vec![format!(
                "  Add an empty [model] table to {}: the file names no model.",
                settings.display()
            )],
        };
        warnings.push(warning(&defaults, "model", "model", &settings, &moves));
    }
    warnings
}

/// The text of the file at `path`; `None` when there is none, or it does
/// not read.
fn read(path: &Path) -> Option<String> {
    fs::read_to_string(path).ok()
}

/// The warning about `old`, whose `what` moves to `[table]` of `settings`
/// with `moves`, one line each.
fn warning(old: &Path, what: &str, table: &str, settings: &Path, moves: &[String]) -> String {
    format!(
        "rig no longer reads {}. Move its {what} to [{table}] in {} (see MIGRATING.md):\n{}\n\
         The file was not changed.",
        old.display(),
        settings.display(),
        moves.join("\n"),
    )
}

/// One `[[plugin]]` table of `plugins.toml`, on, or commented out (off).
struct Entry {
    on: bool,
    /// Its keys and their values, in file order.
    keys: Vec<(String, String)>,
}

impl Entry {
    fn get(&self, key: &str) -> Option<&str> {
        let found = self.keys.iter().find(|(name, _)| name == key);
        found.map(|(_, value)| value.as_str())
    }
}

/// The lines that rebuild the plugin list `text` in `settings`: a
/// `rig plugin enable` or `disable` for each of rig's plugins it turns on or
/// off against today's default, a `[[plugins.extra]]` table for each crate
/// of another source, with a `disable` for one it lists off, and a line for
/// each of rig's plugins that is gone. A plugin of rig's it does not name
/// keeps its default.
fn plugin_moves(text: &str, settings: &Path) -> Vec<String> {
    let entries = entries(text);
    let mut commands = Vec::new();
    let mut gone = Vec::new();
    let mut tables = Vec::new();
    for entry in &entries {
        let Some(plugin) = entry.get("plugin") else {
            continue;
        };
        let own = ["path", "git", "version"]
            .iter()
            .all(|key| entry.get(key).is_none());
        match built_in(plugin) {
            Some(rigs) if rigs.on != entry.on => {
                let verb = if entry.on { "enable" } else { "disable" };
                commands.push(format!("  rig plugin {verb} {plugin}"));
            }
            Some(_) => {}
            None if own => {
                if entry.on {
                    gone.push(format!("  {plugin} is no longer one of rig's plugins."));
                }
            }
            None => {
                tables.push("  [[plugins.extra]]".to_owned());
                for (key, value) in &entry.keys {
                    let value = toml_edit::Value::from(value.as_str());
                    tables.push(format!("  {key} = {value}"));
                }
                if !entry.on {
                    commands.push(format!("  rig plugin disable {plugin}"));
                }
            }
        }
    }
    let mut moves = Vec::new();
    if !tables.is_empty() {
        moves.push(format!(
            "  Add to {}, before the commands below:",
            settings.display()
        ));
        moves.extend(tables);
    }
    moves.extend(commands);
    moves.extend(gone);
    moves
}

/// The `[[plugin]]` tables of `text`, in file order. One that is off has
/// its `[[plugin]]` line and the key lines right after it each behind `#`.
fn entries(text: &str) -> Vec<Entry> {
    let mut entries: Vec<Entry> = Vec::new();
    let mut open = false;
    for line in text.lines() {
        let line = line.trim();
        let (on, bare) = match line.strip_prefix('#') {
            Some(commented) => (false, commented.trim()),
            None => (true, line),
        };
        if bare == "[[plugin]]" {
            entries.push(Entry {
                on,
                keys: Vec::new(),
            });
            open = true;
            continue;
        }
        let key = bare.split_once('=').and_then(|(key, value)| {
            let value = value.trim().parse::<toml_edit::Value>().ok()?;
            Some((key.trim().to_owned(), value.as_str()?.to_owned()))
        });
        match (open, entries.last_mut(), key) {
            (true, Some(entry), Some(key)) if entry.on == on && KEYS.contains(&key.0.as_str()) => {
                entry.keys.push(key);
            }
            _ => open = false,
        }
    }
    entries
}

/// The keys a `[[plugin]]` table of `plugins.toml` held.
const KEYS: [&str; 7] = ["plugin", "crate", "path", "git", "branch", "rev", "version"];

/// The model `defaults.json`'s text names under `"model"`; `None` for
/// `null` or a file that names none.
fn model(text: &str) -> Option<String> {
    let after = text.split_once("\"model\"")?.1.trim_start();
    let value = after.strip_prefix(':')?.trim_start();
    let body = value.strip_prefix('"')?;
    // The closing quote is the first one no backslash escapes.
    let mut escaped = false;
    let end = body.char_indices().find_map(|(at, c)| {
        let closes = c == '"' && !escaped;
        escaped = c == '\\' && !escaped;
        closes.then_some(at)
    })?;
    let literal = format!("\"{}\"", body.get(..end)?);
    let parsed = literal.parse::<toml_edit::Value>().ok()?;
    parsed.as_str().map(str::to_owned)
}
