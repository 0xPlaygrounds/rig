//! The files rig 0.46 kept in `RIG_HOME` that rig no longer reads:
//! `plugins.toml`, the plugin list, and `defaults.json`, the model a new
//! session starts with. While `settings.toml` has no table for what one
//! held, the start and `rig plugin` warn of it with the commands and lines
//! that move it. The files are never changed.

use std::path::Path;

use rig_harness_protocol::config::{SettingsFile, built_in, renamed, toml_edit};
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
        && let Ok(text) = fs::read_to_string(&plugins)
    {
        let moves = plugin_moves(&text, &settings);
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
        && let Ok(text) = fs::read_to_string(&defaults)
    {
        let moves = match model(&text) {
            Some(model) => {
                let mut lines = vec![
                    format!("  Add to {}:", settings.display()),
                    "  [model]".to_owned(),
                    format!("  default = {}", toml_edit::Value::from(model)),
                ];
                lines.extend(
                    effort(&text)
                        .map(|effort| format!("  effort = {}", toml_edit::Value::from(effort))),
                );
                lines
            }
            None => vec![format!(
                "  Add an empty [model] table to {}: the file names no model.",
                settings.display()
            )],
        };
        warnings.push(warning(&defaults, "model", "model", &settings, &moves));
    }
    warnings
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

/// The lines that rebuild the plugin list `text` in `settings`. A
/// `rig plugin add` for each crate of another source comes first; a
/// relative path is made absolute against the folder of `settings`, which
/// held the list too. Then a `rig plugin enable` or `disable` for each of
/// rig's plugins, by the name it has now, it turns on or off against
/// today's default (with the approval plugin beside MCP on, as v0.46.0's
/// MCP plugin asked before a project's server ran), a `disable` for a
/// crate of its own it lists off, and a line for each of rig's plugins that
/// is gone. A plugin of rig's it
/// does not name keeps its default. When nothing is to be added, the step
/// that ends the warning: an empty table.
fn plugin_moves(text: &str, settings: &Path) -> Vec<String> {
    let empty = format!(
        "  Add an empty [plugins] table to {}: the list was rig's defaults.",
        settings.display()
    );
    let entries = match entries(text) {
        Ok(entries) => entries,
        Err(why) => {
            return vec![format!(
                "  It is not TOML ({}), so its plugins are not listed here: move them by hand, \
                 then add [plugins] to {}.",
                why.message(),
                settings.display()
            )];
        }
    };
    let mut commands = Vec::new();
    let mut gone = Vec::new();
    let mut adds = Vec::new();
    for entry in &entries {
        let Some(plugin) = entry.get("plugin") else {
            continue;
        };
        let own = ["path", "git", "version"]
            .iter()
            .all(|key| entry.get(key).is_none());
        let plugin = renamed(plugin).filter(|_| own).unwrap_or(plugin);
        match built_in(plugin) {
            Some(rigs) if rigs.on != entry.on => {
                let verb = if entry.on { "enable" } else { "disable" };
                commands.push(format!("  rig plugin {verb} {plugin}"));
                if entry.on
                    && plugin == MCP
                    && built_in(MCP_APPROVAL).is_some_and(|approval| !approval.on)
                {
                    commands.push(format!("  rig plugin enable {MCP_APPROVAL}"));
                }
            }
            Some(_) => {}
            None if own => {
                if entry.on {
                    gone.push(format!("  {plugin} is no longer one of rig's plugins."));
                }
            }
            None => {
                let root = plugin.split("::").next().unwrap_or_default();
                let mut add = format!("  rig plugin add {}", shell_word(plugin));
                for (key, value) in &entry.keys {
                    let value = match key.as_str() {
                        "plugin" => continue,
                        "crate" if value == root => continue,
                        "path" => (settings.parent().unwrap_or(Path::new(".")))
                            .join(value)
                            .display()
                            .to_string(),
                        _ => value.clone(),
                    };
                    add.push_str(&format!(" --{key} {}", shell_word(&value)));
                }
                adds.push(add);
                if !entry.on {
                    commands.push(format!("  rig plugin disable {plugin}"));
                }
            }
        }
    }
    let mut moves = adds;
    if moves.is_empty() && commands.is_empty() {
        moves.push(empty);
    }
    moves.extend(commands);
    moves.extend(gone);
    moves
}

/// `word` as one shell word: as it is when it holds only characters no
/// shell reads specially, else in single quotes.
fn shell_word(word: &str) -> String {
    let plain = !word.is_empty()
        && (word.chars()).all(|c| c.is_ascii_alphanumeric() || "-_./:@+=,%".contains(c));
    if plain {
        word.to_owned()
    } else {
        format!("'{}'", word.replace('\'', "'\\''"))
    }
}

/// rig's MCP plugin, and the plugin that asks before a project's server
/// runs, which v0.46.0's MCP plugin did itself.
const MCP: &str = "rig_mcp_client::McpPlugin";
const MCP_APPROVAL: &str = "rig_mcp_client::approval::McpApprovalPlugin";

/// The `[[plugin]]` tables of `text`, as v0.46.0 read them: each table of
/// the TOML, on, then each one commented out, off, whose `[[plugin]]` line
/// and the key lines right after it are each behind `#`.
///
/// # Errors
///
/// When `text` is not TOML.
fn entries(text: &str) -> Result<Vec<Entry>, toml_edit::TomlError> {
    let live = text.parse::<toml_edit::DocumentMut>()?;
    let tables = live
        .get("plugin")
        .and_then(toml_edit::Item::as_array_of_tables);
    let mut entries: Vec<Entry> = (tables.into_iter().flat_map(|tables| tables.iter()))
        .map(|table| Entry {
            on: true,
            keys: keys(table),
        })
        .collect();
    let commented = |line: &str| line.trim_start().strip_prefix('#').map(str::to_owned);
    let lines: Vec<&str> = text.lines().collect();
    for (at, line) in lines.iter().enumerate() {
        if commented(line).is_none_or(|rest| rest.trim() != "[[plugin]]") {
            continue;
        }
        let mut table = String::new();
        for line in lines.iter().skip(at + 1) {
            let Some(key) = commented(line).filter(|rest| {
                let parsed = rest.parse::<toml_edit::DocumentMut>().ok();
                parsed.is_some_and(|line| {
                    let keys = keys(line.as_table());
                    keys.len() == 1 && line.as_table().len() == 1
                })
            }) else {
                break;
            };
            table.push_str(&key);
            table.push('\n');
        }
        if let Ok(table) = table.parse::<toml_edit::DocumentMut>() {
            entries.push(Entry {
                on: false,
                keys: keys(table.as_table()),
            });
        }
    }
    Ok(entries)
}

/// The keys of a `[[plugin]]` table v0.46.0 took, with their text.
fn keys(table: &toml_edit::Table) -> Vec<(String, String)> {
    (table.iter())
        .filter(|(key, _)| KEYS.contains(key))
        .filter_map(|(key, item)| Some((key.to_owned(), item.as_str()?.to_owned())))
        .collect()
}

/// The keys a `[[plugin]]` table of `plugins.toml` held.
const KEYS: [&str; 7] = ["plugin", "crate", "path", "git", "branch", "rev", "version"];

/// The model `defaults.json`'s text names under `"model"`; `None` for
/// `null` or a file that names none.
fn model(text: &str) -> Option<String> {
    json_string(after_key(text, "model")?)
}

/// The reasoning setting `defaults.json`'s text holds under `"effort"`, as
/// `[model] effort` writes it: `off`, a level such as `high`, or `4096
/// tokens`; `None` for `null` (the model's default) or a file that has
/// none. v0.46.0 wrote `"off"`, `{"effort": "<level>"}` or `{"budget":
/// {"tokens": <n>}}`.
fn effort(text: &str) -> Option<String> {
    let value = after_key(text, "effort")?;
    if value.starts_with('"') {
        return json_string(value);
    }
    let inner = value.strip_prefix('{')?.trim_start();
    if let Some(level) = after_key(inner, "effort") {
        return json_string(level);
    }
    let budget = after_key(inner, "budget")?.strip_prefix('{')?;
    let tokens = after_key(budget, "tokens")?;
    let digits: String = tokens.chars().take_while(char::is_ascii_digit).collect();
    (!digits.is_empty()).then(|| format!("{digits} tokens"))
}

/// The JSON text after the first `"key":` of `text`, from its value on.
fn after_key<'t>(text: &'t str, key: &str) -> Option<&'t str> {
    let after = text.split_once(&format!("\"{key}\""))?.1.trim_start();
    Some(after.strip_prefix(':')?.trim_start())
}

/// The JSON string `value` starts with; `None` when it starts with none.
fn json_string(value: &str) -> Option<String> {
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
