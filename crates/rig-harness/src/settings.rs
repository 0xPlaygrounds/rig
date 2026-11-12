//! The user's settings for plugins: `RIG_HOME/settings.toml`, one table
//! per plugin, beside the `[plugins]` table that says which run
//! ([`config`](crate::harness_protocol::config)). A plugin names its table and a type that reads it, and gets
//! that type as a resource, its `Default` where the user set nothing. A
//! table that does not read as the type refuses the run, naming the file,
//! the table and the key, as does a file that is not TOML or cannot be
//! read ([`Settings::problems`]). A plugin reads its table in `build`,
//! whatever front takes the run, so a build describing itself checks it
//! as a start does: a table read once the run's front is chosen refuses
//! the run. The harness chooses it as its own plugins finish, so in
//! practice any table read in a `finish` refuses the run. A table no
//! plugin reads is said at
//! startup: an error when it is a likely misspelling of one a plugin reads
//! (`[trafic]` for `[traffic]`), else a note, such as for a disabled
//! plugin's table.
//!
//! A project has no settings: rig never reads its `.rig/settings.toml`.
//!
//! ```
//! use rig_harness::prelude::*;
//! use serde::Deserialize;
//!
//! /// `[confirm]` in `settings.toml`.
//! #[derive(Resource, Deserialize, Default)]
//! #[serde(default, deny_unknown_fields)]
//! struct Confirm {
//!     patterns: Vec<String>,
//! }
//!
//! let mut app = App::new();
//! // A test's settings; the agent reads the user's file.
//! app.insert_resource(Settings::from_toml("[confirm]\npatterns = [\"rm -rf\"]"));
//! app.add_setting::<Confirm>("confirm");
//! assert_eq!(app.world().resource::<Confirm>().patterns, ["rm -rf"]);
//! ```

use std::collections::BTreeSet;
use std::io::ErrorKind;
use std::path::{Path, PathBuf};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use rig_ecs::agent::Notice;
use rig_harness_protocol::config::SettingsFile;
use serde::de::DeserializeOwned;

use crate::describe::described_settings;
use crate::host::cli::Cli;
use crate::run::closest;
use crate::services::RigHome;

/// A settings file's tables, and what was wrong with it. Each problem
/// names the file.
#[derive(Resource, Clone, Debug)]
pub struct Settings {
    file: String,
    tables: toml::Table,
    problems: Vec<String>,
    /// The tables a plugin read.
    read: BTreeSet<String>,
    /// Why a plugin read a table too late, once the run's front was chosen.
    late: Vec<String>,
}

impl Default for Settings {
    /// No settings, as from an empty `settings.toml`.
    fn default() -> Self {
        Self::empty("settings.toml".to_owned())
    }
}

impl Settings {
    /// The settings `text` of `settings.toml` holds; a text that is not
    /// TOML holds none and says why.
    pub fn from_toml(text: &str) -> Self {
        Self::parse("settings.toml".to_owned(), text)
    }

    /// The settings in the file at `path`; none when it does not exist.
    /// Its problems start with `path`.
    pub fn read(path: &Path) -> Self {
        let file = path.display().to_string();
        match std::fs::read_to_string(path) {
            Ok(text) => Self::parse(file, &text),
            Err(failure) if failure.kind() == ErrorKind::NotFound => Self::empty(file),
            Err(failure) => {
                let problem = format!("{file}: could not read: {failure}");
                Self {
                    problems: vec![problem],
                    ..Self::empty(file)
                }
            }
        }
    }

    /// No settings, from `file`.
    fn empty(file: String) -> Self {
        Self {
            file,
            tables: toml::Table::new(),
            problems: Vec::new(),
            read: BTreeSet::new(),
            late: Vec::new(),
        }
    }

    /// The settings `text` of `file` holds; a text that is not TOML holds
    /// none and says why, at its line and column.
    fn parse(file: String, text: &str) -> Self {
        let read = SettingsFile::parse(PathBuf::from(&file), text)
            .map_err(|failure| failure.to_string())
            .and_then(|_| {
                (text.parse::<toml::Table>())
                    .map_err(|failure| format!("{file}: not TOML: {}", failure.message()))
            });
        match read {
            Ok(tables) => Self {
                tables,
                ..Self::empty(file)
            },
            Err(problem) => Self {
                problems: vec![problem],
                ..Self::empty(file)
            },
        }
    }

    /// The `[section]` table as a `T`: `T::default()` without one, and the
    /// reason when it does not read as `T`. A key `T` does not have, such
    /// as one a later rig removed, says to delete it, and from which table
    /// when the file has that table: a nested one is `[section.inner]`.
    pub fn get<T: DeserializeOwned + Default>(&self, section: &str) -> Result<T, String> {
        let Some(table) = self.tables.get(section) else {
            return Ok(T::default());
        };
        table
            .clone()
            .try_into()
            .map_err(|failure: toml::de::Error| {
                let text = failure.to_string();
                let unknown = (failure.message().strip_prefix("unknown field `"))
                    .and_then(|rest| rest.split_once('`'))
                    .map(|(key, _)| {
                        // toml names the table within `section` only in its
                        // text, its keys joined by dots and unquoted.
                        let within = (text.lines())
                            .find_map(|line| line.strip_prefix("in `")?.strip_suffix('`'))
                            .map_or_else(Vec::new, |within| within.split('.').collect());
                        if holds_table(table, &within) {
                            let table = [&[section][..], &within].concat().join(".");
                            format!("; delete `{key}` from [{table}]")
                        } else {
                            format!("; delete `{key}`")
                        }
                    });
                format!(
                    "{} [{section}]: {}{}",
                    self.file,
                    text.trim_end(),
                    unknown.unwrap_or_default()
                )
            })
    }

    /// What was wrong with the file or with a table a plugin read, to show
    /// the user; they refuse the run.
    pub fn problems(&self) -> &[String] {
        &self.problems
    }

    /// The tables read too late and the [`problems`](Self::problems), on
    /// one line, as the run is refused over them; none without either.
    pub(crate) fn refusal(&self) -> Option<String> {
        let problems: Vec<String> = (self.problems.iter())
            .map(|problem| problem.trim_end().replace('\n', " "))
            .collect();
        let problems = (!problems.is_empty())
            .then(|| format!("{}; fix the file and start again", problems.join("; ")));
        let refusals: Vec<String> = self.late.iter().cloned().chain(problems).collect();
        (!refusals.is_empty()).then(|| refusals.join("; "))
    }

    /// Inserts the settings of `world`'s home, unless it has some: those of
    /// the file the launcher names while the build describes itself
    /// ([`described_settings`]), else `RIG_HOME/settings.toml`.
    pub(crate) fn init(world: &mut World) {
        if !world.contains_resource::<Self>() {
            let path = described_settings().unwrap_or_else(|| RigHome::of(world).settings());
            world.insert_resource(Self::read(&path));
        }
    }

    /// A notice for each table of the file no plugin read: an error naming
    /// the read table it likely misspells, within a quarter of its length
    /// (at least one edit), else an info notice.
    pub fn unread(&self) -> Vec<Notice> {
        let read: Vec<&str> = self.read.iter().map(String::as_str).collect();
        // `[plugins]` is the launcher's and the agent's own.
        let unread =
            (self.tables.keys()).filter(|table| *table != "plugins" && !self.read.contains(*table));
        unread
            .map(|table| match closest(table, &read) {
                Some(meant) => Notice::error(
                    None,
                    format!(
                        "{}: no plugin reads [{table}]; did you mean [{meant}]?",
                        self.file
                    ),
                ),
                None => Notice::info(
                    None,
                    format!("{}: no enabled plugin reads [{table}].", self.file),
                ),
            })
            .collect()
    }
}

/// Whether `path` leads from `value` to a table, through any entry of an
/// array of tables.
fn holds_table(value: &toml::Value, path: &[&str]) -> bool {
    match (path.split_first(), value) {
        (None, toml::Value::Table(_)) => true,
        (Some((key, rest)), toml::Value::Table(table)) => table
            .get(*key)
            .is_some_and(|value| holds_table(value, rest)),
        (_, toml::Value::Array(entries)) => entries.iter().any(|entry| holds_table(entry, path)),
        _ => false,
    }
}

/// Reading settings while the app is built.
pub trait AppSettingsExt {
    /// Inserts `T` from the `[section]` table of the [`Settings`], read
    /// from `RIG_HOME/settings.toml` the first time (unless the app
    /// inserted some, as a test does). A table that does not read as `T`
    /// inserts `T::default()` and is one of [`Settings::problems`], which
    /// refuse the run. Call it in the plugin's `build`: a table read once
    /// the run's front is chosen, which in practice is any read in a
    /// `finish`, refuses the run.
    fn add_setting<T: Resource + DeserializeOwned + Default>(&mut self, section: &str)
    -> &mut Self;
}

impl AppSettingsExt for App {
    fn add_setting<T: Resource + DeserializeOwned + Default>(
        &mut self,
        section: &str,
    ) -> &mut Self {
        let world = self.world_mut();
        Settings::init(world);
        let late = world.get_resource::<Cli>().is_some_and(Cli::decided);
        let mut settings = world.resource_mut::<Settings>();
        if late {
            settings.late.push(format!(
                "a plugin read [{section}] once the run's front was chosen; it must read it in its `build`, so a build describing itself checks it"
            ));
        }
        settings.read.insert(section.to_owned());
        let value = settings.get::<T>(section).unwrap_or_else(|problem| {
            warn!("{problem}");
            settings.problems.push(problem);
            T::default()
        });
        self.insert_resource(value)
    }
}

#[cfg(test)]
mod tests;
