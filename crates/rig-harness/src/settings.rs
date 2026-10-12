//! The user's settings for plugins: `RIG_HOME/settings.toml`, one table
//! per plugin. A plugin names its table and a type that reads it, and gets
//! that type as a resource, its `Default` where the user set nothing. A
//! table that does not read as the type leaves the default and says why
//! at startup, as does a file that is not TOML.
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

use std::io::ErrorKind;
use std::path::Path;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use serde::de::DeserializeOwned;

use crate::services::RigHome;

/// A settings file's tables, and what was wrong with it. Each problem
/// names the file.
#[derive(Resource, Clone, Debug)]
pub struct Settings {
    file: String,
    tables: toml::Table,
    problems: Vec<String>,
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
        }
    }

    /// The settings `text` of `file` holds.
    fn parse(file: String, text: &str) -> Self {
        match text.parse::<toml::Table>() {
            Ok(tables) => Self {
                tables,
                ..Self::empty(file)
            },
            Err(failure) => {
                let problem = format!("{file}: not TOML: {failure}");
                Self {
                    problems: vec![problem],
                    ..Self::empty(file)
                }
            }
        }
    }

    /// The `[section]` table as a `T`: `T::default()` without one, and the
    /// reason when it does not read as `T`.
    pub fn get<T: DeserializeOwned + Default>(&self, section: &str) -> Result<T, String> {
        let Some(table) = self.tables.get(section) else {
            return Ok(T::default());
        };
        table
            .clone()
            .try_into()
            .map_err(|failure| format!("{} [{section}]: {failure}", self.file))
    }

    /// What was wrong with the file or with a table a plugin read, to show
    /// the user.
    pub fn problems(&self) -> &[String] {
        &self.problems
    }
}

/// Reading settings while the app is built.
pub trait AppSettingsExt {
    /// Inserts `T` from the `[section]` table of the [`Settings`], read
    /// from `RIG_HOME/settings.toml` the first time (unless the app
    /// inserted some, as a test does). A table that does not read as `T`
    /// inserts `T::default()` and is one of [`Settings::problems`].
    fn add_setting<T: Resource + DeserializeOwned + Default>(&mut self, section: &str)
    -> &mut Self;
}

impl AppSettingsExt for App {
    fn add_setting<T: Resource + DeserializeOwned + Default>(
        &mut self,
        section: &str,
    ) -> &mut Self {
        let world = self.world_mut();
        if !world.contains_resource::<Settings>() {
            let settings = Settings::read(&RigHome::of(world).settings());
            world.insert_resource(settings);
        }
        let mut settings = world.resource_mut::<Settings>();
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
