//! The user's settings for plugins: `RIG_HOME/settings.toml`, one table
//! per plugin. A plugin names its table and a type that reads it, and gets
//! that type as a resource, its `Default` where the user set nothing. A
//! table that does not read as the type leaves the default and says why
//! at startup, as does a file that is not TOML.
//!
//! A plugin may let a project override its table
//! ([`add_project_setting`](AppSettingsExt::add_project_setting)): the same
//! table of the project's `.rig/settings.toml`, merged key by key over the
//! user's when the app starts. Only tables declared that way are read from
//! the project, so a cloned repository cannot change any other setting.
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
use std::marker::PhantomData;
use std::path::{Path, PathBuf};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use serde::de::DeserializeOwned;

use crate::services::{ProjectRoot, RigHome};
use crate::workspace::ProjectWorkspace;

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

    /// As [`add_setting`](Self::add_setting), then, when the app starts
    /// ([`PreStartup`]), the same table of the project's
    /// `.rig/settings.toml`, read through the [`ProjectWorkspace`] under the
    /// [`ProjectRoot`], merged key by key over the user's: so read `T` in
    /// systems, not while building. A project file that is not TOML, or a
    /// merged table that does not read as `T`, keeps the user's value and
    /// is one of [`Settings::problems`].
    ///
    /// ```
    /// use rig_harness::prelude::*;
    /// use serde::Deserialize;
    ///
    /// /// `[large-change]`, which a project may tighten.
    /// #[derive(Resource, Deserialize, Default)]
    /// #[serde(default, deny_unknown_fields)]
    /// struct Limits {
    ///     lines: usize,
    /// }
    ///
    /// let mut app = App::new();
    /// app.insert_resource(Settings::from_toml("[large-change]\nlines = 400"));
    /// app.add_project_setting::<Limits>("large-change");
    /// assert_eq!(app.world().resource::<Limits>().lines, 400);
    /// ```
    fn add_project_setting<T: Resource + DeserializeOwned + Default>(
        &mut self,
        section: &str,
    ) -> &mut Self;
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

    fn add_project_setting<T: Resource + DeserializeOwned + Default>(
        &mut self,
        section: &str,
    ) -> &mut Self {
        self.add_setting::<T>(section);
        if !self.is_plugin_added::<ProjectSettingsPlugin>() {
            self.add_plugins(ProjectSettingsPlugin);
        }
        let world = self.world_mut();
        let read = world
            .query_filtered::<(), With<ReadAs<T>>>()
            .iter(world)
            .next()
            .is_some();
        world.spawn((
            ProjectSetting(section.to_owned()),
            ReadAs::<T>(PhantomData),
            Name::new(format!("project setting [{section}]")),
        ));
        if !read {
            self.add_systems(
                PreStartup,
                apply_project_setting::<T>.in_set(ProjectSettingsSystems::Apply),
            );
        }
        self
    }
}

/// A table of `settings.toml` that a project's `.rig/settings.toml` may
/// override, on an entity of its own for each
/// [`add_project_setting`](AppSettingsExt::add_project_setting): its
/// section's name.
#[derive(Component, Clone, Debug, PartialEq, Eq)]
pub struct ProjectSetting(pub String);

/// Beside a [`ProjectSetting`]: the type its merged table is read into.
#[derive(Component)]
struct ReadAs<T>(PhantomData<fn() -> T>);

/// The project's `.rig/settings.toml`, as read when the app starts, once a
/// plugin declared a [`ProjectSetting`]; absent without that file.
#[derive(Resource, Clone, Debug)]
pub struct ProjectSettings(pub Settings);

/// The phases, in [`PreStartup`], in which the project's settings are read
/// and the tables declared for it are inserted.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ProjectSettingsSystems {
    /// The project's file becomes [`ProjectSettings`].
    Read,
    /// Each [`ProjectSetting`] the project's file has is merged over the
    /// user's table and inserted as its type.
    Apply,
}

/// Reads the project's settings once a table is declared for it.
struct ProjectSettingsPlugin;

impl Plugin for ProjectSettingsPlugin {
    fn build(&self, app: &mut App) {
        app.configure_sets(
            PreStartup,
            (ProjectSettingsSystems::Read, ProjectSettingsSystems::Apply).chain(),
        )
        .add_systems(
            PreStartup,
            read_project_settings.in_set(ProjectSettingsSystems::Read),
        );
    }
}

/// Reads the project's `.rig/settings.toml` into [`ProjectSettings`],
/// saying what was wrong with it.
fn read_project_settings(
    root: Option<Res<ProjectRoot>>,
    workspace: Option<Res<ProjectWorkspace>>,
    settings: Option<ResMut<Settings>>,
    mut commands: Commands,
) {
    let root = root.map_or_else(|| ProjectRoot::default().0, |root| root.0.clone());
    let workspace = workspace.map(|it| it.clone()).unwrap_or_default();
    let path: PathBuf = root.join(".rig").join("settings.toml");
    let file = path.display().to_string();
    let project = match workspace.0.read(&path, rig_tools::MAX_FILE_BYTES) {
        Ok(contents) => Settings::parse(file, &String::from_utf8_lossy(&contents.bytes)),
        Err(failure) if failure.kind() == ErrorKind::NotFound => return,
        Err(failure) => Settings {
            problems: vec![format!("{file}: could not read: {failure}")],
            ..Settings::empty(file)
        },
    };
    for problem in &project.problems {
        warn!("{problem}");
    }
    if let Some(mut settings) = settings {
        settings.problems.extend(project.problems.iter().cloned());
    }
    commands.insert_resource(ProjectSettings(project));
}

/// Inserts `T` from each table declared as read into it that the project
/// overrides, merged over the user's; a table that does not read as `T`
/// keeps the user's value and is one of [`Settings::problems`].
fn apply_project_setting<T: Resource + DeserializeOwned>(
    declared: Query<&ProjectSetting, With<ReadAs<T>>>,
    project: Option<Res<ProjectSettings>>,
    mut settings: Option<ResMut<Settings>>,
    mut commands: Commands,
) {
    let Some(ProjectSettings(project)) = project.as_deref() else {
        return;
    };
    for ProjectSetting(section) in &declared {
        let Some(over) = project.tables.get(section) else {
            continue;
        };
        let user = settings
            .as_ref()
            .and_then(|settings| settings.tables.get(section).cloned());
        let merged = match user {
            Some(user) => merge(user, over.clone()),
            None => over.clone(),
        };
        match merged.try_into::<T>() {
            Ok(value) => commands.insert_resource(value),
            Err(failure) => {
                let problem = format!("{} [{section}]: {failure}", project.file);
                warn!("{problem}");
                if let Some(settings) = settings.as_mut() {
                    settings.problems.push(problem);
                }
            }
        }
    }
}

/// `over` merged key by key onto `base`: a table's keys one by one, a
/// table inside it the same way, anything else replaced.
fn merge(base: toml::Value, over: toml::Value) -> toml::Value {
    match (base, over) {
        (toml::Value::Table(mut base), toml::Value::Table(over)) => {
            for (key, value) in over {
                let merged = match base.remove(&key) {
                    Some(was) => merge(was, value),
                    None => value,
                };
                base.insert(key, merged);
            }
            toml::Value::Table(base)
        }
        (_, over) => over,
    }
}

#[cfg(test)]
mod tests;
