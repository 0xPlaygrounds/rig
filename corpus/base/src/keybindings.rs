//! Plugin corpus: 53. Custom key bindings (spec `53-keybindings.md`).
//!
//! `RIG_HOME/keys.toml` binds keys of the terminal view's input:
//!
//! ```toml
//! "alt+m" = { cycle_model = ["deepseek/deepseek-flash", "deepseek/deepseek-v4-pro"] }
//! "ctrl+g" = { command = "/compact" }
//! "ctrl+t" = { insert = "Think step by step." }
//! "ctrl+o" = { send = "Go on.", label = "Continue" }
//! "ctrl+e" = { editor = true, override = true }
//! ```
//!
//! Each is a `rig_tui` [`KeyBinding`] in the input's scope. One whose keys
//! another binding takes, such as the view's own `ctrl+e` (the end of the
//! line), replaces it only with `override = true`; otherwise it is refused
//! at startup with a notice naming the binding in the way. Other plugins
//! bind keys with `rig_tui`'s `add_key_binding`, and `/keys` lists every
//! binding with the plugin it came from. The file is read when the app is
//! built, so `/reload` takes a change.

use std::path::PathBuf;
use std::process::Command;

use rig_harness::prelude::*;
use rig_tui::{
    AppKeyBindingsExt, Editor, KeyBinding, KeyPressed, KeyScope, Keys, MainEditor, MainInput,
    RunInTerminal, TerminalExit, TuiPlugin, run_command,
};
use serde::Deserialize;

/// The user's bindings, in `RIG_HOME`.
pub const USER_FILE: &str = "keys.toml";

/// Binds the keys of `RIG_HOME/keys.toml` and adds `/keys`.
#[derive(Default)]
pub struct KeyBindingsPlugin;

/// On a binding from [`USER_FILE`].
#[derive(Component, Debug)]
pub struct UserKey;

/// A program writing the input's draft in this file.
#[derive(Component)]
struct Draft(PathBuf);

impl Plugin for KeyBindingsPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TuiPlugin>() {
            app.add_plugins(TuiPlugin);
        }
        app.init_resource::<RigHome>()
            .add_command(
                "keys",
                "List the key bindings and where they came from",
                list,
            )
            .add_systems(Update, take_draft);
    }

    /// Binds the user's keys once every plugin's `build` spawned its own,
    /// so a conflict with any of them is found.
    fn finish(&self, app: &mut App) {
        let path = RigHome::of(app.world()).root().join(USER_FILE);
        let mut refused: Vec<String> = Vec::new();
        for (key, entry) in read(&path, &mut refused) {
            if let Err(why) = entry.bind(app, &key) {
                refused.push(format!("{USER_FILE}: `{key}` refused: {why}."));
            }
        }
        if !refused.is_empty() {
            app.add_systems(Startup, move |mut notices: MessageWriter<Notice>| {
                notices.write_batch(refused.drain(..).map(|why| Notice::error(None, why)));
            });
        }
    }
}

/// One entry of `keys.toml`: one action, an optional label and `override`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Entry {
    #[serde(default)]
    editor: bool,
    command: Option<String>,
    insert: Option<String>,
    cycle_model: Option<Vec<String>>,
    send: Option<String>,
    label: Option<String>,
    #[serde(default, rename = "override")]
    overrides: bool,
}

/// The entries of the file at `path`, with what is wrong with it in
/// `refused`; none when there is no file.
fn read(path: &std::path::Path, refused: &mut Vec<String>) -> Vec<(String, Entry)> {
    let table = match std::fs::read_to_string(path) {
        Ok(text) => text.parse::<toml::Table>().map_err(|why| why.to_string()),
        Err(why) if why.kind() == std::io::ErrorKind::NotFound => Ok(toml::Table::new()),
        Err(why) => Err(why.to_string()),
    };
    let table = table.unwrap_or_else(|why| {
        refused.push(format!("{USER_FILE}: {why}"));
        toml::Table::new()
    });
    table
        .into_iter()
        .filter_map(|(key, value)| match value.try_into::<Entry>() {
            Ok(entry) => Some((key, entry)),
            Err(why) => {
                refused.push(format!("{USER_FILE}: `{key}`: {why}"));
                None
            }
        })
        .collect()
}

/// What a key of `keys.toml` does.
enum Action {
    Editor,
    Command(String),
    Insert(String),
    CycleModel(Vec<String>),
    Send(String),
}

impl Entry {
    /// The entry's one action, and its label.
    fn action(&mut self) -> Result<(Action, String), String> {
        let actions = [
            self.editor.then_some(Action::Editor),
            self.command.take().map(Action::Command),
            self.insert.take().map(Action::Insert),
            self.cycle_model.take().map(Action::CycleModel),
            self.send.take().map(Action::Send),
        ];
        let mut given = actions.into_iter().flatten();
        let (Some(action), None) = (given.next(), given.next()) else {
            return Err("it needs exactly one action".to_owned());
        };
        let label = match &action {
            Action::Editor => "Write the input in $EDITOR".to_owned(),
            Action::Command(line) => format!("Run {line}"),
            Action::Insert(text) => format!("Insert {text:?}"),
            Action::CycleModel(models) if models.is_empty() => {
                return Err("it cycles through no models".to_owned());
            }
            Action::CycleModel(models) => format!("Cycle {}", models.join(", ")),
            Action::Send(text) => format!("Send {text:?}"),
        };
        Ok((action, self.label.take().unwrap_or(label)))
    }

    /// Binds `key` in the input to this entry's action, replacing the
    /// bindings in its way if it overrides them.
    fn bind(mut self, app: &mut App, key: &str) -> Result<(), String> {
        let keys: Keys = key.parse()?;
        let (action, label) = self.action()?;
        let binding = KeyBinding::new(keys, label).in_scope(KeyScope::Input);

        let world = app.world_mut();
        let mut bound = world.query::<(Entity, &KeyBinding)>();
        let (taken, names): (Vec<Entity>, Vec<String>) = bound
            .iter(world)
            .filter(|(_, other)| other.conflicts(&binding))
            .map(|(entity, other)| (entity, format!("`{}` ({})", other.keys, other.label)))
            .unzip();
        if !taken.is_empty() && !self.overrides {
            return Err(format!(
                "{} is bound; add `override = true` to replace it",
                names.join(", ")
            ));
        }
        for entity in taken {
            world.entity_mut(entity).insert(Disabled);
        }

        let bound = match action {
            Action::Editor => app.add_key_binding(binding, edit),
            Action::Command(line) => {
                let line = line.trim_start_matches('/').to_owned();
                app.add_key_binding(binding, run_command(line))
            }
            Action::Insert(text) => app.add_key_binding(binding, insert(text)),
            Action::CycleModel(models) => app.add_key_binding(binding, cycle_model(models)),
            Action::Send(text) => app.add_key_binding(binding, send(text)),
        };
        if let Some(entity) = bound {
            app.world_mut().entity_mut(entity).insert(UserKey);
        }
        Ok(())
    }
}

/// Inserts `text` at the input's cursor.
fn insert(text: String) -> impl FnMut(On<KeyPressed>, MainInput) {
    move |_: On<KeyPressed>, mut input: MainInput| input.insert(&text)
}

/// Sends `text` to the agent shown, as the user.
fn send(text: String) -> impl FnMut(On<KeyPressed>, Commands) {
    move |pressed: On<KeyPressed>, mut commands: Commands| {
        if let Some(agent) = pressed.agent {
            commands.trigger(Deliver::new(agent, text.clone(), DeliveryMode::Steer));
        }
    }
}

/// Steps the agent shown to the model after its own in `models`, or to the
/// first; `rig_basics::models` shows it in the status line.
fn cycle_model(
    models: Vec<String>,
) -> impl FnMut(On<KeyPressed>, Query<&ModelChoice>, Res<Models>, Commands) {
    move |pressed: On<KeyPressed>,
          choices: Query<&ModelChoice>,
          catalog: Res<Models>,
          mut commands: Commands| {
        let Some(agent) = pressed.agent else {
            return;
        };
        let current = choices.get(agent).ok().map(|choice| choice.0.as_str());
        let reference = |model: &String| {
            let found = catalog.0.catalog().resolve(model).ok();
            found.map(|found| found.spec.reference())
        };
        let at = models
            .iter()
            .position(|model| reference(model).as_deref() == current);
        let next = at.map_or(0, |at| (at + 1) % models.len());
        if let Some(model) = models.get(next) {
            commands.trigger(SetModel {
                entity: agent,
                model: model.clone(),
            });
        }
    }
}

/// Writes the input's draft to a file and opens `$VISUAL` or `$EDITOR`
/// (else `vi`) on it, with the terminal.
fn edit(
    pressed: On<KeyPressed>,
    input: Single<&Editor, With<MainEditor>>,
    (paths, home): (Option<Res<SessionPaths>>, Res<RigHome>),
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let dir = paths.map_or_else(
        || home.0.root().to_path_buf(),
        |paths| paths.path().to_path_buf(),
    );
    let path = dir.join("draft.md");
    if let Err(why) = std::fs::write(&path, input.text()) {
        notices.write(Notice::error(
            pressed.agent,
            format!("Could not write the draft: {why}."),
        ));
        return;
    }
    let editor = std::env::var("VISUAL")
        .or_else(|_| std::env::var("EDITOR"))
        .unwrap_or_else(|_| "vi".to_owned());
    let mut words = editor.split_whitespace();
    let mut command = Command::new(words.next().unwrap_or("vi"));
    command.args(words).arg(&path);
    commands.spawn((RunInTerminal(command), Draft(path)));
}

/// Puts the draft as the editor left it in the input, unless the editor
/// failed.
fn take_draft(
    done: Query<(Entity, &TerminalExit, &Draft), Added<TerminalExit>>,
    mut input: MainInput,
    mut commands: Commands,
) {
    for (run, exit, Draft(path)) in &done {
        if let (true, Ok(text)) = (exit.success(), std::fs::read_to_string(path)) {
            input.set(text.trim_end());
        }
        std::fs::remove_file(path).ok();
        commands.entity(run).despawn();
    }
}

/// `/keys`: every key binding with the plugin it came from.
fn list(
    In(args): In<CommandArgs>,
    bindings: Query<(&KeyBinding, Option<&ProvidedBy>, Has<UserKey>)>,
    names: Query<&Name>,
    mut notices: MessageWriter<Notice>,
) {
    let mut lines: Vec<String> = bindings
        .iter()
        .map(|(binding, by, user)| {
            let plugin = by.and_then(|by| names.get(by.0).ok()).map(Name::as_str);
            let source = if user { Some(USER_FILE) } else { plugin };
            format!(
                "{:<14} {}  ({})",
                binding.keys.to_string(),
                binding.label,
                source.unwrap_or("unknown")
            )
        })
        .collect();
    lines.sort();
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

#[cfg(test)]
mod tests;
