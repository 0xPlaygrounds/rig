//! SN-18: vim-style modal editing in the terminal view's input box.
//!
//! The input starts in insert mode (`[vim] start = "normal"` in
//! `settings.toml` changes that), where keys type as without the plugin.
//! Esc goes to normal mode, where the printable keys and Ctrl-r are this
//! plugin's [`KeyBinding`]s in the input's scope, run through
//! [`editor::Editor`]: motions, counts, operators, registers, undo and a
//! visual selection, written back into [`TuiInput`]. Because they are
//! input-scoped bindings, a picker, a question or a panel with the keys
//! gets them as before. Enter is the view's own key, so it sends in either
//! mode. Esc replaces the view's Esc (which stops the agent); twice within
//! [`DOUBLE_ESC`] in normal mode with an empty input it sends the view's
//! own [`InputAction::Stop`]. The mode shows in the status line, and
//! `/vim on|off` toggles it, kept in `RIG_HOME/vim.json` for every session.

pub mod editor;

use std::path::PathBuf;
use std::process::Stdio;
use std::sync::Mutex;
use std::sync::mpsc::{Receiver, Sender, channel};

use bevy_ecs::query::Allow;
use rig_harness::prelude::*;
use rig_tui::crossterm::event::KeyCode;
use rig_tui::{
    AppKeyBindingsExt, AppStatus, CopyToClipboard, InputAction, KeyBinding, KeyPress, KeyPressed,
    KeyScope, Side, StatusItem, StatusSystems, Tone, TuiInput, TuiPlugin,
};
use serde::{Deserialize, Serialize};

use editor::{Buffer, Editor, Effect, Key, Mode};

/// How soon a second Esc in normal mode, with nothing typed, does what Esc
/// does without the plugin.
pub const DOUBLE_ESC: Duration = Duration::from_millis(500);

/// The mode's place in the status line: after the agent's status.
const MODE_ITEM: StatusItem = StatusItem::at(Side::Left, 55, 9);

/// The file under `RIG_HOME` that keeps `/vim on|off` across sessions.
const SAVED: &str = "vim.json";

/// The `[vim]` table of `settings.toml`.
#[derive(Resource, Deserialize, Default, Clone, Copy)]
#[serde(default, deny_unknown_fields)]
pub struct VimSettings {
    /// The mode the input starts in, and returns to after a message is
    /// sent.
    pub start: StartMode,
}

/// The mode the input starts in.
#[derive(Deserialize, Default, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum StartMode {
    /// Typing, as without the plugin.
    #[default]
    Insert,
    /// Motions and operators.
    Normal,
}

/// What `RIG_HOME/vim.json` keeps.
#[derive(Serialize, Deserialize)]
struct Saved {
    enabled: bool,
}

/// The modal editor's state.
#[derive(Resource)]
pub struct Vim {
    /// Whether vim keys are on (`/vim on|off`).
    pub enabled: bool,
    /// Off while a question is answered in the input, which takes the
    /// keys as without the plugin.
    pub suspended: bool,
    editor: Editor,
    start: Mode,
    /// When the last Esc with nothing to cancel was pressed, on the real
    /// clock.
    last_esc: Option<Duration>,
    /// Where `vim.json` is.
    saved: PathBuf,
    /// The view's own Esc binding, freed while vim keys are on.
    view_esc: Option<Entity>,
}

impl Vim {
    /// The input's mode.
    pub fn mode(&self) -> Mode {
        self.editor.mode()
    }

    /// Whether the plugin's keys apply now.
    pub fn active(&self) -> bool {
        self.enabled && !self.suspended
    }
}

/// On each of the plugin's key bindings: the key it gives the editor.
#[derive(Component, Clone, Copy)]
struct VimKey(Key);

/// System-clipboard reads in flight for `"+p`: each sends the text, or
/// `None`, and whether it goes after the cursor.
#[derive(Resource)]
struct ClipboardReads {
    sender: Sender<(Option<String>, bool)>,
    results: Mutex<Receiver<(Option<String>, bool)>>,
}

impl Default for ClipboardReads {
    fn default() -> Self {
        let (sender, results) = channel();
        Self {
            sender,
            results: Mutex::new(results),
        }
    }
}

/// Vim-style modal editing in the terminal view's input.
#[derive(Default)]
pub struct VimPlugin;

impl Plugin for VimPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TuiPlugin>() {
            app.add_plugins(TuiPlugin);
        }
        app.add_setting::<VimSettings>("vim");
        let start = match app.world().get_resource::<VimSettings>().map(|s| s.start) {
            Some(StartMode::Normal) => Mode::Normal,
            _ => Mode::Insert,
        };
        let saved = RigHome::of(app.world()).root().join(SAVED);
        let enabled = std::fs::read_to_string(&saved)
            .ok()
            .and_then(|text| serde_json::from_str::<Saved>(&text).ok())
            .is_none_or(|saved| saved.enabled);
        app.insert_resource(Vim {
            enabled,
            suspended: false,
            editor: Editor::new(start),
            start,
            last_esc: None,
            saved,
            view_esc: None,
        })
        .init_resource::<ClipboardReads>()
        .add_command("vim", "Vim keys in the input: /vim on|off", toggle)
        .add_observer(after_send)
        .add_systems(Startup, arm_keys)
        .add_systems(
            Update,
            (
                suspend_for_questions,
                paste_clipboard,
                arm_keys.run_if(resource_changed::<Vim>),
            )
                .chain()
                .after(AgentLoop),
        )
        .add_systems(
            PostUpdate,
            show_mode
                .run_if(resource_changed::<Vim>)
                .in_set(StatusSystems),
        );
        let keys = (' '..='~')
            .map(|character| {
                (
                    KeyPress::key(KeyCode::Char(character)),
                    Key::Char(character),
                )
            })
            .chain([(KeyPress::ctrl('r'), Key::Redo)]);
        for (press, key) in keys {
            bind(app, press, key);
        }
    }

    /// Frees the view's Esc, which stops the agent, for the plugin's own,
    /// once every plugin's bindings are spawned.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let esc = KeyPress::key(KeyCode::Esc);
        let mut bindings = world.query::<(Entity, &KeyBinding, &InputAction)>();
        let view_esc = bindings.iter(world).find_map(|(entity, binding, action)| {
            let is_esc = binding.scope == KeyScope::Input && binding.keys.presses() == [esc];
            (is_esc && *action == InputAction::Stop).then_some(entity)
        });
        if let Some(entity) = view_esc {
            world.entity_mut(entity).insert(Disabled);
        }
        world.resource_mut::<Vim>().view_esc = view_esc;
        bind(app, esc, Key::Esc);
    }
}

/// Binds `press` in the input to give the editor `key`.
fn bind(app: &mut App, press: KeyPress, key: Key) {
    let label = match key {
        Key::Esc => "Vim: normal mode; twice, stop the agent".to_owned(),
        Key::Redo => "Vim: redo".to_owned(),
        Key::Char(character) => format!("Vim: {character}"),
    };
    let binding = KeyBinding::new(press, label).in_scope(KeyScope::Input);
    if let Some(entity) = app.add_key_binding(binding, on_key) {
        app.world_mut().entity_mut(entity).insert(VimKey(key));
    }
}

/// Feeds a pressed key to the editor and writes the result into the input.
#[allow(clippy::too_many_arguments)]
fn on_key(
    pressed: On<KeyPressed>,
    keys: Query<&VimKey>,
    (mut vim, mut input, time): (ResMut<Vim>, ResMut<TuiInput>, Res<Time<Real>>),
    (reads, processes, wake): (
        Res<ClipboardReads>,
        Option<Res<Processes>>,
        Option<Res<Wake>>,
    ),
    mut copies: MessageWriter<CopyToClipboard>,
    mut actions: MessageWriter<InputAction>,
) {
    let Ok(&VimKey(key)) = keys.get(pressed.entity) else {
        return;
    };
    if pressed.released {
        return;
    }
    let mut buffer = Buffer {
        text: input.text().to_owned(),
        cursor: input.cursor(),
    };
    let effect = vim.editor.key(key, &mut buffer);
    if buffer.text != input.text() {
        input.set(buffer.text);
    }
    if buffer.cursor != input.cursor() {
        input.set_cursor(buffer.cursor);
    }
    match effect {
        Some(Effect::Copy(text)) => {
            copies.write(CopyToClipboard(text));
        }
        Some(Effect::PasteClipboard { after }) => {
            let launcher = processes.map(|p| p.launcher()).unwrap_or_default();
            let sender = reads.sender.clone();
            let wake = wake.map(|wake| wake.clone());
            let spawned = std::thread::Builder::new()
                .name("vim-clipboard".to_owned())
                .spawn(move || {
                    sender.send((read_clipboard_text(&launcher), after)).ok();
                    if let Some(wake) = wake {
                        wake.wake();
                    }
                });
            if spawned.is_err() {
                reads.sender.send((None, after)).ok();
            }
        }
        Some(Effect::BareEsc) => {
            let now = time.elapsed();
            let twice = vim
                .last_esc
                .is_some_and(|last| now.saturating_sub(last) <= DOUBLE_ESC);
            if twice && input.is_empty() {
                vim.last_esc = None;
                actions.write(InputAction::Stop);
                if let Some(wake) = wake {
                    wake.wake();
                }
            } else {
                vim.last_esc = Some(now);
            }
        }
        None => {}
    }
}

/// The system clipboard's text, read by `wl-paste`, `xclip` or `pbpaste`.
/// Blocking.
fn read_clipboard_text(launcher: &rig_harness::rig_tools::process::Launcher) -> Option<String> {
    let mut readers: Vec<(&str, &[&str])> = Vec::new();
    if std::env::var_os("WAYLAND_DISPLAY").is_some() {
        readers.push(("wl-paste", &["--no-newline"]));
    }
    if std::env::var_os("DISPLAY").is_some() {
        readers.push(("xclip", &["-selection", "clipboard", "-out"]));
    }
    if cfg!(target_os = "macos") {
        readers.push(("pbpaste", &[]));
    }
    readers.iter().find_map(|(program, args)| {
        let output = launcher
            .command(program)
            .args(*args)
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).into_owned())
    })
}

/// Pastes what the clipboard reads brought back.
fn paste_clipboard(
    reads: Res<ClipboardReads>,
    mut vim: ResMut<Vim>,
    mut input: ResMut<TuiInput>,
    shown: Option<Res<Shown>>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(results) = reads.results.lock() else {
        return;
    };
    for (text, after) in results.try_iter() {
        let Some(text) = text else {
            let agent = shown.as_ref().and_then(|shown| shown.0);
            notices.write(Notice::info(agent, "No clipboard text to paste."));
            continue;
        };
        let mut buffer = Buffer {
            text: input.text().to_owned(),
            cursor: input.cursor(),
        };
        vim.editor.put_text(&mut buffer, &text, after);
        input.set(buffer.text);
        input.set_cursor(buffer.cursor);
    }
}

/// Leaves the keys to a question answered in the input while one is open.
fn suspend_for_questions(asks: Query<&Ask, Without<Outcome>>, mut vim: ResMut<Vim>) {
    let in_input = asks
        .iter()
        .any(|ask| !ask.inline && (ask.initial.is_some() || ask.multiline));
    if vim.suspended != in_input {
        vim.suspended = in_input;
    }
}

/// Binds the plugin's keys as the mode needs: Esc while vim keys are on,
/// the others in normal and visual mode only, so insert mode types as the
/// view does; the view's Esc while they are off.
fn arm_keys(
    vim: Res<Vim>,
    keys: Query<(Entity, &VimKey, Has<Disabled>), Allow<Disabled>>,
    view_keys: Query<Has<Disabled>, (With<KeyBinding>, Without<VimKey>, Allow<Disabled>)>,
    mut commands: Commands,
) {
    let active = vim.active();
    let mut arm = |entity: Entity, on: bool, disabled: bool| match (on, disabled) {
        (true, true) => {
            commands.entity(entity).remove::<Disabled>();
        }
        (false, false) => {
            commands.entity(entity).insert(Disabled);
        }
        _ => {}
    };
    for (entity, &VimKey(key), disabled) in &keys {
        let on = active && (key == Key::Esc || vim.mode() != Mode::Insert);
        arm(entity, on, disabled);
    }
    // The view's Esc is free exactly while the plugin's is armed.
    if let Some(view_esc) = vim.view_esc
        && let Ok(disabled) = view_keys.get(view_esc)
    {
        arm(view_esc, !active, disabled);
    }
}

/// A message the user sent leaves the input in the start mode.
fn after_send(deliver: On<Deliver>, mut vim: ResMut<Vim>, input: Res<TuiInput>) {
    if deliver.message.origin.kind == OriginKind::User && input.is_empty() {
        let start = vim.start;
        if vim.mode() != start {
            vim.editor.reset(start);
        }
    }
}

/// Shows the mode in the status line while vim keys are on.
fn show_mode(vim: Res<Vim>, mut status: ResMut<AppStatus>) {
    let text = if vim.enabled { vim.mode().label() } else { "" };
    status.0.show(MODE_ITEM.says(text, Tone::Cyan));
}

/// `/vim on|off`, or `/vim` to switch; kept for every session.
fn toggle(In(args): In<CommandArgs>, mut vim: ResMut<Vim>, mut notices: MessageWriter<Notice>) {
    let enabled = match args.args.trim() {
        "on" => true,
        "off" => false,
        "" => !vim.enabled,
        other => {
            notices.write(Notice::error(
                args.agent,
                format!("/vim takes on or off, not `{other}`"),
            ));
            return;
        }
    };
    vim.enabled = enabled;
    let start = vim.start;
    vim.editor.reset(start);
    let saved = serde_json::to_string(&Saved { enabled }).unwrap_or_default();
    if let Err(failure) = std::fs::write(&vim.saved, saved) {
        let why = format!("Could not keep the vim setting: {failure}.");
        notices.write(Notice::warn(args.agent, why));
    }
    let said = if enabled {
        "Vim keys on."
    } else {
        "Vim keys off."
    };
    notices.write(Notice::info(args.agent, said));
}

#[cfg(test)]
mod tests;
