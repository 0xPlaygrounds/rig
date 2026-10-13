//! SN-18: vim-style modal editing in the terminal view's input box.
//!
//! The input starts in insert mode (`[vim] start = "normal"` in
//! `settings.toml` changes that), where keys type as without the plugin.
//! A filter in [`InputSystems::Filter`] reads the frame's
//! [`TerminalEvents`] in order: Esc goes to normal mode, where it takes the
//! keys and runs them through [`editor::Editor`] (motions, counts,
//! operators, registers, undo and a visual selection), written back into
//! [`TuiInput`]. While a picker, a question or a panel has the keys
//! ([`TuiInputState`]), it leaves them alone. Enter stays the view's key,
//! so it sends in either mode. Twice within [`DOUBLE_ESC`] in normal mode
//! with an empty input, Esc does the view's own [`InputAction::Stop`]. The
//! mode shows in the status line, and `/vim on|off` toggles it, kept in
//! [`PluginData::user`] for every session.
//!
//! A filter sees each key with the input the keys before it left. Once a
//! key it lets through may have changed the input, the keys after it that
//! an earlier filter made of one go back to the view, which does them
//! next.

pub mod editor;

use rig_harness::prelude::*;
use rig_harness::rig_tools::clipboard;
use rig_tui::crossterm::event::{Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use rig_tui::{
    AppStatus, CopyToClipboard, InputAction, InputSystems, KeyInput, KeyScope, Sent, Side,
    StatusItem, StatusSystems, TerminalEvents, Tone, TuiInput, TuiInputState, TuiPlugin,
};
use serde::{Deserialize, Serialize};

use editor::{Buffer, Editor, Effect, Key, Mode};

/// How soon a second Esc in normal mode, with nothing typed, does what Esc
/// does without the plugin.
pub const DOUBLE_ESC: Duration = Duration::from_millis(500);

/// The mode's place in the status line: after the agent's status.
const MODE_ITEM: StatusItem = StatusItem::new("sn-vim-input", Side::Left, 55, 9);

/// The file in [`PluginData::user`] that keeps `/vim on|off`.
pub const SAVED: &str = "vim/saved.json";

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

/// What [`SAVED`] keeps.
#[derive(Serialize, Deserialize)]
struct Saved {
    enabled: bool,
}

/// The modal editor's state.
#[derive(Resource)]
pub struct Vim {
    /// Whether vim keys are on (`/vim on|off`).
    pub enabled: bool,
    editor: Editor,
    start: Mode,
    /// When the last Esc with nothing to cancel was pressed, on the real
    /// clock.
    last_esc: Option<Duration>,
    /// The rest of a frame whose keys may have changed the input, for the
    /// next frame.
    held: Vec<KeyInput>,
}

impl Vim {
    /// The input's mode.
    pub fn mode(&self) -> Mode {
        self.editor.mode()
    }
}

/// A read of the system clipboard for `"+p` or `"+P`, which ends as a
/// [`Done`] with its text.
#[derive(Component)]
struct ClipboardPaste {
    /// After the cursor rather than before it.
    after: bool,
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
        app.insert_resource(Vim {
            enabled: true,
            editor: Editor::new(start),
            start,
            last_esc: None,
            held: Vec::new(),
        })
        .add_command("vim", "Vim keys in the input: /vim on|off", toggle)
        .add_observer(paste_clipboard)
        .add_systems(Startup, load)
        .add_systems(
            rig_tui::TuiKeys,
            (
                edit.in_set(InputSystems::Filter),
                hold.after(InputSystems::Dispatch),
            ),
        )
        .add_systems(Update, after_send)
        .add_systems(
            PostUpdate,
            show_mode
                .run_if(resource_changed::<Vim>)
                .in_set(StatusSystems::Show),
        );
    }
}

/// Whether `/vim off` was kept from an earlier session.
fn load(data: Res<PluginData>, mut vim: ResMut<Vim>) {
    let saved = data.user.read_file(SAVED).ok();
    let saved = saved.and_then(|bytes| serde_json::from_slice::<Saved>(&bytes).ok());
    vim.enabled = saved.is_none_or(|saved| saved.enabled);
}

/// What the filter does with an item of the frame.
enum Take {
    /// Leaves it to the view; whether it may change the input.
    Pass { changes: bool },
    /// Gives the editor this key.
    Edit(Key),
    /// Enter in normal mode: sends, closing a completion first so it does
    /// not accept it.
    Send,
}

/// Takes the frame's keys as the mode says, in their order.
#[allow(clippy::too_many_arguments)]
fn edit(
    mut events: ResMut<TerminalEvents>,
    state: Res<TuiInputState>,
    mut vim: ResMut<Vim>,
    mut input: ResMut<TuiInput>,
    time: Res<Time<Real>>,
    (processes, wake): (Option<Res<Processes>>, Res<Wake>),
    mut copies: MessageWriter<CopyToClipboard>,
    mut commands: Commands,
) {
    if !vim.enabled || events.0.is_empty() {
        return;
    }
    let vim = &mut *vim;
    let mut buffer = Buffer {
        text: input.text().to_owned(),
        cursor: input.cursor(),
    };
    let elsewhere = state.picking().is_some()
        || state.answering().is_some()
        || matches!(state.scope(), KeyScope::Panel(_));
    let mut completion = state.completion().is_some();
    // Whether the input and the state are still as the filter knows them.
    let mut known = true;
    let mut kept = Vec::with_capacity(events.0.len());
    let mut items = std::mem::take(&mut events.0).into_iter();
    while let Some(item) = items.next() {
        let take = match &item {
            KeyInput::Event(Event::Key(key)) if key.kind != KeyEventKind::Release => {
                if !known && might_take(key, vim.mode()) {
                    vim.held.push(item);
                    vim.held.extend(items);
                    break;
                }
                take(key, vim.mode(), elsewhere)
            }
            KeyInput::Event(Event::Key(_) | Event::Resize(..) | Event::Mouse(_)) => {
                Take::Pass { changes: false }
            }
            KeyInput::Event(Event::FocusGained | Event::FocusLost) => Take::Pass { changes: false },
            KeyInput::Event(Event::Paste(_)) | KeyInput::Action(_) => Take::Pass { changes: true },
        };
        match take {
            Take::Pass { changes } => {
                known &= !changes;
                kept.push(item);
            }
            Take::Send => {
                if completion {
                    kept.push(KeyInput::Action(InputAction::Stop));
                }
                kept.push(item);
                known = false;
            }
            Take::Edit(key) => {
                // Esc in insert mode closes a completion as the view's does.
                if key == Key::Esc && vim.mode() == Mode::Insert && completion {
                    kept.push(KeyInput::Action(InputAction::Stop));
                    completion = false;
                }
                match vim.editor.key(key, &mut buffer) {
                    Some(Effect::Copy(text)) => {
                        copies.write(CopyToClipboard(text));
                    }
                    Some(Effect::PasteClipboard { after }) => {
                        let launcher = processes.as_deref().map(Processes::launcher);
                        let launcher = launcher.unwrap_or_default();
                        let pool =
                            bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
                        let read = blocking(move || Ok(clipboard::read_text(&launcher).ok()));
                        let task = async move { read.await.ok().flatten() };
                        commands
                            .spawn((ClipboardPaste { after }, Running::spawn(pool, &wake, task)));
                    }
                    Some(Effect::BareEsc) => {
                        let now = time.elapsed();
                        let twice = vim
                            .last_esc
                            .is_some_and(|last| now.saturating_sub(last) <= DOUBLE_ESC);
                        if twice && buffer.text.is_empty() {
                            vim.last_esc = None;
                            kept.push(KeyInput::Action(InputAction::Stop));
                            known = false;
                        } else {
                            vim.last_esc = Some(now);
                        }
                    }
                    None => {}
                }
            }
        }
    }
    events.0 = kept;
    if buffer.text != input.text() {
        input.set(buffer.text);
    }
    if buffer.cursor != input.cursor() {
        input.set_cursor(buffer.cursor);
    }
}

/// What the filter does with `key` in `mode`, the input's state known.
fn take(key: &KeyEvent, mode: Mode, elsewhere: bool) -> Take {
    let plain = key.modifiers.difference(KeyModifiers::SHIFT).is_empty();
    match (key.code, mode) {
        _ if elsewhere => Take::Pass { changes: true },
        (KeyCode::Esc, _) if plain => Take::Edit(Key::Esc),
        (_, Mode::Insert) => Take::Pass { changes: true },
        (KeyCode::Enter, _) if plain => Take::Send,
        (KeyCode::Char(character), _) if plain => Take::Edit(Key::Char(character)),
        (KeyCode::Char('r'), _) if key.modifiers == KeyModifiers::CONTROL => Take::Edit(Key::Redo),
        _ => Take::Pass { changes: true },
    }
}

/// Whether the editor might take `key` in `mode`, whatever the input's
/// state: Esc in insert mode, any key in the others.
fn might_take(key: &KeyEvent, mode: Mode) -> bool {
    mode != Mode::Insert || key.code == KeyCode::Esc
}

/// Puts the keys after one that changed the input back, which the view
/// does next, each with the input it left.
fn hold(mut vim: ResMut<Vim>, mut events: ResMut<TerminalEvents>) {
    if vim.held.is_empty() {
        return;
    }
    events.0.append(&mut vim.held);
}

/// Pastes what a clipboard read brought back.
fn paste_clipboard(
    read: On<Insert<Done<Option<String>>>>,
    reads: Query<(&ClipboardPaste, &Done<Option<String>>)>,
    (mut vim, mut input): (ResMut<Vim>, ResMut<TuiInput>),
    shown: Option<Res<Shown>>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((&ClipboardPaste { after }, Done(text))) = reads.get(read.entity) else {
        return;
    };
    commands.entity(read.entity).despawn();
    let Some(text) = text else {
        let agent = shown.and_then(|shown| shown.0);
        notices.write(Notice::info(agent, "No clipboard text to paste."));
        return;
    };
    let mut buffer = Buffer {
        text: input.text().to_owned(),
        cursor: input.cursor(),
    };
    vim.editor.put_text(&mut buffer, text, after);
    input.set(buffer.text);
    input.set_cursor(buffer.cursor);
}

/// A message the view sent leaves the input in the start mode.
fn after_send(mut sent: MessageReader<Sent>, mut vim: ResMut<Vim>) {
    if sent.read().count() > 0 && vim.mode() != vim.start {
        let start = vim.start;
        vim.editor.reset(start);
    }
}

/// Shows the mode in the status line while vim keys are on.
fn show_mode(vim: Res<Vim>, mut status: ResMut<AppStatus>) {
    let text = if vim.enabled { vim.mode().label() } else { "" };
    status.0.show(MODE_ITEM.says(text, Tone::Cyan));
}

/// `/vim on|off`, or `/vim` to switch; kept for every session.
fn toggle(
    In(args): In<CommandArgs>,
    mut vim: ResMut<Vim>,
    data: Res<PluginData>,
    mut notices: MessageWriter<Notice>,
) {
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
    let saved = serde_json::to_vec(&Saved { enabled }).unwrap_or_default();
    if let Err(failure) = data.user.write_file(SAVED, &saved) {
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
