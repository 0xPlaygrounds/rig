//! Plugin corpus: 53. Custom key bindings (spec `53-keybindings.md`).
//!
//! `RIG_HOME/keys.toml` maps keys to actions:
//!
//! ```toml
//! "alt+m" = { cycle_model = ["deepseek/deepseek-flash", "deepseek/deepseek-v4-pro"] }
//! "ctrl+g" = { command = "/compact" }
//! "ctrl+t" = { insert = "Think step by step." }
//! "ctrl+o" = { send = "Go on.", label = "continue" }
//! "ctrl+e" = { editor = true, override = true }
//! ```
//!
//! A binding of a key the terminal view already uses is refused at startup
//! with a notice, unless it says `override = true`. Other plugins add
//! bindings with [`AppKeyBindingsExt::add_key_binding`]; `/keys` lists them
//! all with where each came from. The file is read at startup, so `/reload`
//! picks up a change.
//!
//! A front hands a key to the bindings with [`KeyPressed`]. The terminal
//! view (`rig-tui`) reads its keys privately and never does, and its input
//! line is private too, so `insert` and `editor` cannot reach it.

use std::fmt;
use std::path::PathBuf;

use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use rig_harness::prelude::*;
use serde::Deserialize;

/// Where the user's bindings come from in `/keys`.
const USER_FILE: &str = "keys.toml";

/// A key with its modifiers, as `ctrl+e`, `alt+m`, `enter` or `f5`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct KeyChord {
    /// The key; a letter is lowercase.
    pub code: KeyCode,
    /// Control, Alt and Shift only.
    pub modifiers: KeyModifiers,
}

impl KeyChord {
    /// Parses `ctrl+alt+x`, `shift+up`, `enter`, `f5`, `space`.
    pub fn parse(text: &str) -> Result<Self, String> {
        let text = text.trim().to_lowercase();
        let mut parts: Vec<&str> = text.split('+').collect();
        let key = parts.pop().filter(|key| !key.is_empty());
        let key = key.ok_or_else(|| format!("`{text}` names no key"))?;
        let mut modifiers = KeyModifiers::NONE;
        for part in parts {
            modifiers |= match part {
                "ctrl" | "control" => KeyModifiers::CONTROL,
                "alt" | "meta" => KeyModifiers::ALT,
                "shift" => KeyModifiers::SHIFT,
                other => return Err(format!("`{other}` in `{text}` is not a modifier")),
            };
        }
        let code = match key {
            "enter" | "return" => KeyCode::Enter,
            "esc" | "escape" => KeyCode::Esc,
            "tab" => KeyCode::Tab,
            "backspace" => KeyCode::Backspace,
            "delete" | "del" => KeyCode::Delete,
            "up" => KeyCode::Up,
            "down" => KeyCode::Down,
            "left" => KeyCode::Left,
            "right" => KeyCode::Right,
            "home" => KeyCode::Home,
            "end" => KeyCode::End,
            "pageup" => KeyCode::PageUp,
            "pagedown" => KeyCode::PageDown,
            "space" => KeyCode::Char(' '),
            _ => match (key.chars().next(), key.chars().count()) {
                (Some(character), 1) => KeyCode::Char(character),
                (Some('f'), _) => match key.get(1..).and_then(|n| n.parse().ok()) {
                    Some(n) if (1..=24).contains(&n) => KeyCode::F(n),
                    _ => return Err(format!("`{key}` is not a key")),
                },
                _ => return Err(format!("`{key}` is not a key")),
            },
        };
        Ok(Self { code, modifiers })
    }

    /// The chord of a key event.
    pub fn of(key: &KeyEvent) -> Self {
        let code = match key.code {
            KeyCode::Char(character) => KeyCode::Char(character.to_ascii_lowercase()),
            code => code,
        };
        let modifiers =
            key.modifiers & (KeyModifiers::CONTROL | KeyModifiers::ALT | KeyModifiers::SHIFT);
        Self { code, modifiers }
    }

    /// What the terminal view does with this chord in its input line, if
    /// anything: a copy of `rig-tui`'s `input_key`, which is private.
    pub fn built_in(&self) -> Option<&'static str> {
        let control = self.modifiers.contains(KeyModifiers::CONTROL);
        let alt = self.modifiers.contains(KeyModifiers::ALT);
        let shift = self.modifiers.contains(KeyModifiers::SHIFT);
        let plain = !control && !alt;
        Some(match self.code {
            KeyCode::Char(character) if control && !alt => match character {
                'c' => "clear the input",
                'j' => "new line",
                'a' => "line start",
                'e' => "line end",
                'b' => "cursor left",
                'f' => "cursor right",
                'k' => "delete to line end",
                'u' => "delete to line start",
                'w' => "delete word",
                'p' => "line up",
                'n' => "line down",
                'v' => "paste an image",
                _ => return None,
            },
            KeyCode::Char(character) if alt && !control => match character {
                'b' => "word left",
                'f' => "word right",
                'd' => "delete next word",
                _ => return None,
            },
            KeyCode::Char(_) if plain => "typing",
            KeyCode::Backspace | KeyCode::Delete | KeyCode::Left | KeyCode::Right => "editing",
            KeyCode::Home | KeyCode::End if plain => "editing",
            KeyCode::Tab if plain => "completion and queueing",
            KeyCode::Enter if shift || alt => "new line",
            KeyCode::Enter if plain => "send",
            KeyCode::Esc if plain => "interrupt",
            KeyCode::Up | KeyCode::Down if plain => "history and scrolling",
            KeyCode::PageUp | KeyCode::PageDown if plain => "scrolling",
            _ => return None,
        })
    }
}

impl fmt::Display for KeyChord {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (modifier, name) in [
            (KeyModifiers::CONTROL, "ctrl+"),
            (KeyModifiers::ALT, "alt+"),
            (KeyModifiers::SHIFT, "shift+"),
        ] {
            if self.modifiers.contains(modifier) {
                f.write_str(name)?;
            }
        }
        match self.code {
            KeyCode::Char(' ') => f.write_str("space"),
            KeyCode::Char(character) => write!(f, "{character}"),
            KeyCode::F(n) => write!(f, "f{n}"),
            code => f.write_str(&format!("{code:?}").to_lowercase()),
        }
    }
}

/// What a key does.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum KeyAction {
    /// Edit the input in `$EDITOR`.
    Editor,
    /// Run a slash command, such as `/compact`.
    Command(String),
    /// Insert text at the cursor.
    Insert(String),
    /// Step the agent's model through these `vendor/model` references.
    CycleModel(Vec<String>),
    /// Send a message.
    Send(String),
}

impl fmt::Display for KeyAction {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Editor => f.write_str("edit in $EDITOR"),
            Self::Command(line) => write!(f, "run {line}"),
            Self::Insert(text) => write!(f, "insert {text:?}"),
            Self::CycleModel(models) => write!(f, "cycle {}", models.join(", ")),
            Self::Send(text) => write!(f, "send {text:?}"),
        }
    }
}

/// A key bound to an action, with its label and where it came from.
#[derive(Clone, Debug)]
pub struct KeyBinding {
    /// The key.
    pub chord: KeyChord,
    /// What it does.
    pub action: KeyAction,
    /// A short label for `/keys`.
    pub label: String,
    /// `keys.toml`, or the plugin that added it.
    pub source: String,
    /// Whether it may replace the terminal view's own use of the key.
    pub overrides: bool,
}

impl KeyBinding {
    /// `key` doing `action`, labelled `label`, added by `source`.
    pub fn new(key: &str, action: KeyAction, label: &str, source: &str) -> Result<Self, String> {
        Ok(Self {
            chord: KeyChord::parse(key)?,
            action,
            label: label.to_owned(),
            source: source.to_owned(),
            overrides: false,
        })
    }

    /// This binding, allowed to replace a built-in key.
    pub fn overriding(self) -> Self {
        Self {
            overrides: true,
            ..self
        }
    }
}

/// The bindings in force: after startup, only those accepted.
#[derive(Resource, Default, Debug)]
pub struct KeyBindings(pub Vec<KeyBinding>);

impl KeyBindings {
    /// The binding of `chord`.
    pub fn get(&self, chord: KeyChord) -> Option<&KeyBinding> {
        self.0.iter().find(|binding| binding.chord == chord)
    }
}

/// The file the user's bindings are read from: `RIG_HOME/keys.toml`.
#[derive(Resource, Clone, Debug)]
pub struct KeysFile(pub PathBuf);

/// Adds key bindings from another plugin.
pub trait AppKeyBindingsExt {
    /// Adds `binding`; it is checked with the user's at startup.
    fn add_key_binding(&mut self, binding: KeyBinding) -> &mut Self;
}

impl AppKeyBindingsExt for App {
    fn add_key_binding(&mut self, binding: KeyBinding) -> &mut Self {
        self.init_resource::<KeyBindings>();
        self.world_mut()
            .resource_mut::<KeyBindings>()
            .0
            .push(binding);
        self
    }
}

/// A key pressed in a front's input for the agent `entity`.
#[derive(EntityEvent, Clone, Debug)]
pub struct KeyPressed {
    /// The agent shown.
    pub entity: Entity,
    /// The key.
    pub key: KeyEvent,
}

/// Reads `keys.toml` at startup, runs bound keys, and adds `/keys`.
#[derive(Default)]
pub struct KeyBindingsPlugin;

impl Plugin for KeyBindingsPlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<KeysFile>() {
            app.insert_resource(KeysFile(Home::from_env().root().join(USER_FILE)));
        }
        app.init_resource::<KeyBindings>()
            .add_systems(Startup, load)
            .add_observer(press)
            .add_command(
                "keys",
                "List the key bindings and where they came from",
                keys,
            );
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

impl Entry {
    fn binding(self, key: &str) -> Result<KeyBinding, String> {
        let mut actions = Vec::new();
        if self.editor {
            actions.push(KeyAction::Editor);
        }
        actions.extend(self.command.map(KeyAction::Command));
        actions.extend(self.insert.map(KeyAction::Insert));
        actions.extend(self.cycle_model.map(KeyAction::CycleModel));
        actions.extend(self.send.map(KeyAction::Send));
        let action = match actions.pop() {
            Some(action) if actions.is_empty() => action,
            _ => return Err(format!("`{key}` needs exactly one action")),
        };
        if let KeyAction::CycleModel(models) = &action
            && models.is_empty()
        {
            return Err(format!("`{key}` cycles through no models"));
        }
        let label = self.label.unwrap_or_else(|| action.to_string());
        let binding = KeyBinding::new(key, action, &label, USER_FILE)?;
        Ok(if self.overrides {
            binding.overriding()
        } else {
            binding
        })
    }
}

/// Reads the user's bindings, then keeps each binding, the plugins' and
/// the user's, that neither takes a built-in key without `override` nor
/// one bound before it; each refused one is a notice.
fn load(
    file: Res<KeysFile>,
    mut bindings: ResMut<KeyBindings>,
    mut notices: MessageWriter<Notice>,
) {
    let mut candidates = std::mem::take(&mut bindings.0);
    match std::fs::read_to_string(&file.0) {
        Ok(text) => match toml::from_str::<toml::Table>(&text) {
            Ok(table) => {
                for (key, value) in table {
                    let entry = value.try_into::<Entry>().map_err(|why| why.to_string());
                    match entry.and_then(|entry| entry.binding(&key)) {
                        Ok(binding) => candidates.push(binding),
                        Err(why) => {
                            notices.write(Notice::error(None, format!("{USER_FILE}: {why}.")));
                        }
                    }
                }
            }
            Err(why) => {
                notices.write(Notice::error(None, format!("{USER_FILE}: {why}")));
            }
        },
        Err(why) if why.kind() == std::io::ErrorKind::NotFound => {}
        Err(why) => {
            notices.write(Notice::error(None, format!("{USER_FILE}: {why}.")));
        }
    }
    for binding in candidates {
        let refusal = match (binding.chord.built_in(), bindings.get(binding.chord)) {
            (_, Some(taken)) => Some(format!("it is bound by {} already", taken.source)),
            (Some(built_in), None) if !binding.overrides => Some(format!(
                "the terminal view uses it for {built_in}; add `override = true` to replace it"
            )),
            _ => None,
        };
        match refusal {
            Some(why) => {
                let text = format!("Key {} ({}) refused: {why}.", binding.chord, binding.source);
                notices.write(Notice::error(None, text));
            }
            None => bindings.0.push(binding),
        }
    }
}

/// Runs the action bound to a pressed key, if any.
fn press(
    pressed: On<KeyPressed>,
    bindings: Res<KeyBindings>,
    choices: Query<Option<&ModelChoice>, With<Agent>>,
    models: Res<Models>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = pressed.entity;
    let Some(binding) = bindings.get(KeyChord::of(&pressed.key)) else {
        return;
    };
    match &binding.action {
        KeyAction::Command(line) | KeyAction::Send(line) => {
            commands.trigger(Deliver::new(agent, line.clone(), DeliveryMode::Steer));
        }
        KeyAction::CycleModel(list) => {
            let Ok(current) = choices.get(agent) else {
                return;
            };
            let resolve = |reference: &str| {
                models
                    .0
                    .catalog()
                    .resolve(reference)
                    .map(|found| found.spec.reference())
                    .ok()
            };
            let at = current.and_then(|current| {
                list.iter()
                    .position(|model| resolve(model).as_deref() == Some(current.0.as_str()))
            });
            let next = at.map_or(0, |at| (at + 1) % list.len());
            if let Some(model) = list.get(next) {
                commands.trigger(SetModel {
                    entity: agent,
                    model: model.clone(),
                });
            }
        }
        KeyAction::Insert(_) | KeyAction::Editor => {
            let why = format!(
                "{}: the terminal view's input line cannot be changed by a plugin.",
                binding.chord
            );
            notices.write(Notice::error(agent, why));
        }
    }
}

/// `/keys`: every binding in force, with where it came from.
fn keys(In(args): In<CommandArgs>, bindings: Res<KeyBindings>, mut notices: MessageWriter<Notice>) {
    let text = if bindings.0.is_empty() {
        format!("No key bindings: add some to {USER_FILE}.")
    } else {
        bindings
            .0
            .iter()
            .map(|binding| {
                format!(
                    "{:<12} {}  ({})",
                    binding.chord.to_string(),
                    binding.label,
                    binding.source
                )
            })
            .collect::<Vec<_>>()
            .join("\n")
    };
    notices.write(Notice::info(args.agent, text));
}

#[cfg(test)]
mod tests;
