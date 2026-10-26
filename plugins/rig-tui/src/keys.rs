//! Key bindings: every key the view acts on, its own editing keys
//! included, is a [`KeyBinding`] entity named `key:<keys>`, so a plugin
//! binds a key or a sequence (`ctrl+x ctrl+e`) as the view does, and
//! listings read them as data.
//!
//! The first binding of a key stays ([`AppKeyBindingsExt::add_key_binding`]
//! warns on a conflict); `Disabled` on a binding, in `Plugin::finish`,
//! frees its keys. [`KeyScope`] says where a binding applies and which
//! shadows which. A pressed binding triggers [`KeyPressed`] on its entity,
//! unless it has an [`InputAction`], which the view does instead. A
//! [`TuiPanel`](crate::TuiPanel) takes the keys while it is the
//! [`KeyFocus`]. For keys before any binding, filter
//! [`PendingInput`](crate::PendingInput) in
//! [`InputSystems::Filter`](crate::InputSystems::Filter).
//!
//! ```no_run
//! use rig_harness::prelude::*;
//! use rig_tui::{AppKeyBindingsExt, KeyBinding, KeyPress, run_command};
//!
//! fn build(app: &mut App) {
//!     // Alt+K compacts the conversation of the shown agent.
//!     app.add_key_binding(KeyBinding::new(KeyPress::alt('k'), "Compact"), run_command("compact"));
//! }
//! ```

use std::fmt;
use std::str::FromStr;

use bevy_app::App;
use bevy_ecs::entity_disabling::Disabled;
use bevy_ecs::prelude::*;
use bevy_ecs::query::Allow;
use bevy_ecs::system::IntoObserverSystem;
use bevy_log::warn;
use bevy_reflect::prelude::*;
use crossterm::event::{KeyCode, KeyEvent, KeyModifiers};
use rig_harness::commands::RunCommand;

use super::view::TuiView;

/// One key with its modifiers, as the terminal reports it. A character
/// carries its case instead of Shift (`A` is Shift+a); Shift+Tab is
/// [`KeyCode::BackTab`].
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct KeyPress {
    /// The key.
    pub code: KeyCode,
    /// Ctrl, Alt, Shift and Super held with it.
    pub modifiers: KeyModifiers,
}

impl KeyPress {
    /// `code` with `modifiers`, in the form the terminal reports it.
    pub fn new(code: KeyCode, modifiers: KeyModifiers) -> Self {
        let modifiers = match code {
            KeyCode::Char(_) | KeyCode::BackTab => modifiers.difference(KeyModifiers::SHIFT),
            _ => modifiers,
        };
        Self { code, modifiers }
    }

    /// `code` alone.
    pub fn key(code: KeyCode) -> Self {
        Self::new(code, KeyModifiers::NONE)
    }

    /// Ctrl+`character`.
    pub fn ctrl(character: char) -> Self {
        Self::new(KeyCode::Char(character), KeyModifiers::CONTROL)
    }

    /// Alt+`character`.
    pub fn alt(character: char) -> Self {
        Self::new(KeyCode::Char(character), KeyModifiers::ALT)
    }

    /// `code` with Shift.
    pub fn shift(code: KeyCode) -> Self {
        Self::new(code, KeyModifiers::SHIFT)
    }

    /// The press `event` reports.
    pub fn of(event: &KeyEvent) -> Self {
        Self::new(event.code, event.modifiers)
    }

    /// Whether it types its character into the input: a character with
    /// neither Ctrl nor Alt.
    pub fn typed(&self) -> Option<char> {
        match self.code {
            KeyCode::Char(character)
                if !self
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                Some(character)
            }
            _ => None,
        }
    }

    /// The same key without modifiers, for a key that is no character
    /// pressed with some: Ctrl+Home is Home where nothing binds it.
    fn bare(&self) -> Option<Self> {
        let character = matches!(self.code, KeyCode::Char(_));
        (!character && !self.modifiers.is_empty()).then(|| Self::key(self.code))
    }
}

impl fmt::Display for KeyPress {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (modifier, name) in [
            (KeyModifiers::CONTROL, "ctrl+"),
            (KeyModifiers::ALT, "alt+"),
            (KeyModifiers::SUPER, "super+"),
            (KeyModifiers::SHIFT, "shift+"),
        ] {
            if self.modifiers.contains(modifier) {
                out.write_str(name)?;
            }
        }
        match self.code {
            KeyCode::Char(' ') => out.write_str("space"),
            KeyCode::Char(character) => write!(out, "{character}"),
            KeyCode::F(number) => write!(out, "f{number}"),
            KeyCode::BackTab => out.write_str("shift+tab"),
            code => {
                let name = NAMES
                    .iter()
                    .find(|(_, named)| *named == code)
                    .map_or("?", |(name, _)| name);
                out.write_str(name)
            }
        }
    }
}

/// The keys known by name, the first of each code being how it is shown.
const NAMES: &[(&str, KeyCode)] = &[
    ("enter", KeyCode::Enter),
    ("return", KeyCode::Enter),
    ("esc", KeyCode::Esc),
    ("escape", KeyCode::Esc),
    ("tab", KeyCode::Tab),
    ("backtab", KeyCode::BackTab),
    ("backspace", KeyCode::Backspace),
    ("delete", KeyCode::Delete),
    ("del", KeyCode::Delete),
    ("insert", KeyCode::Insert),
    ("home", KeyCode::Home),
    ("end", KeyCode::End),
    ("pageup", KeyCode::PageUp),
    ("pgup", KeyCode::PageUp),
    ("pagedown", KeyCode::PageDown),
    ("pgdn", KeyCode::PageDown),
    ("up", KeyCode::Up),
    ("down", KeyCode::Down),
    ("left", KeyCode::Left),
    ("right", KeyCode::Right),
];

impl FromStr for KeyPress {
    type Err = String;

    /// Reads a key such as `ctrl+e`, `alt+shift+left`, `f5`, `space` or
    /// `?`: modifiers (`ctrl`, `alt`, `shift`, `super`), then the key,
    /// joined by `+`, ignoring case except in a lone character.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let mut modifiers = KeyModifiers::NONE;
        let mut rest = text.trim();
        while let Some((modifier, after)) =
            rest.split_once('+').filter(|(_, after)| !after.is_empty())
        {
            modifiers |= match modifier.to_lowercase().as_str() {
                "ctrl" | "control" => KeyModifiers::CONTROL,
                "alt" | "meta" | "option" => KeyModifiers::ALT,
                "shift" => KeyModifiers::SHIFT,
                "super" | "cmd" => KeyModifiers::SUPER,
                _ => return Err(format!("`{modifier}` in `{text}` is not a modifier")),
            };
            rest = after;
        }
        let mut characters = rest.chars();
        let code = match (characters.next(), characters.next()) {
            (Some(character), None) if modifiers.contains(KeyModifiers::SHIFT) => {
                KeyCode::Char(character.to_ascii_uppercase())
            }
            (Some(character), None) => KeyCode::Char(character),
            _ => {
                let name = rest.to_lowercase();
                let numbered = name.strip_prefix('f').and_then(|n| n.parse::<u8>().ok());
                match (name.as_str(), numbered) {
                    ("space", _) => KeyCode::Char(' '),
                    (_, Some(number @ 1..=24)) => KeyCode::F(number),
                    _ => NAMES
                        .iter()
                        .find(|(named, _)| *named == name)
                        .map(|(_, code)| *code)
                        .ok_or_else(|| format!("`{rest}` in `{text}` is not a key"))?,
                }
            }
        };
        let code = match code {
            KeyCode::Tab if modifiers.contains(KeyModifiers::SHIFT) => KeyCode::BackTab,
            code => code,
        };
        Ok(Self::new(code, modifiers))
    }
}

/// A key or a sequence of keys pressed one after another, such as
/// `ctrl+x ctrl+e`.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct Keys(Vec<KeyPress>);

impl Keys {
    /// The presses, in order.
    pub fn presses(&self) -> &[KeyPress] {
        &self.0
    }

    /// Whether these keys start with `presses`, or are them.
    fn starts_with(&self, presses: &[KeyPress]) -> bool {
        self.0.starts_with(presses)
    }
}

impl From<KeyPress> for Keys {
    fn from(press: KeyPress) -> Self {
        Self(vec![press])
    }
}

impl<const N: usize> From<[KeyPress; N]> for Keys {
    fn from(presses: [KeyPress; N]) -> Self {
        Self(presses.to_vec())
    }
}

impl FromStr for Keys {
    type Err = String;

    /// Reads keys separated by spaces, each as [`KeyPress`] reads it.
    fn from_str(text: &str) -> Result<Self, Self::Err> {
        let presses = text
            .split_whitespace()
            .map(KeyPress::from_str)
            .collect::<Result<Vec<_>, _>>()?;
        if presses.is_empty() {
            return Err("no key given".to_owned());
        }
        Ok(Self(presses))
    }
}

impl fmt::Display for Keys {
    fn fmt(&self, out: &mut fmt::Formatter<'_>) -> fmt::Result {
        for (index, press) in self.0.iter().enumerate() {
            if index > 0 {
                out.write_str(" ")?;
            }
            write!(out, "{press}")?;
        }
        Ok(())
    }
}

/// Where a [`KeyBinding`] applies.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash)]
pub enum KeyScope {
    /// Wherever the keys go, as long as no picker or question is open.
    #[default]
    Global,
    /// In the input, while no panel has the [`KeyFocus`]: the view's own
    /// editing keys. A global or transcript binding of the same keys
    /// shadows one.
    Input,
    /// On the transcript, while the person reads it: an item of it is
    /// selected ([`TranscriptView::selected`](crate::TranscriptView::selected))
    /// and no panel has the [`KeyFocus`]. Its bindings are found after the
    /// global ones and before the input's, whose keys work otherwise, so
    /// moving the selection and scrolling go on reading. The
    /// reading ends when the input's text changes or is sent, and on Esc,
    /// which the view binds here to give the keys back to the input.
    Transcript,
    /// In this panel, while it has the [`KeyFocus`].
    Panel(Entity),
}

impl KeyScope {
    /// Whether a key can be bound in both scopes at once: not the same
    /// keys in one scope, nor globally and in a panel or the transcript.
    /// In the input and globally or on the transcript it can: the other
    /// binding shadows the input's.
    fn overlaps(self, other: Self) -> bool {
        match (self, other) {
            (Self::Global | Self::Transcript, Self::Input)
            | (Self::Input, Self::Global | Self::Transcript) => false,
            (Self::Global, _) | (_, Self::Global) => true,
            _ => self == other,
        }
    }

    /// The scopes whose bindings apply while the keys go to this one, in
    /// the order they are found: the global ones, then this one's, then,
    /// while reading the transcript, the input's.
    pub(crate) fn applying(self) -> impl Iterator<Item = Self> {
        let reading = (self == Self::Transcript).then_some(Self::Input);
        [Self::Global, self].into_iter().chain(reading)
    }
}

/// Keys bound to something, on an entity of its own named `key:<keys>`.
/// Disable it to free its keys. See [`keys`](crate::keys).
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq)]
#[reflect(opaque, Component, Clone, Debug, PartialEq)]
pub struct KeyBinding {
    /// The keys.
    pub keys: Keys,
    /// What it does, in a few words, for listings.
    pub label: String,
    /// Where it applies.
    pub scope: KeyScope,
    /// Whether letting go of the key triggers [`KeyPressed`] too, with
    /// `released`, for holding a key down (push to talk). A key held down
    /// then triggers once, not again for each repeat; a binding of a
    /// sequence never hears its release. While one asks for releases the
    /// terminal is asked to report them, which not every terminal does.
    pub releases: bool,
}

impl KeyBinding {
    /// `keys` bound everywhere, doing what `label` says.
    pub fn new(keys: impl Into<Keys>, label: impl Into<String>) -> Self {
        Self {
            keys: keys.into(),
            label: label.into(),
            scope: KeyScope::Global,
            releases: false,
        }
    }

    /// The same binding, applying only in `scope`.
    pub fn in_scope(mut self, scope: KeyScope) -> Self {
        self.scope = scope;
        self
    }

    /// The same binding, also triggered when its key is let go, where the
    /// terminal reports that.
    pub fn with_releases(mut self) -> Self {
        self.releases = true;
        self
    }

    /// Whether this and `other` cannot both be bound: their scopes
    /// overlap, and one's keys are the other's or start them.
    pub fn conflicts(&self, other: &Self) -> bool {
        self.scope.overlaps(other.scope)
            && (self.keys.starts_with(other.keys.presses())
                || other.keys.starts_with(self.keys.presses()))
    }
}

/// Triggered on a [`KeyBinding`]'s entity when its keys are pressed, or,
/// for one with [`KeyBinding::releases`], let go.
#[derive(EntityEvent, Reflect, Clone, Copy, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct KeyPressed {
    /// The binding.
    pub entity: Entity,
    /// The shown agent, if any.
    pub agent: Option<Entity>,
    /// Whether the key was let go rather than pressed.
    pub released: bool,
}

/// The panel that takes the keys, or `None` for the input. Set it to a
/// [`TuiPanel`](crate::TuiPanel)'s entity to give the panel the keys; the view sets it back
/// to `None` when that entity is no panel, or gone, and when Esc is
/// pressed that the panel does not bind.
#[derive(Resource, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Clone, Debug, Default, PartialEq)]
pub struct KeyFocus(pub Option<Entity>);

/// Gives the keys back to the input as soon as the panel that has them
/// stops being one, or goes.
pub(crate) fn release_focus(removed: On<Remove<crate::TuiPanel>>, mut focus: ResMut<KeyFocus>) {
    if focus.0 == Some(removed.entity) {
        focus.0 = None;
    }
}

/// Binds keys on an [`App`].
pub trait AppKeyBindingsExt {
    /// Spawns `binding` with `on_press`, an observer of the [`KeyPressed`]
    /// it triggers, and returns its entity, unless another binding
    /// conflicts with it ([`KeyBinding::conflicts`]): that one stays, with
    /// a warning.
    fn add_key_binding<M>(
        &mut self,
        binding: KeyBinding,
        on_press: impl IntoObserverSystem<KeyPressed, M>,
    ) -> Option<Entity>;
}

impl AppKeyBindingsExt for App {
    fn add_key_binding<M>(
        &mut self,
        binding: KeyBinding,
        on_press: impl IntoObserverSystem<KeyPressed, M>,
    ) -> Option<Entity> {
        let mut bound = bind(self.world_mut(), binding)?;
        bound.observe(on_press);
        Some(bound.id())
    }
}

/// Spawns `binding`, unless one conflicts with it: that one stays, with a
/// warning. A query skips a disabled binding.
fn bind(world: &mut World, binding: KeyBinding) -> Option<EntityWorldMut<'_>> {
    let mut bindings = world.query::<&KeyBinding>();
    if let Some(taken) = bindings.iter(world).find(|bound| bound.conflicts(&binding)) {
        warn!(
            "key binding `{}` ({}) not added: `{}` ({}) is bound; insert `Disabled` on it to \
             replace it",
            binding.keys, binding.label, taken.keys, taken.label
        );
        return None;
    }
    let name = Name::new(format!("key:{}", binding.keys));
    Some(world.spawn((name, binding)))
}

/// An observer for [`AppKeyBindingsExt::add_key_binding`] that runs the
/// slash command `line` (without its `/`) for the shown agent.
pub fn run_command(
    line: impl Into<String>,
) -> impl FnMut(On<KeyPressed>, Commands) + Send + Sync + 'static {
    let line = line.into();
    move |pressed: On<KeyPressed>, mut commands: Commands| {
        if let (Some(entity), false) = (pressed.agent, pressed.released) {
            commands.trigger(RunCommand::new(entity, line.clone()));
        }
    }
}

/// A built-in action of the input. On a [`KeyBinding`]'s entity it is what
/// the binding does, in order with the other keys of the frame, instead of
/// triggering [`KeyPressed`]: the view's own keys are such bindings, and a
/// plugin rebinds one by spawning its own. Pushed onto the
/// [`PendingInput`](crate::PendingInput) as
/// [`KeyInput::Action`](crate::KeyInput::Action), it is done where it
/// stands among the keys.
///
/// ```no_run
/// use rig_harness::prelude::*;
/// use rig_tui::{InputAction, KeyBinding, KeyPress};
///
/// fn build(app: &mut App) {
///     // Ctrl+K browses back as Up does: the completion, the line, the history.
///     app.world_mut().spawn((KeyBinding::new(KeyPress::ctrl('k'), "Up"), InputAction::Up));
/// }
/// ```
#[derive(Component, Clone, Copy, Debug, PartialEq, Eq)]
pub enum InputAction {
    /// Empties the input.
    Clear,
    /// On an empty input, arms the exit for a moment; pressed again while
    /// armed, quits as `/quit` does.
    Exit,
    /// Types a newline.
    NewLine,
    /// The start of the cursor's line.
    LineStart,
    /// The end of the cursor's line.
    LineEnd,
    /// A character back.
    Left,
    /// A character forward.
    Right,
    /// The start of the word before the cursor.
    WordLeft,
    /// The end of the word after the cursor.
    WordRight,
    /// Deletes the character before the cursor.
    Backspace,
    /// Deletes the character after the cursor.
    Delete,
    /// Deletes the word before the cursor.
    DeleteWordBefore,
    /// Deletes the word after the cursor.
    DeleteWordAfter,
    /// Deletes to the start of the line, or the newline before it.
    DeleteToLineStart,
    /// Deletes to the end of the line, or the newline after it.
    DeleteToLineEnd,
    /// The line above, or an earlier prompt of the history: Ctrl+P.
    LineUp,
    /// The line below, or a later prompt of the history: Ctrl+N.
    LineDown,
    /// The completion's previous item, else the line above, else an earlier
    /// prompt.
    Up,
    /// The completion's next item, else the line below, else a later
    /// prompt.
    Down,
    /// Scrolls the transcript a row up.
    ScrollUp,
    /// Scrolls the transcript a row down.
    ScrollDown,
    /// Scrolls the transcript a page up.
    PageUp,
    /// Scrolls the transcript a page down.
    PageDown,
    /// Completes, or with a turn running and nothing to complete, sends
    /// for after the turn.
    Tab,
    /// Completes a partly typed command or path, continues a line ending
    /// in `\`, else sends.
    Send,
    /// Closes the completion, else interrupts the shown agent.
    Stop,
    /// Saves the clipboard's image in the session's
    /// [`Scratch::images`](rig_harness::prelude::Scratch::images) and types
    /// `@path` for it, which attaches it to the message, or says there is
    /// none. A pasted path to an image file (a file dropped on the
    /// terminal) becomes `@path` too.
    PasteImage,
    /// Selects the item of the transcript above the one selected, or the
    /// newest.
    SelectUp,
    /// Selects the item below the one selected; past the newest, none.
    SelectDown,
    /// Selects none, which gives the keys back to the input: Esc while
    /// the person reads the transcript ([`KeyScope::Transcript`]).
    Deselect,
    /// Answers the question shown beside the work with its choice of this
    /// number, counted from 1, or the number after its choices: types its
    /// answer in the input.
    AnswerInline(u8),
    /// Closes the question shown beside the work, refused.
    DismissInline,
}

/// On the bindings of the keys that answer a question shown beside the
/// work: bound only while one shows.
#[derive(Component)]
pub(crate) struct InlineKey;

/// The view's own keys, as `(keys, label, action)`.
fn built_in() -> Vec<(Keys, &'static str, InputAction)> {
    use InputAction as Do;
    use KeyCode as K;
    let key = |code| Keys::from(KeyPress::key(code));
    let ctrl = |character| Keys::from(KeyPress::ctrl(character));
    let alt = |character| Keys::from(KeyPress::alt(character));
    let with = |modifiers, code| Keys::from(KeyPress::new(code, modifiers));
    let (control, meta) = (KeyModifiers::CONTROL, KeyModifiers::ALT);
    vec![
        (ctrl('c'), "Clear the input", Do::Clear),
        (ctrl('d'), "Quit, on an empty input", Do::Exit),
        (ctrl('j'), "New line", Do::NewLine),
        (with(KeyModifiers::SHIFT, K::Enter), "New line", Do::NewLine),
        (with(meta, K::Enter), "New line", Do::NewLine),
        (ctrl('a'), "Start of the line", Do::LineStart),
        (key(K::Home), "Start of the line", Do::LineStart),
        (ctrl('e'), "End of the line", Do::LineEnd),
        (key(K::End), "End of the line", Do::LineEnd),
        (ctrl('b'), "Back a character", Do::Left),
        (key(K::Left), "Back a character", Do::Left),
        (ctrl('f'), "Forward a character", Do::Right),
        (key(K::Right), "Forward a character", Do::Right),
        (alt('b'), "Back a word", Do::WordLeft),
        (with(control, K::Left), "Back a word", Do::WordLeft),
        (with(meta, K::Left), "Back a word", Do::WordLeft),
        (alt('f'), "Forward a word", Do::WordRight),
        (with(control, K::Right), "Forward a word", Do::WordRight),
        (with(meta, K::Right), "Forward a word", Do::WordRight),
        (key(K::Backspace), "Delete back", Do::Backspace),
        (key(K::Delete), "Delete forward", Do::Delete),
        (ctrl('w'), "Delete the word back", Do::DeleteWordBefore),
        (
            with(control, K::Backspace),
            "Delete the word back",
            Do::DeleteWordBefore,
        ),
        (
            with(meta, K::Backspace),
            "Delete the word back",
            Do::DeleteWordBefore,
        ),
        (alt('d'), "Delete the word forward", Do::DeleteWordAfter),
        (
            with(control, K::Delete),
            "Delete the word forward",
            Do::DeleteWordAfter,
        ),
        (
            with(meta, K::Delete),
            "Delete the word forward",
            Do::DeleteWordAfter,
        ),
        (
            ctrl('u'),
            "Delete to the line's start",
            Do::DeleteToLineStart,
        ),
        (ctrl('k'), "Delete to the line's end", Do::DeleteToLineEnd),
        (ctrl('p'), "Line up, or an earlier prompt", Do::LineUp),
        (ctrl('n'), "Line down, or a later prompt", Do::LineDown),
        (key(K::Up), "Previous completion, line or prompt", Do::Up),
        (key(K::Down), "Next completion, line or prompt", Do::Down),
        (with(KeyModifiers::SHIFT, K::Up), "Scroll up", Do::ScrollUp),
        (
            with(KeyModifiers::SHIFT, K::Down),
            "Scroll down",
            Do::ScrollDown,
        ),
        (key(K::PageUp), "Scroll a page up", Do::PageUp),
        (key(K::PageDown), "Scroll a page down", Do::PageDown),
        (key(K::Tab), "Complete, or send after this turn", Do::Tab),
        (key(K::Enter), "Send, or steer the running turn", Do::Send),
        (
            key(K::Esc),
            "Close the completion, or stop the agent",
            Do::Stop,
        ),
        (ctrl('v'), "Paste an image", Do::PasteImage),
        (with(meta, K::Up), "Select the item above", Do::SelectUp),
        (with(meta, K::Down), "Select the item below", Do::SelectDown),
    ]
}

/// Spawns the view's own key bindings, in the input's scope, and Esc on
/// the transcript, which gives the keys back to the input; the keys that
/// answer a question shown beside the work, Alt+1 to Alt+9 and Alt+0,
/// stay disabled until one shows. A plugin's global binding of the same
/// keys wins over them.
pub(crate) fn add_built_in(app: &mut App) {
    app.init_resource::<KeyFocus>();
    for (keys, label, action) in built_in() {
        let binding = KeyBinding::new(keys, label).in_scope(KeyScope::Input);
        if let Some(mut bound) = bind(app.world_mut(), binding) {
            bound.insert(action);
        }
    }
    let back = KeyBinding::new(KeyPress::key(KeyCode::Esc), "Back to the input");
    if let Some(mut bound) = bind(app.world_mut(), back.in_scope(KeyScope::Transcript)) {
        bound.insert(InputAction::Deselect);
    }
    let inline = (1..=9u8).map(|number| (number, "Answer the question beside the work"));
    let inline = inline.chain([(0, "Dismiss the question beside the work")]);
    for (number, label) in inline {
        let digit = char::from(b'0' + number);
        let binding = KeyBinding::new(KeyPress::alt(digit), label).in_scope(KeyScope::Input);
        let action = match number {
            0 => InputAction::DismissInline,
            number => InputAction::AnswerInline(number),
        };
        if let Some(mut bound) = bind(app.world_mut(), binding) {
            bound.insert((action, InlineKey, Disabled));
        }
    }
}

/// Binds the keys that answer a question shown beside the work while one
/// shows, and frees them otherwise.
pub(crate) fn arm_inline_keys(
    shown: Res<TuiView>,
    keys: Query<(Entity, Has<Disabled>), (With<InlineKey>, Allow<Disabled>)>,
    mut commands: Commands,
) {
    let showing = !shown.inline.is_empty();
    for (key, disabled) in &keys {
        match (showing, disabled) {
            (true, true) => {
                commands.entity(key).remove::<Disabled>();
            }
            (false, false) => {
                commands.entity(key).insert(Disabled);
            }
            _ => {}
        }
    }
}

/// What the keys pressed so far match, as [`find`] says.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Found {
    /// This binding's keys.
    Binding(Entity),
    /// The start of a binding's keys: wait for the next.
    Partial,
    /// No binding.
    Nothing,
}

/// The binding that `typed` matches in the scopes that apply while the
/// keys go to `focus` (`KeyScope::applying`), so a global binding
/// shadows the input's; the oldest entity of two bound around the conflict
/// check. A single key that is no character, pressed with modifiers that
/// nothing binds, is that key alone.
pub fn find(bindings: &[(Entity, &KeyBinding)], typed: &[KeyPress], focus: KeyScope) -> Found {
    let scoped = |scope: KeyScope| {
        bindings
            .iter()
            .filter(move |(_, binding)| binding.scope == scope)
    };
    let exact = |typed: &[KeyPress], scope: KeyScope| {
        scoped(scope)
            .filter(|(_, binding)| binding.keys.presses() == typed)
            .map(|(entity, _)| *entity)
            .min()
    };
    for scope in focus.applying() {
        if let Some(entity) = exact(typed, scope) {
            return Found::Binding(entity);
        }
        if scoped(scope).any(|(_, binding)| binding.keys.starts_with(typed)) {
            return Found::Partial;
        }
    }
    let bare = match typed {
        [press] => press.bare(),
        _ => None,
    };
    bare.and_then(|bare| focus.applying().find_map(|scope| exact(&[bare], scope)))
        .map_or(Found::Nothing, Found::Binding)
}

#[cfg(test)]
mod tests;
