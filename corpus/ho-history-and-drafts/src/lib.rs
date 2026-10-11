//! Plugin corpus (held-out): H21: Prompt history across sessions and saved
//! drafts (spec `H21-history-and-drafts.md`).
//!
//! **History.** Every message the user sends, in any front (`Incoming`
//! from `Origin::user()`), is appended to
//! `RIG_HOME/history/<project-hash>.jsonl` as `{text, time, session}`. At
//! most [`MAX_ENTRIES`] are kept per project, the oldest trimmed. Not
//! recorded: a repeat of the previous entry, a message starting with a
//! space, and one matching a secret pattern (`[history] secrets = [..]`
//! in `settings.toml`, regular expressions; `sk-…` by default).
//!
//! **Browsing.** In the terminal view, Up and Down (and Ctrl+P, Ctrl+N) at
//! the input's first or last line browse this project's history, this
//! session's prompts first; elsewhere they move between the input's lines.
//! They replace the view's own bindings of these keys, which browse the
//! view's history of every project. Ctrl+R opens a picker of the history
//! to filter by typing.
//!
//! **Drafts.** What is typed for an agent and not sent is its [`Draft`],
//! saved with the session once typing paused for [`QUIET`], so `/reload`,
//! a restart or `/resume` puts it back in the input with the cursor at its
//! end. Showing another agent keeps the draft of the one left and shows
//! the other's. Sending (the input emptied) clears it at once. A `--print`
//! run types nothing in the input, so it has no draft.

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use regex::Regex;
use rig_harness::prelude::*;
use rig_tui::crossterm::event::KeyCode;
use rig_tui::{
    AppKeyBindingsExt, KeyBinding, KeyPress, KeyPressed, KeyScope, Keys, TuiInput, TuiPlugin,
    TuiSystems,
};
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

/// The entries kept per project.
pub const MAX_ENTRIES: usize = 1000;

/// How long typing pauses before the draft is saved.
pub const QUIET: Duration = Duration::from_millis(500);

/// The `[history]` table of `settings.toml`.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct HistorySettings {
    /// Regular expressions: a message any of them matches is not recorded.
    pub secrets: Vec<String>,
}

impl Default for HistorySettings {
    fn default() -> Self {
        Self {
            secrets: vec![r"sk-[A-Za-z0-9_-]{4,}".to_owned()],
        }
    }
}

/// One line of the history file.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq, Eq)]
pub struct Entry {
    /// What the user sent.
    pub text: String,
    /// When, in seconds since the Unix epoch.
    pub time: u64,
    /// The session it was sent in, when the session has an id.
    #[serde(default)]
    pub session: Option<String>,
}

/// This project's history, as read at startup and appended to since.
#[derive(Resource, Default)]
pub struct History {
    file: Option<PathBuf>,
    entries: Vec<Entry>,
    session: Option<String>,
    secrets: Vec<Regex>,
}

impl History {
    /// The file it is kept in.
    pub fn file(&self) -> Option<&Path> {
        self.file.as_deref()
    }

    /// The entries, oldest first.
    pub fn entries(&self) -> &[Entry] {
        &self.entries
    }

    /// Whether `text` is recorded.
    fn keeps(&self, text: &str) -> bool {
        !text.trim().is_empty()
            && !text.starts_with(' ')
            && self.entries.last().is_none_or(|last| last.text != text)
            && !self.secrets.iter().any(|secret| secret.is_match(text))
    }

    /// Appends `text` to the entries and the file, trimming both to
    /// [`MAX_ENTRIES`].
    fn append(&mut self, text: &str) -> std::io::Result<()> {
        let time = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |since| since.as_secs());
        let entry = Entry {
            text: text.to_owned(),
            time,
            session: self.session.clone(),
        };
        self.entries.push(entry.clone());
        let excess = self.entries.len().saturating_sub(MAX_ENTRIES);
        self.entries.drain(..excess);
        let Some(file) = &self.file else {
            return Ok(());
        };
        if let Some(parent) = file.parent() {
            fs::create_dir_all(parent)?;
        }
        if excess > 0 {
            let mut kept = String::new();
            for entry in &self.entries {
                kept.push_str(&serde_json::to_string(entry)?);
                kept.push('\n');
            }
            let temporary = file.with_extension("jsonl.new");
            fs::write(&temporary, kept)?;
            return fs::rename(&temporary, file);
        }
        let line = serde_json::to_string(&entry)?;
        let mut out = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(file)?;
        writeln!(out, "{line}")
    }

    /// The order Up goes through: this session's prompts, newest first,
    /// then the earlier sessions', newest first.
    fn browse_order(&self) -> Vec<String> {
        let ours = |entry: &&Entry| self.session.is_some() && entry.session == self.session;
        let mine = self.entries.iter().rev().filter(ours);
        let others = self.entries.iter().rev().filter(|entry| !ours(entry));
        mine.chain(others).map(|entry| entry.text.clone()).collect()
    }

    /// Every prompt once, newest first.
    fn search_order(&self) -> Vec<String> {
        let mut seen = Vec::new();
        for entry in self.entries.iter().rev() {
            if !seen.contains(&entry.text) {
                seen.push(entry.text.clone());
            }
        }
        seen
    }
}

/// The input's text not yet sent for this agent, kept with the session.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default)]
pub struct Draft(pub String);

/// Whose draft the input holds, what it held when last looked at, and a
/// count of its edits, so a delayed save knows whether typing went on.
#[derive(Resource, Default)]
struct Typing {
    owner: Option<Entity>,
    seen: String,
    edits: u64,
}

/// Where Up and Down are in the history.
#[derive(Resource, Default)]
struct Browsing {
    /// The position in [`History::browse_order`] shown.
    at: Option<usize>,
    /// What the input held before browsing.
    stash: String,
    /// The text browsing put in the input; another means it was edited.
    showing: String,
}

/// A Ctrl+R picker's prompts, by item.
#[derive(Component)]
struct Recall(Vec<String>);

/// Per-project prompt history across sessions, and saved drafts.
#[derive(Default)]
pub struct HistoryPlugin;

impl Plugin for HistoryPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<TuiPlugin>() {
            app.add_plugins(TuiPlugin);
        }
        app.add_setting::<HistorySettings>("history")
            .init_resource::<RigHome>()
            .init_resource::<ProjectRoot>()
            .init_resource::<Shown>()
            .init_resource::<History>()
            .init_resource::<Typing>()
            .init_resource::<Browsing>()
            .register_required_components::<Agent, Draft>()
            .add_systems(Startup, open_history)
            .add_systems(PostUpdate, follow_input.before(TuiSystems::Layout))
            .add_observer(record);
    }

    /// Takes Up, Down, Ctrl+P and Ctrl+N over from the view, once its
    /// bindings exist, and binds Ctrl+R.
    fn finish(&self, app: &mut App) {
        let up = [KeyPress::key(KeyCode::Up), KeyPress::ctrl('p')];
        let down = [KeyPress::key(KeyCode::Down), KeyPress::ctrl('n')];
        let taken: Vec<Keys> = up.iter().chain(&down).copied().map(Keys::from).collect();
        let world = app.world_mut();
        let mut bindings = world.query::<(Entity, &KeyBinding)>();
        let replaced: Vec<Entity> = bindings
            .iter(world)
            .filter(|(_, binding)| {
                binding.scope == KeyScope::Input && taken.contains(&binding.keys)
            })
            .map(|(entity, _)| entity)
            .collect();
        for binding in replaced {
            world.entity_mut(binding).insert(Disabled);
        }
        let input =
            |keys: KeyPress, label: &str| KeyBinding::new(keys, label).in_scope(KeyScope::Input);
        for keys in up {
            app.add_key_binding(
                input(keys, "Line up, or an earlier prompt of this project"),
                back,
            );
        }
        for keys in down {
            app.add_key_binding(
                input(keys, "Line down, or a later prompt of this project"),
                forward,
            );
        }
        app.add_key_binding(
            input(KeyPress::ctrl('r'), "Search this project's prompts"),
            search,
        );
    }
}

/// The history file of the project at `root`: its path, hashed.
fn history_file(home: &Home, root: &Path) -> PathBuf {
    let root = fs::canonicalize(root).unwrap_or_else(|_| root.to_path_buf());
    let digest = Sha256::digest(root.to_string_lossy().as_bytes());
    let hash: String = digest
        .iter()
        .take(8)
        .map(|byte| format!("{byte:02x}"))
        .collect();
    home.root().join("history").join(format!("{hash}.jsonl"))
}

/// Reads this project's history and the secret patterns.
fn open_history(
    home: Res<RigHome>,
    root: Res<ProjectRoot>,
    paths: Option<Res<SessionPaths>>,
    settings: Res<HistorySettings>,
    mut history: ResMut<History>,
    mut notices: MessageWriter<Notice>,
) {
    let file = history_file(&home.0, &root.0);
    let text = fs::read_to_string(&file).unwrap_or_default();
    let mut entries: Vec<Entry> = text
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    let excess = entries.len().saturating_sub(MAX_ENTRIES);
    entries.drain(..excess);
    let mut secrets = Vec::new();
    for pattern in &settings.secrets {
        match Regex::new(pattern) {
            Ok(secret) => secrets.push(secret),
            Err(why) => {
                notices.write(Notice::error(
                    None,
                    format!("[history] secret `{pattern}`: {why}"),
                ));
            }
        }
    }
    *history = History {
        file: Some(file),
        entries,
        session: paths.and_then(|paths| paths.id()).map(|id| id.to_string()),
        secrets,
    };
}

/// Records each message the user sends, in any front.
fn record(
    sent: On<Add<Incoming>>,
    messages: Query<&Incoming>,
    mut history: ResMut<History>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(message) = messages.get(sent.entity) else {
        return;
    };
    // A titled message is a plugin's on the user's behalf, not typed.
    let typed = message.origin.kind == OriginKind::User && message.origin.title.is_none();
    if typed
        && history.keeps(&message.text)
        && let Err(why) = history.append(&message.text)
    {
        let why = format!("Could not save the prompt history: {why}");
        notices.write(Notice::error(None, why));
    }
}

/// Keeps the input's text as the draft of the agent it is typed for: at
/// once when it is emptied or another agent is shown (whose draft the
/// input then holds), else once typing paused for [`QUIET`].
fn follow_input(
    shown: Res<Shown>,
    mut input: ResMut<TuiInput>,
    mut typing: ResMut<Typing>,
    mut drafts: Query<&mut Draft>,
    mut commands: Commands,
) {
    typing.edits += u64::from(input.text() != typing.seen || typing.owner != shown.0);
    if typing.owner != shown.0 {
        if let Some(mut left) = typing.owner.and_then(|owner| drafts.get_mut(owner).ok()) {
            left.set_if_neq(Draft(input.text().to_owned()));
        }
        let draft = shown.0.and_then(|agent| drafts.get(agent).ok());
        let text = draft.map(|draft| draft.0.clone()).unwrap_or_default();
        if input.text() != text {
            input.set(text.clone());
        }
        typing.owner = shown.0;
        typing.seen = text;
        return;
    }
    if input.text() == typing.seen {
        return;
    }
    typing.seen = input.text().to_owned();
    if typing.seen.is_empty() {
        if let Some(mut draft) = typing.owner.and_then(|owner| drafts.get_mut(owner).ok()) {
            draft.set_if_neq(Draft::default());
        }
        return;
    }
    let edit = typing.edits;
    commands
        .delayed()
        .duration(QUIET)
        .queue(move |world: &mut World| save_if_quiet(world, edit));
}

/// Saves the input as its agent's draft, unless it was edited after
/// `edit`.
fn save_if_quiet(world: &mut World, edit: u64) {
    let Some(typing) = world.get_resource::<Typing>() else {
        return;
    };
    let (Some(owner), true) = (typing.owner, typing.edits == edit) else {
        return;
    };
    let text = typing.seen.clone();
    if let Some(mut draft) = world.get_mut::<Draft>(owner) {
        draft.set_if_neq(Draft(text));
    }
}

/// The byte offset of `column` (in characters) on the line starting at
/// `start`, or the line's end.
fn column_offset(text: &str, start: usize, column: usize) -> usize {
    let line = text.get(start..).unwrap_or_default();
    let line = line.split('\n').next().unwrap_or_default();
    start
        + line
            .char_indices()
            .nth(column)
            .map_or(line.len(), |(index, _)| index)
}

/// The cursor's line start and its column in characters.
fn line_and_column(text: &str, cursor: usize) -> (usize, usize) {
    let before = text.get(..cursor).unwrap_or_default();
    let start = before.rfind('\n').map_or(0, |index| index + 1);
    let column = before.get(start..).map_or(0, |line| line.chars().count());
    (start, column)
}

/// Up: the line above, else an earlier prompt.
fn back(
    pressed: On<KeyPressed>,
    mut input: ResMut<TuiInput>,
    history: Res<History>,
    mut browsing: ResMut<Browsing>,
) {
    if pressed.released {
        return;
    }
    let text = input.text().to_owned();
    let (start, column) = line_and_column(&text, input.cursor());
    if start > 0 {
        let before = text.get(..start - 1).unwrap_or_default();
        let above = before.rfind('\n').map_or(0, |index| index + 1);
        input.set_cursor(column_offset(&text, above, column));
        return;
    }
    let next = match browsing.at {
        Some(at) if browsing.showing == text => at + 1,
        _ => {
            browsing.stash = text;
            0
        }
    };
    if let Some(prompt) = history.browse_order().into_iter().nth(next) {
        browsing.at = Some(next);
        browsing.showing.clone_from(&prompt);
        input.set(prompt);
    }
}

/// Down: the line below, else a later prompt, and after the newest what
/// was typed before browsing.
fn forward(
    pressed: On<KeyPressed>,
    mut input: ResMut<TuiInput>,
    history: Res<History>,
    mut browsing: ResMut<Browsing>,
) {
    if pressed.released {
        return;
    }
    let text = input.text().to_owned();
    let cursor = input.cursor();
    if let Some(newline) = text.get(cursor..).and_then(|after| after.find('\n')) {
        let (_, column) = line_and_column(&text, cursor);
        input.set_cursor(column_offset(&text, cursor + newline + 1, column));
        return;
    }
    let Some(at) = browsing.at.filter(|_| browsing.showing == text) else {
        return;
    };
    if at == 0 {
        browsing.at = None;
        let stash = std::mem::take(&mut browsing.stash);
        input.set(stash);
        return;
    }
    if let Some(prompt) = history.browse_order().into_iter().nth(at - 1) {
        browsing.at = Some(at - 1);
        browsing.showing.clone_from(&prompt);
        input.set(prompt);
    }
}

/// Ctrl+R: a picker of the project's prompts, newest first, filtered by
/// what is typed; the one picked goes in the input.
fn search(pressed: On<KeyPressed>, history: Res<History>, mut commands: Commands) {
    if pressed.released {
        return;
    }
    let prompts = history.search_order();
    let picker = prompts.iter().fold(
        Picker::new("Prompt history").body("Type to filter; Enter puts the prompt in the input."),
        |picker, prompt| picker.item(prompt.replace('\n', " ⏎ "), ""),
    );
    commands.spawn((picker, Recall(prompts))).observe(
        |picked: On<Picked>, recalls: Query<&Recall>, mut input: ResMut<TuiInput>| {
            if let Some(Chosen::Item(index)) = picked.chosen
                && let Ok(Recall(prompts)) = recalls.get(picked.entity)
                && let Some(prompt) = prompts.get(index)
            {
                input.set(prompt.clone());
            }
        },
    );
}

#[cfg(test)]
mod tests;
