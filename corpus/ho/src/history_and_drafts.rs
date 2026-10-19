//! Plugin corpus (held-out): H21: Prompt history across sessions and saved
//! drafts (spec `H21-history-and-drafts.md`).
//!
//! **History.** Every message the user sends, in any front (`Incoming`
//! from `Author::user()`), is appended to
//! `RIG_HOME/history/<project-hash>.jsonl` as `{text, time, session}`. At
//! most [`MAX_ENTRIES`] are kept per project, the oldest trimmed. Not
//! recorded: a repeat of the previous entry, a message starting with a
//! space, one holding a well-known secret (`rig_core::redact`), and one
//! matching a pattern of `[history] secrets = [..]` in `settings.toml`
//! (regular expressions; `sk-…` by default).
//!
//! **Browsing.** The terminal view's Up and Down browse its
//! [`PromptHistory`]; this plugin supplies its entries, this project's
//! prompts with this session's newest, and turns the view's own file off,
//! so nothing the rules above leave out is saved anywhere. Ctrl+R opens a
//! fuzzy search of the history: typed characters match in order, the
//! closest together first; Enter puts the prompt in the input.
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
use rig_harness::rig_core::redact;
use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::text::Line;
use rig_tui::ratatui::widgets::Paragraph;
use rig_tui::{
    AppKeyBindingsExt, HistorySystems, InputSystems, KeyBinding, KeyFocus, KeyInput, KeyPress,
    KeyPressed, KeyScope, MainInput, PanelCanvas, PendingInput, Placement, PromptHistory,
    RequestRedraw, Theme, TuiPanel, TuiPlugin, TuiSystems,
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
    /// Regular expressions: a message any of them matches is not recorded,
    /// besides the secrets `rig_core::redact` knows.
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
            && !redact::has_secret(text)
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

    /// What Up and Down browse, oldest first: the earlier sessions'
    /// prompts, then this session's, so Up reaches this session's first.
    fn browse_order(&self) -> Vec<String> {
        let ours = |entry: &&Entry| self.session.is_some() && entry.session == self.session;
        let others = self.entries.iter().filter(|entry| !ours(entry));
        let mine = self.entries.iter().filter(ours);
        others.chain(mine).map(|entry| entry.text.clone()).collect()
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
            .init_resource::<ShownAgent>()
            .init_resource::<History>()
            .init_resource::<Typing>()
            .register_required_components::<Agent, Draft>()
            .add_systems(Startup, open_history)
            .add_systems(
                Update,
                offer
                    .run_if(resource_changed::<History>)
                    .after(HistorySystems::Record)
                    .before(HistorySystems::Save),
            )
            .add_systems(rig_tui::TuiKeys, type_search.in_set(InputSystems::Filter))
            .add_systems(
                PostUpdate,
                (
                    follow_input.before(TuiSystems::Layout),
                    draw_search.in_set(TuiSystems::Draw),
                ),
            )
            .add_observer(record);
        app.add_key_binding(
            KeyBinding::new(KeyPress::ctrl('r'), "Search this project's prompts")
                .in_scope(KeyScope::Input),
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

/// Reads this project's history and the secret patterns, and turns the
/// view's own history file off: this plugin's file is the history.
fn open_history(
    home: Res<RigHome>,
    root: Res<ProjectRoot>,
    paths: Option<Res<SessionPaths>>,
    settings: Res<HistorySettings>,
    mut history: ResMut<History>,
    mut prompts: ResMut<PromptHistory>,
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
    prompts.file = None;
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
    if message.author.kind != AuthorKind::User || message.author.title.is_some() {
        return;
    }
    // The view recorded what it sent as it is; [`offer`] puts this
    // project's history back in its place, kept or not.
    history.set_changed();
    if history.keeps(message.text())
        && let Err(why) = history.append(message.text())
    {
        let why = format!("Could not save the prompt history: {why}");
        notices.write(Notice::error(None, why));
    }
}

/// Gives the view's Up and Down this project's history, in place of what
/// the view recorded itself.
fn offer(history: Res<History>, mut prompts: ResMut<PromptHistory>) {
    prompts.entries = history.browse_order();
}

/// Keeps the input's text as the draft of the agent it is typed for: at
/// once when it is emptied or another agent is shown (whose draft the
/// input then holds), else once typing paused for [`QUIET`].
fn follow_input(
    shown: Res<ShownAgent>,
    mut input: MainInput,
    mut typing: ResMut<Typing>,
    mut drafts: Query<&mut Draft>,
    mut commands: Commands,
) {
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
        typing.edits += 1;
        return;
    }
    if input.text() == typing.seen {
        return;
    }
    typing.seen = input.text().to_owned();
    typing.edits += 1;
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

/// Ctrl+R: a search box over the transcript holding the keys.
fn search(
    pressed: On<KeyPressed>,
    history: Res<History>,
    mut focus: ResMut<KeyFocus>,
    mut commands: Commands,
) {
    if pressed.released {
        return;
    }
    let placement = Placement::Over {
        width: Constraint::Percentage(80),
        height: Constraint::Length(SHOWN as u16 + 3),
    };
    let search = Search {
        prompts: history.search_order(),
        filter: String::new(),
        selected: 0,
    };
    focus.0 = Some(commands.spawn((search, TuiPanel::new(placement))).id());
}

/// The prompts a search shows at most.
const SHOWN: usize = 12;

/// An open Ctrl+R search: this project's prompts, newest first, what is
/// typed to find one, and the match selected.
#[derive(Component)]
struct Search {
    prompts: Vec<String>,
    filter: String,
    selected: usize,
}

impl Search {
    /// The prompts holding the filter's characters in order, the closest
    /// together first, then the newest.
    fn matches(&self) -> Vec<&str> {
        let mut found: Vec<(usize, &str)> = self
            .prompts
            .iter()
            .filter_map(|prompt| Some((spread(prompt, &self.filter)?, prompt.as_str())))
            .collect();
        found.sort_by_key(|&(spread, _)| spread);
        found
            .into_iter()
            .take(SHOWN)
            .map(|(_, prompt)| prompt)
            .collect()
    }
}

/// How far apart `prompt` holds the characters of `filter`, in order and
/// ignoring case and the filter's spaces, or `None` when it does not.
fn spread(prompt: &str, filter: &str) -> Option<usize> {
    let mut wanted = filter
        .chars()
        .filter(|ch| !ch.is_whitespace())
        .flat_map(char::to_lowercase)
        .peekable();
    let (mut first, mut last) = (None, 0);
    for (at, ch) in prompt.chars().flat_map(char::to_lowercase).enumerate() {
        if wanted.peek() == Some(&ch) {
            wanted.next();
            first.get_or_insert(at);
            last = at;
        }
    }
    wanted
        .peek()
        .is_none()
        .then(|| first.map_or(0, |first| last - first))
}

/// Types into the open search: characters filter, Up and Down select,
/// Enter puts the selected prompt in the input. Esc, which the view
/// answers by taking the keys back, closes it.
fn type_search(
    events: Res<PendingInput>,
    mut searches: Query<(Entity, &mut Search)>,
    focus: Res<KeyFocus>,
    mut input: Option<MainInput>,
    mut redraw: MessageWriter<RequestRedraw>,
    mut commands: Commands,
) {
    let Ok((panel, mut search)) = searches.single_mut() else {
        return;
    };
    if focus.0 != Some(panel) {
        commands.entity(panel).despawn();
        return;
    }
    for item in &events.0 {
        let KeyInput::Event(Event::Key(key)) = item else {
            continue;
        };
        if key.kind == KeyEventKind::Release {
            continue;
        }
        match key.code {
            KeyCode::Enter => {
                if let (Some(prompt), Some(input)) =
                    (search.matches().get(search.selected), input.as_mut())
                {
                    input.set(*prompt);
                }
                commands.entity(panel).despawn();
                return;
            }
            KeyCode::Up => search.selected = search.selected.saturating_sub(1),
            KeyCode::Down => {
                let last = search.matches().len().saturating_sub(1);
                search.selected = (search.selected + 1).min(last);
            }
            KeyCode::Backspace => {
                search.filter.pop();
                search.selected = 0;
            }
            KeyCode::Char(ch)
                if !key
                    .modifiers
                    .intersects(KeyModifiers::CONTROL | KeyModifiers::ALT) =>
            {
                search.filter.push(ch);
                search.selected = 0;
            }
            _ => {}
        }
    }
    redraw.write(RequestRedraw);
}

/// Draws the open search: the filter, then the matches, the selected one
/// on the selected color.
fn draw_search(mut panels: Query<(&Search, &mut PanelCanvas)>, theme: Res<Theme>) {
    for (search, mut canvas) in &mut panels {
        let mut lines = vec![Line::from(format!("› {}", search.filter))];
        for (at, prompt) in search.matches().into_iter().enumerate() {
            let line = Line::from(prompt.replace('\n', " ⏎ "));
            lines.push(if at == search.selected {
                line.patch_style(theme.selection())
            } else {
                line
            });
        }
        let title = "Prompt history: type to search, Enter to use, Esc to close";
        canvas.render(Paragraph::new(lines).block(theme.raised(title)));
    }
}

#[cfg(test)]
mod tests;
