//! The prompt history.

use std::collections::HashSet;
use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_log::warn;
use rig_ecs::agent::{Agent, Conversation};
use rig_harness::prelude::{Files, PluginData, ShownAgent};
use rig_tools::attach::typed_prompts;

use super::jsonl::{self, Kept};

/// The file of the history in its [`PromptHistory::file`].
const HISTORY_FILE: &str = "tui/history.jsonl";

/// The prompts Up and Down browse, oldest first, each text once, kept
/// across sessions in the project's [`PluginData`] as `tui/history.jsonl`
/// (`file: None` saves nothing). Prompts are kept as typed, slash commands
/// included.
///
/// The view loads it in `PreStartup`, puts each prompt [`Sent`] as the
/// newest in [`HistorySystems::Record`], and in [`HistorySystems::Save`]
/// adds to the file those recorded since that are still in `entries`,
/// keeping the newest [`Self::KEPT`]. A plugin filters `entries` between
/// the two, such as to keep secrets out. After another agent is shown,
/// [`HistorySystems::Seed`] [seeds](Self::seed) its conversation's typed
/// prompts as the newest before the next key; a filter that should also
/// keep them from Up runs after it too.
///
/// ```no_run
/// use rig_harness::prelude::*;
/// use rig_tui::{HistorySystems, InputSystems, PromptHistory, TuiKeys};
///
/// /// Prompts that look like they hold a key are never kept.
/// fn scrub(mut history: ResMut<PromptHistory>) {
///     history.entries.retain(|prompt| !prompt.contains("sk-"));
/// }
///
/// fn build(app: &mut App) {
///     app.add_systems(Update, scrub.after(HistorySystems::Record).before(HistorySystems::Save))
///         .add_systems(
///             TuiKeys,
///             scrub.after(HistorySystems::Seed).before(InputSystems::Dispatch),
///         );
/// }
/// ```
#[derive(Resource, Default)]
pub struct PromptHistory {
    /// The prompts, oldest first.
    pub entries: Vec<String>,
    /// The files prompts are saved in, as `tui/history.jsonl`, one JSON
    /// string per line.
    pub file: Option<Arc<dyn Files>>,
    /// The prompts recorded since the last save.
    unsaved: Vec<String>,
}

impl PromptHistory {
    /// How many prompts are kept.
    pub const KEPT: usize = 500;

    /// The newest [`Self::KEPT`] distinct prompts kept in `files`, which
    /// new ones are saved to. Saving only appends, so the file is
    /// rewritten with them once it holds more than twice that many lines.
    pub fn load(files: Arc<dyn Files>) -> Self {
        let file = Kept(files.clone(), HISTORY_FILE);
        let text = files.read_file(HISTORY_FILE).unwrap_or_default();
        let text = String::from_utf8_lossy(&text);
        let mut seen = HashSet::new();
        let mut entries: Vec<String> = text
            .lines()
            .rev()
            .filter_map(|line| serde_json::from_str::<String>(line).ok())
            .filter(|prompt| seen.insert(prompt.clone()))
            .take(Self::KEPT)
            .collect();
        entries.reverse();
        if text.lines().count() > 2 * Self::KEPT
            && let Err(failure) = jsonl::write(&file, &entries)
        {
            warn!("could not trim {file}: {failure}");
        }
        Self {
            entries,
            file: Some(files),
            unsaved: Vec::new(),
        }
    }

    /// Puts `prompts`, in order, as the newest, moving any already there,
    /// without saving them: the prompts of a conversation shown, which
    /// the file has unless its retention dropped them.
    pub fn seed(&mut self, prompts: impl IntoIterator<Item = String>) {
        for prompt in prompts {
            self.newest(prompt);
        }
    }

    /// Puts `prompt` as the newest, moving it if it is already there.
    fn newest(&mut self, prompt: String) {
        self.entries.retain(|entry| *entry != prompt);
        self.entries.push(prompt);
    }
}

/// Loads the history kept in the project's [`PluginData`], before any
/// plugin's `Startup` system edits it.
pub(crate) fn load(data: Option<Res<PluginData>>, mut history: ResMut<PromptHistory>) {
    if let Some(data) = data {
        *history = PromptHistory::load(data.project.clone());
    }
}

/// Seeds the history with the typed prompts of the agent shown, once each
/// time another one is shown.
pub(crate) fn seed_shown(
    shown: Res<ShownAgent>,
    conversations: Query<&Conversation, With<Agent>>,
    mut seeded: Local<Option<Entity>>,
    mut history: ResMut<PromptHistory>,
) {
    if *seeded == shown.0 {
        return;
    }
    let Some((agent, conversation)) = shown
        .0
        .and_then(|agent| Some((agent, conversations.get(agent).ok()?)))
    else {
        return;
    };
    *seeded = Some(agent);
    history.seed(typed_prompts(conversation));
}

/// Where the terminal view records the prompts sent and saves them, in
/// `Update`, in this order, and seeds them in [`TuiKeys`](crate::TuiKeys).
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HistorySystems {
    /// Puts the prompts sent this frame as the newest of
    /// [`PromptHistory::entries`], moving any already there.
    Record,
    /// Saves the prompts recorded that are still there.
    Save,
    /// In `TuiKeys`, before each key's
    /// [`InputSystems::Filter`](crate::InputSystems::Filter): after
    /// another agent is shown, puts its conversation's typed prompts as
    /// the newest, so a key right after the one that showed it browses
    /// them. A filter runs after it and before
    /// [`InputSystems::Dispatch`](crate::InputSystems::Dispatch).
    Seed,
}

/// A prompt sent, which [`HistorySystems::Record`] puts as the newest of
/// the [`PromptHistory`]: the input writes one for each prompt it sends,
/// and a plugin writes one for a prompt it sent for the user, such as
/// from a panel of its own.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct Sent(pub String);

/// Adds each prompt sent to the history as the newest.
pub(crate) fn record(mut sent: MessageReader<Sent>, mut history: ResMut<PromptHistory>) {
    for Sent(text) in sent.read() {
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        history.newest(text.to_owned());
        history.unsaved.push(text.to_owned());
    }
}

/// Adds the prompts recorded since the last save that are still kept to
/// the file, then keeps the newest.
pub(crate) fn save(mut history: ResMut<PromptHistory>) {
    if history.unsaved.is_empty() && history.entries.len() <= PromptHistory::KEPT {
        return;
    }
    let unsaved = std::mem::take(&mut history.unsaved);
    let kept = unsaved.iter().filter(|text| history.entries.contains(text));
    if let Some(file) = &history.file
        && let Err(failure) = jsonl::append(&Kept(file.clone(), HISTORY_FILE), kept)
    {
        warn!("could not save the prompt history: {failure}");
    }
    let excess = history.entries.len().saturating_sub(PromptHistory::KEPT);
    history.entries.drain(..excess);
}

#[cfg(test)]
mod tests;
