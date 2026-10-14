//! The prompt history: the prompts Up and Down browse, kept across
//! sessions in the user's
//! [`PluginData`](rig_harness::prelude::PluginData) as `tui/history.jsonl`. It is the [`PromptHistory`] resource, which a plugin
//! fills, filters or stops saving, such as to keep secrets out of it; a
//! [`Sent`] adds a prompt to it.
//!
//! A prompt is kept as typed, a slash command's included. When the session
//! is rewritten ([`SessionRewritten`](rig_ecs::session_log::SessionRewritten)), the history, in memory and in its
//! file, goes through the same map: a redaction reaches every prompt that
//! held what it removed, those of other sessions too, and so the
//! `/redact <text>` line that asked for it. A command whose arguments must
//! never be kept, even before a rewrite runs, takes them out of `entries`
//! between [`HistorySystems::Record`] and [`HistorySystems::Save`], as
//! below.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//! use rig_tui::{HistorySystems, PromptHistory};
//!
//! /// Prompts that look like they hold a key are never kept.
//! fn scrub(mut history: ResMut<PromptHistory>) {
//!     history.entries.retain(|prompt| !prompt.contains("sk-"));
//! }
//!
//! fn build(app: &mut App) {
//!     app.add_systems(Update, scrub.after(HistorySystems::Record).before(HistorySystems::Save));
//! }
//! ```

use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_log::warn;
use rig_harness::prelude::{Files, PluginData};
use rig_harness::rig_memory::{Place, Rewrite, rewrite_file, rewrite_value};

use super::jsonl::{self, Kept};

/// Prompts kept in the history.
const KEPT: usize = 500;

/// The file of the history in its [`PromptHistory::file`].
const HISTORY_FILE: &str = "tui/history.jsonl";

/// The prompts Up and Down browse, oldest first, and the files new ones
/// are added to; `file: None` saves nothing. The terminal view loads it
/// from the user's [`PluginData`] in `PreStartup`, appends each prompt
/// sent in [`HistorySystems::Record`], and in [`HistorySystems::Save`] adds
/// to the file those recorded since the last save that are still in
/// `entries`, then keeps the newest 500. A plugin edits `entries` between
/// the two to filter what is kept, or replaces them.
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
    /// The newest prompts kept in `files`, which new ones are saved to.
    pub fn load(files: Arc<dyn Files>) -> Self {
        Self {
            entries: jsonl::tail(&Kept(files.clone(), HISTORY_FILE), KEPT),
            file: Some(files),
            unsaved: Vec::new(),
        }
    }
}

impl PromptHistory {
    /// Maps every prompt, in memory and in the file, through `with`.
    pub(crate) fn rewrite(&mut self, with: &dyn Rewrite) {
        let at = Place::File { name: HISTORY_FILE };
        for prompt in self.entries.iter_mut().chain(&mut self.unsaved) {
            // A map that leaves no string leaves no prompt.
            if rewrite_value(with, &at, HISTORY_FILE, prompt).is_err() {
                prompt.clear();
            }
        }
        if let Some(file) = &self.file
            && let Err(failure) = rewrite_file(&**file, HISTORY_FILE, with)
        {
            warn!("could not rewrite the prompt history: {failure}");
        }
    }
}

/// Loads the history kept in the user's [`PluginData`], before any
/// plugin's `Startup` system edits it.
pub(crate) fn load(data: Option<Res<PluginData>>, mut history: ResMut<PromptHistory>) {
    if let Some(data) = data {
        *history = PromptHistory::load(data.user.clone());
    }
}

/// Where the terminal view records the prompts sent and saves them, in
/// `Update`, in this order.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum HistorySystems {
    /// Appends the prompts sent this frame to [`PromptHistory::entries`].
    Record,
    /// Saves the prompts recorded that are still there.
    Save,
}

/// A prompt sent, which [`HistorySystems::Record`] adds to the
/// [`PromptHistory`] unless it repeats the last one: the input writes one
/// for each prompt it sends, and a plugin writes one for a prompt it sent
/// for the user, such as from a panel of its own.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct Sent(pub String);

/// Adds each prompt sent to the history, unless it repeats the last one.
pub(crate) fn record(mut sent: MessageReader<Sent>, mut history: ResMut<PromptHistory>) {
    for Sent(text) in sent.read() {
        let text = text.trim();
        if text.is_empty() || history.entries.last().is_some_and(|last| last == text) {
            continue;
        }
        history.entries.push(text.to_owned());
        history.unsaved.push(text.to_owned());
    }
}

/// Adds the prompts recorded since the last save that are still kept to
/// the file, then keeps the newest.
pub(crate) fn save(mut history: ResMut<PromptHistory>) {
    if history.unsaved.is_empty() && history.entries.len() <= KEPT {
        return;
    }
    let unsaved = std::mem::take(&mut history.unsaved);
    let kept = unsaved.iter().filter(|text| history.entries.contains(text));
    if let Some(file) = &history.file
        && let Err(failure) = jsonl::append(&Kept(file.clone(), HISTORY_FILE), kept)
    {
        warn!("could not save the prompt history: {failure}");
    }
    let excess = history.entries.len().saturating_sub(KEPT);
    history.entries.drain(..excess);
}

#[cfg(test)]
mod tests;
