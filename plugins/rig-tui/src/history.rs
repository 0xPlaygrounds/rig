//! The prompt history: the prompts Up and Down browse, kept in a file
//! across sessions. It is the [`PromptHistory`] resource, which a plugin
//! fills, filters or stops saving, such as to keep secrets out of it; a
//! [`Sent`] adds a prompt to it.
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

use std::path::PathBuf;

use bevy_ecs::prelude::*;
use bevy_log::warn;

use super::jsonl::{self, Kept};

/// Prompts kept in the history.
const KEPT: usize = 500;

/// The prompts Up and Down browse, oldest first, and the file new ones
/// are added to; `file: None` saves nothing. The terminal view loads it
/// from `RIG_HOME` when the app is built, appends each prompt sent in
/// [`HistorySystems::Record`], and in [`HistorySystems::Save`] adds to the
/// file those recorded since the last save that are still in `entries`,
/// then keeps the newest 500. A plugin edits `entries` between the two to
/// filter what is kept, or replaces them.
#[derive(Resource, Default, Debug)]
pub struct PromptHistory {
    /// The prompts, oldest first.
    pub entries: Vec<String>,
    /// Where prompts are saved, one JSON string per line.
    pub file: Option<PathBuf>,
    /// The prompts recorded since the last save.
    unsaved: Vec<String>,
}

impl PromptHistory {
    /// The newest prompts kept in `file`, which new ones are saved to.
    pub fn load(file: PathBuf) -> Self {
        Self {
            entries: jsonl::tail(&Kept::Plain(file.clone()), KEPT),
            file: Some(file),
            unsaved: Vec::new(),
        }
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
        && let Err(failure) = jsonl::append(&Kept::Plain(file.clone()), kept)
    {
        warn!("could not save the prompt history: {failure}");
    }
    let excess = history.entries.len().saturating_sub(KEPT);
    history.entries.drain(..excess);
}

#[cfg(test)]
mod tests;
