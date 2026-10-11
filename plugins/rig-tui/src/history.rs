//! The prompt history: the prompts Up and Down browse, kept in a file
//! across sessions. It is the [`PromptHistory`] resource, which a plugin
//! fills, filters or stops saving, such as to keep secrets out of it.
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

use std::fs;
use std::io::Write as _;
use std::path::{Path, PathBuf};

use bevy_ecs::prelude::*;
use bevy_log::warn;

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
            entries: load(&file),
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

/// A prompt the input sent.
#[derive(Message, Clone, Debug)]
pub(crate) struct Sent(pub(crate) String);

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
    let lines: String = kept
        .filter_map(|text| serde_json::to_string(text).ok())
        .map(|line| line + "\n")
        .collect();
    if let Some(file) = &history.file
        && !lines.is_empty()
        && let Err(failure) = append(file, &lines)
    {
        warn!("could not save the prompt history: {failure}");
    }
    let excess = history.entries.len().saturating_sub(KEPT);
    history.entries.drain(..excess);
}

/// Appends `lines` to `file`, making its directory.
fn append(file: &Path, lines: &str) -> std::io::Result<()> {
    if let Some(parent) = file.parent() {
        fs::create_dir_all(parent)?;
    }
    let mut out = fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(file)?;
    out.write_all(lines.as_bytes())
}

/// The last [`KEPT`] prompts of `file`, rewriting it when it grew past
/// twice that.
fn load(file: &Path) -> Vec<String> {
    let Ok(text) = fs::read_to_string(file) else {
        return Vec::new();
    };
    let mut history: Vec<String> = text
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    let total = history.len();
    history.drain(..total.saturating_sub(KEPT));
    if total > 2 * KEPT {
        let kept: String = history
            .iter()
            .filter_map(|prompt| serde_json::to_string(prompt).ok())
            .map(|line| line + "\n")
            .collect();
        let temporary = file.with_extension("jsonl.new");
        if let Err(failure) =
            fs::write(&temporary, kept).and_then(|()| fs::rename(&temporary, file))
        {
            warn!("could not trim the prompt history: {failure}");
        }
    }
    history
}

#[cfg(test)]
mod tests;
