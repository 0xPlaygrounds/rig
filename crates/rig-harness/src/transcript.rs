//! A plugin's entries in an agent's transcript: a [`TranscriptEntry`]
//! entity [`EntryOf`] the agent is drawn by every front that shows the
//! agent, after the message it follows, as data any front reads, the way
//! an [`Ask`](crate::ask::Ask) is. It is not part of the conversation and
//! never reaches a model. The harness keeps the agent's entries with the
//! session, so they come back on `/resume` and after `/reload`, in a fork
//! too; one with `saved: false` lives only as long as the app.
//!
//! ```
//! use rig_harness::prelude::*;
//! use rig_harness::transcript::{EntryOf, TranscriptEntry};
//!
//! /// Shows a side question's answer after the agent's newest message.
//! fn answered(agent: Entity, conversations: Query<&Conversation>, mut commands: Commands) {
//!     let Ok(conversation) = conversations.get(agent) else {
//!         return;
//!     };
//!     let last = conversation.messages().len().checked_sub(1);
//!     let after = last.and_then(|last| conversation.id(last));
//!     let entry = TranscriptEntry::new("btw", after, "Why tabs?", "The project's style says so.");
//!     commands.spawn((entry, EntryOf(agent)));
//! }
//! ```
//!
//! A front draws an entry with its origin's look; the terminal view lets a
//! plugin choose that look (rig-tui's `add_entry_renderer`).

use std::collections::BTreeSet;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_cassette::journal::MessageId;
use rig_ecs::StopTurns;
use rig_ecs::agent::Agent;
use rig_ecs::journal::ReflectSaved;
use serde::{Deserialize, Serialize};

/// A plugin's entry in an agent's transcript ([`EntryOf`] it): every front
/// that shows the agent draws it after the message it follows. It is not
/// part of the conversation and never reaches a model. The harness saves it
/// with the session unless `saved` is false.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[reflect(Component, Clone, Debug, PartialEq)]
#[require(Name = Name::new("transcript entry"))]
pub struct TranscriptEntry {
    /// Who it is from, such as `btw`: a front draws it with that origin's
    /// look.
    pub origin: String,
    /// The message it follows; `None` puts it before the first. While an
    /// edit such as a compaction has taken that message out, it follows
    /// the message that stands for it.
    pub after: Option<MessageId>,
    /// What it shows collapsed, in a line.
    pub title: String,
    /// What it shows expanded.
    pub text: String,
    /// Whether it is kept with the session.
    pub saved: bool,
}

impl TranscriptEntry {
    /// An entry from `origin` after the message `after`, titled `title`,
    /// showing `text` expanded, kept with the session.
    pub fn new(
        origin: impl Into<String>,
        after: Option<MessageId>,
        title: impl Into<String>,
        text: impl Into<String>,
    ) -> Self {
        Self {
            origin: origin.into(),
            after,
            title: title.into(),
            text: text.into(),
            saved: true,
        }
    }
}

/// On a [`TranscriptEntry`]: the agent whose transcript it is in.
/// Despawning the agent despawns it.
#[derive(Component, Debug)]
#[relationship(relationship_target = TranscriptEntries)]
pub struct EntryOf(pub Entity);

/// An agent's entries, in the order they were added.
#[derive(Component, Debug)]
#[relationship_target(relationship = EntryOf, linked_spawn)]
pub struct TranscriptEntries(Vec<Entity>);

/// The saved entries of an agent: kept by the harness in step with its
/// [`TranscriptEntries`], and spawned back as entries when it is restored.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq)]
#[reflect(Component, Saved, Default)]
struct SavedEntries(Vec<TranscriptEntry>);

/// Keeps every agent's entries with the session.
pub struct TranscriptPlugin;

impl Plugin for TranscriptPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(Last, keep_saved_entries.before(StopTurns))
            .add_observer(restore_entries);
    }
}

/// Brings each agent's [`SavedEntries`] in step with its saved entries,
/// for the agents whose entries were added, changed or removed; an
/// unchanged list is not logged again.
fn keep_saved_entries(
    changed: Query<&EntryOf, Changed<TranscriptEntry>>,
    (regrouped, mut emptied): (
        Query<Entity, Changed<TranscriptEntries>>,
        RemovedComponents<TranscriptEntries>,
    ),
    mut agents: Query<(Option<&TranscriptEntries>, Option<&mut SavedEntries>), With<Agent>>,
    entries: Query<&TranscriptEntry>,
    mut commands: Commands,
) {
    let touched: BTreeSet<Entity> = changed
        .iter()
        .map(|&EntryOf(agent)| agent)
        .chain(regrouped.iter())
        .chain(emptied.read())
        .collect();
    for agent in touched {
        let Ok((list, saved)) = agents.get_mut(agent) else {
            continue;
        };
        let kept: Vec<TranscriptEntry> = list
            .into_iter()
            .flat_map(TranscriptEntries::iter)
            .filter_map(|entry| entries.get(entry).ok())
            .filter(|entry| entry.saved)
            .cloned()
            .collect();
        match saved {
            Some(mut saved) => {
                saved.set_if_neq(SavedEntries(kept));
            }
            None if kept.is_empty() => {}
            None => {
                commands.entity(agent).try_insert(SavedEntries(kept));
            }
        }
    }
}

/// Spawns a restored agent's saved entries again. An agent that has
/// entries already is not a restored one: its list was just saved.
fn restore_entries(
    inserted: On<Insert<SavedEntries>>,
    agents: Query<(&SavedEntries, Has<TranscriptEntries>)>,
    mut commands: Commands,
) {
    let agent = inserted.entity;
    let Ok((SavedEntries(saved), false)) = agents.get(agent) else {
        return;
    };
    for entry in saved {
        commands.spawn((entry.clone(), EntryOf(agent)));
    }
}

#[cfg(test)]
mod tests;
