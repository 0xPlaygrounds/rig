//! The native session views, for any front: [`SessionPicker`] lists saved
//! sessions, the lines of one session, or the entries of one line, in a
//! plain [`Picker`] the harness fills in place as items are read off the
//! main thread, newest first, and previews the highlighted item
//! ([`PickPreview`](crate::pick::PickPreview)). A plugin opens one with its
//! own source, filter, labels and command:
//!
//! ```no_run
//! use rig_harness::prelude::*;
//!
//! fn split_which(agent: Entity, mut commands: Commands) {
//!     commands.spawn((
//!         Picker::new("Split which session?").for_agent(agent),
//!         SessionPicker {
//!             command: "split new --from {session}".into(),
//!             filter: SessionFilter { running: true, ..default() },
//!             ..default()
//!         },
//!     ));
//! }
//! ```

use std::path::PathBuf;
use std::time::SystemTime;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig::harness_protocol::{SessionDir, SessionId};
use rig_core::conversation::{AsideId, EntryRef, MessageId};
use rig_memory::{Line, LineId, Listing};

use crate::host::session::SessionStores;
use crate::pick::Picker;

/// A saved session as a listing shows it.
#[derive(Clone, Debug)]
pub struct SessionEntry {
    /// Its id.
    pub id: SessionId,
    /// The name the user gave it.
    pub name: Option<String>,
    /// What its store says about it.
    pub listing: Listing,
    /// The directory it runs in, if known.
    pub directory: Option<PathBuf>,
    /// When it last changed.
    pub updated: SystemTime,
    /// Whether a rig runs it now, this one or another.
    pub running: bool,
}

impl SessionEntry {
    /// The saved session in `dir`, read from its store as `stores` opens it
    /// and from its directory's files; `None` when it is not a saved
    /// session.
    pub fn read(dir: &SessionDir, stores: &SessionStores) -> Option<Self> {
        let _ = (dir, stores);
        None
    }

    /// One line: its name or title, its labels and its directory.
    pub fn label(&self) -> String {
        self.name.clone().unwrap_or_else(|| self.id.to_string())
    }

    /// What is shown whole beside the label: how long ago it changed, and
    /// the session it was split or forked from.
    pub fn detail(&self) -> String {
        String::new()
    }
}

/// The native session picker: put it on an entity (it requires a
/// [`Picker`]) and the harness fills the picker in place as items are
/// read, newest first, and previews the highlighted one when `preview` is
/// set. Picking runs `command` with `{session}`, `{agent}`, `{line}` and
/// `{entry}` replaced; with an empty `command`, read `Picked` (the item's
/// position is in `Chosen`) and [`SessionPicked`] on the same entity.
#[derive(Component, Clone, Debug)]
#[require(Picker)]
pub struct SessionPicker {
    /// What it lists.
    pub source: SessionSource,
    /// Which of them.
    pub filter: SessionFilter,
    /// The command picking an item runs, its placeholders replaced.
    pub command: String,
    /// The item's label and detail; `None` for the default labels.
    pub show: Option<fn(&SessionItem<'_>) -> (String, Option<String>)>,
    /// Whether the highlighted item is previewed.
    pub preview: bool,
}

/// The saved sessions, no filter, no command, the default labels, and a
/// preview.
impl Default for SessionPicker {
    fn default() -> Self {
        Self {
            source: SessionSource::Saved,
            filter: SessionFilter::default(),
            command: String::new(),
            show: None,
            preview: true,
        }
    }
}

/// One item of a session picker, whichever its source: what
/// [`SessionPicker::show`] labels.
#[derive(Clone, Copy, Debug)]
pub enum SessionItem<'a> {
    /// A saved session.
    Session(&'a SessionEntry),
    /// A line of a session.
    Line {
        /// The session.
        session: &'a SessionEntry,
        /// The line.
        line: &'a Line,
    },
    /// An entry of a line.
    Entry {
        /// The session.
        session: &'a SessionEntry,
        /// The line.
        line: &'a Line,
        /// The entry's position in the line.
        at: usize,
        /// The entry.
        entry: EntryRef<'a>,
    },
}

/// What a [`SessionPicker`] lists.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum SessionSource {
    /// The saved sessions of this home.
    Saved,
    /// These sessions.
    Sessions(Vec<SessionId>),
    /// The lines of an agent (the main one with `None`) of a session (this
    /// one with `None`): rig-memory's `lines`.
    Lines {
        /// The session.
        session: Option<SessionId>,
        /// The agent.
        agent: Option<String>,
    },
    /// The entries of a line, as one-line excerpts: pick a split point, a
    /// turn, a message.
    Entries {
        /// The session; this one with `None`.
        session: Option<SessionId>,
        /// The agent; the main one with `None`.
        agent: Option<String>,
        /// The line; the active one with `None`.
        line: Option<AsideId>,
    },
}

/// Which sessions a [`SessionPicker`] lists.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionFilter {
    /// Whether sessions a rig runs, this one included, are listed too.
    pub running: bool,
    /// Only the sessions that ran in this directory.
    pub project: Option<PathBuf>,
    /// Only this session's family: those it was forked or split from, and
    /// those forked or split from them.
    pub family: Option<SessionId>,
    /// Only the sessions whose label holds this text.
    pub text: String,
}

/// What was picked from a [`SessionPicker`], as data, put on its entity
/// before `Picked` is triggered: the session, agent, line and entry it
/// names.
#[derive(Component, Clone, Debug, Default, PartialEq, Eq)]
pub struct SessionPicked {
    /// The session; `None` for this one.
    pub session: Option<SessionId>,
    /// The agent, for a line or an entry.
    pub agent: Option<String>,
    /// The line, for a line or an entry.
    pub line: Option<LineId>,
    /// The entry.
    pub entry: Option<MessageId>,
}

/// Fills and previews every [`SessionPicker`].
pub struct SessionPickerPlugin;

impl Plugin for SessionPickerPlugin {
    fn build(&self, _app: &mut App) {}
}
