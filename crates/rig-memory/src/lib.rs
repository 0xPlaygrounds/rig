#![cfg_attr(
    test,
    allow(
        clippy::expect_used,
        clippy::indexing_slicing,
        clippy::panic,
        clippy::unwrap_used,
        clippy::unreachable
    )
)]
//! Conversations over time: what of a [`Conversation`] a request sends
//! ([`window`]), how a long one is compacted ([`CompactionPolicy`]), and
//! how one is saved and resumed.
//!
//! A session is kept in a [`Store`]: a log per agent, in the format
//! [`FORMAT`], the blobs its images are kept in, and its [`Files`] by
//! name. [`load`] and [`save`] keep one conversation; [`Session`] reads
//! and saves a whole session; a running host appends with a [`Writer`].
//! [`fork`] copies a session as it was before one of its messages, through
//! a [`Rewrite`], with what the copied messages' asides say now; and
//! [`history`] and [`as_of`] read how a conversation got where it is.
//! Beside its logs a store keeps what a listing shows of the session
//! ([`SessionMeta`]) and the session it was forked from ([`ForkedFrom`]).
//! A saved agent's model and title are kept under [`MODEL`] and [`TITLE`]
//! ([`SavedAgent::model`]), and [`Session::main_agent`] is the agent the
//! user talks to.
//! [`MemoryStore`] keeps a session in memory; with feature `fs`,
//! [`DirStore`] keeps it in a directory, and [`DirFiles`] keeps plain
//! named files under one. Reading is strict: a session it
//! cannot read whole is refused, and nothing is written.
//!
//! Saved history is appended to, and rewritten only through [`rewrite`]
//! (or [`Writer::rewrite`] for a running session): a map of every value
//! the session keeps ([`Rewrite`]; [`Strings`] for a redaction or a path
//! remap), each record kept in its place with its stamp. What it promises:
//! 1. In one process a rewrite and a writer's saves never interleave: they
//!    share the writer's lock, and the rewrite writes what was queued first.
//! 2. Each log is the old one or the new one, whole, to any reader, and
//!    through a crash. Across logs a reader may see some rewritten and some
//!    not; each always reads.
//! 3. A value read before the rewrite no longer continues a changed log:
//!    [`save`] of it gives [`LoadError::Changed`], and the caller reads it
//!    again.
//! 4. Another process must not rewrite a running session; rig-memory cannot
//!    tell, so a program checks first that nothing runs it (for a rig
//!    session, its `SessionDir::runs_elsewhere`).
//! 5. Values a host keeps in memory, and files it keeps from the session
//!    outside its store, are the host's to read again from [`Rewritten`]
//!    or to map with the same [`Rewrite`] ([`rewrite_value`],
//!    [`rewrite_file`]); rig's kernel does it for conversations and saved
//!    values, and each plugin for what it keeps.
//!
//! ```
//! use rig_core::message::Message;
//! use rig_memory::window::{LastMessages, view};
//! use rig_memory::{MemoryStore, load, save};
//!
//! let store = MemoryStore::default();
//! let mut conversation = load(&store, "user-123")?;
//! conversation.extend([Message::user("My name is Alice."), Message::assistant("Hi Alice!")]);
//! save(&store, "user-123", &mut conversation)?;
//!
//! let resumed = load(&store, "user-123")?;
//! assert_eq!(resumed, conversation);
//! assert_eq!(view(resumed.messages(), &LastMessages(1)).len(), 1);
//! # Ok::<(), rig_memory::LoadError>(())
//! ```
//!
//! [`Conversation`]: rig_core::conversation::Conversation

mod blobs;
mod compaction;
#[cfg(feature = "fs")]
mod dir;
mod fork;
mod format;
mod meta;
mod rewrite;
mod saved;
mod session;
mod store;
mod tokens;
pub mod window;
mod writer;

pub use blobs::{load_images, store_images};
pub use compaction::{
    COMPACTED, ClearToolOutputs, Cleared, CompactReason, Compacted, CompactionPolicy, Cut,
    Summarizer, SummaryError, SummaryLimits, SummaryPlan, SummaryPrompts, SummaryState,
    TrackArgument, TrackedSet, Uncompacted, UndoError, apply_summary, summarized_prefix,
    summary_author, summary_in, undo_summary,
};
#[cfg(feature = "fs")]
pub use dir::{DirFiles, DirStore};
pub use fork::{Forked, fork};
pub use format::{FORMAT, LoadError, Repair, Rewrote, SESSION, Stamp, now_ms};
pub use meta::{ForkedFrom, META, PARENT, SessionMeta, title_of};
pub use rewrite::{
    Keep, Messages, Place, Rewrite, Rewritten, RewrittenLog, Strings, map_strings, rewrite,
    rewrite_file, rewrite_value,
};
pub use saved::{MODEL, TITLE, main_order};
pub use session::{Record, SavedAgent, Session, Unreadable, as_of, history, load, save};
pub use store::{Files, MemoryStore, Store, check_name};
pub use tokens::{HeuristicTokenCounter, TokenCounter};
pub use writer::Writer;
