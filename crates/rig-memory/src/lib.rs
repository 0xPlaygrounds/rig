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
//! [`fork`] copies a session as it was before one of its messages, and
//! [`history`] and [`as_of`] read how a conversation got where it is.
//! [`MemoryStore`] keeps a session in memory; with feature `fs`,
//! [`DirStore`] keeps it in a directory. Reading is strict: a session it
//! cannot read whole is refused, and nothing is written.
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
mod format;
mod session;
mod store;
mod tokens;
pub mod window;
mod writer;

pub use blobs::{load_images, store_images};
pub use compaction::{
    ClearToolOutputs, Cleared, CompactReason, CompactionPolicy, Cut, Summarizer, SummaryError,
    SummaryLimits, SummaryPlan, SummaryPrompts, SummaryState, TrackArgument, TrackedSet,
    apply_summary, summarized_prefix, summary_author, summary_in,
};
#[cfg(feature = "fs")]
pub use dir::DirStore;
pub use format::{FORMAT, LoadError, Repair, SESSION, Stamp, now_ms};
pub use session::{Record, SavedAgent, Session, Unreadable, as_of, fork, history, load, save};
pub use store::{Files, MemoryStore, Store, check_name};
pub use tokens::{HeuristicTokenCounter, TokenCounter};
pub use writer::Writer;
