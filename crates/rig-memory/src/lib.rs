#![deny(missing_docs)]
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
//! Conversations over time, over rig-core's [`Conversation`]:
//! - [`recent`]: which newest messages a request sends.
//! - [`CompactionPolicy`], [`apply_summary`], [`undo_summary`]: compacting a long conversation.
//! - [`load`], [`save`], [`Session`], [`Writer`]: saving and resuming, in the one format
//!   [`FORMAT`]. Reading is strict: a session of another format, or that does not read whole,
//!   is refused and nothing is written, unless [`migration`] brings it to the format first.
//!   [`load_async`] and [`save_async`] keep a conversation
//!   in any [`AsyncStore`], such as a database or IndexedDB.
//! - [`history`], [`as_of`]: how a conversation got where it is.
//! - [`rewrite`] (what it promises is on it): rewriting a saved session.
//! - [`SessionMeta`], [`read_conversation`]: what a session list shows, read without folding
//!   every log.
//! - [`Store`], [`MemoryStore`], [`DirStore`] (feature `fs`): where a session is kept; [`Files`],
//!   [`DirFiles`] and [`PluginFiles`] for named files on their own. Every [`Store`] is an
//!   [`AsyncStore`], its calls done when made. A log write appends only if the log is as long as
//!   the writer read it, the store checking and writing in one step, so of two saves that read
//!   one log the second is refused ([`SessionError::Changed`]) and writes nothing to it.
//!   `rig-sqlite` and `rig-postgres` keep sessions in a database with their `memory` feature, and
//!   `conformance` (feature `test-utils`) checks a store against the contract.
//!
//! ```
//! use rig_core::message::Message;
//! use rig_memory::recent::{LastMessages, view};
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
//! # Ok::<(), rig_memory::SessionError>(())
//! ```
//!
//! [`Conversation`]: rig_core::conversation::Conversation

mod blobs;
mod compaction;
#[cfg(any(test, feature = "test-utils"))]
pub mod conformance;
#[cfg(feature = "fs")]
mod dir;
mod fork;
mod format;
mod listing;
mod meta;
pub mod migration;
pub mod recent;
mod rewrite;
mod saved;
mod session;
mod store;
mod tokens;
mod writer;

pub use blobs::{BLOBS, load_images, store_images};
pub use compaction::{
    AFTER_OVERFLOW, COMPACTED, ClearPlan, ClearToolOutputs, Cleared, CompactReason, Compacted,
    CompactionPolicy, NearWindow, Summarizer, SummaryCut, SummaryError, SummaryLimits, SummaryPlan,
    SummaryPrompts, SummaryState, TrackArgument, TrackedSet, Uncompacted, UndoError, apply_summary,
    estimate_tokens, known_limits, summarized_in_log, summarized_prefix, summary_author,
    summary_in, summary_state, undo_summary,
};
#[cfg(feature = "fs")]
pub use dir::{DirFiles, DirStore};
pub use format::{FORMAT, LogRewrite, Repair, SESSION, SessionError, Stamp};
pub use listing::read_conversation;
pub use meta::{META, SessionMeta, excerpt, title_of};
pub use rewrite::{
    Messages, Place, Rewrite, Rewritten, RewrittenLog, Strings, map_strings, rewrite, rewrite_file,
    rewrite_value,
};
pub use saved::{MODEL, TITLE, main_order};
pub use session::{
    LogRecord, SavedAgent, Session, Unreadable, as_of, history, load, load_async, replay, save,
    save_async,
};
pub use store::{
    AsyncStore, Files, LogWriteError, MemoryStore, PLUGINS, PluginFiles, Store, check_log_name,
    check_name,
};
pub use tokens::{
    ATTACHMENT_TOKENS, Anchor, HeuristicTokenCounter, OUTPUT_MARGIN, Prepared, RequestStart,
    TURN_OUTPUT, TokenCounter, next_request_tokens,
};
pub use writer::Writer;
