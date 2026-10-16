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
//!   is refused and nothing is written.
//! - [`history`], [`as_of`]: how a conversation got where it is.
//! - [`rewrite`] (what it promises is on it), [`fork`], [`split`], [`lines`]: rewriting,
//!   copying and splitting a saved session.
//! - [`SessionMeta`], [`ForkedFrom`], [`Listing`], [`read_line`]: what a session list shows,
//!   read without folding the logs.
//! - [`Store`], [`MemoryStore`], [`DirStore`] (feature `fs`): where a session is kept; [`Files`],
//!   [`DirFiles`] and [`PluginFiles`] for named files on their own.
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
#[cfg(feature = "fs")]
mod dir;
mod fork;
mod format;
mod listing;
mod meta;
pub mod recent;
mod rewrite;
mod saved;
mod session;
mod split;
mod store;
mod tokens;
mod writer;

pub use blobs::{load_images, store_images};
pub use compaction::{
    COMPACTED, ClearToolOutputs, Cleared, CompactReason, Compacted, CompactionOverrides,
    CompactionPolicy, Cut, CutRule, CutRules, Summarizer, SummaryCut, SummaryError, SummaryLimits,
    SummaryPlan, SummaryPrompts, SummaryState, TokenTally, TrackArgument, TrackedSet, Uncompacted,
    UndoError, apply_summary, estimate_tokens, summarized_prefix, summary_author, summary_in,
    summary_state, undo_summary,
};
#[cfg(feature = "fs")]
pub use dir::{DirFiles, DirStore};
pub use fork::{Forked, fork};
pub use format::{FORMAT, LogRewrite, Repair, SESSION, SessionError, Stamp};
pub use listing::{Listing, family_root, read_line};
pub use meta::{ForkedFrom, META, PARENT, SessionMeta, excerpt, title_of};
pub use rewrite::{
    Messages, NoRewrite, Place, Rewrite, Rewritten, RewrittenLog, Strings, map_strings, rewrite,
    rewrite_file, rewrite_value,
};
pub use saved::{MODEL, TITLE, main_order};
pub use session::{LogRecord, SavedAgent, Session, Unreadable, as_of, history, load, save};
pub use split::{Line, LineId, SPLIT, SplitOf, SplitPointer, lines, split};
pub use store::{Files, MemoryStore, PLUGINS, PluginFiles, Store, check_name};
pub use tokens::{HeuristicTokenCounter, TokenCounter};
pub use writer::Writer;
