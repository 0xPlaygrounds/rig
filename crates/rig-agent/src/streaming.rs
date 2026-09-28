//! Portable streaming types re-exported for classic runtime users.
//!
//! ```
//! use rig_agent::streaming::{Item, StreamEvent};
//!
//! fn text(item: &Item<StreamEvent>) -> Option<&str> {
//!     match item {
//!         Item::Event(StreamEvent::Text { text, .. }) => Some(text),
//!         _ => None,
//!     }
//! }
//! # let _ = text;
//! ```

pub use rig_core::streaming::*;
