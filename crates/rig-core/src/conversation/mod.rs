//! Conversations: the rules a list of messages keeps so a model can read
//! it. [`pairing`] pairs each tool call with its result.
//!
//! ```
//! use rig_core::{conversation::pairing, message::Message};
//!
//! assert!(pairing::pending_calls(&[Message::user("Hello")]).is_empty());
//! ```

pub mod pairing;
