//! Streaming values for the classic agent runtime.
//!
//! The streaming entry points are inherent on [`Agent`](crate::agent::Agent):
//! [`stream_prompt`](crate::agent::Agent::stream_prompt) and
//! [`stream_chat`](crate::agent::Agent::stream_chat) return the runner, whose
//! [`stream`](crate::agent::AgentRunner::stream) yields
//! [`MultiTurnStreamItem`](crate::agent::MultiTurnStreamItem)s.

pub use rig_core::streaming::*;
