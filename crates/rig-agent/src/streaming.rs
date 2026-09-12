//! Streaming values for the classic agent runtime.
//!
//! Streaming is a terminal of the runner: [`Agent::prompt`](crate::agent::Agent::prompt)
//! returns it, and its [`stream`](crate::agent::AgentRunner::stream) yields
//! [`MultiTurnStreamItem`](crate::agent::MultiTurnStreamItem)s.

pub use rig_core::streaming::*;
