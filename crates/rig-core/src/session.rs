//! Backend-neutral durable agent session events.

use crate::{
    completion::{Message, Usage},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

/// Typed durable event payload.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum SessionEventKind {
    /// Conversation message.
    Message(Message),
    /// Model call metadata.
    ModelCall {
        model: Option<String>,
        config: Value,
        usage: Usage,
    },
    /// Tool call with ancestry.
    ToolCall {
        name: String,
        arguments: Value,
        internal_call_id: String,
        parent_internal_call_id: Option<String>,
    },
    /// Model response metadata, preserving normalized/raw finish reasons.
    ModelResponse {
        usage: Usage,
        finish_reason: Option<crate::runtime::TerminalReason>,
        raw_finish_reason: Option<String>,
    },
    /// Tool result.
    ToolResult {
        outcome: String,
        content: Value,
        internal_call_id: String,
        parent_internal_call_id: Option<String>,
    },
    /// Run lifecycle transition.
    Lifecycle {
        status: String,
        detail: Option<Value>,
    },
    /// Serialized safe-boundary checkpoint.
    Checkpoint { snapshot: String },
    /// Named bookmark/checkpoint.
    Bookmark {
        name: String,
        checkpoint: Option<String>,
    },
    /// Non-destructive compaction summary.
    Compaction {
        through_sequence: u64,
        summary: String,
    },
    /// Interrupted run recovery marker.
    Interrupted {
        run_id: String,
        checkpoint: Option<String>,
    },
    /// Host-defined entry.
    Custom { kind: String, payload: Value },
}

/// Ordered event returned by a session store.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct SessionEvent {
    /// Session identifier.
    pub session_id: String,
    /// Branch identifier.
    pub branch: String,
    /// Branch-local monotonic sequence.
    pub sequence: u64,
    /// Parent sequence when branching or correlating.
    pub parent_sequence: Option<u64>,
    /// Run/call correlation identifier.
    pub correlation_id: Option<String>,
    /// Typed payload.
    pub kind: SessionEventKind,
}

/// Session backend error.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SessionError {
    /// Backend failure.
    #[error("session backend error: {0}")]
    Backend(String),
    /// A durable checkpoint could not be decoded.
    #[error("invalid durable checkpoint: {0}")]
    InvalidCheckpoint(String),
    /// The latest interruption occurred at an unsafe in-flight boundary.
    #[error("interrupted run has no safely resumable checkpoint")]
    UnsafeRecovery,
}

/// Append-only branchable durable session store.
pub trait SessionStore: WasmCompatSend + WasmCompatSync {
    /// Load a branch in sequence order.
    fn load<'a>(
        &'a self,
        session: &'a str,
        branch: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<SessionEvent>, SessionError>>;
    /// Atomically append and assign sequence.
    fn append<'a>(
        &'a self,
        session: &'a str,
        branch: &'a str,
        parent: Option<u64>,
        correlation: Option<String>,
        kind: SessionEventKind,
    ) -> WasmBoxedFuture<'a, Result<SessionEvent, SessionError>>;
    /// Create a branch from an inclusive prefix.
    fn branch<'a>(
        &'a self,
        session: &'a str,
        source: &'a str,
        target: &'a str,
        through: u64,
    ) -> WasmBoxedFuture<'a, Result<(), SessionError>>;
}
