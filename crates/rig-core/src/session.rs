//! Backend-neutral append-only agent session event storage.
//!
//! Sessions intentionally remain separate from [`ConversationMemory`](crate::memory::ConversationMemory):
//! memory shapes model-visible messages, while this log preserves operational
//! history, branches, checkpoints, usage, and interrupted turns.

use std::{
    collections::HashMap,
    sync::{Arc, Mutex},
};

use serde::{Deserialize, Serialize};

use crate::{
    completion::{Message, Usage},
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

/// Stable host/session identifier.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionId(pub String);

/// Identity of one append-only event.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct SessionEventId(pub String);

/// Versioned session event payload.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SessionEventKind {
    /// Model-visible conversation message.
    Message { message: Message },
    /// A model request began.
    ModelCallStarted {
        model: String,
        config: serde_json::Value,
    },
    /// A model request completed with usage and normalized provider metadata.
    ModelCallFinished {
        usage: Usage,
        metadata: serde_json::Value,
    },
    /// Tool execution began.
    ToolCall {
        name: String,
        internal_call_id: String,
        args: serde_json::Value,
    },
    /// Tool execution settled.
    ToolResult {
        internal_call_id: String,
        outcome: String,
        result: serde_json::Value,
    },
    /// Named resumable safe boundary.
    Checkpoint { name: String },
    /// Compaction summary retaining a link to full history.
    Compaction {
        summary: String,
        through: Option<SessionEventId>,
    },
    /// Host-defined versioned event.
    Custom {
        namespace: String,
        payload: serde_json::Value,
    },
    /// A turn was interrupted before normal completion.
    Interrupted { reason: String },
}

/// One record in a session tree.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SessionEvent {
    /// Event identity.
    pub id: SessionEventId,
    /// Session containing the event.
    pub session_id: SessionId,
    /// Parent event; choosing an earlier parent creates a branch.
    pub parent_id: Option<SessionEventId>,
    /// Milliseconds since Unix epoch supplied by the host/store.
    pub timestamp_ms: u64,
    /// Event payload.
    pub kind: SessionEventKind,
}

/// Boxed session backend failure.
#[cfg(not(target_family = "wasm"))]
pub type SessionBackendError = Box<dyn std::error::Error + Send + Sync + 'static>;
/// Boxed session backend failure.
#[cfg(target_family = "wasm")]
pub type SessionBackendError = Box<dyn std::error::Error + 'static>;

/// Session storage failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SessionStoreError {
    /// Backend operation failed.
    #[error("session backend error: {0}")]
    Backend(SessionBackendError),
    /// Imported data violated tree invariants.
    #[error("invalid session event: {0}")]
    Invalid(String),
}

/// Append-only backend-neutral session/event store.
pub trait SessionEventStore: WasmCompatSend + WasmCompatSync {
    /// Append an event, returning its allocated identity.
    fn append<'a>(
        &'a self,
        session_id: &'a SessionId,
        parent_id: Option<SessionEventId>,
        kind: SessionEventKind,
    ) -> WasmBoxedFuture<'a, Result<SessionEventId, SessionStoreError>>;

    /// Load every event in append order. Parent IDs reconstruct branches.
    fn load<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> WasmBoxedFuture<'a, Result<Vec<SessionEvent>, SessionStoreError>>;

    /// Import already-versioned events, preserving identities.
    fn import<'a>(
        &'a self,
        session_id: &'a SessionId,
        events: Vec<SessionEvent>,
    ) -> WasmBoxedFuture<'a, Result<(), SessionStoreError>>;

    /// Delete a session and all branches.
    fn clear<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> WasmBoxedFuture<'a, Result<(), SessionStoreError>>;
}

/// Thread-safe process-local reference implementation of [`SessionEventStore`].
#[derive(Clone, Default)]
pub struct InMemorySessionEventStore {
    sessions: Arc<Mutex<HashMap<SessionId, Vec<SessionEvent>>>>,
}

impl SessionEventStore for InMemorySessionEventStore {
    fn append<'a>(
        &'a self,
        session_id: &'a SessionId,
        parent_id: Option<SessionEventId>,
        kind: SessionEventKind,
    ) -> WasmBoxedFuture<'a, Result<SessionEventId, SessionStoreError>> {
        Box::pin(async move {
            let mut sessions = self
                .sessions
                .lock()
                .unwrap_or_else(|error| error.into_inner());
            let events = sessions.entry(session_id.clone()).or_default();
            if let Some(parent) = &parent_id
                && !events.iter().any(|event| &event.id == parent)
            {
                return Err(SessionStoreError::Invalid(format!(
                    "unknown parent event `{}`",
                    parent.0
                )));
            }
            let id = SessionEventId(crate::id::generate());
            let timestamp_ms = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map_or(0, |duration| duration.as_millis() as u64);
            events.push(SessionEvent {
                id: id.clone(),
                session_id: session_id.clone(),
                parent_id,
                timestamp_ms,
                kind,
            });
            Ok(id)
        })
    }

    fn load<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> WasmBoxedFuture<'a, Result<Vec<SessionEvent>, SessionStoreError>> {
        Box::pin(async move {
            Ok(self
                .sessions
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .get(session_id)
                .cloned()
                .unwrap_or_default())
        })
    }

    fn import<'a>(
        &'a self,
        session_id: &'a SessionId,
        events: Vec<SessionEvent>,
    ) -> WasmBoxedFuture<'a, Result<(), SessionStoreError>> {
        Box::pin(async move {
            let mut known = std::collections::HashSet::new();
            for event in &events {
                if &event.session_id != session_id {
                    return Err(SessionStoreError::Invalid(
                        "event belongs to another session".into(),
                    ));
                }
                if let Some(parent) = &event.parent_id
                    && !known.contains(parent)
                {
                    return Err(SessionStoreError::Invalid(format!(
                        "parent `{}` precedes no imported event",
                        parent.0
                    )));
                }
                if !known.insert(event.id.clone()) {
                    return Err(SessionStoreError::Invalid(format!(
                        "duplicate event `{}`",
                        event.id.0
                    )));
                }
            }
            self.sessions
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .insert(session_id.clone(), events);
            Ok(())
        })
    }

    fn clear<'a>(
        &'a self,
        session_id: &'a SessionId,
    ) -> WasmBoxedFuture<'a, Result<(), SessionStoreError>> {
        Box::pin(async move {
            self.sessions
                .lock()
                .unwrap_or_else(|error| error.into_inner())
                .remove(session_id);
            Ok(())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn append_and_branch_preserve_parent_links() {
        let store = InMemorySessionEventStore::default();
        let session = SessionId("session".into());
        let root = store
            .append(
                &session,
                None,
                SessionEventKind::Message {
                    message: Message::user("root"),
                },
            )
            .await
            .expect("append root");
        let left = store
            .append(
                &session,
                Some(root.clone()),
                SessionEventKind::Checkpoint {
                    name: "left".into(),
                },
            )
            .await
            .expect("append left");
        let right = store
            .append(
                &session,
                Some(root.clone()),
                SessionEventKind::Checkpoint {
                    name: "right".into(),
                },
            )
            .await
            .expect("append right");

        let events = store.load(&session).await.expect("load");
        assert_eq!(events.len(), 3);
        assert_eq!(
            events.get(1).and_then(|event| event.parent_id.as_ref()),
            Some(&root)
        );
        assert_eq!(
            events.get(2).and_then(|event| event.parent_id.as_ref()),
            Some(&root)
        );
        assert_ne!(left, right);
    }
}
