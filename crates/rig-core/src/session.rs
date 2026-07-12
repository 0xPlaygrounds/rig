//! Backend-neutral append-only agent session records.
//!
//! Session storage intentionally remains separate from
//! [`ConversationMemory`](crate::memory::ConversationMemory): memory supplies
//! model prompt history, while this module preserves attempts, branches,
//! checkpoints, compaction records, and host events.

use std::{
    sync::{Arc, Mutex},
    time::{SystemTime, UNIX_EPOCH},
};

use serde::{Deserialize, Serialize};

use crate::{
    completion::Message,
    wasm_compat::{WasmBoxedFuture, WasmCompatSend, WasmCompatSync},
};

/// Typed payload retained in an append-only session.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
#[non_exhaustive]
pub enum SessionEventData {
    /// Conversation message accepted into history.
    Message(Message),
    /// Model-call request/response metadata owned by the host.
    ModelCall(serde_json::Value),
    /// Tool invocation metadata.
    ToolCall(serde_json::Value),
    /// Tool result metadata.
    ToolResult(serde_json::Value),
    /// Non-destructive compaction summary.
    Compaction {
        summary: String,
        covered_event_ids: Vec<String>,
    },
    /// Named durable checkpoint, optionally carrying an `AgentRun` snapshot.
    Checkpoint {
        name: String,
        snapshot: Option<serde_json::Value>,
    },
    /// Bookmark pointing at another event.
    Bookmark { name: String, event_id: String },
    /// Host-defined typed entry.
    Custom {
        kind: String,
        value: serde_json::Value,
    },
}

/// Input for appending one event.
#[derive(Clone, Debug)]
pub struct NewSessionEvent {
    /// Session receiving the event.
    pub session_id: String,
    /// Parent event; choosing an older parent creates a branch.
    pub parent_id: Option<String>,
    /// Event payload.
    pub data: SessionEventData,
    /// Optional model/config/usage metadata.
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

impl NewSessionEvent {
    /// Construct an event at the current active leaf.
    pub fn new(session_id: impl Into<String>, data: SessionEventData) -> Self {
        Self {
            session_id: session_id.into(),
            parent_id: None,
            data,
            metadata: Default::default(),
        }
    }

    /// Explicitly branch from a parent event.
    pub fn with_parent(mut self, parent_id: impl Into<String>) -> Self {
        self.parent_id = Some(parent_id.into());
        self
    }
}

/// Stored append-only session event.
#[derive(Clone, Debug, Deserialize, Serialize, PartialEq)]
pub struct SessionEvent {
    /// Stable event identifier.
    pub id: String,
    /// Session identifier.
    pub session_id: String,
    /// Parent event in the branch graph.
    pub parent_id: Option<String>,
    /// Unix timestamp in milliseconds.
    pub timestamp_ms: u64,
    /// Typed payload.
    pub data: SessionEventData,
    /// Host metadata.
    pub metadata: serde_json::Map<String, serde_json::Value>,
}

/// Session backend failure.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SessionStoreError {
    /// Referenced session/event does not exist.
    #[error("session record not found: {0}")]
    NotFound(String),
    /// Backend failure.
    #[error("session backend error: {0}")]
    Backend(String),
}

/// Append-only backend contract for resumable, branchable sessions.
pub trait SessionStore: WasmCompatSend + WasmCompatSync {
    /// Append one event atomically and return its stored identity.
    fn append(
        &self,
        event: NewSessionEvent,
    ) -> WasmBoxedFuture<'_, Result<SessionEvent, SessionStoreError>>;

    /// Read all events for a session in append order, retaining every branch.
    fn events<'a>(
        &'a self,
        session_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<SessionEvent>, SessionStoreError>>;

    /// Resolve an event by identifier.
    fn get<'a>(
        &'a self,
        event_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Option<SessionEvent>, SessionStoreError>>;
}

impl<S> SessionStore for Arc<S>
where
    S: SessionStore + ?Sized,
{
    fn append(
        &self,
        event: NewSessionEvent,
    ) -> WasmBoxedFuture<'_, Result<SessionEvent, SessionStoreError>> {
        (**self).append(event)
    }
    fn events<'a>(
        &'a self,
        id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<SessionEvent>, SessionStoreError>> {
        (**self).events(id)
    }
    fn get<'a>(
        &'a self,
        id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Option<SessionEvent>, SessionStoreError>> {
        (**self).get(id)
    }
}

/// In-process reference session store.
#[derive(Clone, Default)]
pub struct InMemorySessionStore {
    events: Arc<Mutex<Vec<SessionEvent>>>,
}

impl SessionStore for InMemorySessionStore {
    fn append(
        &self,
        event: NewSessionEvent,
    ) -> WasmBoxedFuture<'_, Result<SessionEvent, SessionStoreError>> {
        Box::pin(async move {
            let mut events = self.events.lock().unwrap_or_else(|e| e.into_inner());
            if let Some(parent) = event.parent_id.as_ref()
                && !events
                    .iter()
                    .any(|stored| &stored.id == parent && stored.session_id == event.session_id)
            {
                return Err(SessionStoreError::NotFound(parent.clone()));
            }
            let stored = SessionEvent {
                id: crate::id::generate(),
                session_id: event.session_id,
                parent_id: event.parent_id,
                timestamp_ms: SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_millis() as u64,
                data: event.data,
                metadata: event.metadata,
            };
            events.push(stored.clone());
            Ok(stored)
        })
    }

    fn events<'a>(
        &'a self,
        session_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<SessionEvent>, SessionStoreError>> {
        Box::pin(async move {
            Ok(self
                .events
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .filter(|event| event.session_id == session_id)
                .cloned()
                .collect())
        })
    }

    fn get<'a>(
        &'a self,
        event_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Option<SessionEvent>, SessionStoreError>> {
        Box::pin(async move {
            Ok(self
                .events
                .lock()
                .unwrap_or_else(|e| e.into_inner())
                .iter()
                .find(|event| event.id == event_id)
                .cloned())
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[allow(clippy::panic_in_result_fn)]
    async fn appends_and_branches_without_destroying_history() -> Result<(), SessionStoreError> {
        let store = InMemorySessionStore::default();
        let root = store
            .append(NewSessionEvent::new(
                "s",
                SessionEventData::Message(Message::user("root")),
            ))
            .await?;
        store
            .append(
                NewSessionEvent::new("s", SessionEventData::Message(Message::user("a")))
                    .with_parent(root.id.clone()),
            )
            .await?;
        store
            .append(
                NewSessionEvent::new("s", SessionEventData::Message(Message::user("b")))
                    .with_parent(root.id),
            )
            .await?;
        assert_eq!(store.events("s").await?.len(), 3);
        Ok(())
    }
}
