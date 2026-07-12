//! Durable SQLite conversation memory and branchable session event storage.

use rig_core::{
    completion::Message,
    memory::{ConversationMemory, MemoryError},
    session::{SessionError, SessionEvent, SessionEventKind, SessionStore},
    wasm_compat::WasmBoxedFuture,
};
use rusqlite::OptionalExtension;
use serde::{Deserialize, Serialize};
use serde_json::Value;
use tokio_rusqlite::Connection;

const MESSAGE_FORMAT_VERSION: i64 = 1;

fn memory_error(error: impl std::fmt::Display) -> MemoryError {
    MemoryError::backend(std::io::Error::other(error.to_string()))
}

/// Durable ordered [`ConversationMemory`] backed by SQLite.
#[derive(Clone)]
pub struct SqliteConversationMemory {
    connection: Connection,
}

impl SqliteConversationMemory {
    /// Open a database and apply idempotent schema migrations.
    pub async fn open(path: impl AsRef<std::path::Path>) -> Result<Self, MemoryError> {
        let connection = Connection::open(path).await.map_err(memory_error)?;
        Self::from_connection(connection).await
    }

    /// Use an existing connection and apply idempotent schema migrations.
    pub async fn from_connection(connection: Connection) -> Result<Self, MemoryError> {
        connection
            .call(|conn| {
                conn.execute_batch(
                    "PRAGMA foreign_keys=ON;
                     PRAGMA busy_timeout=5000;
                     CREATE TABLE IF NOT EXISTS rig_memory_messages (
                       conversation_id TEXT NOT NULL,
                       sequence INTEGER NOT NULL,
                       format_version INTEGER NOT NULL,
                       message_json TEXT NOT NULL,
                       PRIMARY KEY(conversation_id, sequence)
                     );
                     CREATE INDEX IF NOT EXISTS rig_memory_conversation_order
                       ON rig_memory_messages(conversation_id, sequence);",
                )?;
                Ok(())
            })
            .await
            .map_err(memory_error)?;
        Ok(Self { connection })
    }

    /// Access the shared SQLite connection.
    pub fn connection(&self) -> &Connection {
        &self.connection
    }
}

impl ConversationMemory for SqliteConversationMemory {
    fn load<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        let connection = self.connection.clone();
        let conversation_id = conversation_id.to_owned();
        Box::pin(async move {
            connection
                .call(move |conn| {
                    let mut statement = conn.prepare(
                        "SELECT format_version, message_json
                         FROM rig_memory_messages
                         WHERE conversation_id=?1 ORDER BY sequence ASC",
                    )?;
                    let rows = statement.query_map([conversation_id], |row| {
                        Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?))
                    })?;
                    let mut messages = Vec::new();
                    for row in rows {
                        let (version, json) = row?;
                        if version != MESSAGE_FORMAT_VERSION {
                            return Err(tokio_rusqlite::Error::Rusqlite(
                                rusqlite::Error::InvalidQuery,
                            ));
                        }
                        messages.push(serde_json::from_str(&json).map_err(|error| {
                            rusqlite::Error::FromSqlConversionFailure(
                                1,
                                rusqlite::types::Type::Text,
                                Box::new(error),
                            )
                        })?);
                    }
                    Ok(messages)
                })
                .await
                .map_err(memory_error)
        })
    }

    fn append<'a>(
        &'a self,
        conversation_id: &'a str,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        let connection = self.connection.clone();
        let conversation_id = conversation_id.to_owned();
        Box::pin(async move {
            let encoded = messages
                .into_iter()
                .map(|message| serde_json::to_string(&message))
                .collect::<Result<Vec<_>, _>>()
                .map_err(memory_error)?;
            connection
                .call(move |conn| {
                    let transaction =
                        conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                    let next: i64 = transaction.query_row(
                        "SELECT COALESCE(MAX(sequence) + 1, 0)
                         FROM rig_memory_messages WHERE conversation_id=?1",
                        [&conversation_id],
                        |row| row.get(0),
                    )?;
                    for (offset, json) in encoded.into_iter().enumerate() {
                        transaction.execute(
                            "INSERT INTO rig_memory_messages
                             (conversation_id, sequence, format_version, message_json)
                             VALUES (?1, ?2, ?3, ?4)",
                            rusqlite::params![
                                conversation_id,
                                next + offset as i64,
                                MESSAGE_FORMAT_VERSION,
                                json
                            ],
                        )?;
                    }
                    transaction.commit()?;
                    Ok(())
                })
                .await
                .map_err(memory_error)
        })
    }

    fn clear<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        let connection = self.connection.clone();
        let conversation_id = conversation_id.to_owned();
        Box::pin(async move {
            connection
                .call(move |conn| {
                    conn.execute(
                        "DELETE FROM rig_memory_messages WHERE conversation_id=?1",
                        [conversation_id],
                    )?;
                    Ok(())
                })
                .await
                .map_err(memory_error)
        })
    }
}

/// Typed payload stored in a durable session event.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", content = "data", rename_all = "snake_case")]
pub enum SessionEntry {
    /// A serialized conversation message.
    Message(Message),
    /// Model invocation metadata and usage.
    ModelCall {
        model: Option<String>,
        config: Value,
        usage: Value,
    },
    /// Tool invocation with parent correlation.
    ToolCall {
        name: String,
        arguments: Value,
        internal_call_id: String,
        parent_internal_call_id: Option<String>,
    },
    /// Model response metadata.
    ModelResponse {
        usage: Value,
        finish_reason: Option<String>,
        raw_finish_reason: Option<String>,
    },
    /// Tool result payload.
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
    /// Bookmark or named checkpoint.
    Bookmark {
        name: String,
        checkpoint: Option<String>,
    },
    /// Compaction summary. Earlier events remain intact.
    Compaction {
        through_sequence: u64,
        summary: String,
    },
    /// Application-defined event.
    Custom { kind: String, payload: Value },
    /// Marks an interrupted turn that may be recovered.
    Interrupted {
        run_id: String,
        checkpoint: Option<String>,
    },
}

/// Ordered event in a session branch.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct StoredSessionEvent {
    pub session_id: String,
    pub branch: String,
    pub sequence: u64,
    pub parent_sequence: Option<u64>,
    pub correlation_id: Option<String>,
    pub entry: SessionEntry,
}

/// Durable append-only SQLite session store.
#[derive(Clone)]
pub struct SqliteSessionStore {
    connection: Connection,
}

impl SqliteSessionStore {
    /// Open a store and migrate its schema.
    pub async fn open(path: impl AsRef<std::path::Path>) -> Result<Self, MemoryError> {
        let connection = Connection::open(path).await.map_err(memory_error)?;
        Self::from_connection(connection).await
    }

    /// Create a store over an existing connection.
    pub async fn from_connection(connection: Connection) -> Result<Self, MemoryError> {
        connection
            .call(|conn| {
                conn.execute_batch(
                    "CREATE TABLE IF NOT EXISTS rig_session_events (
                   session_id TEXT NOT NULL,
                   branch TEXT NOT NULL,
                   sequence INTEGER NOT NULL,
                   parent_sequence INTEGER,
                   correlation_id TEXT,
                   format_version INTEGER NOT NULL,
                   entry_json TEXT NOT NULL,
                   PRIMARY KEY(session_id, branch, sequence)
                 );",
                )?;
                Ok(())
            })
            .await
            .map_err(memory_error)?;
        Ok(Self { connection })
    }

    /// Atomically append and assign the next branch-local sequence.
    pub async fn append(
        &self,
        session_id: &str,
        branch: &str,
        parent_sequence: Option<u64>,
        correlation_id: Option<String>,
        entry: SessionEntry,
    ) -> Result<StoredSessionEvent, MemoryError> {
        let connection = self.connection.clone();
        let session_id = session_id.to_owned();
        let branch = branch.to_owned();
        let json = serde_json::to_string(&entry).map_err(memory_error)?;
        connection
            .call(move |conn| {
                let tx =
                    conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                let sequence: i64 = tx.query_row(
                    "SELECT COALESCE(MAX(sequence)+1,0) FROM rig_session_events
                 WHERE session_id=?1 AND branch=?2",
                    rusqlite::params![session_id, branch],
                    |row| row.get(0),
                )?;
                tx.execute(
                    "INSERT INTO rig_session_events VALUES (?1,?2,?3,?4,?5,1,?6)",
                    rusqlite::params![
                        session_id,
                        branch,
                        sequence,
                        parent_sequence,
                        correlation_id,
                        json
                    ],
                )?;
                tx.commit()?;
                Ok(StoredSessionEvent {
                    session_id,
                    branch,
                    sequence: sequence as u64,
                    parent_sequence,
                    correlation_id,
                    entry,
                })
            })
            .await
            .map_err(memory_error)
    }

    /// Load a branch in sequence order.
    pub async fn load(
        &self,
        session_id: &str,
        branch: &str,
    ) -> Result<Vec<StoredSessionEvent>, MemoryError> {
        let connection = self.connection.clone();
        let session_id = session_id.to_owned();
        let branch = branch.to_owned();
        connection
            .call(move |conn| {
                let mut stmt = conn.prepare(
                "SELECT sequence,parent_sequence,correlation_id,entry_json FROM rig_session_events
                 WHERE session_id=?1 AND branch=?2 ORDER BY sequence")?;
                let rows = stmt.query_map(rusqlite::params![session_id, branch], |row| {
                    let json: String = row.get(3)?;
                    let entry = serde_json::from_str(&json).map_err(|error| {
                        rusqlite::Error::FromSqlConversionFailure(
                            3,
                            rusqlite::types::Type::Text,
                            Box::new(error),
                        )
                    })?;
                    Ok(StoredSessionEvent {
                        session_id: session_id.clone(),
                        branch: branch.clone(),
                        sequence: row.get::<_, i64>(0)? as u64,
                        parent_sequence: row.get::<_, Option<i64>>(1)?.map(|v| v as u64),
                        correlation_id: row.get(2)?,
                        entry,
                    })
                })?;
                let collected: Result<Vec<_>, rusqlite::Error> = rows.collect();
                Ok(collected?)
            })
            .await
            .map_err(memory_error)
    }

    /// Create a branch by copying an inclusive prefix while retaining parent links.
    pub async fn branch(
        &self,
        session_id: &str,
        source: &str,
        target: &str,
        through: u64,
    ) -> Result<(), MemoryError> {
        let connection = self.connection.clone();
        let session_id = session_id.to_owned();
        let source = source.to_owned();
        let target = target.to_owned();
        connection.call(move |conn| {
            let tx = conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
            tx.execute(
                "INSERT INTO rig_session_events
                 SELECT session_id,?1,sequence,parent_sequence,correlation_id,format_version,entry_json
                 FROM rig_session_events WHERE session_id=?2 AND branch=?3 AND sequence<=?4",
                rusqlite::params![target, session_id, source, through])?;
            tx.commit()?;
            Ok(())
        }).await.map_err(memory_error)
    }

    /// Export a complete branch as versioned JSON.
    pub async fn export_json(&self, session_id: &str, branch: &str) -> Result<String, MemoryError> {
        serde_json::to_string(&self.load(session_id, branch).await?).map_err(memory_error)
    }

    /// Import events transactionally while preserving sequences and parent links.
    ///
    /// Re-importing an identical event is idempotent. An occupied sequence with
    /// different metadata or payload, or a missing parent, rejects the entire
    /// import without writing any events.
    pub async fn import_json(&self, json: &str) -> Result<(), MemoryError> {
        let events: Vec<StoredSessionEvent> = serde_json::from_str(json).map_err(memory_error)?;
        let connection = self.connection.clone();
        connection
            .call(move |conn| {
                let tx =
                    conn.transaction_with_behavior(rusqlite::TransactionBehavior::Immediate)?;
                for event in events {
                    let sequence = i64::try_from(event.sequence).map_err(|_| {
                        rusqlite::Error::InvalidParameterName(
                            "sequence exceeds SQLite range".into(),
                        )
                    })?;
                    let parent_sequence = event
                        .parent_sequence
                        .map(i64::try_from)
                        .transpose()
                        .map_err(|_| {
                            rusqlite::Error::InvalidParameterName(
                                "parent sequence exceeds SQLite range".into(),
                            )
                        })?;
                    let entry_json = serde_json::to_string(&event.entry).map_err(|error| {
                        rusqlite::Error::ToSqlConversionFailure(Box::new(error))
                    })?;
                    let existing = tx
                        .query_row(
                            "SELECT parent_sequence,correlation_id,entry_json
                             FROM rig_session_events
                             WHERE session_id=?1 AND branch=?2 AND sequence=?3",
                            rusqlite::params![event.session_id, event.branch, sequence],
                            |row| {
                                Ok((
                                    row.get::<_, Option<i64>>(0)?,
                                    row.get::<_, Option<String>>(1)?,
                                    row.get::<_, String>(2)?,
                                ))
                            },
                        )
                        .optional()?;
                    if let Some((stored_parent, stored_correlation, stored_json)) = existing {
                        if stored_parent != parent_sequence
                            || stored_correlation != event.correlation_id
                            || stored_json != entry_json
                        {
                            return Err(rusqlite::Error::InvalidParameterName(format!(
                                "conflicting imported event {}:{branch}:{sequence}",
                                event.session_id,
                                branch = event.branch,
                            ))
                            .into());
                        }
                        continue;
                    }
                    if let Some(parent) = parent_sequence {
                        let parent_exists: bool = tx.query_row(
                            "SELECT EXISTS(SELECT 1 FROM rig_session_events
                             WHERE session_id=?1 AND branch=?2 AND sequence=?3)",
                            rusqlite::params![event.session_id, event.branch, parent],
                            |row| row.get(0),
                        )?;
                        if !parent_exists {
                            return Err(rusqlite::Error::InvalidParameterName(format!(
                                "missing parent {parent} for imported event {}:{branch}:{sequence}",
                                event.session_id,
                                branch = event.branch,
                            ))
                            .into());
                        }
                    }
                    tx.execute(
                        "INSERT INTO rig_session_events VALUES (?1,?2,?3,?4,?5,1,?6)",
                        rusqlite::params![
                            event.session_id,
                            event.branch,
                            sequence,
                            parent_sequence,
                            event.correlation_id,
                            entry_json,
                        ],
                    )?;
                }
                tx.commit()?;
                Ok(())
            })
            .await
            .map_err(memory_error)
    }
}

impl SessionStore for SqliteSessionStore {
    fn load<'a>(
        &'a self,
        session: &'a str,
        branch: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<SessionEvent>, SessionError>> {
        Box::pin(async move {
            let events = SqliteSessionStore::load(self, session, branch)
                .await
                .map_err(|error| SessionError::Backend(error.to_string()))?;
            events
                .into_iter()
                .map(|event| {
                    let kind = serde_json::from_value(
                        serde_json::to_value(event.entry)
                            .map_err(|error| SessionError::Backend(error.to_string()))?,
                    )
                    .map_err(|error| SessionError::Backend(error.to_string()))?;
                    Ok(SessionEvent {
                        session_id: event.session_id,
                        branch: event.branch,
                        sequence: event.sequence,
                        parent_sequence: event.parent_sequence,
                        correlation_id: event.correlation_id,
                        kind,
                    })
                })
                .collect()
        })
    }

    fn append<'a>(
        &'a self,
        session: &'a str,
        branch: &'a str,
        parent: Option<u64>,
        correlation: Option<String>,
        kind: SessionEventKind,
    ) -> WasmBoxedFuture<'a, Result<SessionEvent, SessionError>> {
        Box::pin(async move {
            let entry: SessionEntry = serde_json::from_value(
                serde_json::to_value(&kind)
                    .map_err(|error| SessionError::Backend(error.to_string()))?,
            )
            .map_err(|error| SessionError::Backend(error.to_string()))?;
            let event =
                SqliteSessionStore::append(self, session, branch, parent, correlation, entry)
                    .await
                    .map_err(|error| SessionError::Backend(error.to_string()))?;
            Ok(SessionEvent {
                session_id: event.session_id,
                branch: event.branch,
                sequence: event.sequence,
                parent_sequence: event.parent_sequence,
                correlation_id: event.correlation_id,
                kind,
            })
        })
    }

    fn branch<'a>(
        &'a self,
        session: &'a str,
        source: &'a str,
        target: &'a str,
        through: u64,
    ) -> WasmBoxedFuture<'a, Result<(), SessionError>> {
        Box::pin(async move {
            SqliteSessionStore::branch(self, session, source, target, through)
                .await
                .map_err(|error| SessionError::Backend(error.to_string()))
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use futures::StreamExt;
    use rig_core::{
        agent::AgentBuilder,
        completion::{
            CompletionError, CompletionModel, CompletionRequest, CompletionResponse, GetTokenUsage,
            PromptError, Usage,
        },
        streaming::{RawStreamingChoice, RawStreamingToolCall, StreamingCompletionResponse},
        tool::Tool,
    };
    use std::{
        convert::Infallible,
        sync::{
            Arc,
            atomic::{AtomicUsize, Ordering},
        },
    };

    #[derive(Clone, Debug, Serialize, Deserialize)]
    struct TestResponse;

    impl GetTokenUsage for TestResponse {
        fn token_usage(&self) -> Usage {
            Usage::new()
        }
    }

    #[derive(Clone)]
    struct PendingModel {
        started: Arc<tokio::sync::Notify>,
        calls: Arc<AtomicUsize>,
    }

    impl CompletionModel for PendingModel {
        type Response = TestResponse;
        type StreamingResponse = TestResponse;
        type Client = ();

        fn make(_: &Self::Client, _: impl Into<String>) -> Self {
            Self {
                started: Arc::new(tokio::sync::Notify::new()),
                calls: Arc::new(AtomicUsize::new(0)),
            }
        }

        async fn completion(
            &self,
            _: CompletionRequest,
        ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.notify_one();
            futures::future::pending().await
        }

        async fn stream(
            &self,
            _: CompletionRequest,
        ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.notify_one();
            futures::future::pending().await
        }
    }

    #[derive(Clone, Default)]
    struct ToolCallModel {
        calls: Arc<AtomicUsize>,
    }

    impl CompletionModel for ToolCallModel {
        type Response = TestResponse;
        type StreamingResponse = TestResponse;
        type Client = ();

        fn make(_: &Self::Client, _: impl Into<String>) -> Self {
            Self::default()
        }

        async fn completion(
            &self,
            _: CompletionRequest,
        ) -> Result<CompletionResponse<Self::Response>, CompletionError> {
            futures::future::pending().await
        }

        async fn stream(
            &self,
            _: CompletionRequest,
        ) -> Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError> {
            let call = self.calls.fetch_add(1, Ordering::SeqCst);
            let choice = if call == 0 {
                RawStreamingChoice::ToolCall(RawStreamingToolCall {
                    id: "call".into(),
                    internal_call_id: "internal-call".into(),
                    call_id: None,
                    name: "pending".into(),
                    arguments: serde_json::json!({}),
                    signature: None,
                    additional_params: None,
                })
            } else {
                RawStreamingChoice::Message("must not run".into())
            };
            Ok(StreamingCompletionResponse::stream(Box::pin(
                futures::stream::iter([Ok(choice)]),
            )))
        }
    }

    #[derive(Clone)]
    struct PendingTool {
        started: Arc<tokio::sync::Notify>,
        calls: Arc<AtomicUsize>,
    }

    impl Tool for PendingTool {
        const NAME: &'static str = "pending";
        type Error = Infallible;
        type Args = serde_json::Value;
        type Output = String;

        fn description(&self) -> String {
            "wait forever".into()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({"type": "object"})
        }

        async fn call(&self, _: Self::Args) -> Result<Self::Output, Self::Error> {
            self.calls.fetch_add(1, Ordering::SeqCst);
            self.started.notify_one();
            futures::future::pending().await
        }
    }

    fn session_path(name: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!(
            "rig-session-{name}-{}-{}.sqlite",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ))
    }

    async fn assert_unsafe_recovery<M>(agent: rig_core::agent::Agent<M>)
    where
        M: CompletionModel + 'static,
    {
        let mut reopened = agent.runner("resume").stream().await;
        let error = match reopened.next().await {
            Some(Err(error)) => error,
            _ => panic!("unsafe pending work must emit one terminal error"),
        };
        assert!(matches!(
            error,
            rig_core::agent::StreamingError::Prompt(error)
                if matches!(*error, PromptError::SessionError(SessionError::UnsafeRecovery))
        ));
    }

    #[tokio::test]
    async fn streaming_cancellation_persists_interrupted_before_consumer_drop() {
        let model_path = session_path("pending-model");
        let model_started = Arc::new(tokio::sync::Notify::new());
        let model_calls = Arc::new(AtomicUsize::new(0));
        let model = PendingModel {
            started: model_started.clone(),
            calls: model_calls.clone(),
        };
        let store = SqliteSessionStore::open(&model_path).await.unwrap();
        let runner = AgentBuilder::new(model.clone())
            .session(store, "pending-model")
            .build()
            .runner("start");
        let control = runner.control_handle();
        let request = runner.stream();
        let task = tokio::spawn(async move {
            let mut stream = request.await;
            let error = loop {
                if let Some(Err(error)) = stream.next().await {
                    break error;
                }
            };
            drop(stream);
            error
        });
        model_started.notified().await;
        assert!(control.cancel());
        assert!(matches!(
            task.await.unwrap(),
            rig_core::agent::StreamingError::Prompt(_)
        ));

        let reopened = SqliteSessionStore::open(&model_path).await.unwrap();
        let events = reopened.load("pending-model", "main").await.unwrap();
        assert!(matches!(
            events.last().map(|event| &event.entry),
            Some(SessionEntry::Interrupted {
                checkpoint: None,
                ..
            })
        ));
        assert_unsafe_recovery(
            AgentBuilder::new(model)
                .session(reopened, "pending-model")
                .build(),
        )
        .await;
        assert_eq!(model_calls.load(Ordering::SeqCst), 1);
        drop(std::fs::remove_file(model_path));

        let tool_path = session_path("pending-tool");
        let tool_started = Arc::new(tokio::sync::Notify::new());
        let tool_calls = Arc::new(AtomicUsize::new(0));
        let store = SqliteSessionStore::open(&tool_path).await.unwrap();
        let model = ToolCallModel::default();
        let runner = AgentBuilder::new(model.clone())
            .tool(PendingTool {
                started: tool_started.clone(),
                calls: tool_calls.clone(),
            })
            .session(store, "pending-tool")
            .build()
            .runner("start")
            .max_turns(2);
        let control = runner.control_handle();
        let request = runner.stream();
        let task = tokio::spawn(async move {
            let mut stream = request.await;
            let error = loop {
                if let Some(Err(error)) = stream.next().await {
                    break error;
                }
            };
            drop(stream);
            error
        });
        tool_started.notified().await;
        assert!(control.cancel());
        assert!(matches!(
            task.await.unwrap(),
            rig_core::agent::StreamingError::Prompt(_)
        ));

        let reopened = SqliteSessionStore::open(&tool_path).await.unwrap();
        let events = reopened.load("pending-tool", "main").await.unwrap();
        assert!(matches!(
            events.last().map(|event| &event.entry),
            Some(SessionEntry::Interrupted {
                checkpoint: None,
                ..
            })
        ));
        assert_unsafe_recovery(
            AgentBuilder::new(model)
                .tool(PendingTool {
                    started: tool_started,
                    calls: tool_calls.clone(),
                })
                .session(reopened, "pending-tool")
                .build(),
        )
        .await;
        assert_eq!(tool_calls.load(Ordering::SeqCst), 1);
        drop(std::fs::remove_file(tool_path));
    }

    async fn connection(name: &str) -> Connection {
        Connection::open(format!("file:{name}?mode=memory&cache=shared"))
            .await
            .unwrap()
    }

    #[tokio::test]
    async fn memory_orders_concurrent_appends_and_clears() {
        let memory = SqliteConversationMemory::from_connection(connection("memory_order").await)
            .await
            .unwrap();
        let writes = (0..16).map(|value| {
            let memory = memory.clone();
            async move {
                memory
                    .append("conversation", vec![Message::user(value.to_string())])
                    .await
                    .unwrap()
            }
        });
        futures::future::join_all(writes).await;
        let loaded = memory.load("conversation").await.unwrap();
        assert_eq!(loaded.len(), 16);
        memory.clear("conversation").await.unwrap();
        assert!(memory.load("conversation").await.unwrap().is_empty());
    }

    #[tokio::test]
    async fn file_backed_multi_connection_appends_survive_reopen_in_order() {
        let path = std::env::temp_dir().join(format!(
            "rig-sqlite-durable-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        ));
        let first = SqliteConversationMemory::open(&path).await.unwrap();
        let second = SqliteConversationMemory::open(&path).await.unwrap();
        let writes = (0..24).map(|value| {
            let memory = if value % 2 == 0 {
                first.clone()
            } else {
                second.clone()
            };
            async move {
                memory
                    .append("conversation", vec![Message::user(value.to_string())])
                    .await
            }
        });
        for result in futures::future::join_all(writes).await {
            result.unwrap();
        }
        drop(first);
        drop(second);
        let reopened = SqliteConversationMemory::open(&path).await.unwrap();
        let loaded = reopened.load("conversation").await.unwrap();
        assert_eq!(loaded.len(), 24);
        drop(reopened);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn file_backed_session_multi_connection_order_survives_reopen() {
        let path = std::env::temp_dir().join(format!(
            "rig-session-durable-{}-{}.db",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos(),
        ));
        let first = SqliteSessionStore::open(&path).await.unwrap();
        let second = SqliteSessionStore::open(&path).await.unwrap();
        let writes = (0..24).map(|value| {
            let store = if value % 2 == 0 {
                first.clone()
            } else {
                second.clone()
            };
            async move {
                store
                    .append(
                        "session",
                        "main",
                        None,
                        Some(format!("run-{value}")),
                        SessionEntry::Custom {
                            kind: "value".into(),
                            payload: serde_json::json!(value),
                        },
                    )
                    .await
            }
        });
        for result in futures::future::join_all(writes).await {
            result.unwrap();
        }
        drop(first);
        drop(second);
        let reopened = SqliteSessionStore::open(&path).await.unwrap();
        let events = reopened.load("session", "main").await.unwrap();
        assert_eq!(events.len(), 24);
        assert_eq!(
            events
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            (0..24).collect::<Vec<_>>()
        );
        drop(reopened);
        std::fs::remove_file(path).unwrap();
    }

    #[tokio::test]
    async fn sessions_branch_export_and_recover_interruption() {
        let store = SqliteSessionStore::from_connection(connection("sessions").await)
            .await
            .unwrap();
        store
            .append(
                "s",
                "main",
                None,
                Some("run".into()),
                SessionEntry::Message(Message::user("hi")),
            )
            .await
            .unwrap();
        store
            .append(
                "s",
                "main",
                Some(0),
                Some("run".into()),
                SessionEntry::Interrupted {
                    run_id: "run".into(),
                    checkpoint: Some("state".into()),
                },
            )
            .await
            .unwrap();
        store
            .append(
                "s", "main", Some(1), Some("run".into()),
                SessionEntry::ModelResponse {
                    usage: serde_json::json!({"input_tokens": 1, "output_tokens": 2, "total_tokens": 3}),
                    finish_reason: Some("stop".into()), raw_finish_reason: Some("STOP".into()),
                },
            )
            .await
            .unwrap();
        store
            .append(
                "s",
                "main",
                Some(2),
                Some("run".into()),
                SessionEntry::Checkpoint {
                    snapshot: "safe-state".into(),
                },
            )
            .await
            .unwrap();
        store.branch("s", "main", "retry", 3).await.unwrap();
        store
            .append(
                "s",
                "retry",
                Some(3),
                Some("run-2".into()),
                SessionEntry::Bookmark {
                    name: "resumed".into(),
                    checkpoint: None,
                },
            )
            .await
            .unwrap();
        let exported = store.export_json("s", "retry").await.unwrap();
        let imported = SqliteSessionStore::from_connection(connection("sessions_import").await)
            .await
            .unwrap();
        imported.import_json(&exported).await.unwrap();
        imported
            .import_json(&exported)
            .await
            .expect("an identical repeated import is idempotent");
        let loaded = imported.load("s", "retry").await.unwrap();
        assert_eq!(loaded.len(), 5);
        assert_eq!(
            loaded
                .iter()
                .map(|event| event.sequence)
                .collect::<Vec<_>>(),
            vec![0, 1, 2, 3, 4]
        );
        assert_eq!(
            loaded
                .iter()
                .map(|event| event.parent_sequence)
                .collect::<Vec<_>>(),
            vec![None, Some(0), Some(1), Some(2), Some(3)]
        );
        assert!(matches!(
            loaded[2].entry,
            SessionEntry::ModelResponse { .. }
        ));
        assert!(matches!(loaded[3].entry, SessionEntry::Checkpoint { .. }));

        let mut conflicting = loaded[2].clone();
        conflicting.correlation_id = Some("different-run".into());
        let conflict_json = serde_json::to_string(&vec![
            StoredSessionEvent {
                sequence: 5,
                parent_sequence: Some(4),
                entry: SessionEntry::Custom {
                    kind: "must-roll-back".into(),
                    payload: serde_json::json!(true),
                },
                ..loaded[4].clone()
            },
            conflicting,
        ])
        .unwrap();
        imported
            .import_json(&conflict_json)
            .await
            .expect_err("a conflicting occupied sequence rejects the transaction");
        assert_eq!(imported.load("s", "retry").await.unwrap(), loaded);
        assert_eq!(store.load("s", "main").await.unwrap().len(), 4);
    }

    #[tokio::test]
    async fn core_session_events_preserve_finish_and_rig_correlation_metadata() {
        use rig_core::session::{SessionEventKind, SessionStore};
        let store = SqliteSessionStore::from_connection(connection("core_metadata").await)
            .await
            .unwrap();
        let call = SessionStore::append(
            &store,
            "s",
            "main",
            None,
            Some("run".into()),
            SessionEventKind::ToolCall {
                name: "add".into(),
                arguments: serde_json::json!({}),
                internal_call_id: "rig-call".into(),
                parent_internal_call_id: Some("rig-parent".into()),
            },
        )
        .await
        .unwrap();
        SessionStore::append(
            &store,
            "s",
            "main",
            Some(call.sequence),
            Some("run".into()),
            SessionEventKind::ModelResponse {
                usage: rig_core::completion::Usage::new(),
                finish_reason: Some(rig_core::runtime::TerminalReason::Completed),
                raw_finish_reason: Some("STOP_RAW".into()),
            },
        )
        .await
        .unwrap();
        let loaded = SessionStore::load(&store, "s", "main").await.unwrap();
        assert!(matches!(&loaded[0].kind, SessionEventKind::ToolCall {
            internal_call_id, parent_internal_call_id: Some(parent), ..
        } if internal_call_id == "rig-call" && parent == "rig-parent"));
        assert!(matches!(&loaded[1].kind, SessionEventKind::ModelResponse {
            finish_reason: Some(reason), raw_finish_reason: Some(raw), ..
        } if *reason == rig_core::runtime::TerminalReason::Completed && raw == "STOP_RAW"));
    }
}
