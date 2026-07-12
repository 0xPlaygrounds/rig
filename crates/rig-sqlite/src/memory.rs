//! Durable SQLite conversation memory.

use rig_core::{
    completion::Message,
    memory::{ConversationMemory, MemoryError},
    wasm_compat::WasmBoxedFuture,
};
use tokio_rusqlite::Connection;

/// Durable ordered [`ConversationMemory`] backed by SQLite.
///
/// Appends use one transaction, so messages from concurrent turns never
/// interleave within an appended batch. The schema stores versioned JSON to
/// permit future migrations without conflating message history with a session
/// event log.
#[derive(Clone)]
pub struct SqliteConversationMemory {
    connection: Connection,
}

impl SqliteConversationMemory {
    /// Open a database and create the memory table when absent.
    pub async fn open(path: impl AsRef<std::path::Path>) -> Result<Self, MemoryError> {
        let connection = Connection::open(path).await.map_err(MemoryError::backend)?;
        Self::from_connection(connection).await
    }

    /// Use an existing connection and create the memory table when absent.
    pub async fn from_connection(connection: Connection) -> Result<Self, MemoryError> {
        connection
            .call(|conn| {
                conn.execute_batch(
                    "PRAGMA foreign_keys = ON;
                     CREATE TABLE IF NOT EXISTS rig_conversation_messages (
                       sequence INTEGER PRIMARY KEY AUTOINCREMENT,
                       conversation_id TEXT NOT NULL,
                       format_version INTEGER NOT NULL DEFAULT 1,
                       message_json TEXT NOT NULL
                     );
                     CREATE INDEX IF NOT EXISTS rig_conversation_messages_lookup
                       ON rig_conversation_messages(conversation_id, sequence);",
                )?;
                Ok(())
            })
            .await
            .map_err(MemoryError::backend)?;
        Ok(Self { connection })
    }

    /// Borrow the shared SQLite connection.
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
            let encoded: Vec<String> = connection
                .call(move |conn| {
                    let mut statement = conn.prepare(
                        "SELECT message_json FROM rig_conversation_messages
                         WHERE conversation_id = ?1 ORDER BY sequence ASC",
                    )?;
                    let rows =
                        statement.query_map([conversation_id], |row| row.get::<_, String>(0))?;
                    Ok(rows.collect::<Result<Vec<_>, _>>()?)
                })
                .await
                .map_err(MemoryError::backend)?;
            encoded
                .into_iter()
                .map(|json| serde_json::from_str(&json).map_err(MemoryError::backend))
                .collect()
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
                .map(|message| serde_json::to_string(&message).map_err(MemoryError::backend))
                .collect::<Result<Vec<_>, _>>()?;
            connection
                .call(move |conn| {
                    let transaction = conn.transaction()?;
                    {
                        let mut statement = transaction.prepare(
                            "INSERT INTO rig_conversation_messages
                             (conversation_id, format_version, message_json)
                             VALUES (?1, 1, ?2)",
                        )?;
                        for message in encoded {
                            statement.execute(rusqlite::params![conversation_id, message])?;
                        }
                    }
                    transaction.commit()?;
                    Ok(())
                })
                .await
                .map_err(MemoryError::backend)
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
                        "DELETE FROM rig_conversation_messages WHERE conversation_id = ?1",
                        [conversation_id],
                    )?;
                    Ok(())
                })
                .await
                .map_err(MemoryError::backend)
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    #[allow(clippy::panic_in_result_fn)]
    async fn persists_order_and_clears_conversations() -> anyhow::Result<()> {
        let memory =
            SqliteConversationMemory::from_connection(Connection::open_in_memory().await?).await?;
        memory
            .append("a", vec![Message::user("one"), Message::assistant("two")])
            .await?;
        memory.append("a", vec![Message::user("three")]).await?;
        assert_eq!(memory.load("a").await?.len(), 3);
        assert!(memory.load("b").await?.is_empty());
        memory.clear("a").await?;
        assert!(memory.load("a").await?.is_empty());
        Ok(())
    }
}
