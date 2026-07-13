//! Durable SQLite conversation memory.

use rig_core::{
    completion::Message,
    memory::{ConversationMemory, MemoryError},
    wasm_compat::WasmBoxedFuture,
};
use rusqlite::params;
use tokio_rusqlite::Connection;

/// Durable ordered conversation memory backed by SQLite.
#[derive(Clone)]
pub struct SqliteConversationMemory {
    connection: Connection,
    table: String,
}

impl std::fmt::Debug for SqliteConversationMemory {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SqliteConversationMemory")
            .field("table", &self.table)
            .finish_non_exhaustive()
    }
}

impl SqliteConversationMemory {
    /// Create the default memory store and idempotently migrate its schema.
    pub async fn new(connection: Connection) -> Result<Self, MemoryError> {
        let memory = Self {
            connection,
            table: "rig_conversation_messages".to_owned(),
        };
        memory.migrate().await?;
        Ok(memory)
    }

    /// Select a custom table name.
    ///
    /// Only ASCII SQL identifiers are accepted because SQLite cannot bind table
    /// names as parameters.
    pub fn with_table(mut self, table: impl Into<String>) -> Result<Self, MemoryError> {
        let table = table.into();
        validate_identifier(&table)?;
        self.table = table;
        Ok(self)
    }

    /// Create the configured table if needed.
    pub async fn migrate(&self) -> Result<(), MemoryError> {
        let table = self.table.clone();
        validate_identifier(&table)?;
        self.connection
            .call(move |connection| {
                connection.execute_batch(&format!(
                    "CREATE TABLE IF NOT EXISTS {table} (\
                     conversation_id TEXT NOT NULL,\
                     sequence INTEGER NOT NULL,\
                     message_json TEXT NOT NULL,\
                     schema_version INTEGER NOT NULL DEFAULT 1,\
                     PRIMARY KEY (conversation_id, sequence)\
                     );"
                ))?;
                Ok(())
            })
            .await
            .map_err(MemoryError::backend)
    }
}

impl ConversationMemory for SqliteConversationMemory {
    fn load<'a>(
        &'a self,
        conversation_id: &'a str,
    ) -> WasmBoxedFuture<'a, Result<Vec<Message>, MemoryError>> {
        Box::pin(async move {
            let table = self.table.clone();
            let conversation_id = conversation_id.to_owned();
            let rows = self
                .connection
                .call(move |connection| {
                    let mut statement = connection.prepare(&format!(
                        "SELECT message_json FROM {table} \
                         WHERE conversation_id = ?1 ORDER BY sequence ASC"
                    ))?;
                    let rows = statement
                        .query_map(params![conversation_id], |row| row.get::<_, String>(0))?
                        .collect::<Result<Vec<_>, _>>()?;
                    Ok(rows)
                })
                .await
                .map_err(MemoryError::backend)?;

            rows.into_iter()
                .map(|row| serde_json::from_str(&row).map_err(MemoryError::backend))
                .collect()
        })
    }

    fn append<'a>(
        &'a self,
        conversation_id: &'a str,
        messages: Vec<Message>,
    ) -> WasmBoxedFuture<'a, Result<(), MemoryError>> {
        Box::pin(async move {
            if messages.is_empty() {
                return Ok(());
            }
            let serialized = messages
                .into_iter()
                .map(|message| serde_json::to_string(&message).map_err(MemoryError::backend))
                .collect::<Result<Vec<_>, _>>()?;
            let table = self.table.clone();
            let conversation_id = conversation_id.to_owned();
            self.connection
                .call(move |connection| {
                    let transaction = connection.transaction()?;
                    let last: Option<i64> = transaction.query_row(
                        &format!("SELECT MAX(sequence) FROM {table} WHERE conversation_id = ?1"),
                        params![conversation_id],
                        |row| row.get(0),
                    )?;
                    let mut sequence = last.unwrap_or(-1) + 1;
                    for message in serialized {
                        transaction.execute(
                            &format!(
                                "INSERT INTO {table} \
                                 (conversation_id, sequence, message_json, schema_version) \
                                 VALUES (?1, ?2, ?3, 1)"
                            ),
                            params![conversation_id, sequence, message],
                        )?;
                        sequence += 1;
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
        Box::pin(async move {
            let table = self.table.clone();
            let conversation_id = conversation_id.to_owned();
            self.connection
                .call(move |connection| {
                    connection.execute(
                        &format!("DELETE FROM {table} WHERE conversation_id = ?1"),
                        params![conversation_id],
                    )?;
                    Ok(())
                })
                .await
                .map_err(MemoryError::backend)
        })
    }
}

fn validate_identifier(identifier: &str) -> Result<(), MemoryError> {
    let mut chars = identifier.chars();
    let valid_first = chars
        .next()
        .is_some_and(|character| character.is_ascii_alphabetic() || character == '_');
    if valid_first && chars.all(|character| character.is_ascii_alphanumeric() || character == '_') {
        Ok(())
    } else {
        Err(MemoryError::Internal(format!(
            "invalid SQLite table identifier `{identifier}`"
        )))
    }
}

#[cfg(test)]
#[allow(clippy::panic_in_result_fn)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn round_trip_isolated_conversations_and_clear() -> anyhow::Result<()> {
        let connection = Connection::open_in_memory().await?;
        let memory = SqliteConversationMemory::new(connection).await?;
        memory
            .append("a", vec![Message::user("one"), Message::assistant("two")])
            .await?;
        memory.append("b", vec![Message::user("other")]).await?;

        let loaded = memory.load("a").await?;
        assert_eq!(loaded.len(), 2);
        assert!(matches!(loaded.first(), Some(Message::User { .. })));
        assert!(matches!(loaded.get(1), Some(Message::Assistant { .. })));
        assert_eq!(memory.load("b").await?.len(), 1);
        memory.clear("a").await?;
        assert!(memory.load("a").await?.is_empty());
        Ok(())
    }

    #[tokio::test]
    async fn migration_is_idempotent_and_names_are_validated() -> anyhow::Result<()> {
        let connection = Connection::open_in_memory().await?;
        let memory = SqliteConversationMemory::new(connection).await?;
        memory.migrate().await?;
        assert!(memory.clone().with_table("bad; DROP TABLE x").is_err());
        Ok(())
    }
}
