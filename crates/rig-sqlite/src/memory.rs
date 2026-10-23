//! [`SqliteSessionStore`]: rig-memory sessions kept in a SQLite database,
//! so a conversation outlives the process and every process that opens the
//! file shares it. It is a rig-memory [`AsyncStore`], for
//! [`load_async`](rig_memory::load_async), [`save_async`](rig_memory::save_async)
//! and an agent's memory. The store makes its three `rig_memory_*` tables
//! when it is opened and leaves every other table alone.
//!
//! ```no_run
//! use rig_core::message::Message;
//! use rig_memory::{load_async, save_async};
//! use rig_sqlite::SqliteSessionStore;
//! use tokio_rusqlite::Connection;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let store = SqliteSessionStore::open(Connection::open("sessions.db").await?).await?;
//! let mut conversation = load_async(&store, "user-123").await?;
//! conversation.push(Message::user("My name is Alice."));
//! save_async(&store, "user-123", &mut conversation).await?;
//! # Ok(()) }
//! ```

use std::io;

use rig_memory::{AsyncStore, LogWriteError, check_log_name, check_name};
use rusqlite::{OptionalExtension, TransactionBehavior, params};
use tokio_rusqlite::Connection;

/// A log's head (its length) and its appends, kept apart from the vector
/// store's tables. A log is the bytes of its chunks in `start` order; an
/// append inserts one chunk, so it writes only its own bytes.
const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS rig_memory_logs (
    name TEXT PRIMARY KEY NOT NULL,
    len INTEGER NOT NULL
);
CREATE TABLE IF NOT EXISTS rig_memory_log_chunks (
    name TEXT NOT NULL,
    start INTEGER NOT NULL,
    bytes BLOB NOT NULL,
    PRIMARY KEY (name, start)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS rig_memory_files (
    name TEXT PRIMARY KEY NOT NULL,
    bytes BLOB NOT NULL
);
";

/// rig-memory sessions in a SQLite database: one log per conversation and
/// the files, such as image blobs, they name. Cloning shares the
/// connection.
///
/// Each log write is one `IMMEDIATE` transaction that checks the log's
/// length and writes, so of two writers that read a log at one length, in
/// this process or another, one writes and the other gets
/// [`LogWriteError::Changed`]. A crash or a dropped call leaves the whole
/// write or none of it, durable as the connection's `synchronous` setting
/// makes it (SQLite's default, `FULL`, survives power loss). A writer that
/// finds the database locked waits for the connection's busy timeout,
/// 5 seconds unless the caller set another.
#[derive(Clone, Debug)]
pub struct SqliteSessionStore {
    conn: Connection,
}

impl SqliteSessionStore {
    /// The sessions in the database `conn` is open on, making the store's
    /// tables when they are missing.
    ///
    /// # Errors
    ///
    /// When the tables cannot be made.
    pub async fn open(conn: Connection) -> io::Result<Self> {
        conn.call(|conn| Ok(conn.execute_batch(SCHEMA)?))
            .await
            .map_err(io::Error::other)?;
        Ok(Self { conn })
    }

    /// Writes `bytes` to the log `name` if it is `len` bytes long, after
    /// the chunks it keeps, or in their place with `replace`.
    async fn write_log(
        &self,
        name: &str,
        len: u64,
        bytes: &[u8],
        replace: bool,
    ) -> Result<(), LogWriteError> {
        check_log_name(name)?;
        let start = if replace { 0 } else { len };
        // No log is longer than `i64::MAX`, so a write naming or reaching
        // such a length is refused.
        let (Ok(held), Ok(start), Some(Ok(end))) = (
            i64::try_from(len),
            i64::try_from(start),
            start.checked_add(bytes.len() as u64).map(i64::try_from),
        ) else {
            return Err(LogWriteError::Changed);
        };
        let (name, bytes) = (name.to_owned(), bytes.to_vec());
        let wrote = self
            .conn
            .call(move |conn| {
                let tx = conn.transaction_with_behavior(TransactionBehavior::Immediate)?;
                let now: Option<i64> = tx
                    .query_row(
                        "SELECT len FROM rig_memory_logs WHERE name = ?1",
                        [&name],
                        |row| row.get(0),
                    )
                    .optional()?;
                if now.unwrap_or(0) != held {
                    return Ok(false);
                }
                if replace {
                    tx.execute("DELETE FROM rig_memory_log_chunks WHERE name = ?1", [&name])?;
                }
                tx.execute(
                    "INSERT INTO rig_memory_logs (name, len) VALUES (?1, ?2)
                     ON CONFLICT (name) DO UPDATE SET len = excluded.len",
                    params![name, end],
                )?;
                if !bytes.is_empty() {
                    tx.execute(
                        "INSERT INTO rig_memory_log_chunks (name, start, bytes) VALUES (?1, ?2, ?3)",
                        params![name, start, bytes],
                    )?;
                }
                tx.commit()?;
                Ok(true)
            })
            .await
            .map_err(io::Error::other)?;
        if wrote {
            Ok(())
        } else {
            Err(LogWriteError::Changed)
        }
    }
}

impl AsyncStore for SqliteSessionStore {
    /// The head and the chunks are read in one transaction, so a log is
    /// read as one write left it.
    async fn read_log_async(&self, name: &str) -> io::Result<Vec<u8>> {
        check_log_name(name)?;
        let name = name.to_owned();
        let log = self
            .conn
            .call(move |conn| {
                let tx = conn.transaction()?;
                let len: Option<i64> = tx
                    .query_row(
                        "SELECT len FROM rig_memory_logs WHERE name = ?1",
                        [&name],
                        |row| row.get(0),
                    )
                    .optional()?;
                let Some(len) = len else {
                    return Ok(None);
                };
                let mut chunks = tx.prepare(
                    "SELECT bytes FROM rig_memory_log_chunks WHERE name = ?1 ORDER BY start",
                )?;
                let mut log = Vec::new();
                for chunk in chunks.query_map([&name], |row| row.get::<_, Vec<u8>>(0))? {
                    log.extend_from_slice(&chunk?);
                }
                Ok(Some((len, log)))
            })
            .await
            .map_err(io::Error::other)?;
        match log {
            None => Err(io::ErrorKind::NotFound.into()),
            Some((len, log)) if u64::try_from(len).ok() == Some(log.len() as u64) => Ok(log),
            Some((len, log)) => Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("the log is {len} bytes long, its chunks {}", log.len()),
            )),
        }
    }

    async fn append_log_async(
        &self,
        name: &str,
        len: u64,
        bytes: &[u8],
    ) -> Result<(), LogWriteError> {
        self.write_log(name, len, bytes, false).await
    }

    async fn replace_log_async(
        &self,
        name: &str,
        len: u64,
        bytes: &[u8],
    ) -> Result<(), LogWriteError> {
        self.write_log(name, len, bytes, true).await
    }

    async fn read_file_async(&self, name: &str) -> io::Result<Vec<u8>> {
        check_name(name)?;
        let name = name.to_owned();
        self.conn
            .call(move |conn| {
                Ok(conn
                    .query_row(
                        "SELECT bytes FROM rig_memory_files WHERE name = ?1",
                        [&name],
                        |row| row.get::<_, Vec<u8>>(0),
                    )
                    .optional()?)
            })
            .await
            .map_err(io::Error::other)?
            .ok_or_else(|| io::ErrorKind::NotFound.into())
    }

    async fn write_file_async(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        check_name(name)?;
        let (name, bytes) = (name.to_owned(), bytes.to_vec());
        self.conn
            .call(move |conn| {
                conn.execute(
                    "INSERT INTO rig_memory_files (name, bytes) VALUES (?1, ?2)
                     ON CONFLICT (name) DO UPDATE SET bytes = excluded.bytes",
                    params![name, bytes],
                )?;
                Ok(())
            })
            .await
            .map_err(io::Error::other)
    }
}

#[cfg(test)]
mod tests;
