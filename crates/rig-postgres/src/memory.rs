//! [`PostgresSessionStore`]: rig-memory sessions kept in PostgreSQL, so a
//! conversation outlives the process and every worker connected to the
//! database shares it. It is a rig-memory [`AsyncStore`], for
//! [`load_async`](rig_memory::load_async), [`save_async`](rig_memory::save_async)
//! and an agent's memory. The store makes its three `rig_memory_*` tables
//! when it is opened, in the connection's current schema, and leaves every
//! other table alone. It needs no extension.
//!
//! ```no_run
//! use rig_core::message::Message;
//! use rig_memory::{load_async, save_async};
//! use rig_postgres::PostgresSessionStore;
//! use sqlx::PgPool;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let pool = PgPool::connect("postgres://localhost/app").await?;
//! let store = PostgresSessionStore::open(pool).await?;
//! let mut conversation = load_async(&store, "user-123").await?;
//! conversation.push(Message::user("My name is Alice."));
//! save_async(&store, "user-123", &mut conversation).await?;
//! # Ok(()) }
//! ```

use std::io;

use rig_memory::{AsyncStore, LogWriteError, check_log_name, check_name};
use sqlx::PgPool;

/// A log's head (its length) and its appends. A log is the bytes of its
/// chunks in `start` order; an append inserts one chunk, so it writes only
/// its own bytes. The advisory lock lets workers that start at once make
/// the tables one at a time.
const SCHEMA: &str = "
SELECT pg_advisory_xact_lock(hashtext('rig_memory_schema'));
CREATE TABLE IF NOT EXISTS rig_memory_logs (
    name TEXT PRIMARY KEY,
    len BIGINT NOT NULL
);
CREATE TABLE IF NOT EXISTS rig_memory_log_chunks (
    name TEXT NOT NULL,
    start BIGINT NOT NULL,
    bytes BYTEA NOT NULL,
    PRIMARY KEY (name, start)
);
CREATE TABLE IF NOT EXISTS rig_memory_files (
    name TEXT PRIMARY KEY,
    bytes BYTEA NOT NULL
);
";

/// rig-memory sessions in PostgreSQL: one log per conversation and the
/// files, such as image blobs, they name. Cloning shares the pool.
///
/// Each log write is one transaction whose first statement moves the log's
/// length only if it is still the length the writer read, so of two
/// writers that read a log at one length, on any connection, one writes
/// and the other gets [`LogWriteError::Changed`]. A crash or a dropped
/// call leaves the whole write or none of it, durable once committed under
/// the server's default `synchronous_commit`.
#[derive(Clone, Debug)]
pub struct PostgresSessionStore {
    pool: PgPool,
}

impl PostgresSessionStore {
    /// The sessions in the database `pool` connects to, making the store's
    /// tables when they are missing.
    ///
    /// # Errors
    ///
    /// When the tables cannot be made.
    pub async fn open(pool: PgPool) -> io::Result<Self> {
        let mut tx = pool.begin().await.map_err(io::Error::other)?;
        sqlx::raw_sql(SCHEMA)
            .execute(&mut *tx)
            .await
            .map_err(io::Error::other)?;
        tx.commit().await.map_err(io::Error::other)?;
        Ok(Self { pool })
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
        let (Ok(held), Ok(added)) = (i64::try_from(len), i64::try_from(bytes.len())) else {
            return Err(LogWriteError::Changed);
        };
        let start = if replace { 0 } else { held };
        let mut tx = self.pool.begin().await.map_err(io::Error::other)?;
        // A log not there yet is started by an insert that a racing one
        // turns into the conditional update, so one of them wins either way.
        let head = if held == 0 {
            sqlx::query(
                "INSERT INTO rig_memory_logs (name, len) VALUES ($1, $2)
                 ON CONFLICT (name) DO UPDATE SET len = EXCLUDED.len
                 WHERE rig_memory_logs.len = 0",
            )
            .bind(name)
            .bind(start + added)
        } else {
            sqlx::query("UPDATE rig_memory_logs SET len = $2 WHERE name = $1 AND len = $3")
                .bind(name)
                .bind(start + added)
                .bind(held)
        };
        let moved = head.execute(&mut *tx).await.map_err(io::Error::other)?;
        if moved.rows_affected() != 1 {
            return Err(LogWriteError::Changed);
        }
        if replace {
            sqlx::query("DELETE FROM rig_memory_log_chunks WHERE name = $1")
                .bind(name)
                .execute(&mut *tx)
                .await
                .map_err(io::Error::other)?;
        }
        if !bytes.is_empty() {
            sqlx::query(
                "INSERT INTO rig_memory_log_chunks (name, start, bytes) VALUES ($1, $2, $3)",
            )
            .bind(name)
            .bind(start)
            .bind(bytes)
            .execute(&mut *tx)
            .await
            .map_err(io::Error::other)?;
        }
        tx.commit().await.map_err(io::Error::other)?;
        Ok(())
    }
}

impl AsyncStore for PostgresSessionStore {
    /// The head and the chunks are read by one statement, so a log is read
    /// as one write left it.
    async fn read_log_async(&self, name: &str) -> io::Result<Vec<u8>> {
        check_log_name(name)?;
        let rows: Vec<(i64, Option<Vec<u8>>)> = sqlx::query_as(
            "SELECT l.len, c.bytes FROM rig_memory_logs l
             LEFT JOIN rig_memory_log_chunks c ON c.name = l.name
             WHERE l.name = $1 ORDER BY c.start",
        )
        .bind(name)
        .fetch_all(&self.pool)
        .await
        .map_err(io::Error::other)?;
        let Some(&(len, _)) = rows.first() else {
            return Err(io::ErrorKind::NotFound.into());
        };
        let log: Vec<u8> = rows
            .into_iter()
            .filter_map(|(_, chunk)| chunk)
            .flatten()
            .collect();
        if u64::try_from(len).ok() != Some(log.len() as u64) {
            let why = format!("the log is {len} bytes long, its chunks {}", log.len());
            return Err(io::Error::new(io::ErrorKind::InvalidData, why));
        }
        Ok(log)
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
        let file: Option<(Vec<u8>,)> =
            sqlx::query_as("SELECT bytes FROM rig_memory_files WHERE name = $1")
                .bind(name)
                .fetch_optional(&self.pool)
                .await
                .map_err(io::Error::other)?;
        file.map(|(bytes,)| bytes)
            .ok_or_else(|| io::ErrorKind::NotFound.into())
    }

    async fn write_file_async(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        check_name(name)?;
        sqlx::query(
            "INSERT INTO rig_memory_files (name, bytes) VALUES ($1, $2)
             ON CONFLICT (name) DO UPDATE SET bytes = EXCLUDED.bytes",
        )
        .bind(name)
        .bind(bytes)
        .execute(&self.pool)
        .await
        .map_err(io::Error::other)?;
        Ok(())
    }
}
