//! Two workers sharing one conversation in PostgreSQL, then a restart. Each
//! worker has its own pool, as separate processes would, and adds turns to
//! the session `support-42` at the same time. A save made on a stale copy is
//! refused with `SessionError::Changed`, so the worker loads the session
//! again and retries: no turn is lost or written twice. A fresh pool then
//! loads every turn. It needs no model or credentials, only a database:
//!
//! ```not_rust
//! docker run -d -e POSTGRES_PASSWORD=postgres -p 5432:5432 postgres:17
//! export DATABASE_URL=postgres://postgres:postgres@localhost:5432/postgres
//! cargo run -p rig-postgres --features memory --example session_store_postgres
//! ```

use rig_core::message::Message;
use rig_memory::{SessionError, load_async, save_async};
use rig_postgres::PostgresSessionStore;
use sqlx::PgPool;

const SESSION: &str = "support-42";

async fn open() -> anyhow::Result<PostgresSessionStore> {
    let pool = PgPool::connect(&std::env::var("DATABASE_URL")?).await?;
    Ok(PostgresSessionStore::open(pool).await?)
}

/// Adds `turns` turns to the session, each on the session as it is now.
async fn worker(name: &str, turns: usize) -> anyhow::Result<usize> {
    let store = open().await?;
    let mut retries = 0;
    for turn in 1..=turns {
        loop {
            let mut conversation = load_async(&store, SESSION).await?;
            conversation.extend([
                Message::user(format!("{name}: question {turn}")),
                Message::assistant(format!("{name}: answer {turn}")),
            ]);
            match save_async(&store, SESSION, &mut conversation).await {
                Ok(()) => break,
                Err(SessionError::Changed { .. }) => retries += 1,
                Err(error) => return Err(error.into()),
            }
        }
    }
    Ok(retries)
}

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let (first, second) = tokio::try_join!(worker("first", 5), worker("second", 5))?;
    println!(
        "both workers saved; {} saves were refused and retried",
        first + second
    );

    // A restart: a new pool and store over the same database.
    let conversation = load_async(&open().await?, SESSION).await?;
    println!(
        "{SESSION} now holds {} messages",
        conversation.messages().len()
    );
    Ok(())
}
