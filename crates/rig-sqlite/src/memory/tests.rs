use std::path::PathBuf;

use rig_memory::conformance::{Violation, check_store};
use tokio_rusqlite::Connection;

use super::SqliteSessionStore;

fn failed(check: &'static str) -> impl Fn(std::io::Error) -> Violation {
    move |error| Violation {
        check,
        detail: error.to_string(),
    }
}

async fn open(path: &PathBuf) -> Result<SqliteSessionStore, Violation> {
    let conn = Connection::open(path)
        .await
        .map_err(|error| failed("the database opens")(std::io::Error::other(error)))?;
    SqliteSessionStore::open(conn)
        .await
        .map_err(failed("the store makes its tables"))
}

/// Two connections to one database file, as two processes hold.
#[tokio::test]
async fn sqlite_session_store_keeps_the_contract() -> Result<(), Violation> {
    let path = std::env::temp_dir().join(format!("rig-sqlite-memory-{}.db", std::process::id()));
    let _ = std::fs::remove_file(&path);
    let checked = check_store(&open(&path).await?, &open(&path).await?).await;
    let reopened = open(&path).await;
    let _ = std::fs::remove_file(&path);
    checked?;
    reopened.map(drop)
}
