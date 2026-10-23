//! A conversation that survives a restart: each run opens the session in a
//! SQLite file, shows what earlier runs said, adds a turn and saves it. Run
//! it twice and the second run goes on from the first. It needs no model,
//! credentials or network.
//!
//! ```not_rust
//! cargo run -p rig-sqlite --features memory --example session_store_sqlite [sessions.db]
//! ```

use rig_core::message::Message;
use rig_memory::{load_async, save_async};
use rig_sqlite::SqliteSessionStore;
use tokio_rusqlite::Connection;

#[tokio::main]
async fn main() -> anyhow::Result<()> {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "sessions.db".to_owned());
    let store = SqliteSessionStore::open(Connection::open(&path).await?).await?;

    let mut conversation = load_async(&store, "alice").await?;
    println!("{path} holds {} messages", conversation.messages().len());
    for message in conversation.messages() {
        let text = message.rag_text().or_else(|| message.final_answer());
        println!("  {}", text.unwrap_or_default());
    }

    let turn = conversation.messages().len() / 2 + 1;
    conversation.extend([
        Message::user(format!("This is run {turn}.")),
        Message::assistant(format!("Noted: run {turn}.")),
    ]);
    save_async(&store, "alice", &mut conversation).await?;
    println!("saved run {turn}; run again to go on from it");
    Ok(())
}
