//! A chat agent that remembers: two turns, then a new agent over the same
//! store, as a restarted program would build, goes on from them. Once with
//! rig-memory's local `DirStore`, and once with a store reached
//! asynchronously, standing in for a database or IndexedDB. The model is
//! scripted, so it needs no credentials or network.
//!
//! ```not_rust
//! cargo run -p rig-agent --example persistent_chat [session-dir]
//! ```

use std::io;
use std::path::PathBuf;

use anyhow::Result;
use rig_agent::AgentBuilder;
use rig_core::test_utils::{MockCompletionModel, MockTurn};
use rig_memory::{AsyncStore, DirStore, Files, LogWriteError, MemoryStore, Store};

/// A store behind a network: every call is a round trip, here a yield to
/// the runtime over a `MemoryStore` standing in for the remote end.
#[derive(Clone, Default)]
struct Remote(MemoryStore);

impl AsyncStore for Remote {
    async fn read_log_async(&self, name: &str) -> io::Result<Vec<u8>> {
        tokio::task::yield_now().await;
        self.0.read_log(name)
    }

    /// The remote end checks the length and appends in one step, as a
    /// conditional `UPDATE` or one transaction does.
    async fn append_log_async(
        &self,
        name: &str,
        len: u64,
        bytes: &[u8],
    ) -> Result<(), LogWriteError> {
        tokio::task::yield_now().await;
        self.0.append_log(name, len, bytes)
    }

    async fn replace_log_async(
        &self,
        name: &str,
        len: u64,
        bytes: &[u8],
    ) -> Result<(), LogWriteError> {
        tokio::task::yield_now().await;
        self.0.replace_log(name, len, bytes)
    }

    async fn read_file_async(&self, name: &str) -> io::Result<Vec<u8>> {
        tokio::task::yield_now().await;
        self.0.read_file(name)
    }

    async fn write_file_async(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        tokio::task::yield_now().await;
        self.0.write_file(name, bytes)
    }
}

/// Two turns with one agent, then one with a new agent over `store`, a
/// clone of it as a restarted program would open.
async fn chat<S: AsyncStore + Clone + Send + Sync + 'static>(store: S) -> Result<()> {
    let model = MockCompletionModel::from_turns([
        MockTurn::text("Nice to meet you, Alice."),
        MockTurn::text("You are Alice."),
    ]);
    let agent = AgentBuilder::new(model)
        .memory(store.clone())
        .conversation("alice")
        .build();
    for prompt in ["Hi, I am Alice.", "Who am I?"] {
        let response = agent.prompt(prompt).await?;
        println!("  user: {prompt}\n  agent: {response}");
    }

    let model = MockCompletionModel::text("Welcome back, Alice.");
    let reloaded = AgentBuilder::new(model.clone())
        .memory(store)
        .conversation("alice")
        .build();
    let response = reloaded.prompt("Do you remember me?").await?;
    let history = model
        .requests()
        .first()
        .map_or(0, |request| request.chat_history.len());
    println!("  (reloaded: the model was sent {history} messages, the 4 kept and the prompt)");
    println!("  agent: {response}");
    Ok(())
}

#[tokio::main]
async fn main() -> Result<()> {
    let dir = std::env::args()
        .nth(1)
        .map_or_else(|| PathBuf::from("target/persistent_chat"), PathBuf::from);
    if dir.exists() {
        std::fs::remove_dir_all(&dir)?;
    }
    println!("A local store, in {}:", dir.display());
    chat(DirStore::open(&dir)?).await?;

    println!("A store reached asynchronously:");
    chat(Remote::default()).await
}
