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
use rig_memory::{AsyncStore, DirStore, Files, MemoryStore, Store};

/// A store behind a network: every call is a round trip, here a yield to
/// the runtime over a `MemoryStore` standing in for the remote end.
#[derive(Clone, Default)]
struct Remote(MemoryStore);

impl AsyncStore for Remote {
    async fn logs(&self) -> io::Result<Vec<String>> {
        tokio::task::yield_now().await;
        Store::logs(&self.0)
    }

    async fn read_log(&self, name: &str) -> io::Result<Vec<u8>> {
        tokio::task::yield_now().await;
        Store::read_log(&self.0, name)
    }

    async fn append_log(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        tokio::task::yield_now().await;
        Store::append_log(&self.0, name, bytes)
    }

    async fn replace_log(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        tokio::task::yield_now().await;
        Store::replace_log(&self.0, name, bytes)
    }

    async fn files(&self) -> io::Result<Vec<String>> {
        tokio::task::yield_now().await;
        Files::files(&self.0)
    }

    async fn read_file(&self, name: &str) -> io::Result<Vec<u8>> {
        tokio::task::yield_now().await;
        Files::read_file(&self.0, name)
    }

    async fn write_file(&self, name: &str, bytes: &[u8]) -> io::Result<()> {
        tokio::task::yield_now().await;
        Files::write_file(&self.0, name, bytes)
    }
}

/// Two turns with one agent, then one with a new agent over `reopen()`.
async fn chat<S: AsyncStore + Send + Sync + 'static>(reopen: impl Fn() -> Result<S>) -> Result<()> {
    let model = MockCompletionModel::from_turns([
        MockTurn::text("Nice to meet you, Alice."),
        MockTurn::text("You are Alice."),
    ]);
    let agent = AgentBuilder::new(model)
        .memory(reopen()?)
        .conversation("alice")
        .build();
    for prompt in ["Hi, I am Alice.", "Who am I?"] {
        let response = agent.prompt(prompt).await?;
        println!("  user: {prompt}\n  agent: {response}");
    }

    let model = MockCompletionModel::text("Welcome back, Alice.");
    let reloaded = AgentBuilder::new(model.clone())
        .memory(reopen()?)
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
    chat(|| Ok(DirStore::open(&dir)?)).await?;

    println!("A store reached asynchronously:");
    let remote = Remote::default();
    chat(|| Ok(remote.clone())).await
}
