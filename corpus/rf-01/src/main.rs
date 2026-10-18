//! `shared-thread demo`: two simulated clients type into one thread at
//! once, over a store with latency, and the merged transcript is printed.
//! The model is rig's scripted mock, so the demo runs offline.

use std::sync::Arc;
use std::time::Duration;

use corpus_rf_01::{InMemoryStore, SharedThread};
use rig_agent::AgentBuilder;
use rig_core::message::Message;
use rig_core::test_utils::{MockCompletionModel, MockTurn};

const THREAD: &str = "customer-7";

#[tokio::main]
async fn main() {
    if std::env::args().nth(1).as_deref() != Some("demo") {
        eprintln!("usage: shared-thread demo");
        std::process::exit(2);
    }
    let turns = (1..=6).map(|n| MockTurn::text(format!("Support reply #{n}")));
    let agent = AgentBuilder::new(MockCompletionModel::from_turns(turns)).build();
    let store = InMemoryStore::with_latency(Duration::from_millis(20));
    let shared = Arc::new(SharedThread::new(agent, store.clone()));

    let clients = ["web", "slack"].map(|client| {
        let shared = Arc::clone(&shared);
        tokio::spawn(async move {
            for n in 1..=3 {
                let question = format!("[{client}] question {n}");
                if let Err(error) = shared.ask(THREAD, &question).await {
                    eprintln!("{client}: {error}");
                }
            }
        })
    });
    for client in clients {
        if let Err(error) = client.await {
            eprintln!("client task failed: {error}");
        }
    }

    for stored in store.messages(THREAD) {
        let line = match &stored.message {
            Message::Assistant(_) => stored.message.final_answer().map_or_else(
                || "assistant: [tool call]".to_owned(),
                |t| format!("assistant: {t}"),
            ),
            other => format!("user: {}", other.rag_text().unwrap_or_default()),
        };
        println!("{line}");
    }
}
