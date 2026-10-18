//! Three messages through `ChatService` over the app's in-memory
//! repository, then the conversation as the app's admin listing shows it.
//! The model is scripted, so it needs no credentials or network.
//!
//! ```not_rust
//! cargo run -p corpus-rf-02 --example app_chat
//! ```

use std::sync::Arc;

use corpus_rf_02::{ChatService, Conversations, InMemoryConversations, decode};
use rig_agent::AgentBuilder;
use rig_core::message::Message;
use rig_core::test_utils::{MockCompletionModel, MockTurn};

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let repo = Arc::new(InMemoryConversations::default());
    repo.create("welcome", "alice", "Getting started", &["onboarding"]);
    let model = MockCompletionModel::from_turns([
        MockTurn::text("Hi Alice, how can I help?"),
        MockTurn::text("Your plan renews on the 1st."),
        MockTurn::text("Glad to help."),
    ]);
    let helper = AgentBuilder::new(model).preamble("You help.").build();
    let shared: Arc<dyn Conversations> = repo.clone();
    let service = ChatService::new(shared, [("helper", helper)]);
    for text in ["Hello", "When does my plan renew?", "Thanks"] {
        service.send("alice", "welcome", text, "helper").await?;
    }

    let row = repo.row("welcome")?;
    println!("{} (owner {}, tags {:?})", row.title, row.owner, row.tags);
    for stored in row.messages {
        let (role, text) = match decode(stored.body)? {
            Message::System { content } => ("system", content),
            Message::Assistant(turn) => ("assistant", turn.text()),
            message => ("user", message.rag_text().unwrap_or_default()),
        };
        println!("  #{} {role}: {text}", stored.id);
    }
    Ok(())
}
