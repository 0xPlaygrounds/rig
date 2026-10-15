//! A conversation kept over time with rig-core and rig-memory alone, no
//! model needed: it is saved and resumed, a rule of recent messages
//! chooses what a request sends, and old tool outputs are cleared to free context.
//!
//! ```sh
//! cargo run -p rig-memory --example recent_messages_and_compaction
//! ```

use rig_core::completion::Message;
use rig_core::message::{
    AssistantContent, AssistantMessage, CallId, ToolCall, ToolFunction, ToolName, UserContent,
};
use rig_memory::recent::{LastMessages, TokenBudget, view};
use rig_memory::{ClearToolOutputs, MemoryStore, load, save, summary_author};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let store = MemoryStore::default();

    // Each turn is loaded, extended and saved, as a server would per request.
    for turn in 0..3 {
        let mut conversation = load(&store, "user-123")?;
        let id = CallId::from_wire(format!("call-{turn}"));
        let read = ToolFunction::new(
            ToolName::new("read")?,
            serde_json::json!({ "path": "a.rs" }),
        );
        let output = vec![rig_core::message::ToolResultContent::text(
            "fn main() {}\n".repeat(400),
        )];
        conversation.extend([
            Message::user(format!("Question {turn}")),
            Message::Assistant(AssistantMessage::new(vec![AssistantContent::ToolCall(
                ToolCall::new(id.clone(), read),
            )])),
            Message::User {
                content: vec![UserContent::tool_result(id, ToolName::new("read")?, output)],
            },
            Message::assistant(format!("Answer {turn}")),
        ]);
        save(&store, "user-123", &mut conversation)?;
    }

    let mut conversation = load(&store, "user-123")?;
    println!(
        "saved and resumed {} messages",
        conversation.messages().len()
    );

    // What a request would send under two rules.
    let last = view(conversation.messages(), &LastMessages(5));
    let budget = view(conversation.messages(), &TokenBudget::new(2_000));
    println!("the last 5 messages keep {}", last.len());
    println!("a 2,000-token budget keeps {}", budget.len());

    // Clearing old tool outputs is a recorded splice that keeps every id.
    let cleared =
        ClearToolOutputs::new(1_000).clear_conversation(&mut conversation, summary_author());
    save(&store, "user-123", &mut conversation)?;
    println!(
        "cleared {} tool outputs (about {} tokens)",
        cleared.results, cleared.tokens
    );
    Ok(())
}
