use std::sync::Arc;

use rig_core::completion::{CompletionRequest, Message, ToolDefinition};

use super::*;

fn tools() -> Vec<Arc<ToolDefinition>> {
    let schema = serde_json::json!({
        "type": "object",
        "properties": {"path": {"type": "string", "description": "x".repeat(4_000)}},
    });
    let name = rig_core::message::ToolName::new("read").expect("a name");
    vec![Arc::new(ToolDefinition::new(name, "Reads a file.", schema))]
}

/// A request built to send counts its tools and each of its messages: a system prompt, the
/// user's message and a reply.
#[test]
fn a_request_built_to_send_counts_its_tools_and_messages() {
    let messages = [
        Message::system("You are a coding agent."),
        Message::user("read the file"),
        Message::assistant("x".repeat(400)),
    ];
    let mut request = CompletionRequest::new(Message::user("unused"));
    request.chat_history = messages.iter().cloned().map(Arc::new).collect();
    request.tools.clone_from(&tools());
    let offered = request_estimate(&request);
    request.tools.clear();
    assert!(request_estimate(&request) < offered, "the tools count");
    let counter = HeuristicTokenCounter::default();
    let each: usize = messages.iter().map(|message| counter.count(message)).sum();
    assert_eq!(request_estimate(&request), each as u64);
}

#[test]
fn an_image_costs_about_twelve_hundred_tokens() {
    let image = rig_core::message::UserContent::image_base64(
        "aGk=",
        Some(rig_core::message::ImageMediaType::PNG),
        None,
    );
    let counter = HeuristicTokenCounter::default();
    assert_eq!(counter.count_user(&image), ATTACHMENT_TOKENS);
}
