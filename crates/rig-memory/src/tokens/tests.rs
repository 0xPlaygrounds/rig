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

/// A request built to send counts its tools and its messages.
#[test]
fn a_request_built_to_send_counts_its_tools_and_messages() {
    let mut request = CompletionRequest::new(Message::user("read the file"));
    request.tools.clone_from(&tools());
    let offered = request_estimate(&request);
    request.tools.clear();
    assert!(request_estimate(&request) < offered, "the tools count");
    let counter = HeuristicTokenCounter::default();
    assert_eq!(
        request_estimate(&request),
        counter.count(&Message::user("read the file")) as u64
    );
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
