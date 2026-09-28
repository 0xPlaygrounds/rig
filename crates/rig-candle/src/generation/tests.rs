use super::*;
use rig_core::message::Message;

/// Pure parse/emission regression: no model artifact or provider call is needed.
#[test]
fn parsed_missing_ids_keep_their_identity_and_provenance_through_stream_emission() {
    let request = CompletionRequest {
        model: None,
        chat_history: rig_core::NonEmpty::new(Message::user("tools")),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };
    let raw = r#"<tool_call>{"name":"same","arguments":{"n":1}}</tool_call><tool_call>{"id":"tool-0","name":"same","arguments":{"n":2}}</tool_call><tool_call>{"name":"same","arguments":{"n":3}}</tool_call>"#;
    let parsed = crate::protocol::parse_assistant(raw, &request, ConversationProtocol::Qwen3)
        .expect("valid parsed tool event");
    let expected = parsed.items.clone();
    let mut emitted = Vec::new();
    emit_parsed_items(parsed.items, |event| {
        let GenerationEvent::ToolCall(call) = event else {
            return Err(CandleError::Inference("expected only tool calls".into()));
        };
        emitted.push(rig_core::message::AssistantContent::ToolCall(call));
        Ok(())
    })
    .expect("valid parsed tool event");
    assert_eq!(emitted.len(), 3);
    // The call that named its id keeps it; the others keep the ids rig
    // issued when parsing.
    let ids: Vec<_> = emitted
        .iter()
        .filter_map(|content| match content {
            rig_core::message::AssistantContent::ToolCall(call) => Some(call.id.clone()),
            _ => None,
        })
        .collect();
    assert_eq!(ids[1].to_string(), "tool-0");
    assert!(ids[0].provider().is_none() && ids[2].provider().is_none());
    assert_eq!(
        serde_json::to_value(emitted).expect("valid parsed tool event"),
        serde_json::to_value(expected).expect("valid parsed tool event")
    );
}
