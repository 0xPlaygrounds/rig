use super::*;
use rig_core::{
    message::Message,
    streaming::{BlockAccumulator, BlockClose, BlockKind, StreamEvent},
};

/// Pure parse/emission regression: no model artifact or provider call is needed.
#[test]
fn parsed_missing_ids_keep_their_identity_and_provenance_through_stream_emission() {
    let request = CompletionRequest {
        model: None,
        chat_history: vec![Message::user("tools")],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
        observation: None,
    };
    let raw = r#"<tool_call>{"name":"same","arguments":{"n":1}}</tool_call><tool_call>{"id":"tool-0","name":"same","arguments":{"n":2}}</tool_call><tool_call>{"name":"same","arguments":{"n":3}}</tool_call>"#;
    let parsed = crate::protocol::parse_assistant(raw, &request, ConversationProtocol::Qwen3)
        .expect("valid parsed tool event");
    let expected = parsed.items.clone();
    let mut accumulator = BlockAccumulator::new();
    let mut count = 0;
    emit_parsed_items(parsed.items, |event| {
        let GenerationEvent::ToolCall { id, end } = event else {
            return Err(CandleError::Inference("expected only tool calls".into()));
        };
        if count == 1 {
            assert_eq!(id.wire_str(), Some("tool-0"));
            assert_eq!(end.tool_id.as_deref(), Some("tool-0"));
        } else {
            assert!(id.is_minted());
            assert!(end.tool_id.is_none());
            assert!(end.call_id.is_none());
        }
        accumulator
            .apply(&StreamEvent::BlockStart {
                id: id.clone(),
                kind: BlockKind::ToolCall,
            })
            .expect("valid parsed tool event");
        accumulator
            .apply(&StreamEvent::BlockEnd {
                id,
                end: BlockClose::ToolCall(end),
                block: None,
            })
            .expect("valid parsed tool event");
        count += 1;
        Ok(())
    })
    .expect("valid parsed tool event");
    assert_eq!(count, 3);
    assert_eq!(
        serde_json::to_value(accumulator.finish()).expect("valid parsed tool event"),
        serde_json::to_value(expected).expect("valid parsed tool event")
    );
}
