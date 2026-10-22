use super::*;

fn thinking() -> Value {
    json!({
        "type": "thinking", "thinking": "  signed\nreasoning 🦀\t",
        "signature": "sig_+/==\n", "estimated_tokens": 17,
        "extension": {"estimated_tokens": 23}
    })
}

fn request_thinking() -> Value {
    json!({
        "type": "thinking", "thinking": "  signed\nreasoning 🦀\t",
        "signature": "sig_+/==\n", "extension": {"estimated_tokens": 23}
    })
}

fn assert_replay(response: completion::CompletionResponse) {
    let original = response.clone();
    assert_eq!(
        response.choice.first().and_then(|part| part.native_item()),
        Some(&thinking())
    );
    let Some(turn) = response.message() else {
        panic!("the response must produce a history turn");
    };
    let body = prepared_body(vec![message::Message::user("hello"), turn]);
    assert_eq!(
        body.pointer("/messages/1/content/0"),
        Some(&request_thinking())
    );
    assert_eq!(
        response.choice.first().and_then(|part| part.native_item()),
        Some(&thinking())
    );
    assert_eq!(response.raw, original.raw);
}

/// This synthetic regression checks request projection; no live recording or credentials
/// are available for the reported response-only field.
#[test]
fn unary_response_only_thinking_is_projected_without_mutating_response() {
    let reply = json!({
        "type": "message", "id": "msg_1", "model": CLAUDE_SONNET_4_6,
        "role": "assistant", "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [thinking(), {"type": "text", "text": "done"}]
    });
    let response = match fold_reply(&reply) {
        Ok(response) => response,
        Err(error) => panic!("reply must decode: {error}"),
    };
    assert_eq!(
        response.raw.pointer("/content/0/estimated_tokens"),
        Some(&json!(17))
    );
    assert_replay(response);
}

/// This synthetic stream checks decoder-to-request projection without live credentials.
#[test]
fn streamed_response_only_thinking_is_projected_without_mutating_response() {
    let mut start = thinking();
    if let Some(object) = start.as_object_mut() {
        object.insert("thinking".into(), json!(""));
        object.insert("signature".into(), json!(""));
    }
    let frames = [
        json!({"type": "message_start", "message": {
            "id": "msg_1", "type": "message", "role": "assistant", "model": CLAUDE_SONNET_4_6,
            "content": [], "stop_reason": null, "stop_sequence": null,
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }}),
        json!({"type": "content_block_start", "index": 0, "content_block": start}),
        json!({"type": "content_block_delta", "index": 0,
            "delta": {"type": "thinking_delta", "thinking": "  signed\nreasoning 🦀\t"}}),
        json!({"type": "content_block_delta", "index": 0,
            "delta": {"type": "signature_delta", "signature": "sig_+/==\n"}}),
        json!({"type": "content_block_stop", "index": 0}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn", "stop_sequence": null},
            "usage": {"output_tokens": 17}}),
        json!({"type": "message_stop"}),
    ];
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let response = match crate::test_utils::decode_reply(
        &wire,
        &hello_request(),
        Mode::Streaming,
        frames
            .iter()
            .map(|frame| WireFrame::Text(frame.to_string())),
        Value::Null,
    ) {
        Ok(response) => response,
        Err(error) => panic!("stream must decode: {error}"),
    };
    assert_replay(response);
}

#[test]
fn mixed_response_blocks_project_only_signed_thinking_estimates() {
    let redacted = json!({"type": "redacted_thinking", "data": "opaque_+/==\n",
        "extension": {"estimated_tokens": 9}});
    let unsigned = json!({"type": "thinking", "thinking": "unsigned plan",
        "signature": "", "estimated_tokens": 5});
    let tool = json!({"type": "tool_use", "id": "toolu_1", "name": "add",
        "input": {"estimated_tokens": 42}, "caller": {"type": "direct"},
        "extension": {"trace": "preserve"}});
    let reply = json!({
        "type": "message", "id": "msg_mixed", "model": CLAUDE_SONNET_4_6,
        "role": "assistant", "stop_reason": "tool_use", "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [thinking(), redacted.clone(), unsigned, tool.clone()]
    });
    let response = match fold_reply(&reply) {
        Ok(response) => response,
        Err(error) => panic!("reply must decode: {error}"),
    };
    let original = response.clone();
    let Some(turn) = response.message() else {
        panic!("the response must produce a history turn");
    };
    let Ok(name) = message::ToolName::new("add") else {
        panic!("valid tool name");
    };
    let body = prepared_body(vec![
        message::Message::user("hello"),
        turn,
        message::Message::tool_result(message::CallId::from_wire("toolu_1"), name, "42"),
    ]);
    assert_eq!(
        body.pointer("/messages/1/content"),
        Some(&json!([request_thinking(), redacted,
            {"type": "text", "text": "unsigned plan"}, tool]))
    );
    assert_eq!(response.raw, original.raw);
    assert_eq!(response.raw.get("content"), reply.get("content"));
    assert_eq!(
        response.choice.first().and_then(|part| part.native_item()),
        Some(&thinking())
    );
}

/// Saved native items need request-only projection without rewriting session data.
/// This local serialization invariant does not require a provider cassette.
#[test]
fn saved_response_only_thinking_preserves_other_native_fields() {
    let tool = json!({"type": "tool_use", "id": "toolu_1", "name": "add",
        "input": {"estimated_tokens": 42, "nested": {"estimated_tokens": 43}},
        "estimated_tokens": 44});
    let Ok(name) = message::ToolName::new("add") else {
        panic!("valid tool name");
    };
    let parts = vec![
        message::AssistantContent::reasoning("  signed\nreasoning 🦀\t").with_native(thinking()),
        message::AssistantContent::tool_call(
            "toolu_1",
            name,
            json!({"estimated_tokens": 42, "nested": {"estimated_tokens": 43}}),
        )
        .with_native(tool.clone()),
    ];
    let sent = assistant_wire(parts.clone());
    assert_eq!(sent, vec![request_thinking(), tool]);
    assert_eq!(
        parts.first().and_then(|part| part.native_item()),
        Some(&thinking())
    );
}
