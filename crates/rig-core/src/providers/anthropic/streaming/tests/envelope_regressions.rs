use super::*;

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
#[tokio::test]
async fn raw_sse_unknown_preamble_and_fragmented_utf8_preserve_content() {
    use crate::test_utils::SequencedStreamingHttpClient;

    let mut frames = vec![json!({"type": "ping"}), json!({"type": "gateway_ready"})];
    frames.extend(reply(
        vec![block(0, json!({"type": "text", "text": "héllo 🦀"}), &[])],
        "end_turn",
    ));
    frames.push(json!({"type": "message_stop"}));
    let bytes = frames
        .iter()
        .map(|frame| {
            format!(
                "event: {}\ndata: {frame}\n\n",
                frame
                    .get("type")
                    .and_then(Value::as_str)
                    .unwrap_or_default()
            )
        })
        .collect::<String>();
    let chunks = bytes
        .as_bytes()
        .iter()
        .map(|byte| Ok(bytes::Bytes::copy_from_slice(&[*byte])))
        .collect();
    let model = crate::driver::Model::new(
        AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6),
        SequencedStreamingHttpClient::new(chunks),
    );
    let stream = match model.stream(CompletionRequest::new("hello")) {
        Ok(stream) => stream,
        Err(error) => panic!("stream failed to open: {error}"),
    };
    let result = match stream.finish().await {
        Ok(result) => result,
        Err(error) => panic!("stream failed: {error}"),
    };
    assert!(
        matches!(result.choice.as_slice(), [AssistantContent::Text(text)] if text.text == "héllo 🦀")
    );
    assert_eq!(result.stop(), StopReason::Stop);
}

fn response(frames: &[Value]) -> crate::completion::CompletionResponse {
    match streamed(frames) {
        Ok(response) => response,
        Err(error) => panic!("stream failed: {error}"),
    }
}

#[test]
fn duplicate_envelopes_preserve_first_metadata_and_closed_content() {
    for (content, reason) in [
        (
            block(0, json!({"type": "text", "text": "hello"}), &[]),
            "end_turn",
        ),
        (
            block(
                0,
                json!({"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}}),
                &[json!({"type": "input_json_delta", "partial_json": "{\"query\":\"rust\"}"})],
            ),
            "tool_use",
        ),
    ] {
        let expected = response(&reply(vec![content.clone()], reason));
        for replay in [false, true] {
            let mut blocks = vec![
                content.clone(),
                vec![json!({"type": "message_start", "message": {
                    "id": "msg_duplicate", "model": "wrong-model",
                    "usage": {"input_tokens": 99, "output_tokens": 99}
                }})],
            ];
            if replay {
                blocks.push(content.clone());
            }
            let result = response(&reply(blocks, reason));
            assert_eq!(result.origin.response_id.as_deref(), Some("msg_1"));
            assert_eq!(result.usage, expected.usage);
            assert_eq!(result.choice, expected.choice);
            assert_eq!(result.stop(), expected.stop());
            assert_eq!(result.raw, expected.raw);
        }
    }
}

#[test]
fn duplicate_envelope_does_not_reset_an_open_block() {
    let result = response(&reply(
        vec![vec![
            json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "hello"}}),
            json!({"type": "message_start", "message": {"id": "duplicate", "usage": {"input_tokens": 99}}}),
            json!({"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " world"}}),
            json!({"type": "content_block_stop", "index": 0}),
        ]],
        "end_turn",
    ));
    assert!(
        matches!(result.choice.as_slice(), [AssistantContent::Text(text)] if text.text == "hello world")
    );
    assert_eq!(result.usage.input_tokens, Some(3));
}

#[test]
fn duplicate_block_starts_preserve_partial_text_and_tool_arguments() {
    for (start, first, rest, reason) in [
        (
            json!({"type": "text", "text": ""}),
            json!({"type": "text_delta", "text": "hello"}),
            json!({"type": "text_delta", "text": " world"}),
            "end_turn",
        ),
        (
            json!({"type": "tool_use", "id": "call_1", "name": "lookup", "input": {}}),
            json!({"type": "input_json_delta", "partial_json": "{\"query\":"}),
            json!({"type": "input_json_delta", "partial_json": "\"rust\"}"}),
            "tool_use",
        ),
    ] {
        let expected = response(&reply(
            vec![block(0, start.clone(), &[first.clone(), rest.clone()])],
            reason,
        ));
        let result = response(&reply(
            vec![vec![
                json!({"type": "content_block_start", "index": 0, "content_block": start}),
                json!({"type": "content_block_delta", "index": 0, "delta": first}),
                json!({"type": "content_block_start", "index": 0, "content_block": start}),
                json!({"type": "content_block_delta", "index": 0, "delta": rest}),
                json!({"type": "content_block_stop", "index": 0}),
            ]],
            reason,
        ));
        assert_eq!(result.choice, expected.choice);
        assert_eq!(result.stop(), expected.stop());
        assert_eq!(result.usage, expected.usage);
        assert_eq!(result.raw, expected.raw);
    }
}

#[test]
fn unknown_events_and_pings_do_not_change_the_reply() {
    let mut frames = vec![
        json!({"type": "ping"}),
        json!({"type": "gateway_ready", "data": 17}),
    ];
    frames.extend(reply(
        vec![block(0, json!({"type": "text", "text": "hello"}), &[])],
        "end_turn",
    ));
    frames.extend(reply(
        vec![block(0, json!({"type": "text", "text": "wrong"}), &[])],
        "max_tokens",
    ));
    let result = response(&frames);
    assert!(
        matches!(result.choice.as_slice(), [AssistantContent::Text(text)] if text.text == "hello")
    );
    assert_eq!(result.stop(), StopReason::Stop);
    assert_eq!(result.usage.output_tokens, Some(5));
}

#[test]
fn missing_usage_and_nonterminal_delta_are_tolerated() {
    let result = response(&[
        json!({"type": "message_start", "message": {"id": "msg_no_usage"}}),
        json!({"type": "message_delta"}),
        json!({"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": "hello"}}),
        json!({"type": "message_delta", "delta": {"stop_reason": "end_turn"}}),
    ]);
    assert_eq!(result.usage.input_tokens, None);
    assert_eq!(result.usage.output_tokens, None);
    assert_eq!(result.stop(), StopReason::Stop);
}

#[test]
fn native_server_tools_keep_wire_order_and_fragmented_input() {
    let server = json!({"type": "server_tool_use", "id": "srv_1", "name": "tool_search_tool_regex", "input": {}});
    let search_result = json!({"type": "tool_search_tool_result", "tool_use_id": "srv_1", "content": [{"type": "tool_reference", "tool_name": "lookup"}]});
    let signed =
        json!({"type": "thinking", "thinking": "  exact signed bytes\n", "signature": "sig"});
    let client =
        json!({"type": "tool_use", "id": "call_1", "name": "lookup", "input": {"query": "rust"}});
    let result = response(&reply(
        vec![
            block(0, signed.clone(), &[]),
            block(
                1,
                server,
                &[
                    json!({"type": "input_json_delta", "partial_json": "{\"pattern\":"}),
                    json!({"type": "input_json_delta", "partial_json": "\"lookup\"}"}),
                ],
            ),
            block(2, search_result.clone(), &[]),
            block(3, client.clone(), &[]),
        ],
        "tool_use",
    ));
    let native: Vec<_> = result.choice.iter().filter_map(item).cloned().collect();
    assert_eq!(
        native,
        vec![
            signed,
            json!({"type": "server_tool_use", "id": "srv_1", "name": "tool_search_tool_regex", "input": {"pattern": "lookup"}}),
            search_result,
            client
        ]
    );
    assert_eq!(result.stop(), StopReason::ToolUse);
}

#[test]
fn truncated_streams_never_claim_success_even_after_a_closed_tool() {
    for blocks in [
        vec![],
        vec![block(
            0,
            json!({"type": "tool_use", "id": "done", "name": "lookup", "input": {"query": "complete"}}),
            &[],
        )],
    ] {
        let mut frames = reply(blocks, "tool_use");
        frames.pop();
        frames.extend([
            json!({"type": "content_block_start", "index": 1, "content_block": {"type": "tool_use", "id": "cut", "name": "lookup", "input": {}}}),
            json!({"type": "content_block_delta", "index": 1, "delta": {"type": "input_json_delta", "partial_json": "{\"query\":"}}),
        ]);
        assert!(matches!(streamed(&frames), Err(ProviderError::Truncated)));
    }
}

#[test]
fn malformed_closed_tool_is_not_replayable() {
    for arguments in ["{bad json", "{\"query\":"] {
        let result = response(&reply(
            vec![block(
                0,
                json!({"type": "tool_use", "id": "bad", "name": "lookup", "input": {}}),
                &[json!({"type": "input_json_delta", "partial_json": arguments})],
            )],
            "tool_use",
        ));
        assert_eq!(result.stop(), StopReason::ToolUse);
        assert!(
            matches!(result.choice.as_slice(), [AssistantContent::ToolCall(call)]
            if call.function.invalid_arguments.as_deref() == Some(arguments))
        );
        assert!(result.choice.iter().all(|content| item(content).is_none()));
    }
}
