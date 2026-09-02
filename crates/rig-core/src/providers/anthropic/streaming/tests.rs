use super::super::completion::{
    AnthropicRequestParams, CLAUDE_OPUS_4_8, CacheControl, CacheTtl, Message, SystemContent,
    apply_prompt_cache_control, build_tool_definitions, resolve_top_level_cache_control,
};
use super::*;
use crate::completion::Message as RigMessage;
use crate::completion::request::Document as RigDocument;
use crate::streaming::RawStreamingToolCall;
use async_stream::stream;
use futures::StreamExt;

/// Normalize a hand-built Anthropic raw stream exactly as
/// [`GenericCompletionModel::stream`] does, so aggregation assertions run
/// against the same terminal-record mapping as the real path.
#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
fn to_stream_result(
    stream: impl futures::Stream<
        Item = Result<RawStreamingChoice<StreamingCompletionResponse>, CompletionError>,
    > + Send
    + 'static,
) -> crate::streaming::StreamingResult {
    crate::streaming::normalize_stream(Box::pin(stream), |response| {
        Ok(StreamFinal::from(("anthropic", response)))
    })
}

#[cfg(all(target_arch = "wasm32", target_os = "unknown"))]
fn to_stream_result(
    stream: impl futures::Stream<
        Item = Result<RawStreamingChoice<StreamingCompletionResponse>, CompletionError>,
    > + 'static,
) -> crate::streaming::StreamingResult {
    crate::streaming::normalize_stream(Box::pin(stream), |response| {
        Ok(StreamFinal::from(("anthropic", response)))
    })
}

/// Build the streaming request body the way [`GenericCompletionModel::raw_stream`]
/// does — the shared typed request, then the streaming-only patches — without
/// needing a client to reach the prelude.
fn built_streaming_body(
    model: &str,
    request: CompletionRequest,
    strict_tools: bool,
) -> Result<Value, CompletionError> {
    let typed = AnthropicCompletionRequest::try_from_params::<
        crate::providers::anthropic::client::Anthropic,
    >(
        AnthropicRequestParams {
            model,
            request,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        },
        strict_tools,
    )?;

    streaming_body(&typed)
}

#[test]
fn test_streaming_tool_build_marks_final_combined_tool() {
    let mut additional_params = json!({
        "tools": [{
            "name": "provider_tool",
            "description": "Provider tool",
            "input_schema": {"type": "object"}
        }]
    });

    let mut tools = build_tool_definitions::<crate::providers::anthropic::client::Anthropic>(
        vec![crate::completion::ToolDefinition {
            name: "rig_tool".to_string(),
            description: "Rig tool".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
        }],
        &mut additional_params,
        false,
    )
    .unwrap();
    let mut system: Vec<SystemContent> = Vec::new();
    let mut messages: Vec<Message> = Vec::new();
    apply_prompt_cache_control(&mut system, &mut messages, &mut tools, true, None, None).unwrap();

    assert_eq!(tools.len(), 2);
    assert!(tools[0].get("cache_control").is_none());
    assert_eq!(tools[1]["name"], "provider_tool");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
}

#[test]
fn streaming_request_keeps_documents_after_leading_system_messages() {
    let request = CompletionRequest {
        model: None,
        chat_history: vec![
            RigMessage::system("System prompt"),
            RigMessage::assistant("Earlier assistant turn"),
            RigMessage::system("Mid-conversation instruction"),
            RigMessage::user("Prompt"),
        ],
        documents: vec![RigDocument {
            id: "doc1".to_string(),
            text: "Document text.".to_string(),
            additional_props: Default::default(),
        }],
        tools: vec![],
        temperature: None,
        max_tokens: Some(64),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let body = built_streaming_body(CLAUDE_OPUS_4_8, request, false)
        .expect("streaming request body should build");

    assert_eq!(body["system"][0]["text"], "System prompt");
    assert_eq!(body["system"][1]["text"], "Mid-conversation instruction");
    let messages = body["messages"]
        .as_array()
        .expect("messages should be array");
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "user");
    assert!(
        messages[0].to_string().contains("<file id: doc1>"),
        "document message should follow top-level system: {messages:?}"
    );
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(
        messages
            .iter()
            .filter(|message| message.to_string().contains("<file id: doc1>"))
            .count(),
        1,
        "document message should appear exactly once: {messages:?}"
    );
}

#[test]
fn streaming_body_is_blocking_body_plus_stream_flag_and_carries_output_schema() {
    let schema: schemars::Schema = serde_json::from_value(json!({
        "title": "WeatherResponse",
        "type": "object",
        "properties": { "city": { "type": "string" } }
    }))
    .expect("schema should deserialize");

    let request = CompletionRequest {
        model: None,
        chat_history: vec![
            RigMessage::system("You are helpful"),
            RigMessage::user("What's the weather?"),
        ],
        documents: vec![],
        tools: vec![],
        temperature: Some(0.5),
        max_tokens: Some(64),
        tool_choice: None,
        additional_params: None,
        output_schema: Some(schema),
        record_telemetry_content: false,
    };

    let streaming_body = built_streaming_body(CLAUDE_OPUS_4_8, request.clone(), false)
        .expect("streaming request body should build");

    // The streaming endpoint flag is set.
    assert_eq!(streaming_body["stream"], serde_json::Value::Bool(true));

    // Regression: `output_schema` now reaches the streaming wire as
    // `output_config` (the hand-rolled body dropped it entirely, so this
    // assertion would have failed before the typed-request unification).
    assert_eq!(
        streaming_body["output_config"]["format"]["type"],
        "json_schema"
    );
    assert!(
        streaming_body["output_config"]["format"]["schema"].is_object(),
        "streaming body must carry the structured-output schema: {streaming_body}"
    );

    // Unification invariant: the streaming body is exactly the blocking body
    // (built via the same typed request) plus `stream: true`. Pins the two
    // wire formats together so a future edit can't reintroduce drift.
    let blocking = AnthropicCompletionRequest::try_from(AnthropicRequestParams {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .expect("blocking request body should build");
    let mut expected = serde_json::to_value(&blocking).expect("serialize blocking body");
    expected
        .as_object_mut()
        .expect("body is an object")
        .insert("stream".to_string(), serde_json::Value::Bool(true));

    assert_eq!(streaming_body, expected);
}

#[test]
fn streaming_body_keeps_explicit_tool_choice_auto_when_tools_present_but_unset() {
    let request = CompletionRequest {
        model: None,
        chat_history: vec![RigMessage::user("Add 2 and 3")],
        documents: vec![],
        tools: vec![crate::completion::ToolDefinition {
            name: "add".to_string(),
            description: "Add x and y".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "x": { "type": "integer" } }
            }),
        }],
        temperature: None,
        max_tokens: Some(64),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let body = built_streaming_body(CLAUDE_OPUS_4_8, request, false)
        .expect("streaming request body should build");

    // Tools advertised + `tool_choice` unset must still carry the explicit
    // `auto` the streaming wire format has always sent (parity with recorded
    // fixtures), even though the blocking typed request omits it.
    assert_eq!(body["tool_choice"], json!({ "type": "auto" }));
    assert!(body["tools"].is_array());
}

#[test]
fn streaming_body_applies_strict_tool_opt_in() {
    let request = CompletionRequest {
        model: None,
        chat_history: vec![RigMessage::user("Look this up")],
        documents: vec![],
        tools: vec![crate::completion::ToolDefinition {
            name: "lookup".to_string(),
            description: "Look up a value".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
        }],
        temperature: None,
        max_tokens: Some(64),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let body = built_streaming_body(CLAUDE_OPUS_4_8, request, true)
        .expect("streaming request body should build");

    assert_eq!(body["tools"][0]["strict"], true);
    assert_eq!(
        body["tools"][0]["input_schema"]["additionalProperties"],
        false
    );
    assert_eq!(
        body["tools"][0]["input_schema"]["required"],
        json!(["query"])
    );
}

#[test]
fn streaming_body_drops_tool_choice_when_no_tools_are_advertised() {
    // The typed request serializes a caller-set `tool_choice` regardless of
    // whether tools are present, but the streaming path has always emitted
    // `tool_choice` *only* alongside a non-empty tool set (Anthropic rejects it
    // otherwise). A `tool_choice` set with no tools must not reach the wire.
    let request = CompletionRequest {
        model: None,
        chat_history: vec![RigMessage::user("Hi")],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: Some(64),
        tool_choice: Some(crate::message::ToolChoice::Auto),
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let body = built_streaming_body(CLAUDE_OPUS_4_8, request, false)
        .expect("streaming request body should build");

    assert!(
        body.get("tool_choice").is_none(),
        "tool_choice must be omitted when no tools are advertised: {body}"
    );
    assert!(body.get("tools").is_none());
}

#[test]
fn test_streaming_prompt_cache_control_uses_raw_top_level_ttl() {
    let mut additional_params = json!({
        "cache_control": {"type": "ephemeral", "ttl": "1h"}
    });
    let top_level_cache_control =
        resolve_top_level_cache_control(false, None, &mut additional_params).unwrap();
    let mut tools = build_tool_definitions::<crate::providers::anthropic::client::Anthropic>(
        vec![crate::completion::ToolDefinition {
            name: "rig_tool".to_string(),
            description: "Rig tool".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
        }],
        &mut additional_params,
        false,
    )
    .unwrap();
    let mut system = vec![SystemContent::Text {
        text: "System prompt".to_string(),
        cache_control: None,
    }];
    let mut messages: Vec<Message> = Vec::new();

    apply_prompt_cache_control(
        &mut system,
        &mut messages,
        &mut tools,
        true,
        None,
        top_level_cache_control.as_ref(),
    )
    .unwrap();

    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    match &system[0] {
        SystemContent::Text {
            cache_control: Some(CacheControl::Ephemeral { ttl }),
            ..
        } => assert_eq!(ttl.as_ref(), Some(&CacheTtl::OneHour)),
        other => panic!("expected system cache_control, got {other:?}"),
    }
    assert!(additional_params.get("cache_control").is_none());
}

fn handle_event(
    event: &StreamingEvent,
    current_tool_call: &mut Option<String>,
    current_thinking: &mut Option<ThinkingState>,
) -> Option<Result<RawStreamingChoice<StreamingCompletionResponse>, CompletionError>> {
    let mut server_tool_uses = HashMap::new();
    super::handle_event(
        event,
        current_tool_call,
        &mut server_tool_uses,
        current_thinking,
    )
}

#[test]
fn test_thinking_delta_deserialization() {
    let json = r#"{"type": "thinking_delta", "thinking": "Let me think about this..."}"#;
    let delta: ContentDelta = serde_json::from_str(json).unwrap();

    match delta {
        ContentDelta::ThinkingDelta { thinking } => {
            assert_eq!(thinking, "Let me think about this...");
        }
        _ => panic!("Expected ThinkingDelta variant"),
    }
}

#[test]
fn test_signature_delta_deserialization() {
    let json = r#"{"type": "signature_delta", "signature": "abc123def456"}"#;
    let delta: ContentDelta = serde_json::from_str(json).unwrap();

    match delta {
        ContentDelta::SignatureDelta { signature } => {
            assert_eq!(signature, "abc123def456");
        }
        _ => panic!("Expected SignatureDelta variant"),
    }
}

#[test]
fn test_thinking_delta_streaming_event_deserialization() {
    let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "thinking_delta",
                "thinking": "First, I need to understand the problem."
            }
        }"#;

    let event: StreamingEvent = serde_json::from_str(json).unwrap();

    match event {
        StreamingEvent::ContentBlockDelta { index, delta } => {
            assert_eq!(index, 0);
            match delta {
                ContentDelta::ThinkingDelta { thinking } => {
                    assert_eq!(thinking, "First, I need to understand the problem.");
                }
                _ => panic!("Expected ThinkingDelta"),
            }
        }
        _ => panic!("Expected ContentBlockDelta event"),
    }
}

#[test]
fn test_signature_delta_streaming_event_deserialization() {
    let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "signature_delta",
                "signature": "ErUBCkYICBgCIkCaGbqC85F4"
            }
        }"#;

    let event: StreamingEvent = serde_json::from_str(json).unwrap();

    match event {
        StreamingEvent::ContentBlockDelta { index, delta } => {
            assert_eq!(index, 0);
            match delta {
                ContentDelta::SignatureDelta { signature } => {
                    assert_eq!(signature, "ErUBCkYICBgCIkCaGbqC85F4");
                }
                _ => panic!("Expected SignatureDelta"),
            }
        }
        _ => panic!("Expected ContentBlockDelta event"),
    }
}

#[test]
fn test_handle_thinking_delta_event() {
    let event = StreamingEvent::ContentBlockDelta {
        index: 0,
        delta: ContentDelta::ThinkingDelta {
            thinking: "Analyzing the request...".to_string(),
        },
    };

    let mut tool_call_state = None;
    let mut thinking_state = None;
    let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

    assert!(result.is_some());
    let choice = result.unwrap().unwrap();

    match choice {
        RawStreamingChoice::ReasoningDelta { id, reasoning, .. } => {
            assert_eq!(id, crate::streaming::MintKind::Block.for_wire_index(0));
            assert_eq!(reasoning, "Analyzing the request...");
        }
        _ => panic!("Expected ReasoningDelta choice"),
    }

    // The block is tracked (its signature may still arrive); the text
    // itself accumulates in the shared accumulator, not here.
    assert!(thinking_state.is_some());
}

#[test]
fn test_handle_signature_delta_event() {
    let event = StreamingEvent::ContentBlockDelta {
        index: 0,
        delta: ContentDelta::SignatureDelta {
            signature: "test_signature".to_string(),
        },
    };

    let mut tool_call_state = None;
    let mut thinking_state = None;
    let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

    // SignatureDelta should not yield anything (returns None)
    assert!(result.is_none());

    // But signature should be captured in thinking state
    assert!(thinking_state.is_some());
    assert_eq!(thinking_state.unwrap().signature, "test_signature");
}

#[test]
fn test_handle_redacted_thinking_content_block_start_event() {
    let event = StreamingEvent::ContentBlockStart {
        index: 0,
        content_block: Content::RedactedThinking {
            data: "redacted_blob".to_string(),
        },
    };
    let mut tool_call_state = None;
    let mut thinking_state = None;
    let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

    assert!(result.is_some());
    match result.unwrap().unwrap() {
        RawStreamingChoice::Reasoning {
            content: ReasoningContent::Redacted { data },
            ..
        } => {
            assert_eq!(data, "redacted_blob");
        }
        _ => panic!("Expected Redacted reasoning chunk"),
    }
}

/// The adaptive-thinking wire shape, exactly as recorded in
/// `tests/cassettes/anthropic/opus_4_7/messages_adaptive_thinking_streaming_smoke.yaml`:
/// `content_block_start` opens the block with an EMPTY `thinking` and an
/// EMPTY `signature`, a `signature_delta` carries the whole signature, and
/// no `thinking_delta` ever arrives. The block's only content is its
/// signature, and it must survive `content_block_stop`.
#[test]
fn signature_only_thinking_block_survives_content_block_stop() {
    let mut tool_call_state = None;
    let mut thinking_state = None;

    let start = StreamingEvent::ContentBlockStart {
        index: 0,
        content_block: Content::Thinking {
            thinking: String::new(),
            signature: Some(String::new()),
        },
    };
    assert!(handle_event(&start, &mut tool_call_state, &mut thinking_state).is_none());

    let signature = StreamingEvent::ContentBlockDelta {
        index: 0,
        delta: ContentDelta::SignatureDelta {
            signature: "the_whole_signature".to_string(),
        },
    };
    assert!(handle_event(&signature, &mut tool_call_state, &mut thinking_state).is_none());

    let stop = StreamingEvent::ContentBlockStop { index: 0 };
    let result = handle_event(&stop, &mut tool_call_state, &mut thinking_state)
        .expect("signature-only thinking block must not be dropped")
        .expect("thinking block should not be an error");

    match result {
        RawStreamingChoice::ReasoningEnd { id, signature, .. } => {
            assert_eq!(id, crate::streaming::MintKind::Block.for_wire_index(0));
            assert_eq!(signature.as_deref(), Some("the_whole_signature"));
        }
        other => panic!("Expected a signed lifecycle end, got {other:?}"),
    }
}

/// Forward compat: a block that delivers its whole signature on
/// `content_block_start` and sends no `signature_delta` keeps it.
#[test]
fn signature_delivered_only_on_content_block_start_is_kept() {
    let mut tool_call_state = None;
    let mut thinking_state = None;

    let start = StreamingEvent::ContentBlockStart {
        index: 0,
        content_block: Content::Thinking {
            thinking: String::new(),
            signature: Some("up_front_signature".to_string()),
        },
    };
    assert!(handle_event(&start, &mut tool_call_state, &mut thinking_state).is_none());

    let stop = StreamingEvent::ContentBlockStop { index: 0 };
    match handle_event(&stop, &mut tool_call_state, &mut thinking_state)
        .expect("an up-front signature must not be dropped")
        .expect("thinking block should not be an error")
    {
        RawStreamingChoice::ReasoningEnd { signature, .. } => {
            assert_eq!(signature.as_deref(), Some("up_front_signature"));
        }
        other => panic!("Expected a signed lifecycle end, got {other:?}"),
    }
}

/// The opening `signature` is a fallback, never a prefix the deltas
/// extend: a delta-bearing block must publish exactly what the deltas
/// assembled, or the value replayed to Anthropic is corrupt.
#[test]
fn signature_deltas_supersede_the_opening_signature() {
    let mut tool_call_state = None;
    let mut thinking_state = None;

    let start = StreamingEvent::ContentBlockStart {
        index: 0,
        content_block: Content::Thinking {
            thinking: String::new(),
            signature: Some("opening".to_string()),
        },
    };
    assert!(handle_event(&start, &mut tool_call_state, &mut thinking_state).is_none());

    for fragment in ["delta_", "assembled"] {
        let signature = StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::SignatureDelta {
                signature: fragment.to_string(),
            },
        };
        assert!(handle_event(&signature, &mut tool_call_state, &mut thinking_state).is_none());
    }

    let stop = StreamingEvent::ContentBlockStop { index: 0 };
    match handle_event(&stop, &mut tool_call_state, &mut thinking_state)
        .expect("thinking block should be restated")
        .expect("thinking block should not be an error")
    {
        RawStreamingChoice::ReasoningEnd { signature, .. } => {
            assert_eq!(signature.as_deref(), Some("delta_assembled"));
        }
        other => panic!("Expected a signed lifecycle end, got {other:?}"),
    }
}

/// `content_block_start` can carry the block's opening text; discarding it
/// would truncate the restatement the accumulator supersedes deltas with.
#[test]
fn thinking_block_start_text_streams_as_the_first_delta() {
    let mut tool_call_state = None;
    let mut thinking_state = None;

    let start = StreamingEvent::ContentBlockStart {
        index: 2,
        content_block: Content::Thinking {
            thinking: "opening ".to_string(),
            signature: None,
        },
    };
    // The opening payload's text is a delta like any other; the shared
    // accumulator owns the block's text — no adapter-side restatement
    // buffer exists to seed.
    match handle_event(&start, &mut tool_call_state, &mut thinking_state)
        .expect("the opening text streams")
        .expect("not an error")
    {
        RawStreamingChoice::ReasoningDelta { id, reasoning, .. } => {
            assert_eq!(id, crate::streaming::MintKind::Block.for_wire_index(2));
            assert_eq!(reasoning, "opening ");
        }
        other => panic!("Expected the opening delta, got {other:?}"),
    }

    let delta = StreamingEvent::ContentBlockDelta {
        index: 2,
        delta: ContentDelta::ThinkingDelta {
            thinking: "rest".to_string(),
        },
    };
    assert!(handle_event(&delta, &mut tool_call_state, &mut thinking_state).is_some());

    let stop = StreamingEvent::ContentBlockStop { index: 2 };
    match handle_event(&stop, &mut tool_call_state, &mut thinking_state)
        .expect("the stop emits the lifecycle end")
        .expect("not an error")
    {
        RawStreamingChoice::ReasoningEnd {
            id,
            reasoning: None,
            signature: None,
            wire_sent: true,
        } => {
            assert_eq!(id, crate::streaming::MintKind::Block.for_wire_index(2));
        }
        other => panic!("Expected a bare lifecycle end, got {other:?}"),
    }
}

/// A block with neither text nor signature carries nothing to replay.
#[test]
fn wholly_empty_thinking_block_is_dropped() {
    let mut tool_call_state = None;
    let mut thinking_state = None;

    let start = StreamingEvent::ContentBlockStart {
        index: 0,
        content_block: Content::Thinking {
            thinking: String::new(),
            signature: None,
        },
    };
    assert!(handle_event(&start, &mut tool_call_state, &mut thinking_state).is_none());

    let stop = StreamingEvent::ContentBlockStop { index: 0 };
    // The stop emits a bare lifecycle end; with nothing streamed and no
    // signature, the shared accumulator records no part (a bare end for
    // a never-opened key is a no-op).
    match handle_event(&stop, &mut tool_call_state, &mut thinking_state)
        .expect("the stop emits the lifecycle end")
        .expect("not an error")
    {
        RawStreamingChoice::ReasoningEnd {
            reasoning: None,
            signature: None,
            ..
        } => {}
        other => panic!("Expected a bare lifecycle end, got {other:?}"),
    }
}

#[test]
fn test_handle_text_delta_event() {
    let event = StreamingEvent::ContentBlockDelta {
        index: 0,
        delta: ContentDelta::TextDelta {
            text: "Hello, world!".to_string(),
        },
    };

    let mut tool_call_state = None;
    let mut thinking_state = None;
    let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

    assert!(result.is_some());
    let choice = result.unwrap().unwrap();

    match choice {
        RawStreamingChoice::Message(text) => {
            assert_eq!(text, "Hello, world!");
        }
        _ => panic!("Expected Message choice"),
    }
}

#[test]
fn test_handle_text_block_start_event() {
    let event = StreamingEvent::ContentBlockStart {
        index: 0,
        content_block: Content::Text {
            text: String::new(),
            citations: Vec::new(),
            cache_control: None,
        },
    };

    let mut tool_call_state = None;
    let mut thinking_state = None;
    let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

    assert!(result.is_some());
    let choice = result.unwrap().unwrap();
    assert!(matches!(
        choice,
        RawStreamingChoice::TextStart {
            additional_params: None,
            ..
        }
    ));
}

#[test]
fn test_thinking_delta_does_not_interfere_with_tool_calls() {
    // Thinking deltas should still be processed even if a tool call is in progress
    let event = StreamingEvent::ContentBlockDelta {
        index: 0,
        delta: ContentDelta::ThinkingDelta {
            thinking: "Thinking while tool is active...".to_string(),
        },
    };

    let mut tool_call_state = Some("tool_123".to_string());
    let mut thinking_state = None;

    let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

    assert!(result.is_some());
    let choice = result.unwrap().unwrap();

    match choice {
        RawStreamingChoice::ReasoningDelta { reasoning, .. } => {
            assert_eq!(reasoning, "Thinking while tool is active...");
        }
        _ => panic!("Expected ReasoningDelta choice"),
    }

    // Tool call state should remain unchanged
    assert!(tool_call_state.is_some());
}

#[test]
fn test_handle_input_json_delta_event() {
    let event = StreamingEvent::ContentBlockDelta {
        index: 0,
        delta: ContentDelta::InputJsonDelta {
            partial_json: "{\"arg\":\"value".to_string(),
        },
    };

    let mut tool_call_state = Some("tool_123".to_string());
    let mut thinking_state = None;

    let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

    // Should emit a ToolCallDelta
    assert!(result.is_some());
    let choice = result.unwrap().unwrap();

    match choice {
        RawStreamingChoice::ToolCallDelta { id, content } => {
            assert_eq!(id, crate::streaming::BlockId::wire("tool_123"));
            match content {
                ToolCallDeltaContent::Delta(delta) => assert_eq!(delta, "{\"arg\":\"value"),
                _ => panic!("Expected Delta content"),
            }
        }
        _ => panic!("Expected ToolCallDelta choice, got {choice:?}"),
    }

    // The open block stays open; assembly of the fragment happens in the
    // shared accumulator.
    assert!(tool_call_state.is_some());
}

#[test]
fn test_tool_call_accumulation_with_multiple_deltas() {
    let mut tool_call_state = Some("tool_123".to_string());
    let mut thinking_state = None;

    // First delta
    let event1 = StreamingEvent::ContentBlockDelta {
        index: 0,
        delta: ContentDelta::InputJsonDelta {
            partial_json: "{\"location\":".to_string(),
        },
    };
    let result1 = handle_event(&event1, &mut tool_call_state, &mut thinking_state);
    assert!(result1.is_some());

    // Second delta
    let event2 = StreamingEvent::ContentBlockDelta {
        index: 0,
        delta: ContentDelta::InputJsonDelta {
            partial_json: "\"Paris\",".to_string(),
        },
    };
    let result2 = handle_event(&event2, &mut tool_call_state, &mut thinking_state);
    assert!(result2.is_some());

    // Third delta
    let event3 = StreamingEvent::ContentBlockDelta {
        index: 0,
        delta: ContentDelta::InputJsonDelta {
            partial_json: "\"temp\":\"20C\"}".to_string(),
        },
    };
    let result3 = handle_event(&event3, &mut tool_call_state, &mut thinking_state);
    assert!(result3.is_some());

    assert!(tool_call_state.is_some());

    // Final ContentBlockStop hands the block to the shared accumulator,
    // which finalizes the assembled fragments (`Error` policy: a stopped
    // block promised complete input). End-to-end assembly of exactly this
    // fragment sequence is pinned in `streaming::parts` unit tests.
    let stop_event = StreamingEvent::ContentBlockStop { index: 0 };
    let final_result = handle_event(&stop_event, &mut tool_call_state, &mut thinking_state);
    assert!(final_result.is_some());

    match final_result.unwrap().unwrap() {
        RawStreamingChoice::ToolInputEnd(end) => {
            assert_eq!(end.id, crate::streaming::BlockId::wire("tool_123"));
            assert!(matches!(
                end.on_unparseable,
                crate::streaming::UnparseableToolInput::Error
            ));
        }
        other => panic!("Expected ToolInputEnd, got {other:?}"),
    }

    // Tool call state should be taken
    assert!(tool_call_state.is_none());
}

#[test]
fn test_citations_delta_streaming_event_deserialization() {
    let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "char_location",
                    "cited_text": "The grass is green.",
                    "document_index": 0,
                    "document_title": "Example",
                    "start_char_index": 0,
                    "end_char_index": 20
                }
            }
        }"#;

    let event: StreamingEvent = serde_json::from_str(json).unwrap();
    let StreamingEvent::ContentBlockDelta { index, delta } = event else {
        panic!("expected ContentBlockDelta");
    };
    assert_eq!(index, 0);
    let ContentDelta::CitationsDelta { citation } = delta else {
        panic!("expected CitationsDelta");
    };
    let crate::providers::anthropic::completion::Citation::CharLocation(citation) = citation else {
        panic!("expected CharLocation");
    };
    assert_eq!(citation.start_char_index, 0);
    assert_eq!(citation.end_char_index, 20);
}

#[test]
fn test_search_result_citations_delta_streaming_event_deserialization() {
    let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "search_result_location",
                    "cited_text": "API requests require a key.",
                    "source": "https://docs.example.com/api-reference",
                    "title": "API Reference",
                    "search_result_index": 0,
                    "start_block_index": 0,
                    "end_block_index": 1
                }
            }
        }"#;

    let event: StreamingEvent = serde_json::from_str(json).unwrap();
    let StreamingEvent::ContentBlockDelta { delta, .. } = event else {
        panic!("expected ContentBlockDelta");
    };
    let ContentDelta::CitationsDelta { citation } = delta else {
        panic!("expected CitationsDelta");
    };
    assert!(matches!(
        citation,
        crate::providers::anthropic::completion::Citation::SearchResultLocation(
            crate::providers::anthropic::completion::SearchResultLocationCitation {
                search_result_index: 0,
                start_block_index: 0,
                end_block_index: 1,
                ..
            }
        )
    ));
}

#[test]
fn test_web_search_result_citations_delta_streaming_event_deserialization() {
    let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "web_search_result_location",
                    "cited_text": "Claude Shannon was a mathematician.",
                    "url": "https://example.com/shannon",
                    "title": "Claude Shannon",
                    "encrypted_index": "encrypted-reference"
                }
            }
        }"#;

    let event: StreamingEvent = serde_json::from_str(json).unwrap();
    let StreamingEvent::ContentBlockDelta { delta, .. } = event else {
        panic!("expected ContentBlockDelta");
    };
    let ContentDelta::CitationsDelta { citation } = delta else {
        panic!("expected CitationsDelta");
    };
    assert!(matches!(
        citation,
        crate::providers::anthropic::completion::Citation::WebSearchResultLocation(ref citation)
            if citation.url == "https://example.com/shannon"
                && citation.encrypted_index == "encrypted-reference"
    ));
}

#[test]
fn test_web_search_result_citations_delta_allows_null_title() {
    let json = r#"{
            "type": "content_block_delta",
            "index": 0,
            "delta": {
                "type": "citations_delta",
                "citation": {
                    "type": "web_search_result_location",
                    "cited_text": "Claude Shannon was a mathematician.",
                    "url": "https://example.com/shannon",
                    "title": null,
                    "encrypted_index": "encrypted-reference"
                }
            }
        }"#;

    let event: StreamingEvent = serde_json::from_str(json).unwrap();
    let StreamingEvent::ContentBlockDelta { delta, .. } = event else {
        panic!("expected ContentBlockDelta");
    };
    let ContentDelta::CitationsDelta { citation } = delta else {
        panic!("expected CitationsDelta");
    };
    assert!(matches!(
        citation,
        crate::providers::anthropic::completion::Citation::WebSearchResultLocation(
            crate::providers::anthropic::completion::WebSearchResultLocationCitation {
                title: None,
                ..
            }
        )
    ));
}

#[test]
fn test_text_content_block_start_allows_null_citations() {
    // The Anthropic Messages API emits an explicit `"citations": null` on the
    // first text `content_block_start` event. `#[serde(default)]` alone covers
    // a missing field but not an explicit null, so this must deserialize to an
    // empty citation list rather than failing the whole stream (see #1971).
    let json = r#"{
            "type": "content_block_start",
            "index": 0,
            "content_block": {
                "type": "text",
                "text": "",
                "citations": null
            }
        }"#;

    let event: StreamingEvent = serde_json::from_str(json).unwrap();
    let StreamingEvent::ContentBlockStart { content_block, .. } = event else {
        panic!("expected ContentBlockStart");
    };
    let Content::Text {
        text, citations, ..
    } = content_block
    else {
        panic!("expected text content block");
    };
    assert_eq!(text, "");
    assert!(citations.is_empty());
}

#[test]
fn test_web_search_content_block_start_events_deserialize() {
    let server_tool_use = r#"{
            "type": "content_block_start",
            "index": 1,
            "content_block": {
                "type": "server_tool_use",
                "id": "srvtoolu_01",
                "name": "web_search",
                "input": {
                    "query": "claude shannon birth date"
                }
            }
        }"#;
    let event: StreamingEvent = serde_json::from_str(server_tool_use).unwrap();
    assert!(matches!(
        event,
        StreamingEvent::ContentBlockStart {
            content_block: Content::ServerToolUse {
                ref id,
                ref name,
                ref input
            },
            ..
        } if id == "srvtoolu_01"
            && name == "web_search"
            && input["query"] == "claude shannon birth date"
    ));

    let web_search_tool_result = r#"{
            "type": "content_block_start",
            "index": 2,
            "content_block": {
                "type": "web_search_tool_result",
                "tool_use_id": "srvtoolu_01",
                "content": [{
                    "type": "web_search_result",
                    "url": "https://example.com/shannon",
                    "title": "Claude Shannon",
                    "encrypted_content": "encrypted-content"
                }]
            }
        }"#;
    let event: StreamingEvent = serde_json::from_str(web_search_tool_result).unwrap();
    assert!(matches!(
        event,
        StreamingEvent::ContentBlockStart {
            content_block: Content::WebSearchToolResult {
                ref tool_use_id,
                ref content
            },
            ..
        } if tool_use_id == "srvtoolu_01"
            && content[0]["encrypted_content"] == "encrypted-content"
    ));
}

#[test]
fn test_code_execution_tool_result_block_is_preserved() {
    let event: StreamingEvent = serde_json::from_value(serde_json::json!({
        "type": "content_block_start",
        "index": 1,
        "content_block": {
            "type": "code_execution_tool_result",
            "tool_use_id": "srvtoolu_01",
            "content": {
                "type": "code_execution_result",
                "return_code": 0,
                "stdout": "42\n",
                "stderr": "",
                "content": []
            }
        }
    }))
    .unwrap();
    let mut tool_call_state = None;
    let mut server_tool_uses = HashMap::new();
    let mut thinking_state = None;

    let choice = super::handle_event(
        &event,
        &mut tool_call_state,
        &mut server_tool_uses,
        &mut thinking_state,
    )
    .expect("code_execution_tool_result block should produce raw metadata")
    .unwrap();

    let RawStreamingChoice::TextStart {
        id,
        additional_params: Some(additional_params),
    } = choice
    else {
        panic!("expected text-start metadata for code_execution_tool_result");
    };
    assert_eq!(id, crate::streaming::MintKind::Block.for_wire_index(1));
    assert_eq!(
        additional_params[crate::providers::anthropic::completion::ANTHROPIC_RAW_CONTENT_KEY]["type"],
        "code_execution_tool_result"
    );
    assert_eq!(
        additional_params[crate::providers::anthropic::completion::ANTHROPIC_RAW_CONTENT_KEY]["content"]
            ["stdout"],
        "42\n"
    );
}

#[tokio::test]
async fn test_streaming_web_search_blocks_are_preserved_on_final_choice() {
    let raw_stream = stream! {
        let mut tool_call_state = None;
        let mut server_tool_uses = HashMap::new();
        let mut thinking_state = None;

        let server_tool_use_start = super::handle_event(
            &StreamingEvent::ContentBlockStart {
                index: 0,
                content_block: Content::ServerToolUse {
                    id: "srvtoolu_01".to_string(),
                    name: "web_search".to_string(),
                    input: serde_json::Value::Null,
                },
            },
            &mut tool_call_state,
            &mut server_tool_uses,
            &mut thinking_state,
        );
        assert!(
            server_tool_use_start.is_none(),
            "server_tool_use start should be accumulated until its input JSON is complete"
        );

        let server_tool_use_delta = super::handle_event(
            &StreamingEvent::ContentBlockDelta {
                index: 0,
                delta: ContentDelta::InputJsonDelta {
                    partial_json: r#"{"query":"claude shannon birth date"}"#.to_string(),
                },
            },
            &mut tool_call_state,
            &mut server_tool_uses,
            &mut thinking_state,
        );
        assert!(
            server_tool_use_delta.is_none(),
            "server_tool_use input JSON should not be emitted as a Rig tool-call delta"
        );

        yield super::handle_event(
            &StreamingEvent::ContentBlockStop { index: 0 },
            &mut tool_call_state,
            &mut server_tool_uses,
            &mut thinking_state,
        )
        .expect("server_tool_use stop should produce completed raw metadata");

        yield super::handle_event(
            &StreamingEvent::ContentBlockStart {
                index: 1,
                content_block: Content::WebSearchToolResult {
                    tool_use_id: "srvtoolu_01".to_string(),
                    content: serde_json::json!([{
                        "type": "web_search_result",
                        "url": "https://example.com/shannon",
                        "title": "Claude Shannon",
                        "encrypted_content": "encrypted-content"
                    }]),
                },
            },
            &mut tool_call_state,
            &mut server_tool_uses,
            &mut thinking_state,
        )
        .expect("web_search_tool_result block should produce raw metadata");

        yield super::handle_event(
            &StreamingEvent::ContentBlockStart {
                index: 2,
                content_block: Content::Text {
                    text: String::new(),
                    citations: Vec::new(),
                    cache_control: None,
                },
            },
            &mut tool_call_state,
            &mut server_tool_uses,
            &mut thinking_state,
        )
        .expect("text block start should produce a raw choice");

        yield super::handle_event(
            &StreamingEvent::ContentBlockDelta {
                index: 2,
                delta: ContentDelta::TextDelta {
                    text: "Claude Shannon was born on April 30, 1916.".to_string(),
                },
            },
            &mut tool_call_state,
            &mut server_tool_uses,
            &mut thinking_state,
        )
        .expect("text delta should produce a raw choice");

        yield super::handle_event(
            &StreamingEvent::ContentBlockDelta {
                index: 2,
                delta: ContentDelta::CitationsDelta {
                    citation: crate::providers::anthropic::completion::Citation::WebSearchResultLocation(
                        crate::providers::anthropic::completion::WebSearchResultLocationCitation {
                            cited_text: "Claude Shannon was born on April 30, 1916."
                                .to_string(),
                            url: "https://example.com/shannon".to_string(),
                            title: Some("Claude Shannon".to_string()),
                            encrypted_index: "encrypted-index".to_string(),
                        },
                    ),
                },
            },
            &mut tool_call_state,
            &mut server_tool_uses,
            &mut thinking_state,
        )
        .expect("citation delta should produce a raw choice");

        yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse::default()));
    };

    let mut stream = crate::streaming::StreamingCompletionResponse::stream(
        "anthropic",
        to_stream_result(raw_stream),
    );
    while stream.next().await.is_some() {}

    let choice_items: Vec<crate::message::AssistantContent> =
        stream.choice.clone().into_iter().collect();
    assert_eq!(choice_items.len(), 3);
    assert!(
        choice_items
            .iter()
            .all(|item| !matches!(item, crate::message::AssistantContent::ToolCall(_))),
        "provider-owned web-search blocks must not become Rig client tool calls"
    );

    let Some(crate::message::AssistantContent::Text(server_tool_use)) = choice_items.first() else {
        panic!("expected raw server_tool_use metadata");
    };
    assert_eq!(
        server_tool_use.additional_params.as_ref().unwrap()
            [crate::providers::anthropic::completion::ANTHROPIC_RAW_CONTENT_KEY]["type"],
        "server_tool_use"
    );
    assert_eq!(
        server_tool_use.additional_params.as_ref().unwrap()
            [crate::providers::anthropic::completion::ANTHROPIC_RAW_CONTENT_KEY]["input"]["query"],
        "claude shannon birth date"
    );

    let Some(crate::message::AssistantContent::Text(web_search_result)) = choice_items.get(1)
    else {
        panic!("expected raw web_search_tool_result metadata");
    };
    assert_eq!(
        web_search_result.additional_params.as_ref().unwrap()
            [crate::providers::anthropic::completion::ANTHROPIC_RAW_CONTENT_KEY]["content"][0]["encrypted_content"],
        "encrypted-content"
    );

    let Some(crate::message::AssistantContent::Text(answer)) = choice_items.get(2) else {
        panic!("expected answer text");
    };
    assert_eq!(answer.text, "Claude Shannon was born on April 30, 1916.");
    let citations = crate::providers::anthropic::completion::anthropic_citations(answer)
        .expect("expected preserved citations");
    assert!(matches!(
        citations.first(),
        Some(crate::providers::anthropic::completion::Citation::WebSearchResultLocation(citation))
            if citation.encrypted_index == "encrypted-index"
    ));
}

#[test]
fn test_handle_citations_delta_event_preserves_metadata() {
    let event = StreamingEvent::ContentBlockDelta {
        index: 0,
        delta: ContentDelta::CitationsDelta {
            citation: crate::providers::anthropic::completion::Citation::CharLocation(
                crate::providers::anthropic::completion::CharLocationCitation {
                    cited_text: "The grass is green.".to_string(),
                    document_index: 0,
                    document_title: Some("Example".to_string()),
                    start_char_index: 0,
                    end_char_index: 20,
                },
            ),
        },
    };

    let mut tool_call_state = None;
    let mut thinking_state = None;
    let result = handle_event(&event, &mut tool_call_state, &mut thinking_state);

    assert!(result.is_some());
    let choice = result.unwrap().unwrap();
    let RawStreamingChoice::TextAdditionalParams(additional_params) = choice else {
        panic!("expected TextAdditionalParams choice");
    };
    assert_eq!(additional_params["citations"][0]["type"], "char_location");
}

#[tokio::test]
async fn test_streaming_citation_deltas_are_preserved_on_final_text() {
    let citation = crate::providers::anthropic::completion::Citation::CharLocation(
        crate::providers::anthropic::completion::CharLocationCitation {
            cited_text: "The grass is green.".to_string(),
            document_index: 0,
            document_title: Some("Example".to_string()),
            start_char_index: 0,
            end_char_index: 20,
        },
    );

    let raw_stream = stream! {
        let mut tool_call_state = None;
        let mut thinking_state = None;

        yield handle_event(
            &StreamingEvent::ContentBlockStart {
                index: 0,
                content_block: Content::Text {
                    text: String::new(),
                    citations: Vec::new(),
                    cache_control: None,
                },
            },
            &mut tool_call_state,
            &mut thinking_state,
        )
        .expect("text block start should produce a raw choice");

        yield handle_event(
            &StreamingEvent::ContentBlockDelta {
                index: 0,
                delta: ContentDelta::TextDelta {
                    text: "the grass is green".to_string(),
                },
            },
            &mut tool_call_state,
            &mut thinking_state,
        )
        .expect("text delta should produce a raw choice");

        yield handle_event(
            &StreamingEvent::ContentBlockDelta {
                index: 0,
                delta: ContentDelta::CitationsDelta {
                    citation: crate::providers::anthropic::completion::Citation::CharLocation(
                        crate::providers::anthropic::completion::CharLocationCitation {
                            cited_text: "The grass is green.".to_string(),
                            document_index: 0,
                            document_title: Some("Example".to_string()),
                            start_char_index: 0,
                            end_char_index: 20,
                        },
                    ),
                },
            },
            &mut tool_call_state,
            &mut thinking_state,
        )
        .expect("citation delta should produce a raw choice");

        yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse::default()));
    };

    let mut stream = crate::streaming::StreamingCompletionResponse::stream(
        "anthropic",
        to_stream_result(raw_stream),
    );
    while stream.next().await.is_some() {}

    let choice_items: Vec<crate::message::AssistantContent> =
        stream.choice.clone().into_iter().collect();
    let Some(crate::message::AssistantContent::Text(text)) = choice_items.first() else {
        panic!("expected accumulated text item");
    };

    assert_eq!(text.text, "the grass is green");
    let citations = crate::providers::anthropic::completion::anthropic_citations(text).unwrap();
    assert_eq!(citations, vec![citation]);
}

/// The `#[serde(other)]` policy fallbacks are gone: classification is the
/// only policy site. An unmodeled *top-level* event type is `Unknown`
/// (driver: warn + skip); a `ping` is Known; and a known tag whose payload
/// this client cannot decode is `Corrupt`, never silently demoted to an
/// ignorable unknown. An unmodeled *nested* delta type is the one carved
/// exception (Anthropic's versioning policy reserves the right to add
/// them): it decodes to [`ContentDelta::Unknown`] and stays a Known
/// no-op — see the dedicated tests below.
#[test]
fn classify_dispatches_on_the_known_event_list() {
    let adapter = AnthropicAdapter::default();

    let frame = WireFrame::Text(r#"{"type":"something_new_from_anthropic","field":"x"}"#.into());
    assert!(matches!(
        adapter.classify(frame),
        crate::providers::internal::wire::WireEvent::Unknown { event_type, .. }
            if event_type == "something_new_from_anthropic"
    ));

    let frame = WireFrame::Text(r#"{"type":"ping"}"#.into());
    assert!(matches!(
        adapter.classify(frame),
        crate::providers::internal::wire::WireEvent::Known(StreamingEvent::Ping)
    ));

    let frame = WireFrame::Text("{not json".into());
    assert!(matches!(
        adapter.classify(frame),
        crate::providers::internal::wire::WireEvent::Corrupt(_)
    ));
}

/// Forward compat: a novel nested delta type Anthropic ships tomorrow
/// must not corrupt the whole `content_block_delta` frame — it decodes
/// to [`ContentDelta::Unknown`] and interprets as a warned no-op, so the
/// stream continues.
#[test]
fn novel_nested_delta_type_is_a_known_noop() {
    let adapter = AnthropicAdapter::default();
    let frame = WireFrame::Text(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"banana_delta","x":1}}"#.into(),
    );
    let crate::providers::internal::wire::WireEvent::Known(event) = adapter.classify(frame) else {
        panic!("a novel nested delta type must stay a Known event");
    };

    let mut adapter = AnthropicAdapter::default();
    let mut out = Vec::new();
    adapter.interpret(event, &mut out);
    assert!(out.is_empty(), "an unmodeled nested delta is a no-op");
}

/// Anthropic reports the per-TTL `cache_creation` split on
/// `message_start` only; the terminal `message_delta` usage omits it. The
/// adapter must carry it onto the terminal record. Unit-tested (not a
/// cassette) because the carry-forward is internal adapter state — the
/// wire evidence lives in the recorded `prompt_caching/matrix_*` streaming
/// cassettes, whose `message_start` frames hold the split.
#[test]
fn per_ttl_cache_creation_split_carries_from_message_start_to_terminal() {
    let mut adapter = AnthropicAdapter::default();
    let mut out = Vec::new();

    let start = WireFrame::Text(
        r#"{"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":1,"cache_creation_input_tokens":9702,"cache_read_input_tokens":0,"cache_creation":{"ephemeral_1h_input_tokens":9366,"ephemeral_5m_input_tokens":336}}}}"#
            .into(),
    );
    let crate::providers::internal::wire::WireEvent::Known(event) = adapter.classify(start) else {
        panic!("message_start must classify Known");
    };
    adapter.interpret(event, &mut out);

    let delta = WireFrame::Text(
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":7,"input_tokens":3,"cache_creation_input_tokens":9702,"cache_read_input_tokens":0}}"#
            .into(),
    );
    let crate::providers::internal::wire::WireEvent::Known(event) = adapter.classify(delta) else {
        panic!("message_delta must classify Known");
    };
    adapter.interpret(event, &mut out);

    let terminal = out
        .iter()
        .find_map(|item| match item {
            Ok(crate::streaming::RawStreamingChoice::FinalResponse(response)) => {
                Some(response.clone())
            }
            _ => None,
        })
        .expect("terminal message_delta must yield a final response");
    let split = terminal
        .usage
        .cache_creation
        .expect("terminal usage must carry the message_start cache_creation split");
    assert_eq!(split.ephemeral_1h_input_tokens, 9366);
    assert_eq!(split.ephemeral_5m_input_tokens, 336);
    assert_eq!(terminal.usage.cache_creation_input_tokens, Some(9702));
}

/// A `content_block_delta` whose `delta` omits `type` is malformed, not
/// novel: silently skipping it would turn a compat gateway's untagged
/// text delta into a successful *empty* completion. It classifies
/// `Corrupt`, surfacing in-band while the stream keeps consuming
/// (#2258 B5).
#[test]
fn delta_missing_its_type_is_corrupt_not_skipped() {
    let adapter = AnthropicAdapter::default();
    let frame = WireFrame::Text(
        r#"{"type":"content_block_delta","index":0,"delta":{"text":"hello"}}"#.into(),
    );
    assert!(matches!(
        adapter.classify(frame),
        crate::providers::internal::wire::WireEvent::Corrupt(_)
    ));
}

/// Policy preserved: a *known* nested delta tag with a defective payload
/// is a data-level defect, not an unmodeled delta — the frame classifies
/// `Corrupt` instead of degrading to an `Unknown` no-op.
#[test]
fn known_nested_delta_tag_with_defective_payload_is_corrupt() {
    let adapter = AnthropicAdapter::default();
    let frame = WireFrame::Text(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":42}}"#
            .into(),
    );
    assert!(matches!(
        adapter.classify(frame),
        crate::providers::internal::wire::WireEvent::Corrupt(_)
    ));
}

/// Anthropic's top-level `{"type":"error"}` envelope (e.g.
/// `overloaded_error`) is a Known event that surfaces as a provider error
/// carrying the full envelope — never a warn-skipped unknown — and, since
/// no `message_delta` follows, the stream ends with no terminal record.
#[test]
fn top_level_error_event_surfaces_as_a_provider_error() {
    let adapter = AnthropicAdapter::default();
    let frame = WireFrame::Text(
        r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#.into(),
    );
    let crate::providers::internal::wire::WireEvent::Known(event) = adapter.classify(frame) else {
        panic!("the error envelope must classify as a Known event");
    };

    let mut adapter = AnthropicAdapter::default();
    let mut out = Vec::new();
    adapter.interpret(event, &mut out);

    assert_eq!(out.len(), 1, "the error envelope maps to one error item");
    let Some(Err(error)) = out.pop() else {
        panic!("the error envelope must surface as an Err item");
    };
    let body = error
        .provider_response_body()
        .expect("the provider's error payload must be preserved");
    assert!(
        body.contains("overloaded_error") && body.contains("Overloaded"),
        "the full envelope must survive into the error body, got: {body}"
    );
}

/// Bedrock-compat quirk: `message_start` without a message body is a
/// Known no-op, not a corrupt frame.
#[test]
fn message_start_with_null_message_is_a_known_noop() {
    let adapter = AnthropicAdapter::default();
    let frame = WireFrame::Text(r#"{"type":"message_start","message":null}"#.into());
    let crate::providers::internal::wire::WireEvent::Known(event) = adapter.classify(frame) else {
        panic!("null-message message_start must stay a known event");
    };

    let mut adapter = AnthropicAdapter::default();
    let mut out = Vec::new();
    adapter.interpret(event, &mut out);
    assert!(out.is_empty(), "a message-less message_start is a no-op");
}

#[tokio::test]
async fn terminal_record_normalizes_stop_reason_usage_and_metadata() {
    let raw_stream = stream! {
        yield Ok(RawStreamingChoice::Message("hi".to_string()));
        yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
            usage: PartialUsage {
                output_tokens: 5,
                input_tokens: Some(3),
                cache_creation_input_tokens: None,
                cache_creation: None,
                cache_read_input_tokens: Some(2),
                output_tokens_details: None,
            },
            stop_reason: Some("max_tokens".to_string()),
            stop_sequence: None,
            message_id: Some("msg_1".to_string()),
            model: Some(CLAUDE_OPUS_4_8.to_string()),
            provider_request_id: None,
        }));
    };

    let mut stream = crate::streaming::StreamingCompletionResponse::stream(
        "anthropic",
        to_stream_result(raw_stream),
    );
    while stream.next().await.is_some() {}

    let terminal = stream.response.expect("expected a terminal record");
    assert_eq!(terminal.provider, "anthropic");
    assert_eq!(terminal.message_id.as_deref(), Some("msg_1"));
    assert_eq!(terminal.model.as_deref(), Some(CLAUDE_OPUS_4_8));
    assert_eq!(
        terminal.finish_reason,
        Some(crate::completion::FinishReason::Length)
    );
    assert_eq!(terminal.usage.input_tokens, 3);
    assert_eq!(terminal.usage.output_tokens, 5);
    assert_eq!(terminal.usage.cached_input_tokens, 2);
    assert_eq!(terminal.usage.total_tokens, 10);
}

#[tokio::test]
async fn terminal_record_upgrades_end_turn_to_tool_calls_after_a_streamed_tool_call() {
    // Anthropic normally reports `tool_use`, but the reconciliation
    // `normalize_stream` applies must hold whenever the turn actually
    // emitted a tool call.
    let raw_stream = stream! {
        yield Ok(RawStreamingChoice::ToolCall(RawStreamingToolCall::new(
            "toolu_1".to_string(),
            "add".to_string(),
            json!({"x": 1}),
        )));
        yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
            stop_reason: Some("end_turn".to_string()),
            ..Default::default()
        }));
    };

    let mut stream = crate::streaming::StreamingCompletionResponse::stream(
        "anthropic",
        to_stream_result(raw_stream),
    );
    while stream.next().await.is_some() {}

    let terminal = stream.response.expect("expected a terminal record");
    assert_eq!(
        terminal.finish_reason,
        Some(crate::completion::FinishReason::ToolCalls)
    );
}

#[tokio::test]
async fn unknown_stop_reason_survives_onto_the_terminal_record() {
    let raw_stream = stream! {
        yield Ok(RawStreamingChoice::FinalResponse(StreamingCompletionResponse {
            stop_reason: Some("pause_turn".to_string()),
            ..Default::default()
        }));
    };

    let mut stream = crate::streaming::StreamingCompletionResponse::stream(
        "anthropic",
        to_stream_result(raw_stream),
    );
    while stream.next().await.is_some() {}

    let terminal = stream.response.expect("expected a terminal record");
    assert_eq!(
        terminal.finish_reason,
        Some(crate::completion::FinishReason::Other(
            "pause_turn".to_owned()
        ))
    );
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod terminal_emission {
    use super::super::super::completion::CLAUDE_SONNET_4_6;
    use crate::client::CompletionClient;
    use crate::completion::CompletionModel as _;
    use crate::providers::anthropic::Client;
    use crate::streaming::StreamedAssistantContent;
    use crate::test_utils::MockStreamingClient;
    use futures::StreamExt;

    const MESSAGE_START: &str = r#"{"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":5,"output_tokens":0}}}"#;
    const TEXT_START: &str =
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#;
    const TEXT_DELTA: &str =
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#;
    const MESSAGE_DELTA: &str = r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}"#;

    fn sse(frames: &[&str]) -> bytes::Bytes {
        bytes::Bytes::from(
            frames
                .iter()
                .map(|frame| format!("data: {frame}\n\n"))
                .collect::<String>(),
        )
    }

    async fn collect(
        sse_bytes: bytes::Bytes,
    ) -> (
        Vec<String>,
        bool,
        bool,
        crate::streaming::StreamingCompletionResponse,
    ) {
        let client = Client::builder()
            .api_key("test-key")
            .http_client(MockStreamingClient { sse_bytes })
            .build()
            .expect("build client");
        let model = client.completion_model(CLAUDE_SONNET_4_6);
        let request = model.completion_request("hello").build();
        let mut stream = crate::completion::CompletionModel::stream(&model, request)
            .await
            .expect("stream should open");

        let mut texts = Vec::new();
        let mut saw_error = false;
        let mut saw_terminal = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamedAssistantContent::Text(text)) => texts.push(text.text),
                Ok(StreamedAssistantContent::Final(_)) => saw_terminal = true,
                Ok(_) => {}
                Err(_) => saw_error = true,
            }
        }
        (texts, saw_error, saw_terminal, stream)
    }

    #[tokio::test]
    async fn truncated_stream_yields_content_but_no_terminal_record() {
        let (texts, saw_error, saw_terminal, stream) =
            collect(sse(&[MESSAGE_START, TEXT_START, TEXT_DELTA])).await;

        assert_eq!(texts, ["hi"]);
        assert!(!saw_error);
        assert!(
            !saw_terminal,
            "EOF without message_delta must not synthesize a terminal record"
        );
        assert!(stream.response.is_none());
    }

    #[tokio::test]
    async fn errored_stream_forwards_the_error_and_no_terminal_record() {
        use crate::test_utils::SequencedStreamingHttpClient;

        // A transport failure injected into the byte stream after some
        // content must be forwarded (via `from_stream_transport`) and must
        // not be papered over with a synthesized terminal record.
        let client = Client::builder()
            .api_key("test-key")
            .http_client(SequencedStreamingHttpClient::new(vec![
                Ok(sse(&[MESSAGE_START, TEXT_START, TEXT_DELTA])),
                Err(crate::http_client::Error::InvalidStatusCodeWithMessage(
                    http::StatusCode::BAD_GATEWAY,
                    "connection reset".to_string(),
                )),
            ]))
            .build()
            .expect("build client");
        let model = client.completion_model(CLAUDE_SONNET_4_6);
        let request = model.completion_request("hello").build();
        let mut stream = crate::completion::CompletionModel::stream(&model, request)
            .await
            .expect("stream should open");

        let mut texts = Vec::new();
        let mut saw_error = false;
        let mut saw_terminal = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(StreamedAssistantContent::Text(text)) => texts.push(text.text),
                Ok(StreamedAssistantContent::Final(_)) => saw_terminal = true,
                Ok(_) => {}
                Err(_) => saw_error = true,
            }
        }

        assert_eq!(texts, ["hi"]);
        assert!(saw_error, "the transport failure must reach the consumer");
        assert!(
            !saw_terminal,
            "a failed stream must not synthesize a terminal record"
        );
        assert!(stream.response.is_none());
    }

    #[tokio::test]
    async fn provider_error_event_stops_the_stream_before_a_later_terminal() {
        // The findings-file probe: an in-band provider `error` event
        // followed by a well-formed `message_delta`. The error must reach
        // the consumer and NOTHING may follow it — the adapter is
        // finished, so the later terminal frame must not be interpreted
        // into a successful FinalResponse.
        const ERROR_EVENT: &str =
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
        let (texts, saw_error, saw_terminal, stream) = collect(sse(&[
            MESSAGE_START,
            TEXT_START,
            TEXT_DELTA,
            ERROR_EVENT,
            MESSAGE_DELTA,
        ]))
        .await;

        assert_eq!(texts, ["hi"]);
        assert!(saw_error, "the provider error must reach the consumer");
        assert!(
            !saw_terminal,
            "a message_delta after an in-band provider error must not read as a completed turn"
        );
        assert!(stream.response.is_none());
    }

    /// `input_tokens` precedence between `message_start` and the terminal
    /// `message_delta`, across all three wire splits at once.
    ///
    /// Not a cassette test: one recording can only witness whichever split
    /// the endpoint it was recorded against happens to use, and the defect
    /// here is the *precedence rule* relating three of them — the gateway
    /// split, Anthropic proper, and the inverse. The gateway split is also
    /// covered end-to-end by the recorded
    /// `anthropic::cassette::streaming::gateway_reports_input_tokens_on_message_delta`;
    /// this pins the two cases a single recording structurally cannot show
    /// beside it.
    #[tokio::test]
    async fn input_tokens_prefer_the_terminal_delta_and_fall_back_to_message_start() {
        fn message_start(input_tokens: usize) -> String {
            format!(
                r#"{{"type":"message_start","message":{{"id":"msg_1","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":{input_tokens},"output_tokens":0}}}}}}"#
            )
        }
        fn message_delta(input_tokens: usize) -> String {
            format!(
                r#"{{"type":"message_delta","delta":{{"stop_reason":"end_turn","stop_sequence":null}},"usage":{{"input_tokens":{input_tokens},"output_tokens":3}}}}"#
            )
        }

        for (start, delta, expected, case) in [
            // OpenRouter's Anthropic Messages shape: `message_start`
            // reports a placeholder zero and the real prompt size lands on
            // the terminal `message_delta`.
            (
                message_start(0),
                message_delta(9),
                9,
                "a gateway reporting the prompt size on message_delta must reach the consumer",
            ),
            // A delta that omits `input_tokens` entirely — the Bedrock-compat
            // and older/leaner shapes. (Not current Anthropic, which sends
            // the count on both frames; that case is the one below, since
            // the two always agree.)
            (
                message_start(5),
                MESSAGE_DELTA.to_owned(),
                5,
                "a delta without input_tokens falls back to message_start",
            ),
            // Anthropic proper: both frames carry the same count.
            (
                message_start(5),
                message_delta(5),
                5,
                "agreeing frames report that count",
            ),
            // The inverse split: a zero on the delta must not erase the
            // real count `message_start` already gave us.
            (
                message_start(5),
                message_delta(0),
                5,
                "a zero on the delta must not erase the message_start count",
            ),
        ] {
            let (_texts, _saw_error, saw_terminal, stream) =
                collect(sse(&[&start, TEXT_START, TEXT_DELTA, &delta])).await;

            assert!(saw_terminal, "{case}: the turn must complete");
            let terminal = stream.response.expect("terminal record");
            assert_eq!(terminal.usage.input_tokens, expected, "{case}");
        }
    }

    #[tokio::test]
    async fn malformed_frame_then_eof_yields_error_and_no_terminal_record() {
        let (texts, saw_error, saw_terminal, stream) =
            collect(sse(&[MESSAGE_START, TEXT_START, TEXT_DELTA, "{not json"])).await;

        assert_eq!(texts, ["hi"]);
        assert!(saw_error, "the malformed frame must reach the consumer");
        assert!(
            !saw_terminal,
            "a parse error followed by EOF must not read as a completed turn"
        );
        assert!(stream.response.is_none());
    }

    #[tokio::test]
    async fn malformed_frame_then_real_terminal_still_completes_the_stream() {
        let (texts, saw_error, saw_terminal, stream) = collect(sse(&[
            MESSAGE_START,
            TEXT_START,
            TEXT_DELTA,
            "{not json",
            MESSAGE_DELTA,
        ]))
        .await;

        assert_eq!(texts, ["hi"]);
        assert!(saw_error, "the malformed frame must reach the consumer");
        assert!(
            saw_terminal,
            "a genuine message_delta after a parse error still completes the stream"
        );
        let terminal = stream.response.expect("terminal record");
        assert_eq!(
            terminal.finish_reason,
            Some(crate::completion::FinishReason::Stop)
        );
        assert_eq!(terminal.message_id.as_deref(), Some("msg_1"));
    }

    /// Raw capture on the streaming terminal, through the real
    /// `CompletionModel::stream` seam over the mock transport:
    /// `normalize_stream` serializes the terminal before mapping it, so
    /// the terminal `StreamFinal.raw` is Anthropic's own
    /// `StreamingCompletionResponse`. A `message_delta` with
    /// `stop_sequence` set is used because the normalized terminal folds
    /// it into `FinishReason::Stop` and keeps neither Anthropic's spelling
    /// nor which sequence fired — both are readable only off the capture.
    #[tokio::test]
    async fn terminal_raw_round_trips_into_the_terminal_type() {
        const STOP_SEQUENCE_DELTA: &str = r#"{"type":"message_delta","delta":{"stop_reason":"stop_sequence","stop_sequence":"alpha"},"usage":{"output_tokens":3}}"#;

        let client = Client::builder()
            .api_key("test-key")
            .http_client(MockStreamingClient {
                sse_bytes: sse(&[MESSAGE_START, TEXT_START, TEXT_DELTA, STOP_SEQUENCE_DELTA]),
            })
            .build()
            .expect("build client");
        let model = client.completion_model(CLAUDE_SONNET_4_6);
        let request = model.completion_request("hello").build();
        let mut stream = crate::completion::CompletionModel::stream(&model, request)
            .await
            .expect("stream should open");
        while let Some(item) = stream.next().await {
            item.expect("stream item");
        }
        let terminal = stream.response.expect("terminal record");

        let raw = &terminal.raw;
        let typed: super::super::StreamingCompletionResponse =
            serde_json::from_value(raw.clone()).expect("raw must deserialize");
        assert_eq!(
            serde_json::to_value(&typed).expect("re-serialize"),
            *raw,
            "the capture must be exactly what the terminal type serializes to"
        );
        assert_eq!(typed.stop_reason.as_deref(), Some("stop_sequence"));
        assert_eq!(typed.stop_sequence.as_deref(), Some("alpha"));
        assert_eq!(typed.message_id.as_deref(), Some("msg_1"));

        // Re-normalizing the capture tells the same story as the terminal
        // the stream produced.
        let renormalized = crate::streaming::StreamFinal::from(("anthropic", typed));
        assert_eq!(terminal.identity(), renormalized.identity());
        assert_eq!(terminal.finish_reason, renormalized.finish_reason);
        assert_eq!(terminal.model, renormalized.model);
        assert_eq!(terminal.usage, renormalized.usage);
        assert_eq!(
            terminal.finish_reason,
            Some(crate::completion::FinishReason::Stop)
        );
        assert_eq!(terminal.usage.output_tokens, 3);
    }
}
