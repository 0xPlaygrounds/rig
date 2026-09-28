use super::super::completion::{
    AnthropicCompletionRequest, AnthropicRequestParams, CLAUDE_OPUS_4_8, CacheControl, CacheTtl,
    Message, SystemContent, apply_prompt_cache_control, build_tool_definitions,
    resolve_top_level_cache_control,
};
use super::*;
use crate::completion::CompletionRequest;
use crate::completion::Message as RigMessage;
use crate::completion::request::Document as RigDocument;
use crate::driver::{Decoded, decode_events};
use crate::message::{AssistantContent, Reasoning, ReasoningContent};
use crate::streaming::{PartKind, StreamEvent};

/// A fresh decoder, for its classifier.
fn adapter() -> MessagesDecoder<'static> {
    MessagesDecoder::new()
}

/// Decode `events` through one decoder as one reply, then EOF.
fn decode(events: impl IntoIterator<Item = StreamingEvent>) -> Decoded<Completion> {
    decode_events!(MessagesDecoder::new(), "anthropic", events)
}

/// The event one wire frame classifies to.
fn classified(frame: &str) -> StreamingEvent {
    let crate::wire::WireEvent::Known(event) = adapter().classify(WireFrame::Text(frame.into()))
    else {
        panic!("{frame} must classify Known");
    };
    event
}

/// The reasoning that ended, opened for its issuer.
fn reasoning_of(decoded: &Decoded<Completion>) -> Vec<Reasoning> {
    decoded
        .ended()
        .into_iter()
        .filter_map(|content| match content {
            AssistantContent::Reasoning(reasoning) => reasoning.open(reasoning.issuer()).cloned(),
            _ => None,
        })
        .collect()
}

fn thinking_start(index: usize, thinking: &str, signature: Option<&str>) -> StreamingEvent {
    StreamingEvent::ContentBlockStart {
        index,
        content_block: Content::Thinking {
            thinking: thinking.to_string(),
            signature: signature.map(str::to_owned),
        },
    }
}

fn signature_delta(index: usize, signature: &str) -> StreamingEvent {
    StreamingEvent::ContentBlockDelta {
        index,
        delta: ContentDelta::SignatureDelta {
            signature: signature.to_string(),
        },
    }
}

fn stop(index: usize) -> StreamingEvent {
    StreamingEvent::ContentBlockStop { index }
}

fn tool_use(index: usize, id: &str, name: &str) -> StreamingEvent {
    StreamingEvent::ContentBlockStart {
        index,
        content_block: Content::ToolUse {
            id: id.to_string(),
            name: name.to_string(),
            input: json!({}),
        },
    }
}

fn input_json(index: usize, partial_json: &str) -> StreamingEvent {
    StreamingEvent::ContentBlockDelta {
        index,
        delta: ContentDelta::InputJsonDelta {
            partial_json: partial_json.to_string(),
        },
    }
}

/// The signed text a reasoning part closed with.
fn signed(text: &str, signature: &str) -> Vec<ReasoningContent> {
    vec![ReasoningContent::Text {
        text: text.to_string(),
        signature: Some(signature.to_string()),
    }]
}

/// The one terminal frame: a `message_delta` carrying `stop_reason`.
fn message_delta(stop_reason: &str, usage: PartialUsage) -> StreamingEvent {
    StreamingEvent::MessageDelta {
        delta: MessageDelta {
            stop_reason: Some(stop_reason.to_string()),
            stop_sequence: None,
        },
        usage,
    }
}

/// The streaming request body the [`Messages`](super::super::wire::Messages)
/// wire encodes — the shared typed request plus the streaming-only patches.
///
/// Read back off the encoded HTTP request rather than rebuilt here: the
/// wire's `encode` is the only statement of that body now, so a cell that
/// pins the body pins the thing that runs.
fn built_streaming_body(
    model: &str,
    request: CompletionRequest,
    strict_tools: bool,
) -> Result<Value, ProviderError> {
    use crate::wire::{Body, Mode, Wire};

    let wire =
        crate::providers::anthropic::wire::AnthropicConfig::new("test-key").completion(model);
    let wire = if strict_tools {
        wire.with_strict_tools()
    } else {
        wire
    };
    let encoded = wire.encode(request, Mode::Streaming)?;
    match encoded.request.body() {
        Body::Bytes(bytes) => Ok(serde_json::from_slice(bytes)?),
        Body::Multipart(_) => Err(ProviderError::request("the Messages endpoint takes JSON")),
    }
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

    let mut tools = build_tool_definitions(
        vec![crate::completion::ToolDefinition {
            name: "rig_tool".to_string(),
            description: "Rig tool".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
        }],
        &mut additional_params,
        None,
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
        chat_history: crate::NonEmpty::with_rest(
            RigMessage::system("System prompt"),
            [
                RigMessage::assistant("Earlier assistant turn"),
                RigMessage::system("Mid-conversation instruction"),
                RigMessage::user("Prompt"),
            ],
        ),
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
        chat_history: crate::NonEmpty::with_rest(
            RigMessage::system("You are helpful"),
            [RigMessage::user("What's the weather?")],
        ),
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
        issuers: &[crate::message::Issuer::from_static("anthropic")],
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
        chat_history: crate::NonEmpty::new(RigMessage::user("Add 2 and 3")),
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
        chat_history: crate::NonEmpty::new(RigMessage::user("Look this up")),
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
        chat_history: crate::NonEmpty::new(RigMessage::user("Hi")),
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
    let mut tools = build_tool_definitions(
        vec![crate::completion::ToolDefinition {
            name: "rig_tool".to_string(),
            description: "Rig tool".to_string(),
            parameters: json!({"type": "object", "properties": {}}),
        }],
        &mut additional_params,
        None,
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
    let decoded = decode([StreamingEvent::ContentBlockDelta {
        index: 0,
        delta: ContentDelta::ThinkingDelta {
            thinking: "Analyzing the request...".to_string(),
        },
    }]);
    // An unseen thinking block opens its part before the first fragment.
    assert!(
        matches!(
            decoded.events().as_slice(),
            [
                StreamEvent::Start { kind: PartKind::Reasoning, .. },
                StreamEvent::Reasoning { text, .. },
            ] if text == "Analyzing the request..."
        ),
        "{:?}",
        decoded.events()
    );
}

#[test]
fn test_handle_signature_delta_event() {
    let decoded = decode([signature_delta(0, "test_signature"), stop(0)]);
    let reasoning = reasoning_of(&decoded);
    assert_eq!(reasoning.len(), 1);
    assert_eq!(reasoning[0].content, signed("", "test_signature"));
}

#[test]
fn test_handle_redacted_thinking_content_block_start_event() {
    let decoded = decode([StreamingEvent::ContentBlockStart {
        index: 0,
        content_block: Content::RedactedThinking {
            data: "redacted_blob".to_string(),
        },
    }]);
    // A whole reasoning block is its start and its end.
    let reasoning = reasoning_of(&decoded);
    assert_eq!(reasoning.len(), 1);
    assert_eq!(
        reasoning[0].content,
        vec![ReasoningContent::Redacted {
            data: "redacted_blob".to_string()
        }]
    );
}

/// The adaptive-thinking wire shape, exactly as recorded in
/// `crates/rig-cassette/fixtures/cassettes/anthropic/opus_4_7/messages_adaptive_thinking_streaming_smoke.yaml`:
/// `content_block_start` opens the block with an EMPTY `thinking` and an
/// EMPTY `signature`, a `signature_delta` carries the whole signature, and
/// no `thinking_delta` ever arrives. The block's only content is its
/// signature, and it must survive `content_block_stop`.
#[test]
fn signature_only_thinking_block_survives_content_block_stop() {
    let decoded = decode([
        thinking_start(0, "", Some("")),
        signature_delta(0, "the_whole_signature"),
        stop(0),
    ]);
    let reasoning = reasoning_of(&decoded);
    assert_eq!(reasoning.len(), 1, "the signature-only block is kept");
    assert_eq!(reasoning[0].content, signed("", "the_whole_signature"));
}

/// Forward compat: a block that delivers its whole signature on
/// `content_block_start` and sends no `signature_delta` keeps it.
#[test]
fn signature_delivered_only_on_content_block_start_is_kept() {
    let decoded = decode([thinking_start(0, "", Some("up_front_signature")), stop(0)]);
    let reasoning = reasoning_of(&decoded);
    assert_eq!(reasoning.len(), 1, "an up-front signature is kept");
    assert_eq!(reasoning[0].content, signed("", "up_front_signature"));
}

/// The opening `signature` is a fallback, never a prefix the deltas
/// extend: a delta-bearing block must publish exactly what the deltas
/// assembled, or the value replayed to Anthropic is corrupt.
#[test]
fn signature_deltas_supersede_the_opening_signature() {
    let decoded = decode([
        thinking_start(0, "", Some("opening")),
        signature_delta(0, "delta_"),
        signature_delta(0, "assembled"),
        stop(0),
    ]);
    let reasoning = reasoning_of(&decoded);
    assert_eq!(reasoning[0].content, signed("", "delta_assembled"));
}

/// `content_block_start` can carry the block's opening text; discarding it
/// would truncate the block.
#[test]
fn thinking_block_start_text_streams_as_the_first_delta() {
    let decoded = decode([
        thinking_start(2, "opening ", None),
        StreamingEvent::ContentBlockDelta {
            index: 2,
            delta: ContentDelta::ThinkingDelta {
                thinking: "rest".to_string(),
            },
        },
        stop(2),
    ]);
    let fragments: Vec<&str> = decoded
        .events()
        .into_iter()
        .filter_map(|event| match event {
            StreamEvent::Reasoning { text, .. } => Some(text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(fragments, ["opening ", "rest"]);
    let reasoning = reasoning_of(&decoded);
    assert_eq!(
        reasoning[0].content,
        vec![ReasoningContent::Text {
            text: "opening rest".to_string(),
            signature: None,
        }]
    );
}

/// A block with neither text nor signature carries nothing to replay.
#[test]
fn wholly_empty_thinking_block_is_dropped() {
    let decoded = decode([thinking_start(0, "", None), stop(0)]);
    assert!(decoded.ended().is_empty(), "{:?}", decoded.events());
}

#[test]
fn test_handle_text_delta_event() {
    let decoded = decode([StreamingEvent::ContentBlockDelta {
        index: 0,
        delta: ContentDelta::TextDelta {
            text: "Hello, world!".to_string(),
        },
    }]);
    // A bare text delta with no open text block opens one first.
    assert!(
        matches!(
            decoded.events().as_slice(),
            [
                StreamEvent::Start { kind: PartKind::Text, .. },
                StreamEvent::Text { text, .. },
            ] if text == "Hello, world!"
        ),
        "{:?}",
        decoded.events()
    );
}

#[test]
fn test_handle_text_block_start_event() {
    let decoded = decode([StreamingEvent::ContentBlockStart {
        index: 0,
        content_block: Content::Text {
            text: String::new(),
            citations: Vec::new(),
            cache_control: None,
        },
    }]);
    // A part streams nothing until its first fragment.
    assert!(decoded.events().is_empty(), "{:?}", decoded.events());
}

#[test]
fn test_thinking_delta_does_not_interfere_with_tool_calls() {
    // Thinking still streams while a tool call is in progress.
    let decoded = decode([
        tool_use(0, "tool_123", "lookup"),
        StreamingEvent::ContentBlockDelta {
            index: 1,
            delta: ContentDelta::ThinkingDelta {
                thinking: "Thinking while tool is active...".to_string(),
            },
        },
        input_json(0, "{}"),
        stop(0),
        stop(1),
    ]);
    let ended = decoded.ended();
    assert!(
        ended
            .iter()
            .any(|content| matches!(content, AssistantContent::ToolCall(call) if call.id.to_string() == "tool_123"))
    );
    let reasoning = reasoning_of(&decoded);
    assert_eq!(
        reasoning[0].content,
        vec![ReasoningContent::Text {
            text: "Thinking while tool is active...".to_string(),
            signature: None,
        }]
    );
}

#[test]
fn test_handle_input_json_delta_event() {
    let decoded = decode([
        tool_use(0, "tool_123", "lookup"),
        input_json(0, "{\"arg\":\"value"),
    ]);
    // A call streams nothing until it closes.
    assert!(decoded.events().is_empty(), "{:?}", decoded.events());
}

#[test]
fn test_tool_call_accumulation_with_multiple_deltas() {
    let decoded = decode([
        tool_use(0, "tool_123", "lookup"),
        input_json(0, "{\"location\":"),
        input_json(0, "\"Paris\","),
        input_json(0, "\"temp\":\"20C\"}"),
        stop(0),
    ]);
    let ended = decoded.ended();
    let [AssistantContent::ToolCall(call)] = ended.as_slice() else {
        panic!("one call ended: {:?}", decoded.events());
    };
    assert_eq!(call.id.to_string(), "tool_123");
    assert_eq!(
        call.function.arguments,
        json!({"location": "Paris", "temp": "20C"})
    );
    assert!(decoded.events().into_iter().any(|event| matches!(
        event,
        StreamEvent::Arguments { json, .. } if json == "{\"location\":\"Paris\",\"temp\":\"20C\"}"
    )));
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
    let decoded = decode([event, stop(1)]);
    let ended = decoded.ended();
    let [AssistantContent::Text(text)] = ended.as_slice() else {
        panic!("the result block is a text part: {:?}", decoded.events());
    };
    let additional_params = text.additional_params.as_ref().expect("its raw content");
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

#[test]
fn test_streaming_web_search_blocks_are_preserved_on_final_choice() {
    let decoded = decode([
        StreamingEvent::ContentBlockStart {
            index: 0,
            content_block: Content::ServerToolUse {
                id: "srvtoolu_01".to_string(),
                name: "web_search".to_string(),
                input: serde_json::Value::Null,
            },
        },
        input_json(0, r#"{"query":"claude shannon birth date"}"#),
        stop(0),
        StreamingEvent::ContentBlockStart {
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
        StreamingEvent::ContentBlockStart {
            index: 2,
            content_block: Content::Text {
                text: String::new(),
                citations: Vec::new(),
                cache_control: None,
            },
        },
        StreamingEvent::ContentBlockDelta {
            index: 2,
            delta: ContentDelta::TextDelta {
                text: "Claude Shannon was born on April 30, 1916.".to_string(),
            },
        },
        StreamingEvent::ContentBlockDelta {
            index: 2,
            delta: ContentDelta::CitationsDelta {
                citation:
                    crate::providers::anthropic::completion::Citation::WebSearchResultLocation(
                        crate::providers::anthropic::completion::WebSearchResultLocationCitation {
                            cited_text: "Claude Shannon was born on April 30, 1916.".to_string(),
                            url: "https://example.com/shannon".to_string(),
                            title: Some("Claude Shannon".to_string()),
                            encrypted_index: "encrypted-index".to_string(),
                        },
                    ),
            },
        },
        message_delta("end_turn", PartialUsage::default()),
    ]);
    let choice_items = decoded.outcome.expect("the reply ended").choice;
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
    let decoded = decode([
        StreamingEvent::ContentBlockDelta {
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
        stop(0),
    ]);
    let ended = decoded.ended();
    let [AssistantContent::Text(text)] = ended.as_slice() else {
        panic!("the citation rides a text part: {:?}", decoded.events());
    };
    let additional_params = text.additional_params.as_ref().expect("its citations");
    assert_eq!(additional_params["citations"][0]["type"], "char_location");
}

#[test]
fn test_streaming_citation_deltas_are_preserved_on_final_text() {
    let citation = crate::providers::anthropic::completion::Citation::CharLocation(
        crate::providers::anthropic::completion::CharLocationCitation {
            cited_text: "The grass is green.".to_string(),
            document_index: 0,
            document_title: Some("Example".to_string()),
            start_char_index: 0,
            end_char_index: 20,
        },
    );

    let decoded = decode([
        StreamingEvent::ContentBlockStart {
            index: 0,
            content_block: Content::Text {
                text: String::new(),
                citations: Vec::new(),
                cache_control: None,
            },
        },
        StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::TextDelta {
                text: "the grass is green".to_string(),
            },
        },
        StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::CitationsDelta {
                citation: citation.clone(),
            },
        },
        message_delta("end_turn", PartialUsage::default()),
    ]);
    let choice_items = decoded.outcome.expect("the reply ended").choice;
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
    let adapter = adapter();

    let frame = WireFrame::Text(r#"{"type":"something_new_from_anthropic","field":"x"}"#.into());
    assert!(matches!(
        adapter.classify(frame),
        crate::wire::WireEvent::Unknown { event_type, .. }
            if event_type == "something_new_from_anthropic"
    ));

    let frame = WireFrame::Text(r#"{"type":"ping"}"#.into());
    assert!(matches!(
        adapter.classify(frame),
        crate::wire::WireEvent::Known(StreamingEvent::Ping)
    ));

    let frame = WireFrame::Text("{not json".into());
    assert!(matches!(
        adapter.classify(frame),
        crate::wire::WireEvent::Corrupt(_)
    ));
}

/// Forward compat: a novel nested delta type Anthropic ships tomorrow
/// must not corrupt the whole `content_block_delta` frame — it decodes
/// to [`ContentDelta::Unknown`] and is a warned no-op, so the stream
/// continues.
#[test]
fn novel_nested_delta_type_is_a_known_noop() {
    let event = classified(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"banana_delta","x":1}}"#,
    );
    let decoded = decode([event]);
    assert!(decoded.events().is_empty(), "an unmodeled nested delta is a no-op");
}

/// Anthropic reports the per-TTL `cache_creation` split on
/// `message_start` only; the terminal `message_delta` usage omits it. The
/// decoder must carry it onto the reply's end. Unit-tested (not a cassette)
/// because the carry-forward is internal decoder state — the wire evidence
/// lives in the recorded `prompt_caching/matrix_*` streaming cassettes,
/// whose `message_start` frames hold the split.
#[test]
fn per_ttl_cache_creation_split_carries_from_message_start_to_terminal() {
    let decoded = decode([
        classified(
            r#"{"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":3,"output_tokens":1,"cache_creation_input_tokens":9702,"cache_read_input_tokens":0,"cache_creation":{"ephemeral_1h_input_tokens":9366,"ephemeral_5m_input_tokens":336}}}}"#,
        ),
        classified(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":7,"input_tokens":3,"cache_creation_input_tokens":9702,"cache_read_input_tokens":0}}"#,
        ),
    ]);
    let response = decoded.outcome.expect("the message_delta ends the reply");
    // The native record rides on `raw`; the split is Anthropic-specific, so
    // it is readable only there.
    let native: StreamingCompletionResponse =
        serde_json::from_value(response.raw).expect("raw must be the native terminal");
    let split = native
        .usage
        .cache_creation
        .expect("terminal usage must carry the message_start cache_creation split");
    assert_eq!(split.ephemeral_1h_input_tokens, 9366);
    assert_eq!(split.ephemeral_5m_input_tokens, 336);
    assert_eq!(native.usage.cache_creation_input_tokens, Some(9702));
}

/// A `content_block_delta` whose `delta` omits `type` is malformed, not
/// novel: silently skipping it would turn a compat gateway's untagged
/// text delta into a successful *empty* completion. It classifies
/// `Corrupt`, surfacing in-band while the stream keeps consuming
/// (#2258 B5).
#[test]
fn delta_missing_its_type_is_corrupt_not_skipped() {
    let adapter = adapter();
    let frame = WireFrame::Text(
        r#"{"type":"content_block_delta","index":0,"delta":{"text":"hello"}}"#.into(),
    );
    assert!(matches!(
        adapter.classify(frame),
        crate::wire::WireEvent::Corrupt(_)
    ));
}

/// Policy preserved: a *known* nested delta tag with a defective payload
/// is a data-level defect, not an unmodeled delta — the frame classifies
/// `Corrupt` instead of degrading to an `Unknown` no-op.
#[test]
fn known_nested_delta_tag_with_defective_payload_is_corrupt() {
    let adapter = adapter();
    let frame = WireFrame::Text(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":42}}"#
            .into(),
    );
    assert!(matches!(
        adapter.classify(frame),
        crate::wire::WireEvent::Corrupt(_)
    ));
}

/// Anthropic's top-level `{"type":"error"}` envelope (e.g.
/// `overloaded_error`) is a Known event that ends the reply with a provider
/// error carrying the envelope verbatim — never a warn-skipped unknown.
///
/// Byte-equality is the assertion, and the frame carries the top-level
/// `request_id` recorded replies carry: an envelope re-encoded from the
/// fields this client models loses every sibling key and normalizes the
/// order, which is the provider's body rendered rather than preserved.
#[test]
fn top_level_error_event_surfaces_as_a_provider_error() {
    const ENVELOPE: &str = r#"{"error":{"message":"Overloaded","type":"overloaded_error"},"request_id":"req_011CXYZ","type":"error"}"#;
    let decoded = decode([classified(ENVELOPE)]);
    let error = decoded.outcome.expect_err("the envelope ends the reply");
    assert_eq!(error.provider_response_body(), Some(ENVELOPE));
}

/// Bedrock-compat quirk: `message_start` without a message body is a
/// Known no-op, not a corrupt frame.
#[test]
fn message_start_with_null_message_is_a_known_noop() {
    let decoded = decode([classified(r#"{"type":"message_start","message":null}"#)]);
    assert!(decoded.events().is_empty(), "a message-less message_start is a no-op");
}

#[test]
fn terminal_record_normalizes_stop_reason_usage_and_metadata() {
    let decoded = decode([
        classified(&format!(
            r#"{{"type":"message_start","message":{{"id":"msg_1","role":"assistant","content":[],"model":"{CLAUDE_OPUS_4_8}","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":3,"output_tokens":0}}}}}}"#
        )),
        StreamingEvent::ContentBlockDelta {
            index: 0,
            delta: ContentDelta::TextDelta {
                text: "hi".to_string(),
            },
        },
        message_delta(
            "max_tokens",
            PartialUsage {
                output_tokens: 5,
                input_tokens: Some(3),
                cache_creation_input_tokens: None,
                cache_creation: None,
                cache_read_input_tokens: Some(2),
                output_tokens_details: None,
            },
        ),
    ]);
    let response = decoded.outcome.expect("the reply ended");
    assert_eq!(response.provider, "anthropic");
    assert_eq!(response.message_id.as_deref(), Some("msg_1"));
    assert_eq!(response.model.as_deref(), Some(CLAUDE_OPUS_4_8));
    assert_eq!(
        response.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
    assert_eq!(response.usage.input_tokens, Some(3));
    assert_eq!(response.usage.output_tokens, Some(5));
    assert_eq!(response.usage.cached_input_tokens, Some(2));
    assert_eq!(response.usage.total_tokens, Some(10));
}

#[test]
fn terminal_record_upgrades_end_turn_to_tool_calls_after_a_streamed_tool_call() {
    // Anthropic normally reports `tool_use`, but the finish must report tool
    // calls whenever the turn actually emitted one.
    let decoded = decode([
        tool_use(0, "toolu_1", "add"),
        input_json(0, r#"{"x":1}"#),
        stop(0),
        message_delta("end_turn", PartialUsage::default()),
    ]);
    assert_eq!(
        decoded.outcome.expect("the reply ended").finish_reason(),
        Some(crate::completion::FinishReason::ToolCalls)
    );
}

#[test]
fn unknown_stop_reason_survives_onto_the_terminal_record() {
    let decoded = decode([message_delta("pause_turn", PartialUsage::default())]);
    assert_eq!(
        decoded.outcome.expect("the reply ended").finish_reason(),
        Some(crate::completion::FinishReason::Other(
            "pause_turn".to_owned()
        ))
    );
}

#[cfg(not(all(target_arch = "wasm32", target_os = "unknown")))]
mod terminal_emission {
    use super::super::super::completion::CLAUDE_SONNET_4_6;
    use crate::providers::anthropic::wire::AnthropicConfig;
    use crate::streaming::{Item, StreamEvent};
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

    /// The texts the stream yielded, whether an error item came, and what
    /// it finished with.
    async fn collect(
        sse_bytes: bytes::Bytes,
    ) -> (
        Vec<String>,
        bool,
        Result<crate::completion::CompletionResponse, crate::error::ProviderError>,
    ) {
        let bound = crate::driver::Model::new(
            AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6),
            MockStreamingClient { sse_bytes },
        );
        let request = crate::completion::CompletionRequest::new("hello");
        let stream = bound.stream(request).expect("stream should open");
        drain(stream).await
    }

    async fn drain(
        mut stream: crate::streaming::CompletionStream,
    ) -> (
        Vec<String>,
        bool,
        Result<crate::completion::CompletionResponse, crate::error::ProviderError>,
    ) {
        let mut texts = Vec::new();
        let mut saw_error = false;
        while let Some(item) = stream.next().await {
            match item {
                Ok(Item::Event(StreamEvent::Text { text, .. })) => texts.push(text),
                Ok(_) => {}
                Err(_) => saw_error = true,
            }
        }
        (texts, saw_error, stream.finish().await)
    }

    #[tokio::test]
    async fn truncated_stream_yields_content_then_truncation() {
        let (texts, saw_error, finished) =
            collect(sse(&[MESSAGE_START, TEXT_START, TEXT_DELTA])).await;

        assert_eq!(texts, ["hi"]);
        assert!(saw_error, "the truncation is the stream's last item");
        assert!(
            matches!(finished, Err(crate::error::ProviderError::Truncated)),
            "EOF without message_delta is truncation: {finished:?}"
        );
    }

    #[tokio::test]
    async fn errored_stream_forwards_the_error_and_no_end() {
        use crate::test_utils::SequencedStreamingHttpClient;

        // A transport failure injected into the byte stream after some
        // content must be forwarded and must not be papered over with a
        // synthesized end.
        let bound = crate::driver::Model::new(
            AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6),
            SequencedStreamingHttpClient::new(vec![
                Ok(sse(&[MESSAGE_START, TEXT_START, TEXT_DELTA])),
                Err(crate::http_client::Error::non_success_with_details(
                    http::StatusCode::BAD_GATEWAY,
                    http::HeaderMap::new(),
                    "connection reset".to_string(),
                )),
            ]),
        );
        let request = crate::completion::CompletionRequest::new("hello");
        let stream = bound.stream(request).expect("stream should open");
        let (texts, saw_error, finished) = drain(stream).await;

        assert_eq!(texts, ["hi"]);
        assert!(saw_error, "the transport failure must reach the consumer");
        assert!(finished.is_err(), "a failed stream has no response");
    }

    #[tokio::test]
    async fn provider_error_event_stops_the_stream_before_a_later_terminal() {
        // An in-band provider `error` event followed by a well-formed
        // `message_delta`: the error ends the reply, so the later end frame
        // never reads as a completed turn.
        const ERROR_EVENT: &str =
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#;
        let (texts, saw_error, finished) = collect(sse(&[
            MESSAGE_START,
            TEXT_START,
            TEXT_DELTA,
            ERROR_EVENT,
            MESSAGE_DELTA,
        ]))
        .await;

        assert_eq!(texts, ["hi"]);
        assert!(saw_error, "the provider error must reach the consumer");
        assert!(finished.is_err(), "the error ended the reply");
    }

    /// The streamed surface preserves the in-band envelope with the same
    /// fidelity as the unary one: the provider's own bytes, `request_id`
    /// and key order included.
    ///
    /// No status is asserted, and none is stamped. A preserved in-band
    /// error's `status` is the *classification* the wire read off the body
    /// — Gemini's `error.code` is the case that made the rule, and
    /// `gemini::streaming::tests::in_band_opaque_or_invalid_codes_do_not_invent_http_status`
    /// pins it — so stamping the transport's 200 over every streamed frame
    /// would overwrite that meaning and flip a refusal's retry verdict.
    /// The unary driver's fold-failure decoration is scoped to one reply
    /// and is where `Model::call` supplies it.
    #[tokio::test]
    async fn streamed_error_envelope_preserves_the_verbatim_body() {
        const ENVELOPE: &str = r#"{"error":{"message":"Overloaded","type":"overloaded_error"},"request_id":"req_011CXYZ","type":"error"}"#;
        let bound = crate::driver::Model::new(
            AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6),
            MockStreamingClient {
                sse_bytes: sse(&[MESSAGE_START, ENVELOPE]),
            },
        );
        let request = crate::completion::CompletionRequest::new("hello");
        let mut stream = bound.stream(request).expect("stream should open");

        let error = loop {
            match stream.next().await {
                Some(Ok(_)) => continue,
                Some(Err(error)) => break error,
                None => panic!("the stream ended without the in-band error"),
            }
        };

        assert_eq!(error.provider_response_body(), Some(ENVELOPE));
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
            let (_texts, _saw_error, finished) =
                collect(sse(&[&start, TEXT_START, TEXT_DELTA, &delta])).await;

            let response = finished.expect("the turn must complete");
            assert_eq!(response.usage.input_tokens, Some(expected), "{case}");
        }
    }

    #[tokio::test]
    async fn malformed_frame_then_eof_yields_error_and_no_end() {
        let (texts, saw_error, finished) =
            collect(sse(&[MESSAGE_START, TEXT_START, TEXT_DELTA, "{not json"])).await;

        assert_eq!(texts, ["hi"]);
        assert!(saw_error, "the malformed frame must reach the consumer");
        assert!(finished.is_err(), "a parse error is not a completed turn");
    }

    /// A corrupt frame ends the reply: a genuine `message_delta` after it
    /// is never read.
    #[tokio::test]
    async fn a_malformed_frame_ends_the_reply_before_a_later_end() {
        let (texts, saw_error, finished) = collect(sse(&[
            MESSAGE_START,
            TEXT_START,
            TEXT_DELTA,
            "{not json",
            MESSAGE_DELTA,
        ]))
        .await;

        assert_eq!(texts, ["hi"]);
        assert!(saw_error, "the malformed frame must reach the consumer");
        assert!(finished.is_err(), "the corrupt frame ended the reply");
    }

    /// Raw capture on a streamed reply, through the real `Model::stream`
    /// seam over the mock transport: the response's `raw` is Anthropic's own
    /// `StreamingCompletionResponse`. A `message_delta` with `stop_sequence`
    /// set is used because the normalized finish folds it into
    /// `FinishReason::Stop` and keeps neither Anthropic's spelling nor which
    /// sequence fired — both are readable only off the capture.
    #[tokio::test]
    async fn terminal_raw_round_trips_into_the_terminal_type() {
        const STOP_SEQUENCE_DELTA: &str = r#"{"type":"message_delta","delta":{"stop_reason":"stop_sequence","stop_sequence":"alpha"},"usage":{"output_tokens":3}}"#;

        let (_, saw_error, finished) =
            collect(sse(&[MESSAGE_START, TEXT_START, TEXT_DELTA, STOP_SEQUENCE_DELTA])).await;
        assert!(!saw_error);
        let response = finished.expect("the reply ended");

        let raw = &response.raw;
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

        // The capture maps to the same end the reply finished with.
        let end = super::super::finish_of(&typed);
        assert_eq!(end.message_id, response.message_id);
        assert_eq!(end.model, response.model);
        assert_eq!(end.usage, response.usage);
        assert_eq!(
            response.finish_reason(),
            Some(crate::completion::FinishReason::Stop)
        );
        assert_eq!(response.usage.output_tokens, Some(3));
    }
}

/// A `tool_use` block whose wire id is empty gets an id rig issues: two
/// such blocks in one reply stay distinct calls.
#[test]
fn an_empty_tool_use_id_is_minted_not_keyed_on_the_empty_string() {
    let decoded = decode([
        tool_use(0, "", "add"),
        stop(0),
        tool_use(1, "", "add"),
        stop(1),
        message_delta("tool_use", PartialUsage::default()),
    ]);
    let ids: Vec<_> = decoded
        .outcome
        .expect("the reply ended")
        .tool_calls()
        .map(|call| call.id.clone())
        .collect();
    assert_eq!(ids.len(), 2);
    assert!(ids.iter().all(|id| id.provider().is_none()), "{ids:?}");
    assert_ne!(ids[0], ids[1], "each id-less call is its own call");
}

/// The Messages projection, driven through [`crate::driver`].
///
/// The projector rides on the wire's [`crate::wire::Encoded`], and the HTTP
/// transport calls it on every raw payload — the rejection body included — so
/// these cells exercise the wire and the driver together rather than the
/// projector in isolation. That is the only way the *closure* facts
/// (`Started`, `Response`, `Finished`) are observable at all: they belong to
/// the attempt, not to the payload.
mod projection {

    use std::sync::Arc;

    use crate::completion::CompletionRequest;
    use crate::observe::{
        Action, AdapterContext, AdapterEnding, AdapterErrorBoundary, AdapterErrorEnvelope,
        AdapterEvent, AdapterUsage, AdapterVerdict, ObservationLog, Subject,
    };
    use crate::providers::anthropic::wire::{AnthropicConfig, Messages};
    use crate::test_utils::{MockStreamingClient, RecordingHttpClient};
    use futures::StreamExt;

    fn adapter_events(log: &ObservationLog) -> Vec<AdapterEvent> {
        log.trace()
            .observations
            .iter()
            .filter_map(|o| match &o.action {
                Action::Adapter { observation } => Some(observation.event.clone()),
                _ => None,
            })
            .collect()
    }

    fn context(log: &Arc<ObservationLog>) -> AdapterContext {
        AdapterContext::new(log.clone(), Subject::default(), "call")
    }

    fn wire() -> Messages {
        AnthropicConfig::new("test-key").completion("claude-test")
    }

    fn request() -> CompletionRequest {
        CompletionRequest {
            model: None,
            chat_history: crate::NonEmpty::new(crate::message::Message::user("hello")),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: Some(64),
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        }
    }

    /// A rejected Messages call: the envelope's type and message, and the
    /// closure with the one funnel's classification.
    #[tokio::test]
    async fn messages_rejection_projects_the_envelope() {
        let http = RecordingHttpClient::with_error(
            http::StatusCode::SERVICE_UNAVAILABLE,
            r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
        );
        let log = Arc::new(ObservationLog::default());
        let error = crate::driver::Model::new(wire(), http.clone())
            .call_observed(request(), context(&log))
            .await
            .expect_err("the transport rejects the call");
        assert!(error.is_retryable());
        let events = adapter_events(&log);
        assert!(
            matches!(&events[0], AdapterEvent::Started { method, route } if method == "POST" && route == "/v1/messages")
        );
        assert!(events.contains(&AdapterEvent::Response { status: 503 }));
        assert!(events.contains(&AdapterEvent::ErrorEnvelope {
            error: AdapterErrorEnvelope {
                code: None,
                status: Some("overloaded_error".into()),
                message: Some("Overloaded".into()),
            }
        }));
        assert_eq!(
            events.last(),
            Some(&AdapterEvent::Finished {
                ending: AdapterEnding::Error {
                    boundary: AdapterErrorBoundary::ProviderResponse,
                    kind: "provider_response".into(),
                    status: Some(503),
                    retryable: true,
                }
            })
        );
    }

    /// A Messages stream: `message_start` carries the id, the model and the
    /// prompt usage; `message_delta` carries the stop reason and the answer's
    /// usage; the terminal closes the attempt.
    #[tokio::test]
    async fn messages_stream_projects_usage_stop_reason_and_model() {
        let sse = "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude-sonnet-4-6\",\"content\":[],\"stop_reason\":null,\"usage\":{\"input_tokens\":9,\"output_tokens\":1,\"cache_read_input_tokens\":0}}}\n\n\
    event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n\
    event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"hi\"}}\n\n\
    event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n\
    event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":3,\"output_tokens_details\":{\"thinking_tokens\":2}}}\n\n\
    event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n";
        let http = MockStreamingClient {
            sse_bytes: bytes::Bytes::from(sse),
        };
        let log = Arc::new(ObservationLog::default());
        let stream = crate::driver::tests::stream(&wire(), &http, request(), Some(context(&log)))
            .expect("the streamed request encodes");
        let mut stream = Box::pin(stream);
        while let Some(item) = stream.next().await {
            item.expect("the recorded stream decodes without an in-band error");
        }
        drop(stream);

        let events = adapter_events(&log);
        assert!(events.contains(&AdapterEvent::Usage {
            usage: AdapterUsage {
                input_tokens: Some(9),
                output_tokens: Some(1),
                cached_input_tokens: Some(0),
                ..AdapterUsage::default()
            }
        }));
        // The witnessed reasoning count is the one the response reports.
        assert!(events.contains(&AdapterEvent::Usage {
            usage: AdapterUsage {
                output_tokens: Some(3),
                reasoning_tokens: Some(2),
                ..AdapterUsage::default()
            }
        }));
        assert!(events.iter().any(|event| matches!(
            event,
            AdapterEvent::Provider { verdict: AdapterVerdict { model: Some(model), finish_reason: None, .. } }
                if model == "claude-sonnet-4-6"
        )));
        assert!(events.iter().any(|event| matches!(
            event,
            AdapterEvent::Provider { verdict: AdapterVerdict { finish_reason: Some(reason), .. } }
                if reason == "end_turn"
        )));
        assert_eq!(
            events.last(),
            Some(&AdapterEvent::Finished {
                ending: AdapterEnding::Terminal
            })
        );
    }
}
