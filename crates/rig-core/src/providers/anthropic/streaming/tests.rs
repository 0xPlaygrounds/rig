use super::super::completion::{
    AnthropicCompletionRequest, AnthropicRequestParams, CLAUDE_OPUS_4_8, CLAUDE_SONNET_4_6,
    CacheControl, CacheTtl, Message, SystemContent, anthropic_citations,
    apply_prompt_cache_control, build_tool_definitions, resolve_top_level_cache_control,
};
use super::*;
use crate::completion::CompletionRequest;
use crate::completion::Message as RigMessage;
use crate::completion::request::Document as RigDocument;
use crate::driver::{Decoded, decode_events};
use crate::message::{AssistantContent, Opaque, Reasoning, StopReason};
use crate::providers::anthropic::wire::AnthropicConfig;
use crate::streaming::StreamEvent;
use crate::wire::Mode;
use serde_json::json;

/// A fresh decoder, for its classifier.
fn adapter() -> MessagesDecoder {
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

/// The response the stream of `frames` folds into on the Anthropic wire.
fn streamed(frames: &[Value]) -> Result<crate::completion::CompletionResponse, ProviderError> {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    crate::test_utils::decode_reply(
        &wire,
        &CompletionRequest::new("hello"),
        Mode::Streaming,
        frames
            .iter()
            .map(|frame| WireFrame::Text(frame.to_string())),
        Value::Null,
    )
}

/// The frames of one content block: its start, its deltas and its stop.
fn block(index: usize, start: Value, deltas: &[Value]) -> Vec<Value> {
    std::iter::once(json!({"type": "content_block_start", "index": index, "content_block": start}))
        .chain(
            deltas.iter().map(
                |delta| json!({"type": "content_block_delta", "index": index, "delta": delta}),
            ),
        )
        .chain([json!({"type": "content_block_stop", "index": index})])
        .collect()
}

/// A stream of `blocks` ending with `stop_reason`.
fn reply(blocks: Vec<Vec<Value>>, stop_reason: &str) -> Vec<Value> {
    std::iter::once(json!({"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": CLAUDE_SONNET_4_6,
        "content": [], "stop_reason": null, "stop_sequence": null,
        "usage": {"input_tokens": 3, "output_tokens": 1}
    }}))
    .chain(blocks.into_iter().flatten())
    .chain([json!({"type": "message_delta",
        "delta": {"stop_reason": stop_reason, "stop_sequence": null},
        "usage": {"output_tokens": 5}})])
    .collect()
}

/// The provider item a decoded block holds.
fn item(content: &AssistantContent) -> Option<&Value> {
    match content {
        AssistantContent::Opaque(Opaque { item, .. }) => Some(item),
        content => content.native_item(),
    }
}

/// The message delta that ends a reply with `stop_reason`.
fn message_delta(stop_reason: &str, usage: PartialUsage) -> StreamingEvent {
    StreamingEvent::MessageDelta {
        delta: MessageDelta {
            stop_reason: Some(stop_reason.to_string()),
            stop_sequence: None,
            stop_details: None,
            container: None,
        },
        usage,
    }
}

fn tool_use(index: usize, id: &str, name: &str) -> StreamingEvent {
    classified(
        &json!({"type": "content_block_start", "index": index,
            "content_block": {"type": "tool_use", "id": id, "name": name, "input": {}}})
        .to_string(),
    )
}

fn input_json(index: usize, partial_json: &str) -> StreamingEvent {
    classified(
        &json!({"type": "content_block_delta", "index": index,
            "delta": {"type": "input_json_delta", "partial_json": partial_json}})
        .to_string(),
    )
}

fn stop(index: usize) -> StreamingEvent {
    StreamingEvent::ContentBlockStop { index }
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
            name: crate::message::ToolName::new("rig_tool").expect("tool name"),
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
    let request = CompletionRequest::from(vec![
        RigMessage::system("System prompt"),
        RigMessage::assistant("Earlier assistant turn"),
        RigMessage::system("Mid-conversation instruction"),
        RigMessage::user("Prompt"),
    ])
    .max_tokens(64)
    .documents(vec![RigDocument {
        id: "doc1".to_string(),
        text: "Document text.".to_string(),
        additional_props: Default::default(),
    }]);

    let body = built_streaming_body(CLAUDE_OPUS_4_8, request, false)
        .expect("streaming request body should build");

    assert_eq!(body["system"][0]["text"], "System prompt");
    assert_eq!(body["system"].as_array().map(Vec::len), Some(1));
    let messages = body["messages"]
        .as_array()
        .expect("messages should be array");
    // The misplaced mid-conversation instruction follows the user turn.
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[3]["role"], "system");
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

    let request = CompletionRequest::from(vec![
        RigMessage::system("You are helpful"),
        RigMessage::user("What's the weather?"),
    ])
    .temperature(0.5)
    .max_tokens(64)
    .output_schema(schema);

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
    let request = CompletionRequest::new(RigMessage::user("Add 2 and 3"))
        .max_tokens(64)
        .tools(vec![crate::completion::ToolDefinition {
            name: crate::message::ToolName::new("add").expect("tool name"),
            description: "Add x and y".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "x": { "type": "integer" } }
            }),
        }]);

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
    let request = CompletionRequest::new(RigMessage::user("Look this up"))
        .max_tokens(64)
        .tools(vec![crate::completion::ToolDefinition {
            name: crate::message::ToolName::new("lookup").expect("tool name"),
            description: "Look up a value".to_string(),
            parameters: json!({
                "type": "object",
                "properties": { "query": { "type": "string" } },
                "required": ["query"]
            }),
        }]);

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
    let request = CompletionRequest::new(RigMessage::user("Hi"))
        .max_tokens(64)
        .tool_choice(crate::message::ToolChoice::Auto);

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
            name: crate::message::ToolName::new("rig_tool").expect("tool name"),
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

/// Signature fragments concatenate onto the opening signature, as pi
/// assembles them, and the thinking item holds the whole block.
#[test]
fn thinking_assembles_its_text_and_signature_into_its_item() {
    let frames = reply(
        vec![block(
            0,
            json!({"type": "thinking", "thinking": "", "signature": "open-"}),
            &[
                json!({"type": "thinking_delta", "thinking": "Let me "}),
                json!({"type": "thinking_delta", "thinking": "think."}),
                json!({"type": "signature_delta", "signature": "sig_"}),
                json!({"type": "signature_delta", "signature": "end"}),
            ],
        )],
        "end_turn",
    );
    let response = streamed(&frames).expect("the reply folds");
    let [AssistantContent::Reasoning(reasoning)] = response.choice.as_slice() else {
        panic!("one reasoning block: {:?}", response.choice);
    };
    assert_eq!(reasoning.text, "Let me think.");
    assert!(!reasoning.redacted);
    assert_eq!(
        item(&response.choice[0]),
        Some(
            &json!({"type": "thinking", "thinking": "Let me think.", "signature": "open-sig_end"})
        )
    );
}

/// The adaptive-thinking shape recorded in
/// `anthropic/opus_4_7/messages_adaptive_thinking_streaming_smoke.yaml`: an
/// empty thinking block whose only content is the signature its deltas
/// carry. The block and its signature survive `content_block_stop`.
#[test]
fn signature_only_thinking_block_keeps_its_item() {
    let frames = reply(
        vec![block(
            0,
            json!({"type": "thinking", "thinking": "", "signature": ""}),
            &[json!({"type": "signature_delta", "signature": "the_whole_signature"})],
        )],
        "end_turn",
    );
    let response = streamed(&frames).expect("the reply folds");
    assert_eq!(response.choice.len(), 1);
    assert_eq!(
        item(&response.choice[0]).and_then(|item| item.get("signature")),
        Some(&json!("the_whole_signature"))
    );
}

/// `content_block_start` can carry the block's opening text; it streams as
/// the first fragment.
#[test]
fn thinking_block_start_text_streams_as_the_first_fragment() {
    let decoded = decode([
        classified(
            r#"{"type":"content_block_start","index":2,"content_block":{"type":"thinking","thinking":"opening "}}"#,
        ),
        classified(
            r#"{"type":"content_block_delta","index":2,"delta":{"type":"thinking_delta","thinking":"rest"}}"#,
        ),
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
}

#[test]
fn redacted_thinking_is_a_redacted_block_holding_its_payload() {
    let frames = reply(
        vec![block(
            0,
            json!({"type": "redacted_thinking", "data": "redacted_blob"}),
            &[],
        )],
        "end_turn",
    );
    let response = streamed(&frames).expect("the reply folds");
    let [AssistantContent::Reasoning(Reasoning { text, redacted, .. })] =
        response.choice.as_slice()
    else {
        panic!("one reasoning block: {:?}", response.choice);
    };
    assert!(text.is_empty() && *redacted);
    assert_eq!(
        item(&response.choice[0]),
        Some(&json!({"type": "redacted_thinking", "data": "redacted_blob"}))
    );
}

#[test]
fn text_streams_its_fragments_and_holds_the_whole_block() {
    let frames = reply(
        vec![block(
            0,
            json!({"type": "text", "text": ""}),
            &[
                json!({"type": "text_delta", "text": "Hello, "}),
                json!({"type": "text_delta", "text": "world!"}),
            ],
        )],
        "end_turn",
    );
    let response = streamed(&frames).expect("the reply folds");
    assert_eq!(response.text(), "Hello, world!");
    assert_eq!(
        item(&response.choice[0]),
        Some(&json!({"type": "text", "text": "Hello, world!"}))
    );
}

/// A part streams nothing until its first fragment.
#[test]
fn a_text_block_start_streams_nothing() {
    let decoded = decode([classified(
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
    )]);
    assert!(decoded.events().is_empty(), "{:?}", decoded.events());
}

#[test]
fn a_call_streams_nothing_until_it_closes() {
    let decoded = decode([
        tool_use(0, "tool_123", "lookup"),
        input_json(0, "{\"arg\":\"value"),
    ]);
    assert!(decoded.events().is_empty(), "{:?}", decoded.events());
}

/// A call's item holds the input its fragments assembled, and keeps every
/// field the provider sent, `caller` included.
#[test]
fn a_call_assembles_its_input_into_its_item() {
    let frames = reply(
        vec![block(
            0,
            json!({"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {},
                "caller": {"type": "direct"}}),
            &[
                json!({"type": "input_json_delta", "partial_json": "{\"location\":"}),
                json!({"type": "input_json_delta", "partial_json": "\"Paris\"}"}),
            ],
        )],
        "tool_use",
    );
    let response = streamed(&frames).expect("the reply folds");
    let [AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
        panic!("one call: {:?}", response.choice);
    };
    assert_eq!(call.id.to_string(), "toolu_1");
    assert_eq!(
        call.function.arguments_value(),
        json!({"location": "Paris"})
    );
    assert_eq!(
        item(&response.choice[0]),
        Some(
            &json!({"type": "tool_use", "id": "toolu_1", "name": "lookup",
            "input": {"location": "Paris"}, "caller": {"type": "direct"}})
        )
    );
    assert_eq!(response.stop(), StopReason::ToolUse);
}

#[test]
fn malformed_streamed_call_input_keeps_what_it_states() {
    let frames = reply(
        vec![block(
            0,
            json!({"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": {}}),
            &[json!({"type": "input_json_delta", "partial_json": "{\"x\": \"ab"})],
        )],
        "max_tokens",
    );
    let response = streamed(&frames).expect("a cut-off call does not fail the reply");
    let call = response.tool_calls().next().expect("the call is kept");
    assert_eq!(call.function.arguments_value(), json!({"x": "ab"}));
    assert_eq!(
        call.function.invalid_arguments.as_deref(),
        Some("{\"x\": \"ab")
    );
    assert_eq!(response.stop(), StopReason::Length);
}

#[test]
fn input_to_a_text_block_fails_the_reply() {
    let frames = reply(
        vec![block(
            0,
            json!({"type": "text", "text": ""}),
            &[json!({"type": "input_json_delta", "partial_json": "{}"})],
        )],
        "end_turn",
    );
    assert!(streamed(&frames).is_err());
}

/// Citations arrive as deltas onto a text block that opened with `null` or
/// an empty list, and land on its item in order.
#[test]
fn citation_deltas_land_on_the_text_item() {
    let citation = json!({"type": "char_location", "cited_text": "The grass is green.",
        "document_index": 0, "document_title": "Example", "start_char_index": 0,
        "end_char_index": 20});
    for opening in [json!(null), json!([])] {
        let frames = reply(
            vec![block(
                0,
                json!({"type": "text", "text": "", "citations": opening}),
                &[
                    json!({"type": "citations_delta", "citation": citation}),
                    json!({"type": "text_delta", "text": "the grass is green"}),
                ],
            )],
            "end_turn",
        );
        let response = streamed(&frames).expect("the reply folds");
        let [AssistantContent::Text(text)] = response.choice.as_slice() else {
            panic!("one text block: {:?}", response.choice);
        };
        assert_eq!(text.text, "the grass is green");
        let citations = anthropic_citations(text).expect("the citations parse");
        assert_eq!(
            serde_json::to_value(&citations).expect("citations serialize"),
            json!([citation])
        );
    }
}

#[test]
fn a_known_citation_with_a_defective_payload_is_corrupt() {
    let frame = WireFrame::Text(
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"citations_delta","citation":{"type":"char_location","cited_text":1}}}"#.into(),
    );
    assert!(matches!(
        adapter().classify(frame),
        crate::wire::WireEvent::Corrupt(_)
    ));
}

/// Server tools and their results are provider items with no canonical
/// meaning: each is one opaque block that replays, its streamed input
/// assembled onto it. None becomes a client tool call.
#[test]
fn server_tool_blocks_are_opaque_items_that_replay() {
    let result = json!({"type": "web_search_tool_result", "tool_use_id": "srvtoolu_01",
        "content": [{"type": "web_search_result", "url": "https://example.com/shannon",
            "title": "Claude Shannon", "encrypted_content": "encrypted-content"}]});
    let frames = reply(
        vec![
            block(
                0,
                json!({"type": "server_tool_use", "id": "srvtoolu_01", "name": "web_search", "input": {}}),
                &[json!({"type": "input_json_delta", "partial_json": "{\"query\":\"shannon\"}"})],
            ),
            block(1, result.clone(), &[]),
            block(
                2,
                json!({"type": "code_execution_tool_result", "tool_use_id": "srvtoolu_02",
                    "content": {"type": "code_execution_result", "return_code": 0,
                        "stdout": "42\n", "stderr": "", "content": []}}),
                &[],
            ),
            block(
                3,
                json!({"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "fetch",
                "server_name": "docs", "input": {}}),
                &[],
            ),
            block(
                4,
                json!({"type": "container_upload", "file_id": "file_1"}),
                &[],
            ),
        ],
        "end_turn",
    );
    let response = streamed(&frames).expect("the reply folds");
    assert_eq!(response.choice.len(), 5);
    assert!(response.choice.iter().all(|content| matches!(
        content,
        AssistantContent::Opaque(Opaque { replay: true, .. })
    )));
    assert_eq!(
        item(&response.choice[0]),
        Some(
            &json!({"type": "server_tool_use", "id": "srvtoolu_01", "name": "web_search",
            "input": {"query": "shannon"}})
        )
    );
    assert_eq!(item(&response.choice[1]), Some(&result));
}

/// A compaction block streams its summary as `compaction_delta`s, which
/// merge into its item.
#[test]
fn compaction_deltas_assemble_the_compaction_item() {
    let frames = reply(
        vec![block(
            0,
            json!({"type": "compaction", "content": ""}),
            &[
                json!({"type": "compaction_delta", "content": "Summary "}),
                json!({"type": "compaction_delta", "content": "so far."}),
            ],
        )],
        "end_turn",
    );
    let response = streamed(&frames).expect("the reply folds");
    assert_eq!(
        item(&response.choice[0]),
        Some(&json!({"type": "compaction", "content": "Summary so far."}))
    );
}

/// A delta kind rig has never seen lands in the item it targets, and the
/// stream goes on.
#[test]
fn a_novel_delta_merges_into_its_item() {
    let frames = reply(
        vec![block(
            0,
            json!({"type": "text", "text": ""}),
            &[
                json!({"type": "text_delta", "text": "hi"}),
                json!({"type": "banana_delta", "banana": "ripe"}),
            ],
        )],
        "end_turn",
    );
    let response = streamed(&frames).expect("the reply folds");
    assert_eq!(response.text(), "hi");
    assert_eq!(
        item(&response.choice[0]),
        Some(&json!({"type": "text", "text": "hi", "banana": "ripe"}))
    );
}

/// pi's fallback rule: a leading `fallback` block marks the model that
/// took over and never replays; one after output began fails the reply.
#[test]
fn a_fallback_block_is_kept_first_and_an_error_after_output() {
    let fallback = json!({"type": "fallback", "model": "claude-opus-4-8"});
    let text = json!({"type": "text", "text": ""});
    let leading = streamed(&reply(
        vec![
            block(0, fallback.clone(), &[]),
            block(
                1,
                text.clone(),
                &[json!({"type": "text_delta", "text": "hi"})],
            ),
        ],
        "end_turn",
    ))
    .expect("a leading fallback is legal");
    assert!(matches!(
        leading.choice.first(),
        Some(AssistantContent::Opaque(Opaque { replay: false, .. }))
    ));
    let late = streamed(&reply(
        vec![
            block(0, text, &[json!({"type": "text_delta", "text": "hi"})]),
            block(1, fallback, &[]),
        ],
        "end_turn",
    ));
    assert!(late.is_err(), "{late:?}");
}

/// pi's refusal rule: a refused turn ends in an error carrying Anthropic's
/// explanation, or a default one, and is never replayed.
#[test]
fn a_refusal_ends_the_turn_in_an_error_with_its_explanation() {
    let mut frames = reply(vec![], "refusal");
    if let Some(delta) = frames.last_mut() {
        delta["delta"]["stop_details"] =
            json!({"type": "refusal", "category": "cyber", "explanation": "Not this."});
    }
    let response = streamed(&frames).expect("the reply folds");
    assert_eq!(response.stop(), StopReason::Error("Not this.".to_owned()));

    let response = streamed(&reply(vec![], "refusal")).expect("the reply folds");
    assert_eq!(
        response.stop(),
        StopReason::Error("The model refused to complete the request".to_owned())
    );
}

/// The reply-level `container` is the turn's message-level provider item.
#[test]
fn the_reply_container_is_the_message_item() {
    let container = json!({"id": "container_1", "expires_at": "2026-10-01T00:00:00Z"});
    let mut frames = reply(
        vec![block(0, json!({"type": "text", "text": "done"}), &[])],
        "end_turn",
    );
    if let Some(delta) = frames.last_mut() {
        delta["delta"]["container"] = container.clone();
    }
    let response = streamed(&frames).expect("the reply folds");
    let Some(RigMessage::Assistant(turn)) = response.message() else {
        panic!("an assistant turn");
    };
    assert_eq!(turn.native_item(), Some(&json!({ "container": container })));
}

#[test]
fn a_tool_use_without_its_id_is_corrupt() {
    let frame = WireFrame::Text(
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"tool_use","name":"add","input":{}}}"#.into(),
    );
    assert!(matches!(
        adapter().classify(frame),
        crate::wire::WireEvent::Corrupt(_)
    ));
}

/// Classification is the only policy site. An unmodeled *top-level* event
/// type is `Unknown` (driver: warn + skip); a `ping` is Known; and a known
/// tag whose payload this client cannot decode is `Corrupt`, never silently
/// demoted to an ignorable unknown. An unmodeled *nested* delta type is
/// Known and lands in its item.
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

/// A terminal `message_delta` carrying only the output count (Anthropic's
/// older shape, and Messages gateways) keeps `message_start`'s cache
/// counters, so input still counts the cached prefix: 10 uncached, 6 read
/// and 4 written.
#[test]
fn cache_usage_from_message_start_survives_output_only_terminal_delta() {
    let decoded = decode([
        classified(
            r#"{"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0,"cache_creation_input_tokens":4,"cache_read_input_tokens":6,"cache_creation":{"ephemeral_1h_input_tokens":3,"ephemeral_5m_input_tokens":1}}}}"#,
        ),
        classified(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3}}"#,
        ),
    ]);
    let response = decoded.outcome.expect("the message_delta ends the reply");
    assert_eq!(response.usage.input_tokens, Some(10 + 6 + 4));
    assert_eq!(response.usage.cache_creation_input_tokens, Some(4));
    assert_eq!(response.usage.cached_input_tokens, Some(6));
    assert_eq!(response.usage.output_tokens, Some(3));
    assert_eq!(response.usage.total_tokens, Some(23));
    let native: StreamingCompletionResponse =
        serde_json::from_value(response.raw).expect("raw must be the native terminal");
    assert_eq!(native.usage.cache_creation_input_tokens, Some(4));
    assert_eq!(native.usage.cache_read_input_tokens, Some(6));
    assert_eq!(
        native
            .usage
            .cache_creation
            .map(|cache| cache.ephemeral_1h_input_tokens),
        Some(3)
    );
}

/// An explicit terminal zero is authoritative over `message_start`'s counts.
#[test]
fn terminal_cache_usage_zero_overrides_message_start() {
    let decoded = decode([
        classified(
            r#"{"type":"message_start","message":{"id":"msg_1","role":"assistant","content":[],"model":"claude-sonnet-4-6","stop_reason":null,"stop_sequence":null,"usage":{"input_tokens":10,"output_tokens":0,"cache_creation_input_tokens":4,"cache_read_input_tokens":6}}}"#,
        ),
        classified(
            r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":3,"cache_creation_input_tokens":0,"cache_read_input_tokens":0}}"#,
        ),
    ]);
    let response = decoded.outcome.expect("the message_delta ends the reply");
    assert_eq!(response.usage.cache_creation_input_tokens, Some(0));
    assert_eq!(response.usage.cached_input_tokens, Some(0));
    assert_eq!(response.usage.input_tokens, Some(10));
    assert_eq!(response.usage.total_tokens, Some(13));
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
    assert!(
        decoded.events().is_empty(),
        "a message-less message_start is a no-op"
    );
}

#[test]
fn terminal_record_normalizes_stop_reason_usage_and_metadata() {
    let decoded = decode([
        classified(&format!(
            r#"{{"type":"message_start","message":{{"id":"msg_1","role":"assistant","content":[],"model":"{CLAUDE_OPUS_4_8}","stop_reason":null,"stop_sequence":null,"usage":{{"input_tokens":3,"output_tokens":0}}}}}}"#
        )),
        classified(
            r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        ),
        classified(
            r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"hi"}}"#,
        ),
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
    assert_eq!(response.provider(), "anthropic");
    assert_eq!(response.response_id(), Some("msg_1"));
    assert_eq!(response.model(), Some(CLAUDE_OPUS_4_8));
    assert_eq!(
        response.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
    // Input counts the cache read Anthropic reports beside `input_tokens`.
    assert_eq!(response.usage.input_tokens, Some(5));
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

        let (_, saw_error, finished) = collect(sse(&[
            MESSAGE_START,
            TEXT_START,
            TEXT_DELTA,
            STOP_SEQUENCE_DELTA,
        ]))
        .await;
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
        assert_eq!(typed.message_id.as_deref(), response.response_id());
        assert_eq!(typed.model.as_deref(), response.model());
        assert_eq!(crate::completion::Usage::from(&typed.usage), response.usage);
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
        CompletionRequest::new("hello").max_tokens(64)
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
/// One sample per Messages event, through an exhaustive, wildcard-free
/// index: a new event fails to compile until it is numbered, and fails
/// here until it has a sample.
#[test]
fn every_messages_event_has_a_sample() {
    let samples: Vec<StreamingEvent> = [
        r#"{"type":"message_start","message":null}"#,
        r#"{"type":"message","id":"msg_1","role":"assistant","model":"m","content":[],"stop_reason":"end_turn","stop_sequence":null,"usage":{"input_tokens":1,"output_tokens":1}}"#,
        r#"{"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}"#,
        r#"{"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"x"}}"#,
        r#"{"type":"content_block_stop","index":0}"#,
        r#"{"type":"message_delta","delta":{"stop_reason":"end_turn","stop_sequence":null},"usage":{"output_tokens":1}}"#,
        r#"{"type":"message_stop"}"#,
        r#"{"type":"ping"}"#,
        r#"{"type":"error","error":{"type":"overloaded_error","message":"Overloaded"}}"#,
    ]
    .into_iter()
    .map(classified)
    .collect();
    let index = |event: &StreamingEvent| match event {
        StreamingEvent::MessageStart { .. } => 0,
        StreamingEvent::Message { .. } => 1,
        StreamingEvent::ContentBlockStart { .. } => 2,
        StreamingEvent::ContentBlockDelta { .. } => 3,
        StreamingEvent::ContentBlockStop { .. } => 4,
        StreamingEvent::MessageDelta { .. } => 5,
        StreamingEvent::MessageStop => 6,
        StreamingEvent::Ping => 7,
        StreamingEvent::Error { .. } => 8,
    };
    crate::test_utils::history::assert_every_variant(&samples, index, 9);
}

/// An item type rig has never seen, and a field it has never seen on a
/// known item, survive decoding whole and streamed, and go back verbatim
/// to the model that produced them.
#[test]
fn an_invented_item_and_field_survive_decode_and_same_model_replay() {
    use crate::wire::{Operation, Wire};

    let invented = json!({"type": "frobnicate", "payload": {"x": 1}});
    let text = json!({"type": "text", "text": "hi", "sparkle": true});
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let whole = json!({
        "type": "message", "id": "msg_1", "role": "assistant", "model": CLAUDE_SONNET_4_6,
        "content": [invented, text], "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 3, "output_tokens": 1}
    });
    let unary = crate::test_utils::decode_reply(
        &wire,
        &CompletionRequest::new("hello"),
        Mode::Unary,
        [WireFrame::Text(whole.to_string())],
        whole.clone(),
    )
    .expect("the whole reply folds");
    let stream = streamed(&reply(
        vec![
            block(0, invented.clone(), &[]),
            block(
                1,
                json!({"type": "text", "text": "", "sparkle": true}),
                &[json!({"type": "text_delta", "text": "hi"})],
            ),
        ],
        "end_turn",
    ))
    .expect("the stream folds");
    assert_eq!(unary.message(), stream.message());

    for response in [unary, stream] {
        let turn = response.message().expect("an assistant turn");
        let request = CompletionRequest::from(vec![
            RigMessage::user("hello"),
            turn,
            RigMessage::user("again"),
        ]);
        let request = Completion::prepare(request, &wire.describe()).expect("the history adapts");
        let encoded = wire
            .encode(request, Mode::Unary)
            .expect("the request encodes");
        let body = crate::test_utils::json_body(&encoded.request);
        assert_eq!(body["messages"][1]["content"], json!([invented, text]));
    }
}
