use super::*;
use crate::error::ProviderError;
use crate::providers::anthropic::wire::AnthropicConfig;
use crate::test_utils::json_body;
use crate::wire::WireFrame;

/// The wire settings a request body is built with.
struct Params<'a> {
    model: &'a str,
    request: CompletionRequest,
    prompt_caching: bool,
    automatic_caching: bool,
    automatic_caching_ttl: Option<CacheTtl>,
    static_prefix_cache_ttl: Option<CacheTtl>,
}

/// The body `params` builds, its documents placed as `prepare` places them.
fn request_body(params: Params<'_>) -> Result<Value, EncodeError> {
    request_body_with(params, false)
}

/// [`request_body`], with strict tools when `strict`.
fn request_body_with(params: Params<'_>, strict: bool) -> Result<Value, EncodeError> {
    let mut wire = AnthropicConfig::new("test-key").completion(params.model);
    wire.prompt_caching = params.prompt_caching;
    wire.automatic_caching = params.automatic_caching;
    wire.automatic_caching_ttl = params.automatic_caching_ttl;
    wire.static_prefix_cache_ttl = params.static_prefix_cache_ttl;
    wire.strict_tools = strict;
    let mut request = params.request;
    request.chat_history = request.chat_history_with_documents();
    request.documents.clear();
    body(&wire, request, Mode::Unary)
}

/// `message` alone on the wire.
fn convert(message: message::Message) -> Result<Option<Value>, EncodeError> {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let history = [message];
    let ids = WireIds::for_target(&history, &wire, CLAUDE_SONNET_4_6);
    history
        .iter()
        .map(|message| message_json(message, &wire, &ids))
        .next()
        .unwrap_or(Ok(None))
}

/// The user message holding `part`, as the encoder converts it.
fn user_wire(part: message::UserContent) -> Result<Option<Value>, EncodeError> {
    convert(message::Message::User {
        content: vec![part],
    })
}

/// An assistant turn of `blocks` on the wire.
fn assistant_wire(blocks: Vec<message::AssistantContent>) -> Vec<Value> {
    convert(message::Message::Assistant(message::AssistantMessage::new(
        blocks,
    )))
    .expect("the turn converts")
    .and_then(|message| message["content"].as_array().cloned())
    .unwrap_or_default()
}

/// The one-turn request every reply below is folded against.
fn hello_request() -> CompletionRequest {
    completion_request_with_history(vec![message::Message::user("hello")], None)
}

/// Fold a Messages reply body through the wire's own decoder — the one
/// mapping there is, now that the duplicate `normalize` is gone.
///
/// `POST /v1/messages` answers with a whole `message` object, which the
/// decoder treats as a frame like any other, so a cell that used to call
/// `normalize` on a typed response states the same thing by folding the
/// body the provider would have sent. `type` is injected when the cell
/// built the typed response directly, since that type does not model the
/// tag. Whatever the decoder or the fold rejects surfaces here as the
/// error the driver would report.
fn fold_reply(body: &serde_json::Value) -> Result<completion::CompletionResponse, ProviderError> {
    let mut body = body.clone();
    if let Some(map) = body.as_object_mut() {
        map.entry("type").or_insert_with(|| json!("message"));
    }
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    crate::test_utils::decode_reply(
        &wire,
        &hello_request(),
        crate::wire::Mode::Unary,
        [WireFrame::Text(body.to_string())],
        body,
    )
}

#[test]
fn current_model_default_max_tokens_match_anthropic_limits() {
    assert_eq!(
        default_max_tokens_for_model(CLAUDE_FABLE_5_1),
        Some(128_000)
    );
    assert_eq!(default_max_tokens_for_model(CLAUDE_FABLE_5), Some(128_000));
    assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_5), Some(128_000));
    assert_eq!(default_max_tokens_for_model(CLAUDE_SONNET_5), Some(128_000));
    assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_4_8), Some(128_000));
    assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_4_7), Some(128_000));
    assert_eq!(default_max_tokens_for_model(CLAUDE_OPUS_4_6), Some(128_000));
    assert_eq!(
        default_max_tokens_for_model(CLAUDE_SONNET_4_6),
        Some(128_000)
    );
    assert_eq!(default_max_tokens_for_model(CLAUDE_HAIKU_4_5), Some(64_000));
}

#[test]
fn unknown_model_has_no_documented_default_max_tokens() {
    assert_eq!(default_max_tokens_for_model("claude-unknown"), None);
}

fn generic_tool(name: &str) -> completion::ToolDefinition {
    completion::ToolDefinition {
        name: crate::message::ToolName::new(name).expect("tool name"),
        description: format!("{name} description"),
        parameters: json!({
            "type": "object",
            "properties": {}
        }),
    }
}

fn completion_request_with_tools(
    tools: Vec<completion::ToolDefinition>,
    additional_params: Option<serde_json::Value>,
) -> CompletionRequest {
    CompletionRequest::from(vec![
        message::Message::system("System prompt"),
        message::Message::from("Hello"),
    ])
    .max_tokens(64)
    .tools(tools)
    .additional_params(additional_params)
}

fn completion_request_with_history(
    chat_history: Vec<message::Message>,
    preamble: Option<String>,
) -> CompletionRequest {
    CompletionRequest::from(
        preamble
            .map(message::Message::system)
            .into_iter()
            .chain(chat_history)
            .collect::<Vec<_>>(),
    )
    .max_tokens(64)
}

#[test]
fn rig_tools_are_non_strict_by_default() {
    let request = completion_request_with_tools(vec![generic_tool("lookup")], None);
    let request = request_body(Params {
        model: CLAUDE_SONNET_4_6,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert!(value["tools"][0].get("strict").is_none());
    assert!(
        value["tools"][0]["input_schema"]
            .get("additionalProperties")
            .is_none()
    );
}

/// Anthropic-compatible gateways do not necessarily implement Anthropic's
/// constrained tool schemas, so asking for strict tools on one leaves the
/// tool unchanged. The policy is the dialect's data, not a trait hook, so
/// this is asserted through the wire a gateway builds.
#[test]
fn strict_tool_hook_is_a_noop_for_anthropic_compatible_gateways() {
    use crate::wire::{Mode, Wire};

    let request = completion_request_with_tools(vec![generic_tool("lookup")], None);
    let encoded = crate::providers::anthropic::wire::AnthropicConfig::with_key(
        &crate::providers::anthropic::wire::ZAI,
        "k",
    )
    .completion("some-model")
    .with_strict_tools()
    .encode(request, Mode::Unary)
    .expect("the request encodes");
    let value = json_body(&encoded.request);

    assert!(value["tools"][0].get("strict").is_none());
    assert!(
        value["tools"][0]["input_schema"]
            .get("additionalProperties")
            .is_none()
    );
}

#[test]
fn strict_tools_opt_in_marks_and_sanitizes_rig_tools_only() {
    let mut tool = generic_tool("lookup");
    tool.parameters = json!({
        "type": "object",
        "additionalProperties": true,
        "properties": {
            "query": {
                "type": "string",
                "minLength": 2,
                "maxLength": 20,
                "pattern": "^[a-z]+$",
                "format": "uuid"
            },
            "kind": {
                "type": "string",
                "const": "lookup"
            },
            "legacy_filter": {
                "$ref": "#/definitions/LegacyFilter"
            },
            "options": {
                "type": "object",
                "additionalProperties": true,
                "properties": {
                    "limit": {
                        "type": ["integer", "null"],
                        "minimum": 1,
                        "maximum": 100,
                        "format": "uint32"
                    }
                }
            }
        },
        "definitions": {
            "LegacyFilter": {
                "type": "object",
                "properties": {
                    "term": { "type": "string" }
                }
            }
        },
        "required": ["query"]
    });
    let request = completion_request_with_tools(
        vec![tool],
        Some(json!({
            "tools": [{
                "type": "mcp_toolset",
                "name": "remote_tools"
            }]
        })),
    );
    let request = request_body_with(
        Params {
            model: CLAUDE_SONNET_4_6,
            request,
            prompt_caching: false,
            automatic_caching: false,
            automatic_caching_ttl: None,
            static_prefix_cache_ttl: None,
        },
        true,
    )
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let rig_tool = &value["tools"][0];
    assert_eq!(rig_tool["strict"], true);
    assert_eq!(rig_tool["input_schema"]["additionalProperties"], false);
    let required = rig_tool["input_schema"]["required"]
        .as_array()
        .expect("strict object schema should list required properties");
    assert_eq!(required.len(), 1);
    assert!(required.contains(&json!("query")));
    assert_eq!(
        rig_tool["input_schema"]["properties"]["options"]["additionalProperties"],
        false
    );
    assert!(
        rig_tool["input_schema"]["properties"]["options"]
            .get("required")
            .is_none()
    );
    let query = &rig_tool["input_schema"]["properties"]["query"];
    assert_eq!(query["format"], "uuid");
    for keyword in ["minLength", "maxLength", "pattern"] {
        assert!(query.get(keyword).is_none());
    }
    let query_description = query["description"]
        .as_str()
        .expect("unsupported string constraints should become guidance");
    for guidance in ["minLength: 2", "maxLength: 20", "pattern: ^[a-z]+$"] {
        assert!(query_description.contains(guidance));
    }
    assert_eq!(
        rig_tool["input_schema"]["properties"]["kind"]["const"],
        "lookup"
    );
    assert_eq!(
        rig_tool["input_schema"]["properties"]["legacy_filter"]["$ref"],
        "#/definitions/LegacyFilter"
    );
    assert_eq!(
        rig_tool["input_schema"]["definitions"]["LegacyFilter"]["additionalProperties"],
        false
    );
    let limit = &rig_tool["input_schema"]["properties"]["options"]["properties"]["limit"];
    assert!(limit.get("format").is_none());
    assert!(
        ["minimum", "maximum"]
            .into_iter()
            .all(|keyword| limit.get(keyword).is_none())
    );
    let limit_description = limit["description"]
        .as_str()
        .expect("unsupported numeric constraints should become guidance");
    for guidance in ["minimum: 1", "maximum: 100", "format: uint32"] {
        assert!(limit_description.contains(guidance));
    }

    let provider_tool = &value["tools"][1];
    assert_eq!(provider_tool["type"], "mcp_toolset");
    assert!(provider_tool.get("strict").is_none());
}

fn system_has_cache_control(value: &serde_json::Value) -> bool {
    value["system"]
        .as_array()
        .and_then(|blocks| blocks.last())
        .and_then(|block| block.get("cache_control"))
        .is_some()
}

fn last_message_has_cache_control(value: &serde_json::Value) -> bool {
    value["messages"]
        .as_array()
        .and_then(|messages| messages.last())
        .and_then(|message| message["content"].as_array())
        .and_then(|content| content.last())
        .and_then(|content| content.get("cache_control"))
        .is_some()
}

#[test]
fn opus_4_8_preserves_mid_conversation_system_message() {
    let request = completion_request_with_history(
        vec![
            message::Message::System {
                content: "Global history instruction.".to_string(),
            },
            message::Message::from("Review this code."),
            message::Message::System {
                content: "From now on, require explicit type annotations.".to_string(),
            },
        ],
        Some("Top-level instruction.".to_string()),
    );

    let request = request_body(Params {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert_eq!(value["system"][0]["text"], "Top-level instruction.");
    assert_eq!(value["system"][1]["text"], "Global history instruction.");

    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "system");
    assert_eq!(
        messages[1]["content"][0]["text"],
        "From now on, require explicit type annotations."
    );
}

#[test]
fn opus_4_8_preserves_mid_conversation_system_message_before_assistant_turn() {
    let request = completion_request_with_history(
        vec![
            message::Message::user("Review this code."),
            message::Message::System {
                content: "From now on, require explicit type annotations.".to_string(),
            },
            message::Message::assistant("I will enforce explicit type annotations."),
        ],
        None,
    );

    let request = request_body(Params {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "system");
    assert_eq!(messages[2]["role"], "assistant");
    assert!(value.get("system").is_none());
}

#[test]
fn opus_4_8_hoists_leading_system_message_when_documents_are_present() {
    let mut request = completion_request_with_history(
        vec![
            message::Message::System {
                content: "Global history instruction.".to_string(),
            },
            message::Message::assistant("Acknowledged."),
            message::Message::System {
                content: "Mid-conversation instruction.".to_string(),
            },
            message::Message::user("Answer from the document."),
        ],
        None,
    );
    request.documents = vec![completion::Document {
        id: "doc".to_string(),
        text: "Document context.".to_string(),
        additional_props: Default::default(),
    }];

    let request = request_body(Params {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert_eq!(value["system"][0]["text"], "Global history instruction.");
    assert_eq!(value["system"].as_array().map(Vec::len), Some(1));

    // The misplaced mid-conversation instruction moves after the next user
    // turn instead of into `system`.
    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 4);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[1]["role"], "assistant");
    assert_eq!(messages[2]["role"], "user");
    assert_eq!(messages[3]["role"], "system");
    assert!(
        messages[0].to_string().contains("<file id: doc>"),
        "document message should follow top-level system: {messages:?}"
    );
    assert_eq!(
        messages
            .iter()
            .filter(|message| message.to_string().contains("<file id: doc>"))
            .count(),
        1,
        "document message should appear exactly once: {messages:?}"
    );
    assert_eq!(
        messages
            .iter()
            .filter(|message| message["role"].as_str() == Some("system"))
            .count(),
        1
    );
}

#[test]
fn opus_4_8_preserves_system_message_after_assistant_server_tool_result() {
    let request = completion_request_with_history(
        vec![
            message::Message::Assistant(message::AssistantMessage::new(vec![
                message::AssistantContent::Opaque(message::Opaque {
                    item: json!({
                        "type": "server_tool_use",
                        "id": "srvtoolu_01",
                        "name": "web_search",
                        "input": {
                            "query": "clear daytime sky color"
                        }
                    }),
                    replay: true,
                }),
                message::AssistantContent::Opaque(message::Opaque {
                    item: json!({
                        "type": "web_search_tool_result",
                        "tool_use_id": "srvtoolu_01",
                        "content": {
                            "type": "web_search_tool_result_error",
                            "error_code": "unavailable"
                        }
                    }),
                    replay: true,
                }),
            ])),
            message::Message::System {
                content: "For the rest of this conversation, answer in Spanish.".to_string(),
            },
            message::Message::assistant("Entendido."),
        ],
        None,
    );

    let request = request_body(Params {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert!(value.get("system").is_none());

    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "assistant");
    assert_eq!(messages[0]["content"][0]["type"], "server_tool_use");
    assert_eq!(messages[0]["content"][1]["type"], "web_search_tool_result");
    assert_eq!(messages[1]["role"], "system");
    assert_eq!(
        messages[1]["content"][0]["text"],
        "For the rest of this conversation, answer in Spanish."
    );
    assert_eq!(messages[2]["role"], "assistant");
}

#[test]
fn opus_4_8_preserves_system_message_after_assistant_server_tool_use() {
    let request = completion_request_with_history(
        vec![
            message::Message::Assistant(message::AssistantMessage::new(vec![
                message::AssistantContent::Opaque(message::Opaque {
                    item: json!({
                        "type": "server_tool_use",
                        "id": "srvtoolu_01",
                        "name": "web_search",
                        "input": {
                            "query": "clear daytime sky color"
                        }
                    }),
                    replay: true,
                }),
            ])),
            message::Message::System {
                content: "For the rest of this conversation, answer in Spanish.".to_string(),
            },
            message::Message::assistant("Entendido."),
        ],
        None,
    );

    let request = request_body(Params {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert!(value.get("system").is_none());

    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[0]["role"], "assistant");
    assert_eq!(messages[0]["content"][0]["type"], "server_tool_use");
    assert_eq!(messages[1]["role"], "system");
    assert_eq!(
        messages[1]["content"][0]["text"],
        "For the rest of this conversation, answer in Spanish."
    );
    assert_eq!(messages[2]["role"], "assistant");
}

/// Encode a history for a model that keeps mid-conversation system messages
/// and return the wire's `system` and `messages` as JSON.
fn encode_mid_conversation_history(
    model: &str,
    history: Vec<message::Message>,
) -> serde_json::Value {
    let request = request_body(Params {
        model,
        request: completion_request_with_history(history, None),
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();
    serde_json::to_value(request).unwrap()
}

/// A misplaced system message moves after the next user turn instead of into
/// `system`, so the prompt prefix a cache or a bound thinking block depends on
/// stays the same. Offline because the position rule's every branch is shown
/// here; the recorded proof is the model sessions' mid-conversation phase
/// (`anthropic/models/<model>/session.yaml`), whose system message sits after
/// an assistant turn and before the user's question.
#[test]
fn opus_4_8_moves_a_misplaced_system_message_after_the_next_user_turn() {
    let value = encode_mid_conversation_history(
        CLAUDE_OPUS_4_8,
        vec![
            message::Message::user("Review this code."),
            message::Message::System {
                content: "From now on, require explicit type annotations.".to_string(),
            },
            message::Message::user("Now review this other file."),
        ],
    );

    // The two user turns it separated go as one, as Anthropic reads them.
    assert!(value.get("system").is_none(), "{value}");
    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 2);
    assert_eq!(messages[0]["role"], "user");
    assert_eq!(messages[0]["content"].as_array().map(Vec::len), Some(2));
    assert_eq!(messages[1]["role"], "system");
}

/// The agent-loop placement: a system message between an assistant's
/// `tool_use` and its `tool_result` moves after the tool results, where
/// Anthropic documents it, and a trailing one after an assistant turn (no user
/// turn follows) is still hoisted. Offline for the reason above.
#[test]
fn sonnet_5_5_defers_a_system_message_past_tool_results() {
    let lookup = || message::ToolName::new("lookup").expect("tool name");
    let tool_call = message::Message::Assistant(message::AssistantMessage::new(vec![
        message::AssistantContent::tool_call("toolu_1", lookup(), json!({})),
    ]));
    let tool_result =
        message::Message::tool_result(message::CallId::from_wire("toolu_1"), lookup(), "ok");
    let value = encode_mid_conversation_history(
        CLAUDE_SONNET_5_5,
        vec![
            message::Message::user("Look it up."),
            tool_call,
            message::Message::system("Answer in Spanish."),
            tool_result,
        ],
    );
    assert!(value.get("system").is_none(), "{value}");
    let roles: Vec<_> = value["messages"]
        .as_array()
        .unwrap()
        .iter()
        .map(|message| message["role"].clone())
        .collect();
    assert_eq!(roles, ["user", "assistant", "user", "system"]);

    let value = encode_mid_conversation_history(
        CLAUDE_SONNET_5_5,
        vec![
            message::Message::user("Look it up."),
            message::Message::assistant("Done."),
            message::Message::system("Answer in Spanish."),
        ],
    );
    assert_eq!(value["system"][0]["text"], "Answer in Spanish.");
}

#[test]
fn older_anthropic_models_hoist_mid_conversation_system_message() {
    let request = completion_request_with_history(
        vec![
            message::Message::from("Review this code."),
            message::Message::System {
                content: "From now on, require explicit type annotations.".to_string(),
            },
        ],
        None,
    );

    let request = request_body(Params {
        model: CLAUDE_OPUS_4_7,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert_eq!(
        value["system"][0]["text"],
        "From now on, require explicit type annotations."
    );

    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 1);
    assert_eq!(messages[0]["role"], "user");
}

#[test]
fn test_prompt_caching_skips_final_deferred_tool_in_request() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "regular_tool",
                    "description": "Regular tool",
                    "input_schema": {"type": "object"}
                },
                {
                    "name": "deferred_tool",
                    "description": "Deferred tool",
                    "input_schema": {"type": "object"},
                    "defer_loading": true
                }
            ]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["name"], "regular_tool");
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[1]["name"], "deferred_tool");
    assert!(tools[1].get("cache_control").is_none());
}

#[test]
fn test_prompt_caching_preserves_existing_final_tool_cache_control() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [{
                "name": "cached_tool",
                "description": "Cached tool",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral", "ttl": "1h"}
            }]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
}

#[test]
fn test_prompt_caching_all_deferred_tools_do_not_receive_cache_control() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_deferred_tool",
                    "description": "First deferred tool",
                    "input_schema": {"type": "object"},
                    "defer_loading": true
                },
                {
                    "name": "second_deferred_tool",
                    "description": "Second deferred tool",
                    "input_schema": {"type": "object"},
                    "defer_loading": true
                }
            ]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert!(tools[0].get("cache_control").is_none());
    assert!(tools[1].get("cache_control").is_none());
}

#[test]
fn test_prompt_caching_preserves_earlier_tool_cache_control() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "earlier_tool",
                    "description": "Earlier tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}
                },
                {
                    "name": "later_tool",
                    "description": "Later tool",
                    "input_schema": {"type": "object"}
                }
            ]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_deferred_marker_does_not_suppress_loaded_tool_marker() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "regular_tool",
                    "description": "Regular tool",
                    "input_schema": {"type": "object"}
                },
                {
                    "name": "deferred_cached_tool",
                    "description": "Deferred cached tool",
                    "input_schema": {"type": "object"},
                    "defer_loading": true,
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_errors_when_tool_cache_control_ttl_order_is_invalid() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}
                }
            ]
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("ttl `1h`"));
}

#[test]
fn test_prompt_caching_preserves_valid_mixed_ttl_tool_cache_controls() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral", "ttl": "1h"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    assert!(tools[1]["cache_control"].get("ttl").is_none());
}

#[test]
fn test_prompt_caching_preserves_deferred_tool_cache_control() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [{
                "name": "deferred_cached_tool",
                "description": "Deferred cached tool",
                "input_schema": {"type": "object"},
                "defer_loading": true,
                "cache_control": {"type": "ephemeral"}
            }]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_budget_preserves_three_tool_markers_and_skips_message() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[2]["cache_control"]["type"], "ephemeral");
    assert!(system_has_cache_control(&value));
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_prompt_caching_errors_when_explicit_tool_markers_exceed_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "fourth_cached_tool",
                    "description": "Fourth cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "fifth_cached_tool",
                    "description": "Fifth cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("Too many Anthropic tool"));
}

#[test]
fn test_prompt_caching_errors_when_final_tool_marker_has_no_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "fourth_cached_tool",
                    "description": "Fourth cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "final_uncached_tool",
                    "description": "Final uncached tool",
                    "input_schema": {"type": "object"}
                }
            ]
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("final non-deferred tool"));
}

#[test]
fn test_prompt_caching_replaces_null_final_tool_cache_control() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [{
                "name": "final_tool",
                "description": "Final tool",
                "input_schema": {"type": "object"},
                "cache_control": null
            }]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_ignores_null_tool_cache_control_when_budgeting() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_null_tool",
                    "description": "First null tool",
                    "input_schema": {"type": "object"},
                    "cache_control": null
                },
                {
                    "name": "second_null_tool",
                    "description": "Second null tool",
                    "input_schema": {"type": "object"},
                    "cache_control": null
                },
                {
                    "name": "third_null_tool",
                    "description": "Third null tool",
                    "input_schema": {"type": "object"},
                    "cache_control": null
                },
                {
                    "name": "fourth_null_tool",
                    "description": "Fourth null tool",
                    "input_schema": {"type": "object"},
                    "cache_control": null
                },
                {
                    "name": "final_uncached_tool",
                    "description": "Final uncached tool",
                    "input_schema": {"type": "object"}
                }
            ]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert!(tools[0].get("cache_control").is_none());
    assert!(tools[1].get("cache_control").is_none());
    assert!(tools[2].get("cache_control").is_none());
    assert!(tools[3].get("cache_control").is_none());
    assert_eq!(tools[4]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_preserves_non_null_provider_tool_cache_control_escape_hatch() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [{
                "name": "provider_tool",
                "description": "Provider tool",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "provider_specific"}
            }]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "provider_specific");
}

#[test]
fn test_prompt_caching_automatic_mode_uses_reduced_marker_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: true,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[2]["cache_control"]["type"], "ephemeral");
    assert_eq!(value["cache_control"]["type"], "ephemeral");
    assert!(!system_has_cache_control(&value));
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_prompt_caching_automatic_mode_errors_when_final_tool_marker_has_no_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "final_uncached_tool",
                    "description": "Final uncached tool",
                    "input_schema": {"type": "object"}
                }
            ]
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: true,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("final non-deferred tool"));
}

#[test]
fn test_automatic_caching_errors_when_explicit_tool_markers_exhaust_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "fourth_cached_tool",
                    "description": "Fourth cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: true,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("Too many Anthropic tool"));
}

#[test]
fn test_automatic_caching_1h_errors_with_explicit_five_minute_tool_marker() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "tools": [{
                "name": "cached_tool",
                "description": "Cached tool",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"}
            }]
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: true,
        automatic_caching_ttl: Some(CacheTtl::OneHour),
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("ttl `1h`"));
}

#[test]
fn test_prompt_and_automatic_caching_1h_uses_1h_generated_markers() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: true,
        automatic_caching_ttl: Some(CacheTtl::OneHour),
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    assert_eq!(value["cache_control"]["ttl"], "1h");
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_prompt_and_raw_top_level_automatic_caching_1h_uses_1h_generated_markers() {
    let request = completion_request_with_tools(
        vec![generic_tool("cached_tool")],
        Some(json!({
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "metadata": {"source": "test"}
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: true,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    assert_eq!(value["cache_control"]["ttl"], "1h");
    assert_eq!(value["metadata"]["source"], "test");
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_prompt_caching_uses_raw_top_level_cache_control_ttl() {
    let request = completion_request_with_tools(
        vec![generic_tool("cached_tool")],
        Some(json!({
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "metadata": {"source": "raw-cache-control"}
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    assert_eq!(value["cache_control"]["ttl"], "1h");
    assert_eq!(value["metadata"]["source"], "raw-cache-control");
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_static_prefix_ttl_with_manual_caching_splits_prefix_and_tail() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: Some(CacheTtl::OneHour),
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    // The tail keeps the 5-minute default: a marker with no `ttl` field.
    let tail_cache_control = value["messages"]
        .as_array()
        .and_then(|messages| messages.last())
        .and_then(|message| message["content"].as_array())
        .and_then(|content| content.last())
        .map(|block| &block["cache_control"])
        .unwrap();
    assert_eq!(tail_cache_control["type"], "ephemeral");
    assert!(tail_cache_control.get("ttl").is_none());
    assert!(value.get("cache_control").is_none());
}

#[test]
fn test_static_prefix_ttl_with_automatic_caching_marks_prefix_only() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: true,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: Some(CacheTtl::OneHour),
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    // The moving tail breakpoint is Anthropic's top-level one at the
    // 5-minute default; no explicit message marker exists.
    assert_eq!(value["cache_control"]["type"], "ephemeral");
    assert!(value["cache_control"].get("ttl").is_none());
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_static_prefix_ttl_alone_marks_prefix_without_tail_or_top_level() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: Some(CacheTtl::OneHour),
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["ttl"], "1h");
    assert_eq!(
        value["system"]
            .as_array()
            .and_then(|blocks| blocks.last())
            .and_then(|block| block["cache_control"].get("ttl")),
        Some(&json!("1h"))
    );
    assert!(value.get("cache_control").is_none());
    assert!(!last_message_has_cache_control(&value));
}

#[test]
fn test_static_prefix_ttl_five_minutes_with_automatic_1h_errors_client_side() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let error = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: true,
        automatic_caching_ttl: Some(CacheTtl::OneHour),
        static_prefix_cache_ttl: Some(CacheTtl::FiveMinutes),
    })
    .unwrap_err();

    let message = error.to_string();
    assert!(
        message.contains("with_static_prefix_cache_ttl"),
        "error should name the knob: {message}"
    );
    assert!(
        message.contains("with_automatic_caching_1h"),
        "error should name the conflicting knob: {message}"
    );
}

#[test]
fn test_static_prefix_ttl_five_minutes_matches_automatic_default_ttl() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    // 5m prefix + 5m (default) top-level is uniform, not an inversion.
    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: true,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: Some(CacheTtl::FiveMinutes),
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
    // The explicit knob serializes an explicit `"5m"`, equivalent to the
    // omitted-`ttl` default.
    assert_eq!(tools[0]["cache_control"]["ttl"], "5m");
}

#[test]
fn test_static_prefix_ttl_preserves_marker_budget_arithmetic() {
    // Automatic mode reserves one marker for the top-level breakpoint; the
    // static-prefix knob spends from the same remaining budget as manual
    // caching does — two markers (final tool + system), no more.
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: true,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: Some(CacheTtl::OneHour),
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let marker_count = value["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|tool| !tool["cache_control"].is_null())
        .count()
        + value["system"]
            .as_array()
            .into_iter()
            .flatten()
            .filter(|block| !block["cache_control"].is_null())
            .count()
        + usize::from(!value["cache_control"].is_null());
    assert_eq!(marker_count, 3);
    assert!(marker_count <= MAX_CACHE_CONTROL_MARKERS);
}

#[test]
fn test_raw_top_level_automatic_caching_reduces_marker_budget() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "cache_control": {"type": "ephemeral"},
            "tools": [
                {
                    "name": "first_cached_tool",
                    "description": "First cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "second_cached_tool",
                    "description": "Second cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "third_cached_tool",
                    "description": "Third cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                },
                {
                    "name": "fourth_cached_tool",
                    "description": "Fourth cached tool",
                    "input_schema": {"type": "object"},
                    "cache_control": {"type": "ephemeral"}
                }
            ]
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("Too many Anthropic tool"));
}

#[test]
fn test_raw_top_level_automatic_caching_1h_errors_after_explicit_five_minute_tool_marker() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "cache_control": {"type": "ephemeral", "ttl": "1h"},
            "tools": [{
                "name": "cached_tool",
                "description": "Cached tool",
                "input_schema": {"type": "object"},
                "cache_control": {"type": "ephemeral"}
            }]
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("ttl `1h`"));
}

#[test]
fn test_typed_automatic_caching_ttl_errors_on_conflicting_raw_top_level_ttl() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "cache_control": {"type": "ephemeral"}
        })),
    );

    let err = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        automatic_caching: true,
        automatic_caching_ttl: Some(CacheTtl::OneHour),
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(
        err.to_string()
            .contains("conflicts with the typed automatic caching TTL")
    );
}

#[test]
fn test_prompt_caching_marks_final_tool_in_request() {
    let request = completion_request_with_tools(
        vec![generic_tool("first_tool"), generic_tool("second_tool")],
        None,
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 2);
    assert!(tools[0].get("cache_control").is_none());
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
}

#[test]
fn test_prompt_caching_marks_final_additional_tool_in_request() {
    let request = completion_request_with_tools(
        vec![generic_tool("rig_tool")],
        Some(json!({
            "tools": [{
                "name": "provider_tool",
                "description": "Provider tool",
                "input_schema": {"type": "object"}
            }],
            "metadata": {"source": "test"}
        })),
    );

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools.len(), 2);
    assert!(tools[0].get("cache_control").is_none());
    assert_eq!(tools[1]["name"], "provider_tool");
    assert_eq!(tools[1]["cache_control"]["type"], "ephemeral");
    assert_eq!(value["metadata"]["source"], "test");
}

#[test]
fn test_prompt_caching_without_tools_omits_tools() {
    let request = completion_request_with_tools(Vec::new(), None);

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert!(value.get("tools").is_none());
}

#[test]
fn test_file_id_rig_to_anthropic_conversion() {
    let converted = user_wire(message::UserContent::Document(message::Document {
        data: DocumentSourceKind::FileId("file_abc".to_string()),
        media_type: None,
        additional_params: None,
    }))
    .unwrap()
    .unwrap();
    assert_eq!(converted["role"], "user");
    assert_eq!(
        converted["content"][0],
        json!({"type": "document", "source": {"type": "file", "file_id": "file_abc"}})
    );
}

#[test]
fn test_plaintext_rig_to_anthropic_conversion() {
    let converted = user_wire(message::UserContent::document_text(
        "Some plain text content".to_string(),
        Some(message::DocumentMediaType::TXT),
    ))
    .unwrap()
    .unwrap();
    assert_eq!(
        converted["content"][0]["source"],
        json!({"type": "text", "media_type": "text/plain", "data": "Some plain text content"})
    );
}

/// A block whose provider item is current replays the item verbatim:
/// signed thinking, redacted thinking, a call with its `caller`, cited text.
#[test]
fn current_provider_items_replay_verbatim() {
    let thinking = json!({"type": "thinking", "thinking": "step one", "signature": "sig-1"});
    let redacted = json!({"type": "redacted_thinking", "data": "redacted block"});
    let call = json!({"type": "tool_use", "id": "toolu_1", "name": "add",
        "input": {"x": 1}, "caller": {"type": "direct"}});
    let text = json!({"type": "text", "text": "Two.", "citations": [{"type": "char_location",
        "cited_text": "2", "document_index": 0, "start_char_index": 0, "end_char_index": 1}]});
    let add = message::ToolName::new("add").expect("tool name");
    let wire = assistant_wire(vec![
        message::AssistantContent::reasoning("step one").with_native(thinking.clone()),
        message::AssistantContent::Reasoning(message::Reasoning {
            redacted: true,
            ..message::Reasoning::default()
        })
        .with_native(redacted.clone()),
        message::AssistantContent::tool_call("toolu_1", add, json!({"x": 1}))
            .with_native(call.clone()),
        message::AssistantContent::text("Two.").with_native(text.clone()),
    ]);
    assert_eq!(wire, [thinking, redacted, call, text]);
}

/// A current provider item is sent before any per-kind rule: thinking a
/// dialect (Kimi) sent unsigned goes back to it as the thinking it was,
/// not as text (#1315).
#[test]
fn a_current_unsigned_thinking_item_replays_verbatim() {
    let item = json!({"type": "thinking", "thinking": "thought", "signature": ""});
    let unsigned = message::AssistantContent::reasoning("thought").with_native(item.clone());
    assert_eq!(assistant_wire(vec![unsigned]), [item]);
}

/// pi's rules for blocks with no current item: thinking goes as text,
/// redacted thinking (whose payload lived in its item) is dropped, and an
/// edited block is rebuilt from its fields.
#[test]
fn blocks_without_a_current_item_rebuild_as_pi_does() {
    let unsigned = message::AssistantContent::reasoning("thought");
    let redacted = message::AssistantContent::Reasoning(message::Reasoning {
        redacted: true,
        ..message::Reasoning::default()
    });
    let mut edited = message::AssistantContent::text("before")
        .with_native(json!({"type": "text", "text": "before", "citations": []}));
    if let message::AssistantContent::Text(text) = &mut edited {
        text.text = "after".to_owned();
    }
    let wire = assistant_wire(vec![unsigned, redacted, edited]);
    assert_eq!(
        wire,
        [
            json!({"type": "text", "text": "thought"}),
            json!({"type": "text", "text": "after"})
        ]
    );
}

/// An opaque provider item that survived history adaptation goes back as
/// it came.
#[test]
fn opaque_items_replay_verbatim() {
    let item = json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
        "input": {"query": "rust"}});
    let wire = assistant_wire(vec![message::AssistantContent::Opaque(message::Opaque {
        item: item.clone(),
        replay: true,
    })]);
    assert_eq!(wire, [item]);
}

/// An empty-content reply with the given terminal.
fn empty_reply(stop_reason: &str, stop_sequence: Option<&str>) -> Value {
    json!({
        "content": [], "id": "msg_123", "model": CLAUDE_SONNET_4_6, "role": "assistant",
        "stop_reason": stop_reason, "stop_sequence": stop_sequence,
        "usage": {"input_tokens": 7, "output_tokens": 2}
    })
}

#[test]
fn empty_end_turn_response_normalizes_to_an_empty_choice() {
    let parsed = fold_reply(&empty_reply("end_turn", None)).expect("empty end_turn folds");

    // Anthropic's documented empty `end_turn` is a turn that carried
    // nothing: the empty list is the turn, said honestly.
    assert!(parsed.choice.is_empty());
    assert_eq!(parsed.provider(), "anthropic");
    assert_eq!(parsed.response_id(), Some("msg_123"));
    assert_eq!(parsed.model(), Some(CLAUDE_SONNET_4_6));
    assert_eq!(parsed.finish_reason(), Some(completion::FinishReason::Stop));
}

#[test]
fn empty_stop_sequence_response_naming_its_sequence_is_a_completed_turn() {
    let parsed = fold_reply(&empty_reply("stop_sequence", Some("alpha")))
        .expect("a completed stop-sequence turn must not fold into an error");

    assert!(parsed.choice.is_empty());
    assert_eq!(parsed.finish_reason(), Some(completion::FinishReason::Stop));
}

#[test]
fn end_turn_with_a_tool_call_is_reconciled_to_tool_calls() {
    // Anthropic reports `tool_use`, but the reconciliation the response
    // builder applies must hold for any provider that reports a plain stop
    // alongside a tool call.
    let mut reply = empty_reply("end_turn", None);
    reply["content"] =
        json!([{"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"x": 1}}]);
    let parsed = fold_reply(&reply).expect("tool-use response should fold");
    assert_eq!(
        parsed.finish_reason(),
        Some(completion::FinishReason::ToolCalls)
    );
}

// -------------------------------------------------------------------
// Citations (#1767)
// -------------------------------------------------------------------

#[test]
fn document_serializes_citations_and_metadata() {
    let value = user_wire(message::UserContent::Document(message::Document {
        data: DocumentSourceKind::String("hello".into()),
        media_type: Some(DocumentMediaType::TXT),
        additional_params: Some(json!({"title": "My Doc", "citations": {"enabled": true}})),
    }))
    .unwrap()
    .unwrap();
    let document = &value["content"][0];
    assert_eq!(document["citations"]["enabled"], true);
    assert_eq!(document["title"], "My Doc");
    assert!(
        document.get("context").is_none(),
        "context should be skipped when None"
    );
}

/// Server tools and their results, MCP blocks and whatever Anthropic adds
/// next are opaque blocks that replay; none fails the reply, and each goes
/// back to the model verbatim beside the cited answer.
#[test]
fn hosted_tool_reply_decodes_to_opaque_blocks_that_replay_verbatim() {
    let content = vec![
        json!({"type": "server_tool_use", "id": "srvtoolu_01", "name": "web_search",
            "input": {"query": "claude shannon birth date"}, "caller": {"type": "direct"}}),
        json!({"type": "web_search_tool_result", "tool_use_id": "srvtoolu_01", "content": [{
            "type": "web_search_result", "url": "https://example.com/shannon",
            "title": "Claude Shannon", "encrypted_content": "encrypted-content",
            "page_age": "April 30, 2025"}]}),
        json!({"type": "web_search_tool_result", "tool_use_id": "srvtoolu_02", "content": {
            "type": "web_search_tool_result_error", "error_code": "max_uses_exceeded"}}),
        json!({"type": "web_fetch_tool_result", "tool_use_id": "srvtoolu_03", "content": {
            "type": "web_fetch_result", "url": "https://example.com"}}),
        json!({"type": "code_execution_tool_result", "tool_use_id": "srvtoolu_04", "content": {
            "type": "encrypted_code_execution_result", "return_code": 1, "stderr": "failure",
            "encrypted_stdout": "encrypted-output", "content": []}}),
        json!({"type": "bash_code_execution_tool_result", "tool_use_id": "srvtoolu_05",
            "content": {"type": "bash_code_execution_result", "stdout": "ok", "stderr": "",
                "return_code": 0, "content": []}}),
        json!({"type": "text_editor_code_execution_tool_result", "tool_use_id": "srvtoolu_06",
            "content": {"type": "text_editor_code_execution_view_result", "content": "x"}}),
        json!({"type": "tool_search_tool_result", "tool_use_id": "srvtoolu_07",
            "content": {"type": "tool_search_tool_search_result", "tool_references": []}}),
        json!({"type": "mcp_tool_use", "id": "mcptoolu_1", "name": "fetch",
            "server_name": "docs", "input": {}}),
        json!({"type": "mcp_tool_result", "tool_use_id": "mcptoolu_1", "is_error": false,
            "content": [{"type": "text", "text": "fetched"}]}),
        json!({"type": "container_upload", "file_id": "file_1"}),
        json!({"type": "compaction", "content": "Summary so far."}),
        json!({"type": "text", "text": "Claude Shannon was born on April 30, 1916.",
            "citations": [{"type": "web_search_result_location",
                "cited_text": "Claude Shannon was born on April 30, 1916.",
                "url": "https://example.com/shannon", "title": "Claude Shannon",
                "encrypted_index": "encrypted-index"}]}),
    ];
    let value = json!({
        "id": "msg_web_search", "model": CLAUDE_SONNET_4_6, "role": "assistant",
        "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 10, "output_tokens": 20},
        "content": content,
    });
    let converted = fold_reply(&value).expect("the hosted-tool reply folds");
    assert_eq!(converted.choice.len(), content.len());
    let (answer, hosted) = converted.choice.split_last().expect("blocks");
    assert!(hosted.iter().all(|block| matches!(
        block,
        message::AssistantContent::Opaque(message::Opaque { replay: true, .. })
    )));
    let message::AssistantContent::Text(answer_text) = answer else {
        panic!("expected the text answer last, got {answer:?}");
    };
    assert_eq!(
        answer_text.text,
        "Claude Shannon was born on April 30, 1916."
    );
    assert_eq!(
        answer
            .native_item()
            .and_then(|item| item.pointer("/citations/0/encrypted_index")),
        Some(&json!("encrypted-index"))
    );

    assert_eq!(assistant_wire(converted.choice), content);
}

#[test]
fn document_additional_params_forward_to_anthropic_document() {
    let converted = user_wire(message::UserContent::Document(message::Document {
        data: message::DocumentSourceKind::String("Hello world.".into()),
        media_type: Some(message::DocumentMediaType::TXT),
        additional_params: Some(json!({
            "title": "Doc1",
            "context": "ctx",
            "citations": { "enabled": true }
        })),
    }))
    .unwrap()
    .unwrap();
    let document = &converted["content"][0];
    assert_eq!(document["title"], "Doc1");
    assert_eq!(document["context"], "ctx");
    assert_eq!(document["citations"], json!({"enabled": true}));
}

#[tokio::test]
async fn completion_http_non_success_preserves_status_and_body() {
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"slow down"}}"#;
    let http = RecordingHttpClient::with_error_response(http::StatusCode::TOO_MANY_REQUESTS, body);
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);

    let error = crate::driver::Model::new(wire.clone(), http.clone())
        .call(hello_request())
        .await
        .expect_err("completion should fail with non-success status");

    // rig#2314: a provider with a request-id contract preserves its
    // non-success responses as ProviderResponse, so the transport id has
    // a home on the error; this mock sent no header, so the id is None.
    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(error.provider_request_id(), None);
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::TOO_MANY_REQUESTS)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn completion_2xx_error_envelope_preserves_status_and_body() {
    use crate::test_utils::RecordingHttpClient;

    // Anthropic answers an in-band failure with its top-level error
    // envelope, and it can arrive under a 200: the decoder models `error`
    // as an event of the Messages wire, so the envelope routes through the
    // same `ProviderResponse` funnel as a rejected status rather than
    // reading as a corrupt frame.
    //
    // The body is asserted byte-for-byte, in the key order Anthropic's
    // recorded replies use, and carries the top-level `request_id` the
    // envelope type does not model: a preserved reply that Rig rebuilt from
    // the fields it happens to parse is not the provider's reply. The
    // status is the driver's — a success at the transport layer is still a
    // status the caller must see.
    let body = r#"{"error":{"message":"model overloaded","type":"overloaded_error"},"request_id":"req_011CXYZ","type":"error"}"#;
    let http = RecordingHttpClient::new(body); // 200 OK
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);

    let error = crate::driver::Model::new(wire.clone(), http.clone())
        .call(hello_request())
        .await
        .expect_err("completion should fail with provider error envelope");

    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(error.provider_response_status(), Some(http::StatusCode::OK));
}

#[tokio::test]
async fn completion_streaming_http_non_success_preserves_status_and_body() {
    use crate::test_utils::HttpErrorStreamingClient;
    use futures::StreamExt;

    let body = r#"{"type":"error","error":{"type":"overloaded_error","message":"slow down"}}"#;
    let http = HttpErrorStreamingClient::new(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);

    let stream = crate::driver::tests::stream(&wire, &http, hello_request(), None)
        .expect("the streamed request encodes");
    let mut stream = Box::pin(stream);

    // The transport failure surfaces as the first error item yielded by the stream.
    let error = loop {
        match stream.next().await {
            Some(Ok(_)) => continue,
            Some(Err(error)) => break error,
            None => panic!("stream ended without yielding the transport error"),
        }
    };

    // A rejected SSE handshake is the provider's reply, classified like the
    // unary driver's and the in-band envelopes': one funnel.
    assert!(matches!(error, ProviderError::ProviderResponse(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));

    // The transport failure ends the stream: nothing may follow it that
    // would read as a successfully completed turn.
    assert!(stream.next().await.is_none());
}

// Regression test for issue #1429: PR #1431 added the `DocumentSource::Url`
// wire variant and response-side parsing, but the request-side
// `UserContent::Document` conversion still rejected URL-backed PDFs even
// though the Anthropic Messages API supports
// `"source": {"type": "url", ...}` for PDFs.
// The media type is optional because Anthropic's URL source is implicitly a
// PDF and does not include a media-type field on the wire.
//
// See <https://docs.anthropic.com/en/docs/build-with-claude/pdf-support>
// for URL-sourced PDF documents.
#[test]
fn url_pdf_with_or_without_media_type_converts_to_url_document_source() {
    let pdf_url = "https://example.com/resume.pdf";

    for media_type in [Some(message::DocumentMediaType::PDF), None] {
        let msg = message::Message::User {
            content: vec![message::UserContent::document_url(pdf_url, media_type)],
        };

        let converted = convert(msg)
            .expect("URL PDF should convert")
            .expect("a block survives");
        let json = serde_json::to_value(&converted).expect("message should serialize");

        assert_eq!(
            json.pointer("/content/0/source"),
            Some(&json!({ "type": "url", "url": pdf_url })),
            "URL PDF should map to a url document source: {json:#}"
        );
    }
}

/// Raw-capture tests: `CompletionResponse::raw` driven end to end over a
/// mock transport that hands back a Messages body *and* a `request-id`
/// response header. `raw` is the verbatim reply document — not a
/// re-serialization of the parsed type — so it answers what the normalized
/// response does not, and the transport id the driver read off the headers
/// reaches the normalized response rather than the document.
/// `with_error_response_headers` with `200 OK` is the one unary double
/// that carries response headers.
mod raw_capture {
    use super::*;
    use crate::test_utils::RecordingHttpClient;

    const REQUEST_ID: &str = "req_unit_anthropic_0001";

    /// A Messages body whose `stop_sequence` is set: the normalized
    /// response maps it to `FinishReason::Stop` and drops which sequence
    /// fired, so the capture provably answers more than the fold does.
    const BODY: &str = r#"{
            "id": "msg_raw_1",
            "type": "message",
            "role": "assistant",
            "model": "claude-sonnet-4-6",
            "content": [{"type": "text", "text": "hello"}],
            "stop_reason": "stop_sequence",
            "stop_sequence": "alpha",
            "usage": {"input_tokens": 7, "output_tokens": 2}
        }"#;

    fn http() -> RecordingHttpClient {
        let mut headers = http::HeaderMap::new();
        headers.insert("request-id", http::HeaderValue::from_static(REQUEST_ID));
        RecordingHttpClient::with_error_response_headers(http::StatusCode::OK, BODY, headers)
    }

    /// The load-bearing capture property: `raw` is the reply Anthropic
    /// sent, verbatim — it still carries the `type` tag the wire type does
    /// not model, which is how you can tell it is the document and not a
    /// projection of it — it deserializes into Anthropic's own
    /// `CompletionResponse`, and it answers `stop_sequence`, which the
    /// normalized response drops.
    #[tokio::test]
    async fn completion_captures_the_verbatim_reply_as_raw() {
        let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
        let response = crate::driver::Model::new(wire.clone(), http())
            .call(hello_request())
            .await
            .expect("completion");

        let raw = &response.raw;
        assert_eq!(
            raw["type"], "message",
            "raw must be the verbatim reply, tag included"
        );
        assert_eq!(raw["id"], "msg_raw_1");
        assert_eq!(raw["model"], "claude-sonnet-4-6");
        assert_eq!(raw["stop_reason"], "stop_sequence");
        assert_eq!(raw["stop_sequence"], "alpha");
        assert_eq!(raw["usage"], json!({"input_tokens": 7, "output_tokens": 2}));

        // The transport id is not part of any reply document; the driver
        // read it off the `request-id` header and stamped the normalized
        // response with it.
        assert!(
            raw.get("provider_request_id").is_none(),
            "the document carries no transport id"
        );
        assert_eq!(response.provider_request_id.as_deref(), Some(REQUEST_ID));

        // The normalized response reports the reason and drops which
        // sequence fired — the whole reason `raw` is worth capturing.
        assert_eq!(
            response.finish_reason(),
            Some(completion::FinishReason::Stop)
        );
        assert_eq!(response.model(), Some("claude-sonnet-4-6"));
    }
}

/// A call rig issued an id for is spelled with one request-local alias on
/// the call and on its result.
#[test]
fn a_rig_issued_call_id_is_spelled_alike_on_call_and_result() {
    let call = message::ToolCall::new(
        message::CallId::from_wire(""),
        message::ToolFunction::new(message::ToolName::new("add").expect("tool name"), json!({})),
    );
    let request = completion_request_with_history(
        vec![
            message::Message::user("Add."),
            message::Message::Assistant(message::AssistantMessage::new(vec![
                message::AssistantContent::ToolCall(call.clone()),
            ])),
            message::Message::tool_results(vec![
                call.result(vec![message::ToolResultContent::text("3")]),
            ]),
        ],
        None,
    );
    let wire = request_body(Params {
        model: CLAUDE_SONNET_4_6,
        request,
        prompt_caching: false,
        automatic_caching: false,
        automatic_caching_ttl: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();
    let value = serde_json::to_value(wire).unwrap();
    assert_eq!(value["messages"][1]["content"][0]["id"], "tool-0");
    assert_eq!(value["messages"][2]["content"][0]["tool_use_id"], "tool-0");
}

/// pi's rule, which Z.AI needs: consecutive user messages of tool results
/// go as one user message, which `adapt` makes of them.
#[test]
fn consecutive_tool_results_merge_into_one_user_message() {
    let add = || message::ToolName::new("add").expect("tool name");
    let value = prepared_body(vec![
        message::Message::user("Add twice."),
        message::Message::Assistant(message::AssistantMessage::new(vec![
            message::AssistantContent::tool_call("toolu_1", add(), json!({})),
            message::AssistantContent::tool_call("toolu_2", add(), json!({})),
        ])),
        message::Message::tool_result(message::CallId::from_wire("toolu_1"), add(), "1"),
        message::Message::tool_result(message::CallId::from_wire("toolu_2"), add(), "2"),
    ]);
    let messages = value["messages"].as_array().unwrap();
    assert_eq!(messages.len(), 3);
    assert_eq!(messages[2]["content"][0]["tool_use_id"], "toolu_1");
    assert_eq!(messages[2]["content"][1]["tool_use_id"], "toolu_2");
}

/// A `tool_use` whose `input` is not an object keeps its call, with the
/// arguments normalized, and no provider item: the item could not be sent
/// back as it is.
#[test]
fn a_tool_use_whose_input_is_not_an_object_keeps_its_call() {
    let value = json!({
        "id": "msg_bad", "model": CLAUDE_SONNET_4_6, "role": "assistant",
        "stop_reason": "tool_use", "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [{"type": "tool_use", "id": "toolu_1", "name": "lookup", "input": [1]}]
    });
    let response = fold_reply(&value).expect("the reply folds");
    let [message::AssistantContent::ToolCall(call)] = response.choice.as_slice() else {
        panic!("one call: {:?}", response.choice);
    };
    assert_eq!(call.function.arguments_value(), json!({}));
    assert_eq!(call.function.invalid_arguments.as_deref(), Some("[1]"));
    assert!(response.choice[0].native_item().is_none());
}

/// A failed tool result is sent with `is_error`; a successful one leaves
/// the field to its default.
#[test]
fn a_failed_tool_result_is_sent_as_an_error() {
    let name = message::ToolName::new("lookup").expect("tool name");
    let results = [false, true].map(|is_error| {
        message::UserContent::ToolResult(message::ToolResult {
            call: message::CallId::from_wire("toolu_1"),
            name: name.clone(),
            content: vec![message::ToolResultContent::text("boom")],
            is_error,
        })
    });
    let converted = convert(message::Message::User {
        content: results.to_vec(),
    })
    .expect("the results convert")
    .expect("a message");
    let value = serde_json::to_value(&converted).expect("the message serializes");
    assert!(value["content"][0].get("is_error").is_none());
    assert_eq!(value["content"][1]["is_error"], json!(true));
}

/// A hand-built assistant image, which `adapt` leaves on a same-model
/// turn, is sent as its placeholder: assistant turns take no images.
#[test]
fn an_assistant_image_is_sent_as_its_placeholder() {
    let image = message::AssistantContent::Image(message::Image {
        data: message::DocumentSourceKind::base64("aGk="),
        media_type: Some(message::ImageMediaType::PNG),
        ..message::Image::default()
    });
    assert_eq!(
        assistant_wire(vec![image]),
        [json!({"type": "text", "text": crate::completion::history::ASSISTANT_IMAGE_OMITTED})]
    );
}

/// The container the last same-model turn ran in is the next request's,
/// unless the request names its own; another model's turn names none.
#[test]
fn the_last_turn_container_is_replayed_unless_the_request_names_one() {
    use crate::wire::{Operation, Wire};

    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let turn = |model: &str, container: &str| {
        message::Message::Assistant(message::AssistantMessage {
            content: vec![
                message::AssistantContent::text("ran it"),
                message::AssistantContent::Opaque(message::Opaque {
                    item: json!({
                        "type": "container",
                        "container": {"id": container, "expires_at": "2026-10-02T00:00:00Z"},
                    }),
                    replay: true,
                }),
            ],
            origin: Some(message::Origin::new(
                "anthropic.messages",
                "anthropic",
                model,
            )),
            stop: Some(message::StopReason::Stop),
        })
    };
    let body = |history: Vec<message::Message>, params: Option<serde_json::Value>| {
        let mut request = completion_request_with_history(history, None);
        request.additional_params = params;
        let request = crate::operation::Completion::prepare(request, &wire.describe())
            .expect("the request prepares");
        json_body(
            &wire
                .encode(request, crate::wire::Mode::Unary)
                .expect("the request encodes")
                .request,
        )
    };
    let history = vec![
        message::Message::user("run it"),
        turn(CLAUDE_SONNET_4_6, "container_old"),
        message::Message::user("again"),
        turn(CLAUDE_SONNET_4_6, "container_new"),
        message::Message::user("next"),
    ];
    assert_eq!(
        body(history.clone(), None)["container"],
        json!("container_new")
    );
    assert_eq!(
        body(history, Some(json!({"container": "container_mine"})))["container"],
        json!("container_mine")
    );
    let foreign = vec![
        message::Message::user("run it"),
        turn(CLAUDE_OPUS_4_8, "container_other"),
        message::Message::user("next"),
    ];
    assert!(body(foreign, None).get("container").is_none());
}

/// A tool-result image takes the sources a user image does, a URL among
/// them, rather than being refused.
#[test]
fn a_tool_result_image_by_url_is_sent_by_url() {
    let result = message::UserContent::ToolResult(message::ToolResult {
        call: message::CallId::from_wire("toolu_1"),
        name: message::ToolName::new("screenshot").expect("tool name"),
        content: vec![message::ToolResultContent::Image(message::Image {
            data: message::DocumentSourceKind::Url("https://example.com/shot.png".to_owned()),
            ..message::Image::default()
        })],
        is_error: false,
    });
    let converted = convert(message::Message::User {
        content: vec![result],
    })
    .expect("the result converts")
    .expect("a message");
    let value = serde_json::to_value(&converted).expect("the message serializes");
    assert_eq!(
        value["content"][0]["content"][0]["source"],
        json!({"type": "url", "url": "https://example.com/shot.png"})
    );
}

/// `encodes` is true exactly for the media forms the encoder converts:
/// images by typed base64, URL or file id, in a user turn or a tool result,
/// and documents by file id, PDF data or URL, or text. Audio, video and
/// assistant images are never carried.
#[test]
fn encodes_states_exactly_the_media_the_encoder_carries() {
    use crate::completion::{Media, Place, ReplayTarget};
    use message::{
        Audio, AudioMediaType, Document, DocumentMediaType as Doc, DocumentSourceKind as Source,
        Image, ImageMediaType, Video, VideoMediaType,
    };

    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let image = |data: Source, media_type: Option<ImageMediaType>| Image {
        data,
        media_type,
        ..Image::default()
    };
    let document = |data: Source, media_type: Option<Doc>| Document {
        data,
        media_type,
        additional_params: None,
    };
    let images = [
        (
            image(Source::base64("aGk="), Some(ImageMediaType::PNG)),
            true,
        ),
        (image(Source::base64("aGk="), None), false),
        (
            image(Source::base64("aGk="), Some(ImageMediaType::HEIC)),
            false,
        ),
        (image(Source::url("https://example.com/a.png"), None), true),
        (image(Source::file_id("file_image"), None), true),
        (image(Source::string("not an image"), None), false),
    ];
    for (image, carried) in images {
        let in_result = message::UserContent::ToolResult(message::ToolResult {
            call: message::CallId::from_wire("toolu_1"),
            name: message::ToolName::new("shot").expect("tool name"),
            content: vec![message::ToolResultContent::Image(image.clone())],
            is_error: false,
        });
        for (place, part) in [
            (Place::User, message::UserContent::Image(image.clone())),
            (Place::ToolResult, in_result),
        ] {
            assert_eq!(
                wire.encodes(CLAUDE_SONNET_4_6, Media::Image(&image, place)),
                carried,
                "{image:?} at {place:?}"
            );
            assert_eq!(user_wire(part).is_ok(), carried, "{image:?} at {place:?}");
        }
        assert!(!wire.encodes(CLAUDE_SONNET_4_6, Media::Image(&image, Place::Assistant)));
    }
    let documents = [
        (document(Source::base64("JVBERi0="), Some(Doc::PDF)), true),
        (
            document(Source::url("https://example.com/a.pdf"), Some(Doc::PDF)),
            true,
        ),
        (
            document(Source::url("https://example.com/a.pdf"), None),
            true,
        ),
        (
            document(Source::url("https://example.com/a.txt"), Some(Doc::TXT)),
            false,
        ),
        (document(Source::file_id("file_doc"), None), true),
        (document(Source::string("plain"), Some(Doc::TXT)), true),
        (document(Source::string("plain"), None), true),
        (document(Source::base64("cGxhaW4="), Some(Doc::TXT)), true),
        (
            document(Source::base64("cmlnLG1hdHJpeAo="), Some(Doc::CSV)),
            true,
        ),
        (document(Source::string("<p>hi</p>"), Some(Doc::HTML)), true),
        (document(Source::base64("//79"), Some(Doc::CSV)), false),
        (document(Source::base64("cGxhaW4="), None), false),
    ];
    for (document, carried) in documents {
        assert_eq!(
            wire.encodes(CLAUDE_SONNET_4_6, Media::Document(&document)),
            carried,
            "{document:?}"
        );
        assert_eq!(
            user_wire(message::UserContent::Document(document.clone())).is_ok(),
            carried,
            "{document:?}"
        );
    }
    let audio = Audio {
        data: Source::base64("SUQz"),
        media_type: Some(AudioMediaType::MP3),
    };
    let video = Video {
        data: Source::url("https://example.com/a.mp4"),
        media_type: Some(VideoMediaType::MP4),
        additional_params: None,
    };
    assert!(!wire.encodes(CLAUDE_SONNET_4_6, Media::Audio(&audio)));
    assert!(!wire.encodes(CLAUDE_SONNET_4_6, Media::Video(&video)));
    assert!(user_wire(message::UserContent::Audio(audio)).is_err());
    assert!(user_wire(message::UserContent::Video(video)).is_err());
}

/// A text-family document is sent as a text source, its base64 data
/// decoded, so a CSV or a base64 plain-text file reaches the model as the
/// text it holds.
#[test]
fn a_text_family_document_is_sent_as_its_text() {
    use message::{DocumentMediaType as Doc, DocumentSourceKind as Source};

    for (data, media_type, text) in [
        (
            Source::base64("cmlnLG1hdHJpeAoxLDIK"),
            Some(Doc::CSV),
            "rig,matrix\n1,2\n",
        ),
        (
            Source::base64("cGxhaW4gdGV4dA=="),
            Some(Doc::TXT),
            "plain text",
        ),
        (Source::string("# notes"), Some(Doc::MARKDOWN), "# notes"),
        (Source::string("untyped"), None, "untyped"),
    ] {
        let converted = user_wire(message::UserContent::Document(message::Document {
            data,
            media_type,
            additional_params: None,
        }))
        .expect("the document converts")
        .expect("a block");
        let value = serde_json::to_value(&converted).expect("the message serializes");
        assert_eq!(
            value["content"][0]["source"],
            json!({"type": "text", "media_type": "text/plain", "data": text})
        );
    }
}

/// An image by Files API id is sent as a file source, in a user turn and
/// in a tool result.
#[test]
fn an_image_by_file_id_is_sent_as_a_file_source() {
    let image = message::Image {
        data: message::DocumentSourceKind::file_id("file_image"),
        ..message::Image::default()
    };
    let converted = convert(message::Message::User {
        content: vec![
            message::UserContent::ToolResult(message::ToolResult {
                call: message::CallId::from_wire("toolu_1"),
                name: message::ToolName::new("shot").expect("tool name"),
                content: vec![message::ToolResultContent::Image(image.clone())],
                is_error: false,
            }),
            message::UserContent::Image(image),
        ],
    })
    .expect("the images convert")
    .expect("a message");
    let value = serde_json::to_value(&converted).expect("the message serializes");
    let source = json!({"type": "file", "file_id": "file_image"});
    assert_eq!(value["content"][0]["content"][0]["source"], source);
    assert_eq!(value["content"][1]["source"], source);
}

/// The request body for `history` on Sonnet 4.6, prepared as a run sends it.
fn prepared_body(history: Vec<message::Message>) -> serde_json::Value {
    use crate::wire::{Operation, Wire};

    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let mut request = completion_request_with_history(history, None);
    request.tools = vec![generic_tool("add")];
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    json_body(
        &wire
            .encode(request, crate::wire::Mode::Unary)
            .expect("the request encodes")
            .request,
    )
}

/// A same-model turn that ran in a container and called a tool from it:
/// a leading `fallback` marker, which never replays, then the call.
fn container_turn() -> message::AssistantMessage {
    let reply = json!({
        "id": "msg_1", "model": CLAUDE_SONNET_4_6, "role": "assistant",
        "stop_reason": "tool_use", "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "container": {"id": "container_1", "expires_at": "2026-10-03T00:00:00Z"},
        "content": [
            {"type": "fallback", "model": CLAUDE_SONNET_4_6},
            {"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"x": 1},
                "caller": {"type": "code_execution_20250825", "tool_id": "srvtoolu_1"}}
        ]
    });
    let Some(message::Message::Assistant(turn)) =
        fold_reply(&reply).expect("the reply folds").message()
    else {
        panic!("an assistant turn");
    };
    turn
}

/// The history that answers `turn`'s call.
fn answered(turn: message::AssistantMessage) -> Vec<message::Message> {
    vec![
        message::Message::user("add"),
        message::Message::Assistant(turn),
        message::Message::tool_result(
            message::CallId::from_wire("toolu_1"),
            message::ToolName::new("add").expect("tool name"),
            "2",
        ),
    ]
}

/// The container is conversation state: a same-model turn names it after
/// `adapt` drops its `fallback` block, and after its call is edited.
#[test]
fn the_container_survives_a_dropped_block_and_an_edited_call() {
    let body = prepared_body(answered(container_turn()));
    assert_eq!(body["container"], json!("container_1"));

    let mut edited = container_turn();
    for block in &mut edited.content {
        if let message::AssistantContent::ToolCall(call) = block {
            call.function.arguments = json!({"x": 5}).as_object().cloned().unwrap_or_default();
        }
    }
    let body = prepared_body(answered(edited));
    assert_eq!(body["container"], json!("container_1"));
}

/// A same-model call rebuilt after an edit keeps the `caller` that ties it
/// to the code execution that made it.
#[test]
fn an_edited_call_keeps_its_caller() {
    let mut turn = container_turn();
    for block in &mut turn.content {
        if let message::AssistantContent::ToolCall(call) = block {
            call.function.arguments = json!({"x": 5}).as_object().cloned().unwrap_or_default();
        }
    }
    let body = prepared_body(answered(turn));
    assert_eq!(
        body["messages"][1]["content"][0],
        json!({"type": "tool_use", "id": "toolu_1", "name": "add", "input": {"x": 5},
            "caller": {"type": "code_execution_20250825", "tool_id": "srvtoolu_1"}})
    );
}

/// Text the adapter merges into a user message ahead of its tool results
/// is sent after them: Anthropic requires results first.
#[test]
fn tool_results_lead_a_merged_user_message() {
    let call = message::AssistantContent::tool_call(
        "toolu_1",
        message::ToolName::new("add").expect("tool name"),
        json!({"x": 1}),
    );
    let body = prepared_body(vec![
        message::Message::user("add"),
        message::Message::Assistant(message::AssistantMessage::new(vec![call])),
        message::Message::user("also this"),
        message::Message::User {
            content: vec![
                message::UserContent::text("and this"),
                message::UserContent::tool_result(
                    message::CallId::from_wire("toolu_1"),
                    message::ToolName::new("add").expect("tool name"),
                    vec![message::ToolResultContent::text("2")],
                ),
            ],
        },
    ]);
    let content = &body["messages"][2]["content"];
    assert_eq!(content[0]["type"], json!("tool_result"), "{body:#}");
    assert_eq!(content[0]["tool_use_id"], json!("toolu_1"));
    assert_eq!(content[1]["text"], json!("also this"));
    assert_eq!(content[2]["text"], json!("and this"));
}

/// The turn `reply` folds into on `wire`, for `request`.
fn folded_turn(
    wire: &Messages,
    request: &CompletionRequest,
    reply: serde_json::Value,
) -> message::AssistantMessage {
    let mut reply = reply;
    if let Some(map) = reply.as_object_mut() {
        map.entry("type").or_insert_with(|| json!("message"));
        map.entry("id").or_insert_with(|| json!("msg_1"));
        map.entry("usage")
            .or_insert_with(|| json!({"input_tokens": 1, "output_tokens": 1}));
    }
    let response = crate::test_utils::decode_reply(
        wire,
        request,
        crate::wire::Mode::Unary,
        [WireFrame::Text(reply.to_string())],
        reply,
    )
    .expect("the reply folds");
    let Some(message::Message::Assistant(turn)) = response.message() else {
        panic!("an assistant turn: {:?}", response.choice);
    };
    turn
}

/// The body `history` sends on `wire` with `tools` declared, prepared as a
/// run sends it.
fn sent_with(
    wire: &Messages,
    history: Vec<message::Message>,
    tools: Vec<completion::ToolDefinition>,
) -> serde_json::Value {
    use crate::wire::{Operation, Wire};
    let mut request = completion_request_with_history(history, None);
    request.tools = tools;
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .expect("the request prepares");
    json_body(
        &wire
            .encode(request, crate::wire::Mode::Unary)
            .expect("the request encodes")
            .request,
    )
}

/// #2655: a `tool_use` the provider sent without an id gets an id rig
/// issues, and the replayed item carries the spelling its result gets, so
/// the call and its result always agree, on every dialect.
#[test]
fn an_idless_call_replays_under_the_id_its_result_gets() {
    for dialect in [&super::super::wire::ANTHROPIC, &super::super::wire::MINIMAX] {
        let wire = AnthropicConfig::with_key(dialect, "k").completion(CLAUDE_SONNET_4_6);
        for call in [
            json!({"type": "tool_use", "id": "", "name": "add", "input": {"x": 1}}),
            json!({"type": "tool_use", "name": "add", "input": {"x": 1}}),
        ] {
            let turn = folded_turn(
                &wire,
                &hello_request(),
                json!({"model": CLAUDE_SONNET_4_6, "stop_reason": "tool_use", "content": [call]}),
            );
            let id = turn.tool_calls().next().expect("a call").id.clone();
            assert!(id.is_local(), "{id:?}");
            let body = sent_with(
                &wire,
                vec![
                    message::Message::user("add"),
                    message::Message::Assistant(turn),
                    message::Message::tool_result(
                        id,
                        message::ToolName::new("add").expect("name"),
                        "2",
                    ),
                ],
                vec![generic_tool("add")],
            );
            let use_id = &body["messages"][1]["content"][0]["id"];
            assert_eq!(
                use_id, &body["messages"][2]["content"][0]["tool_use_id"],
                "{body}"
            );
            assert_eq!(use_id, "tool-0", "{body}");
        }
    }
}

/// #2703: Claude Opus 5.5 binds its thinking to the tools and system prompt
/// it was made under. Its turns replay verbatim whatever the tools, and
/// every request asks Anthropic, under the binding beta, to drop a block
/// whose binding no longer matches (pi's `drop_block`). A model that does
/// not bind gets neither.
#[test]
fn a_binding_model_replays_its_thinking_and_asks_for_drop_block() {
    use crate::wire::{Operation, Wire};
    let thinking = json!({"type": "thinking", "thinking": "plan", "signature": "sig_opus"});
    let drop_block = json!({"prefix_mismatch_behavior": "drop_block"});
    for (model, binds) in [(CLAUDE_OPUS_5_5, true), (CLAUDE_SONNET_4_6, false)] {
        let wire = AnthropicConfig::new("test-key").completion(model);
        let made = hello_request().tools(vec![generic_tool("add")]);
        let turn = folded_turn(
            &wire,
            &made,
            json!({"model": model, "stop_reason": "end_turn", "content": [
                thinking.clone(), {"type": "text", "text": "done"}]}),
        );
        let history = vec![
            message::Message::user("go"),
            message::Message::Assistant(turn),
            message::Message::user("next"),
        ];
        let changed = sent_with(
            &wire,
            history.clone(),
            vec![generic_tool("add"), generic_tool("mul")],
        );
        assert_eq!(
            changed["messages"][1]["content"][0], thinking,
            "{model}: {changed}"
        );
        let expected = binds.then(|| json!({"type": "adaptive", "block_binding": drop_block}));
        assert_eq!(
            changed.get("thinking"),
            expected.as_ref(),
            "{model}: {changed}"
        );

        let mut request = completion_request_with_history(history, None);
        request.tools = vec![generic_tool("add")];
        let request = crate::operation::Completion::prepare(request, &wire.describe())
            .expect("the request prepares");
        let encoded = wire
            .encode(request, crate::wire::Mode::Unary)
            .expect("the request encodes");
        let beta = encoded
            .request
            .headers()
            .get("anthropic-beta")
            .and_then(|value| value.to_str().ok());
        assert_eq!(
            beta,
            binds.then_some("thinking-binding-controls-2026-08-01")
        );
    }
}

/// The caller's thinking settings stay, with `drop_block` added; thinking
/// the caller disabled stays disabled.
#[test]
fn drop_block_joins_the_callers_thinking() {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_OPUS_5_5);
    for (thinking, sent) in [
        (
            json!({"type": "adaptive", "display": "summarized"}),
            json!({"type": "adaptive", "display": "summarized",
                "block_binding": {"prefix_mismatch_behavior": "drop_block"}}),
        ),
        (json!({"type": "disabled"}), json!({"type": "disabled"})),
    ] {
        let mut request = completion_request_with_history(vec![message::Message::user("q")], None);
        request.additional_params = Some(json!({ "thinking": thinking }));
        let body = super::body(&wire, request, crate::wire::Mode::Unary).expect("encodes");
        assert_eq!(body["thinking"], sent);
    }
}

/// Anthropic A4: thinking that ended without a signature keeps no item, so
/// it replays as text, as pi sends it. Kimi, which sends thinking unsigned,
/// takes it back as thinking (#1315).
#[test]
fn unsigned_thinking_keeps_its_item_only_where_the_dialect_takes_it() {
    let unsigned = json!({"type": "thinking", "thinking": "plan", "signature": ""});
    for (dialect, model, kept) in [
        (&super::super::wire::ANTHROPIC, CLAUDE_SONNET_4_6, false),
        (&super::super::wire::MOONSHOT, "kimi-k2.6", true),
    ] {
        let wire = AnthropicConfig::with_key(dialect, "k").completion(model);
        let turn = folded_turn(
            &wire,
            &hello_request(),
            json!({"model": model, "stop_reason": "max_tokens", "content": [unsigned.clone()]}),
        );
        assert_eq!(turn.content[0].native_item().is_some(), kept, "{model}");
        let body = sent_with(
            &wire,
            vec![
                message::Message::user("go"),
                message::Message::Assistant(turn),
                message::Message::user("next"),
            ],
            Vec::new(),
        );
        let sent = &body["messages"][1]["content"][0];
        if kept {
            assert_eq!(sent, &unsigned);
        } else {
            assert_eq!(sent, &json!({"type": "text", "text": "plan"}));
        }
    }
}

/// Anthropic A5: a `tool_use` whose item states no `input` (a gateway's
/// zero-argument call) keeps an item that does: the canonical arguments.
/// Input that is not an object leaves the call without an item, rebuilt.
#[test]
fn a_kept_tool_use_item_always_states_an_object_input() {
    let wire = AnthropicConfig::with_key(&super::super::wire::ZAI, "k").completion("glm-4.6");
    for (call, item) in [
        (
            json!({"type": "tool_use", "id": "call_1", "name": "add"}),
            Some(json!({"type": "tool_use", "id": "call_1", "name": "add", "input": {}})),
        ),
        (
            json!({"type": "tool_use", "id": "call_1", "name": "add", "input": null}),
            Some(json!({"type": "tool_use", "id": "call_1", "name": "add", "input": {}})),
        ),
        (
            json!({"type": "tool_use", "id": "call_1", "name": "add", "input": "{\"x\":1}"}),
            None,
        ),
    ] {
        let turn = folded_turn(
            &wire,
            &hello_request(),
            json!({"model": "glm-4.6", "stop_reason": "tool_use", "content": [call]}),
        );
        assert_eq!(turn.content[0].native_item(), item.as_ref());
        let body = sent_with(
            &wire,
            vec![
                message::Message::user("add"),
                message::Message::Assistant(turn),
                message::Message::tool_result(
                    message::CallId::from_wire("call_1"),
                    message::ToolName::new("add").expect("name"),
                    "2",
                ),
            ],
            vec![generic_tool("add")],
        );
        assert!(
            body["messages"][1]["content"][0]["input"].is_object(),
            "{body}"
        );
    }
}

/// Anthropic rejects a blank text block, so a blank one keeps no item and
/// `adapt` drops it; the encoder drops nothing.
#[test]
fn a_blank_text_block_keeps_no_item_and_is_never_sent() {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let turn = folded_turn(
        &wire,
        &hello_request(),
        json!({"model": CLAUDE_SONNET_4_6, "stop_reason": "end_turn", "content": [
            {"type": "text", "text": "  "}, {"type": "text", "text": "real answer"}]}),
    );
    assert!(turn.content[0].native_item().is_none());
    let body = sent_with(
        &wire,
        vec![
            message::Message::user("go"),
            message::Message::Assistant(turn),
            message::Message::user("next"),
        ],
        Vec::new(),
    );
    assert_eq!(
        body["messages"][1]["content"],
        json!([{"type": "text", "text": "real answer"}])
    );
}

/// Anthropic A3: a server tool's result replays only with its use. A use
/// whose input never completed replays neither, and neither does a
/// same-model result whose use is gone.
#[test]
fn a_server_tool_result_never_replays_without_its_use() {
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_4_6);
    let result = json!({"type": "web_search_tool_result", "tool_use_id": "srvtoolu_1",
        "content": [{"type": "web_search_result", "url": "https://x", "title": "x",
            "encrypted_content": "e"}]});
    let mut turn = folded_turn(
        &wire,
        &hello_request(),
        json!({"model": CLAUDE_SONNET_4_6, "stop_reason": "end_turn", "content": [
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
                "input": {"query": "rig"}},
            result.clone(),
            {"type": "text", "text": "done"}]}),
    );
    turn.content.remove(0);
    let body = sent_with(
        &wire,
        vec![
            message::Message::user("go"),
            message::Message::Assistant(turn),
            message::Message::user("next"),
        ],
        Vec::new(),
    );
    assert_eq!(
        body["messages"][1]["content"],
        json!([{"type": "text", "text": "done"}]),
        "{body}"
    );
}

#[test]
fn context_binding_reads_every_spelling_of_a_claude_model() {
    for model in [
        "claude-opus-5-5",
        "claude-opus-5-5-20260101",
        "anthropic/claude-opus-5.5",
        "anthropic.claude-opus-5-5-v1:0",
        "us.anthropic.claude-opus-5-5-20260101-v1:0",
        "us.anthropic.claude-opus-5",
    ] {
        assert!(binds_context(model), "{model}");
    }
    for model in [
        "claude-sonnet-5",
        "us.anthropic.claude-haiku-4-5-20251001-v1:0",
        "gpt-5",
    ] {
        assert!(!binds_context(model), "{model}");
    }
}
