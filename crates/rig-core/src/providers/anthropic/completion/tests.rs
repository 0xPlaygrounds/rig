use super::*;
use crate::completion::CacheRetention;
use crate::error::ProviderError;
use crate::providers::anthropic::wire::AnthropicConfig;
use crate::test_utils::json_body;
use crate::wire::WireFrame;

/// The wire settings a request body is built with.
struct Params<'a> {
    model: &'a str,
    request: CompletionRequest,
    prompt_caching: bool,
    /// The request's cache retention.
    cache: Option<crate::completion::CacheRetention>,
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
    wire.static_prefix_cache_ttl = params.static_prefix_cache_ttl;
    wire.strict_tools = strict;
    use crate::wire::{Operation, Wire};
    // The request as production encodes it: prepared, so adapted.
    let mut request = params.request;
    if let Some(cache) = params.cache {
        request.options = request.options.cache(cache);
    }
    let request = crate::operation::Completion::prepare(request, &wire.describe())
        .map_err(|error| EncodeError::request(error.to_string()))?;
    Ok(serde_json::to_value(body(&wire, &request, Mode::Unary)?)?)
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
        default_max_tokens_for_model(crate::catalog::ModelFacts::builtin(), CLAUDE_FABLE_5_1),
        Some(128_000)
    );
    assert_eq!(
        default_max_tokens_for_model(crate::catalog::ModelFacts::builtin(), CLAUDE_FABLE_5),
        Some(128_000)
    );
    assert_eq!(
        default_max_tokens_for_model(crate::catalog::ModelFacts::builtin(), CLAUDE_OPUS_5),
        Some(128_000)
    );
    assert_eq!(
        default_max_tokens_for_model(crate::catalog::ModelFacts::builtin(), CLAUDE_SONNET_5),
        Some(128_000)
    );
    assert_eq!(
        default_max_tokens_for_model(crate::catalog::ModelFacts::builtin(), CLAUDE_OPUS_4_8),
        Some(128_000)
    );
    assert_eq!(
        default_max_tokens_for_model(crate::catalog::ModelFacts::builtin(), CLAUDE_OPUS_4_7),
        Some(128_000)
    );
    assert_eq!(
        default_max_tokens_for_model(crate::catalog::ModelFacts::builtin(), CLAUDE_OPUS_4_6),
        Some(128_000)
    );
    assert_eq!(
        default_max_tokens_for_model(crate::catalog::ModelFacts::builtin(), CLAUDE_SONNET_4_6),
        Some(128_000)
    );
    assert_eq!(
        default_max_tokens_for_model(crate::catalog::ModelFacts::builtin(), CLAUDE_HAIKU_4_5),
        Some(64_000)
    );
    assert_eq!(
        default_max_tokens_for_model(
            crate::catalog::ModelFacts::builtin(),
            "claude-sonnet-4-20250514"
        ),
        Some(64_000)
    );
    assert_eq!(
        default_max_tokens_for_model(
            crate::catalog::ModelFacts::builtin(),
            "claude-opus-4-1-20250805"
        ),
        Some(32_000),
        "the Models API's limit for Claude Opus 4.1"
    );
    assert_eq!(
        default_max_tokens_for_model(crate::catalog::ModelFacts::builtin(), "claude-3-opus"),
        None
    );
    // An id the catalog does not list takes its family's default.
    assert_eq!(
        default_max_tokens_for_model(crate::catalog::ModelFacts::builtin(), "claude-opus-4.6"),
        Some(64_000)
    );
    assert_eq!(
        default_max_tokens_for_model(
            crate::catalog::ModelFacts::builtin(),
            "claude-sonnet-4-6@20260101"
        ),
        Some(64_000)
    );
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
            cache: None,
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
        cache: None,
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
        cache: None,
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

/// A code-execution turn ends in its `container` block, which goes to the
/// request's top level: a system message after the turn's last result
/// stays in place rather than joining `system`.
#[test]
fn opus_4_8_preserves_system_message_after_a_code_execution_result_in_a_container() {
    let reply = json!({
        "type": "message", "id": "msg_1", "model": CLAUDE_OPUS_4_8, "role": "assistant",
        "stop_reason": "end_turn", "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "container": {"id": "container_1", "expires_at": "2026-10-03T00:00:00Z"},
        "content": [
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "code_execution",
                "input": {"code": "print(1)"}},
            {"type": "code_execution_tool_result", "tool_use_id": "srvtoolu_1",
                "content": {"type": "code_execution_result", "stdout": "1\n", "stderr": "",
                    "return_code": 0, "content": []}}
        ]
    });
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_OPUS_4_8);
    let turn = crate::test_utils::decode_reply(
        &wire,
        &hello_request(),
        crate::wire::Mode::Unary,
        [WireFrame::Text(reply.to_string())],
        reply.clone(),
    )
    .expect("the reply folds")
    .message()
    .expect("a turn");
    let request = completion_request_with_history(
        vec![
            message::Message::user("run it"),
            turn,
            message::Message::System {
                content: "Answer in Spanish.".to_string(),
            },
        ],
        None,
    );
    let value = request_body(Params {
        model: CLAUDE_OPUS_4_8,
        request,
        prompt_caching: false,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();
    assert!(value.get("system").is_none(), "{value:#}");
    assert_eq!(value["messages"][2]["role"], "system", "{value:#}");
    assert_eq!(value["container"], json!("container_1"));
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
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();
    serde_json::to_value(request).unwrap()
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
        cache: None,
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
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    let tools = value["tools"].as_array().unwrap();
    assert_eq!(tools[0]["cache_control"]["type"], "ephemeral");
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
        cache: Some(CacheRetention::Short),
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
        cache: Some(CacheRetention::Short),
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
        cache: Some(CacheRetention::Short),
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
        cache: Some(CacheRetention::Long),
        static_prefix_cache_ttl: None,
    })
    .unwrap_err();

    assert!(err.to_string().contains("ttl `1h`"));
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
        cache: Some(CacheRetention::Short),
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
fn test_static_prefix_ttl_with_manual_caching_splits_prefix_and_tail() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
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
fn test_static_prefix_ttl_five_minutes_with_automatic_1h_errors_client_side() {
    let request = completion_request_with_tools(vec![generic_tool("cached_tool")], None);

    let error = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        cache: Some(CacheRetention::Long),
        static_prefix_cache_ttl: Some(CacheTtl::FiveMinutes),
    })
    .unwrap_err();

    let message = error.to_string();
    assert!(
        message.contains("with_static_prefix_cache_ttl"),
        "error should name the knob: {message}"
    );
    assert!(
        message.contains("CacheRetention::Long"),
        "error should name the conflicting option: {message}"
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
        cache: Some(CacheRetention::Short),
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

/// A raw top-level marker is above the cache option: its keys win, and the
/// mapped keys it does not name stay.
#[test]
fn a_raw_top_level_marker_beats_the_cache_option() {
    let request = completion_request_with_tools(
        Vec::new(),
        Some(json!({
            "cache_control": {"type": "ephemeral", "ttl": "5m"}
        })),
    );

    let value = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: false,
        cache: Some(CacheRetention::Long),
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    assert_eq!(
        value["cache_control"],
        json!({"type": "ephemeral", "ttl": "5m"})
    );
}

#[test]
fn test_prompt_caching_without_tools_omits_tools() {
    let request = completion_request_with_tools(Vec::new(), None);

    let request = request_body(Params {
        model: "claude-sonnet-4-6",
        request,
        prompt_caching: true,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .unwrap();

    let value = serde_json::to_value(request).unwrap();
    assert!(value.get("tools").is_none());
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

// -------------------------------------------------------------------
// Citations (#1767)
// -------------------------------------------------------------------

#[test]
fn document_serializes_citations_and_metadata() {
    let value = user_wire(message::UserContent::Document(message::Document {
        data: message::DocumentData::Text("hello".into()),
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

/// Raw-capture tests: `CompletionResponse::raw` driven end to end over a
/// mock transport that hands back a Messages body *and* a `request-id`
/// response header. `raw` is the verbatim reply document — not a
/// re-serialization of the parsed type — so it answers what the normalized
/// response does not, and the transport id the driver read off the headers
/// reaches the normalized response rather than the document.
/// `with_error_response_headers` with `200 OK` is the one unary double
/// that carries response headers.
mod raw_capture {}

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
        data: data.into(),
        media_type,
        additional_params: None,
    };
    let text = |text: &str, media_type: Option<Doc>| Document {
        data: message::DocumentData::Text(text.into()),
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
        (text("plain", Some(Doc::TXT)), true),
        (text("plain", None), true),
        (document(Source::base64("cGxhaW4="), Some(Doc::TXT)), true),
        (
            document(Source::base64("cmlnLG1hdHJpeAo="), Some(Doc::CSV)),
            true,
        ),
        (text("<p>hi</p>", Some(Doc::HTML)), true),
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
            Source::base64("cmlnLG1hdHJpeAoxLDIK").into(),
            Some(Doc::CSV),
            "rig,matrix\n1,2\n",
        ),
        (
            Source::base64("cGxhaW4gdGV4dA==").into(),
            Some(Doc::TXT),
            "plain text",
        ),
        (
            message::DocumentData::Text("# notes".into()),
            Some(Doc::MARKDOWN),
            "# notes",
        ),
        (
            message::DocumentData::Text("untyped".into()),
            None,
            "untyped",
        ),
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

/// A paused turn goes back as it is to resume it, its running
/// `server_tool_use` included (Anthropic's `pause_turn` contract).
#[test]
fn a_paused_turn_resumes_with_its_running_server_tool_use() {
    let reply = json!({
        "id": "msg_1", "model": CLAUDE_SONNET_4_6, "role": "assistant",
        "stop_reason": "pause_turn", "stop_sequence": null,
        "usage": {"input_tokens": 1, "output_tokens": 1},
        "content": [
            {"type": "text", "text": "Searching."},
            {"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
                "input": {"query": "rig"}}
        ]
    });
    let turn = fold_reply(&reply)
        .expect("the reply folds")
        .message()
        .expect("a turn");
    let body = prepared_body(vec![message::Message::user("search"), turn]);
    assert_eq!(
        body["messages"][1]["content"][1],
        json!({"type": "server_tool_use", "id": "srvtoolu_1", "name": "web_search",
            "input": {"query": "rig"}}),
        "{body:#}"
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
        let body = super::body(&wire, &request, crate::wire::Mode::Unary).expect("encodes");
        assert_eq!(body.get("thinking"), Some(&sent));
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

/// Anthropic's wire reads its own ids and their dated snapshots. Another
/// vendor's spelling is that vendor's catalog row, which carries the
/// Anthropic facts itself.
#[test]
fn context_binding_reads_anthropics_ids_and_their_snapshots() {
    for model in [
        "claude-opus-5-5",
        "claude-opus-5-5-20260101",
        "claude-opus-5",
        "claude-fable-5-1",
        "claude-sonnet-5-5",
    ] {
        assert!(
            binds_context(crate::catalog::ModelFacts::builtin(), model),
            "{model}"
        );
    }
    for model in [
        "claude-sonnet-5",
        "claude-haiku-4-5-20251001",
        "anthropic/claude-opus-5.5",
        "claude-opus-5.5",
        "us.anthropic.claude-opus-5",
        "gpt-5",
    ] {
        assert!(
            !binds_context(crate::catalog::ModelFacts::builtin(), model),
            "{model}"
        );
    }
}

/// Thinking that takes no block binding (`between_tools`, `disabled`) asks
/// Anthropic for no `drop_block`, so a turn made under other tools replays
/// as another model's: its signed thinking goes as text, which Anthropic
/// takes where a mismatched signature is refused (checked live on Claude
/// Sonnet 5.5).
#[test]
fn thinking_without_a_binding_replays_another_contexts_turn_as_text() {
    let thinking = json!({"type": "thinking", "thinking": "plan", "signature": "sig"});
    let wire = AnthropicConfig::new("test-key").completion(CLAUDE_SONNET_5_5);
    let made = hello_request().tools(vec![generic_tool("add")]);
    let turn = folded_turn(
        &wire,
        &made,
        json!({"model": CLAUDE_SONNET_5_5, "stop_reason": "end_turn", "content": [
            thinking.clone(), {"type": "text", "text": "done"}]}),
    );
    let history = vec![
        message::Message::user("go"),
        message::Message::Assistant(turn),
        message::Message::user("next"),
    ];
    let mut request = completion_request_with_history(history, None);
    request.tools = vec![generic_tool("add"), generic_tool("mul")];
    request.additional_params = Some(json!({"thinking": {"type": "between_tools"}}));
    let value = request_body(Params {
        model: CLAUDE_SONNET_5_5,
        request,
        prompt_caching: false,
        cache: None,
        static_prefix_cache_ttl: None,
    })
    .expect("encodes");
    assert_eq!(
        value["thinking"],
        json!({"type": "between_tools"}),
        "{value}"
    );
    assert_eq!(
        value["messages"][1]["content"][0],
        json!({"type": "text", "text": "plan"}),
        "{value}"
    );
}

/// What Anthropic would reject in `body`: an empty message or blank text,
/// a system message outside a slot (after a user turn, before an assistant
/// turn or the end), and a call its next message does not answer first.
fn messages_violations(body: &Value) -> Vec<String> {
    let mut out = Vec::new();
    let messages = body["messages"].as_array().cloned().unwrap_or_default();
    for (i, m) in messages.iter().enumerate() {
        let content = m["content"].as_array().cloned().unwrap_or_default();
        if content.is_empty() {
            out.push(format!("message {i} has empty content"));
        }
        for b in &content {
            if b["type"] == "text" && b["text"].as_str().is_some_and(|t| t.trim().is_empty()) {
                out.push(format!("message {i} has blank text"));
            }
        }
        // A run of system messages shares one slot, which Anthropic takes.
        if m["role"] == "system" {
            let prev = messages[..i]
                .iter()
                .rev()
                .find(|message| message["role"] != "system")
                .map(|message| message["role"].clone());
            let next = messages[i + 1..]
                .iter()
                .find(|message| message["role"] != "system")
                .map(|message| message["role"].clone());
            if prev != Some(json!("user")) || !(next.is_none() || next == Some(json!("assistant")))
            {
                out.push(format!("system at {i} between {prev:?} and {next:?}"));
            }
        }
        if m["role"] == "assistant" {
            let ids: Vec<String> = content
                .iter()
                .filter(|b| b["type"] == "tool_use")
                .filter_map(|b| b["id"].as_str().map(str::to_owned))
                .collect();
            if ids.is_empty() {
                continue;
            }
            let Some(next) = messages.get(i + 1) else {
                out.push(format!("tool_use at {i} ends the request"));
                continue;
            };
            let next_content = next["content"].as_array().cloned().unwrap_or_default();
            let leading: Vec<String> = next_content
                .iter()
                .take_while(|b| b["type"] == "tool_result")
                .filter_map(|b| b["tool_use_id"].as_str().map(str::to_owned))
                .collect();
            if next["role"] != "user" || ids.iter().any(|id| !leading.contains(id)) {
                out.push(format!(
                    "tool_use at {i} not answered first by next message"
                ));
            }
        }
    }
    out
}

/// Histories `adapt` and the Messages encoder must shape into a request
/// Anthropic takes, on a model that takes system messages in place and on
/// one that folds them into the prompt (round-5 F3).
#[test]
fn adversarial_histories_encode_to_requests_anthropic_takes() {
    use crate::message::{
        AssistantMessage, CallId, Origin, StopReason, ToolCall, ToolFunction, ToolName,
    };
    let other = Some(Origin::new("openai.chat", "openai", "gpt-4.1"));
    let asst = |content: Vec<AssistantContent>, stop: StopReason| {
        Message::Assistant(AssistantMessage {
            content,
            origin: other.clone(),
            stop: Some(stop),
        })
    };
    let call = |id: &str| {
        ToolCall::new(
            CallId::from_wire(id),
            ToolFunction::new(ToolName::new("lookup").expect("name"), json!({})),
        )
    };
    let result = |id: &str| Message::User {
        content: vec![UserContent::ToolResult(
            call(id).result(vec![ToolResultContent::text("r")]),
        )],
    };
    let text = |t: &str| AssistantContent::text(t);
    let tc = |id: &str| AssistantContent::ToolCall(call(id));
    let cases: Vec<(&str, Vec<Message>)> = vec![
        (
            "system between assistants",
            vec![
                Message::user("a"),
                asst(vec![text("one")], StopReason::Stop),
                Message::system("mid"),
                asst(vec![text("two")], StopReason::Stop),
                Message::user("q"),
            ],
        ),
        (
            "system while a call waits, then text",
            vec![
                Message::user("a"),
                asst(vec![tc("x")], StopReason::ToolUse),
                Message::system("mid"),
                Message::User {
                    content: vec![
                        UserContent::text("note"),
                        UserContent::ToolResult(
                            call("x").result(vec![ToolResultContent::text("r")]),
                        ),
                    ],
                },
                Message::user("q"),
            ],
        ),
        (
            "blank user after a call",
            vec![
                Message::user("a"),
                asst(vec![tc("x")], StopReason::ToolUse),
                Message::user("  "),
                result("x"),
            ],
        ),
        (
            "two systems around a user",
            vec![
                Message::system("s0"),
                Message::user("a"),
                Message::system("s1"),
                Message::user("b"),
                Message::system("s2"),
                asst(vec![text("ok")], StopReason::Stop),
                Message::user("end"),
            ],
        ),
        (
            "system before results of an aborted turn",
            vec![
                Message::user("a"),
                asst(vec![tc("x")], StopReason::Aborted("cut".into())),
                Message::system("mid"),
                result("x"),
                Message::user("q"),
            ],
        ),
    ];
    let mut failures = Vec::new();
    for model in [CLAUDE_OPUS_5_5, CLAUDE_SONNET_4_6] {
        for (label, history) in &cases {
            let mut request = CompletionRequest::from(history.clone()).max_tokens(64);
            request.tools = vec![completion::ToolDefinition {
                name: ToolName::new("lookup").expect("name"),
                description: "d".into(),
                parameters: json!({"type": "object", "properties": {}}),
            }];
            let body = match request_body(Params {
                model,
                request,
                prompt_caching: false,
                cache: None,
                static_prefix_cache_ttl: None,
            }) {
                Ok(body) => body,
                Err(error) => {
                    failures.push(format!("{model} {label}: {error}"));
                    continue;
                }
            };
            let v = messages_violations(&body);
            if !v.is_empty() {
                failures.push(format!("{model} {label}: {v:?}\n  {}", body["messages"]));
            }
        }
    }
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

/// What `wire` sends for a request with one Rig tool and one provider tool:
/// each tool's `eager_input_streaming`, and the `anthropic-beta` header.
fn tool_input_streaming_sent(
    wire: &crate::providers::anthropic::wire::Messages,
    mode: Mode,
    tools: Vec<completion::ToolDefinition>,
) -> (Vec<Option<Value>>, Option<String>) {
    use crate::wire::Wire;
    let request = completion_request_with_tools(
        tools,
        Some(json!({"tools": [{"type": "web_search_20250305", "name": "web_search"}]})),
    );
    let encoded = wire.encode(request, mode).expect("the request encodes");
    let body = json_body(&encoded.request);
    let eager = body["tools"]
        .as_array()
        .map(|tools| {
            tools
                .iter()
                .map(|tool| tool.get("eager_input_streaming").cloned())
                .collect()
        })
        .unwrap_or_default();
    let beta = encoded
        .request
        .headers()
        .get("anthropic-beta")
        .and_then(|value| value.to_str().ok())
        .map(str::to_owned);
    (eager, beta)
}

/// A request with Rig tools asks every dialect to stream their input as it
/// is written, as pi does, unary or streamed alike so the tool definitions
/// a prompt cache keys on never change; a provider tool is left as given,
/// and a request without Rig tools asks nothing.
#[test]
fn a_request_with_tools_asks_for_eager_tool_input_on_every_dialect() {
    use crate::providers::anthropic::wire::{ANTHROPIC, MINIMAX, MOONSHOT, XIAOMIMIMO, ZAI};
    for dialect in [ANTHROPIC, ZAI, MINIMAX, MOONSHOT, XIAOMIMIMO] {
        let wire = AnthropicConfig::with_key(&dialect, "k").completion("some-model");
        assert_eq!(
            tool_input_streaming_sent(&wire, Mode::Streaming, vec![generic_tool("lookup")]),
            (vec![Some(json!(true)), None], None),
            "{}",
            dialect.name
        );
        assert_eq!(
            tool_input_streaming_sent(&wire, Mode::Unary, vec![generic_tool("lookup")]),
            (vec![Some(json!(true)), None], None),
            "{}",
            dialect.name
        );
        assert_eq!(
            tool_input_streaming_sent(&wire, Mode::Streaming, Vec::new()),
            (vec![None], None),
            "{}",
            dialect.name
        );
    }
}

/// A gateway that rejects the per-tool field gets the beta flag instead,
/// beside the caller's own flags, and one that rejects both gets neither.
#[test]
fn tool_input_streaming_falls_back_to_the_beta_flag_or_off() {
    use crate::providers::anthropic::wire::ToolInputStreaming;
    let provider = AnthropicConfig::new("k").with_beta("files-api-2025-04-14");
    let beta = provider
        .completion(CLAUDE_SONNET_4_6)
        .with_tool_input_streaming(ToolInputStreaming::BetaHeader);
    assert_eq!(
        tool_input_streaming_sent(&beta, Mode::Streaming, vec![generic_tool("lookup")]),
        (
            vec![None, None],
            Some("files-api-2025-04-14,fine-grained-tool-streaming-2025-05-14".to_owned())
        )
    );
    assert_eq!(
        tool_input_streaming_sent(&beta, Mode::Unary, vec![generic_tool("lookup")]),
        (
            vec![None, None],
            Some("files-api-2025-04-14,fine-grained-tool-streaming-2025-05-14".to_owned())
        )
    );
    assert_eq!(
        tool_input_streaming_sent(&beta, Mode::Streaming, Vec::new()),
        (vec![None], Some("files-api-2025-04-14".to_owned()))
    );
    let off = provider
        .completion(CLAUDE_SONNET_4_6)
        .with_tool_input_streaming(ToolInputStreaming::Off);
    assert_eq!(
        tool_input_streaming_sent(&off, Mode::Streaming, vec![generic_tool("lookup")]),
        (vec![None, None], Some("files-api-2025-04-14".to_owned()))
    );
}

/// A wire serialized before the setting existed reloads with eager input.
#[test]
fn a_wire_serialized_without_tool_input_streaming_reloads_eager() {
    use crate::providers::anthropic::wire::{Messages, ToolInputStreaming};
    let wire = AnthropicConfig::new("k")
        .completion(CLAUDE_SONNET_4_6)
        .with_tool_input_streaming(ToolInputStreaming::Off);
    let mut json = serde_json::to_value(&wire).expect("the wire serializes");
    assert_eq!(json["tool_input_streaming"], json!("off"));
    json.as_object_mut()
        .map(|wire| wire.shift_remove("tool_input_streaming"));
    let restored: Messages = serde_json::from_value(json).expect("the wire reloads");
    assert_eq!(restored.tool_input_streaming, ToolInputStreaming::Eager);
}

/// Each setting reads back from its serialized name, and the beta flag is
/// sent once when the caller already asked for it.
#[test]
fn tool_input_streaming_reads_its_names_and_sends_the_beta_flag_once() {
    use crate::providers::anthropic::wire::ToolInputStreaming;
    for (name, streaming) in [
        ("eager", ToolInputStreaming::Eager),
        ("beta_header", ToolInputStreaming::BetaHeader),
        ("off", ToolInputStreaming::Off),
    ] {
        assert_eq!(
            serde_json::from_value::<ToolInputStreaming>(json!(name)).ok(),
            Some(streaming)
        );
        assert_eq!(serde_json::to_value(streaming).ok(), Some(json!(name)));
        assert!(!format!("{streaming:?}").is_empty());
    }
    for unknown in [json!("fine_grained"), json!(1)] {
        assert!(serde_json::from_value::<ToolInputStreaming>(unknown).is_err());
    }
    let wire = AnthropicConfig::new("k")
        .with_beta("fine-grained-tool-streaming-2025-05-14")
        .completion(CLAUDE_SONNET_4_6)
        .with_tool_input_streaming(ToolInputStreaming::BetaHeader);
    assert_eq!(
        tool_input_streaming_sent(&wire, Mode::Streaming, vec![generic_tool("lookup")]).1,
        Some("fine-grained-tool-streaming-2025-05-14".to_owned())
    );
}

/// A raw top-level `cache_control` that is not an ephemeral marker with a
/// known TTL is refused, never sent as written.
#[test]
fn an_invalid_raw_top_level_cache_marker_is_refused() {
    for marker in [
        json!({"type": "persistent"}),
        json!({"type": "ephemeral", "ttl": "1d"}),
    ] {
        let request = completion_request_with_tools(
            vec![generic_tool("cached_tool")],
            Some(json!({ "cache_control": marker })),
        );
        let error = request_body(Params {
            model: "claude-sonnet-4-6",
            request,
            prompt_caching: false,
            cache: None,
            static_prefix_cache_ttl: None,
        })
        .expect_err("refused");
        assert!(
            error
                .to_string()
                .contains("additional_params.cache_control"),
            "{error}"
        );
    }
}
