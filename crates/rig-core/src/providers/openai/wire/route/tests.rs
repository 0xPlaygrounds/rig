//! The route's options, reached the way a caller reaches them.
//!
//! `Bound<OpenAI>::completion` yields a `Bound<OpenAiWire>`, so every
//! per-route option has to be reachable through `Bound::map_wire` or it is
//! not reachable at all. Each test below asserts on the *encoded request*:
//! an option that only sets a field the encoder ignores is not forwarded.

use super::*;
use crate::completion::ToolDefinition;
use crate::driver::Bound;
use crate::message::{AssistantContent, Message, ToolResultContent, UserContent};

use super::super::OPENROUTER;

/// A turn with a system prompt, a tool and a tool result, so each option
/// under test has something in the body it could change.
fn request() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![
            Message::system("be brief"),
            "probe".into(),
            Message::Assistant {
                id: None,
                content: vec![AssistantContent::tool_call(
                    "call_1",
                    "lookup",
                    serde_json::json!({"q": "x"}),
                )],
            },
            Message::User {
                content: vec![UserContent::tool_result_from_wire(
                    "call_1",
                    "lookup",
                    vec![ToolResultContent::text("the answer")],
                )],
            },
        ],
        documents: vec![],
        tools: vec![ToolDefinition {
            name: "lookup".to_owned(),
            description: "look something up".to_owned(),
            parameters: serde_json::json!({
                "type": "object",
                "properties": {"q": {"type": "string"}}
            }),
        }],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

/// What `provider`'s route sends after `option` went through
/// [`Bound::map_wire`].
fn body(provider: OpenAI, option: impl FnOnce(OpenAiWire) -> OpenAiWire) -> serde_json::Value {
    body_with_request(provider, option, request())
}

fn body_with_request(
    provider: OpenAI,
    option: impl FnOnce(OpenAiWire) -> OpenAiWire,
    request: CompletionRequest,
) -> serde_json::Value {
    let bound = Bound::new(provider, ())
        .completion("gpt-5.2")
        .map_wire(option);
    let encoded = bound
        .wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let [request] = encoded.requests.as_slice() else {
        panic!("a completion route sends one request");
    };
    match request.body() {
        Body::Bytes(bytes) => serde_json::from_slice(bytes).expect("the body is JSON"),
        Body::Multipart(_) => panic!("neither completion route sends a multipart body"),
    }
}

/// The wire unchanged, as the baseline every assertion compares against.
fn untouched(wire: OpenAiWire) -> OpenAiWire {
    wire
}

/// `option` changes what `changes` sends and leaves `unchanged` as it was —
/// a route without the option is a no-op, not a panic and not a type error.
fn only_on(changes: OpenAI, unchanged: OpenAI, option: impl Fn(OpenAiWire) -> OpenAiWire + Copy) {
    assert_ne!(
        body(changes.clone(), option),
        body(changes, untouched),
        "the option never reached the route that has it"
    );
    assert_eq!(
        body(unchanged.clone(), option),
        body(unchanged, untouched),
        "the option changed a route that has no such option"
    );
}

fn chat() -> OpenAI {
    OpenAI::new("sk-test").with_route(Route::Chat)
}

fn responses() -> OpenAI {
    OpenAI::new("sk-test").with_route(Route::Responses)
}

fn tool() -> ResponsesToolDefinition {
    ResponsesToolDefinition::function(
        "hosted",
        "a provider-side tool",
        serde_json::json!({"type": "object"}),
    )
}

/// The placement [`OpenAiWire::with_system_instructions_as_messages`] is
/// sugar for.
fn as_messages(wire: OpenAiWire) -> OpenAiWire {
    wire.with_system_instructions_placement(SystemInstructionsPlacement::InputSystemMessages)
}

#[test]
fn map_wire_reaches_strict_tools_on_both_routes() {
    for provider in [chat(), responses()] {
        assert_ne!(
            body(provider.clone(), OpenAiWire::with_strict_tools),
            body(provider, untouched),
        );
    }
}

#[test]
fn map_wire_reaches_tool_result_array_content_on_the_chat_route_only() {
    only_on(
        chat(),
        responses(),
        OpenAiWire::with_tool_result_array_content,
    );
}

/// Prompt caching is OpenRouter's `cache_control` on the chat body, so that
/// is the dialect whose request it changes.
#[test]
fn map_wire_reaches_prompt_caching_on_the_chat_route_only() {
    only_on(
        OpenAI::with_key(&OPENROUTER, "sk-test").with_route(Route::Chat),
        OpenAI::with_key(&OPENROUTER, "sk-test").with_route(Route::Responses),
        OpenAiWire::with_prompt_caching,
    );
}

#[test]
fn changed_system_prefix_changes_both_encoded_openai_routes() {
    for provider in [chat(), responses()] {
        let original = request();
        let mut changed = original.clone();
        *changed.chat_history.first_mut().expect("system message") =
            Message::system("different task rules");
        assert_ne!(
            body_with_request(provider.clone(), untouched, original),
            body_with_request(provider, untouched, changed)
        );
    }
}

#[test]
fn encoded_cache_affinity_and_retention_preserve_configured_values() {
    for provider in [chat(), responses()] {
        for key in ["first-task-key", "changed-task-key"] {
            let mut request = request();
            request.additional_params =
                Some(serde_json::json!({"prompt_cache_key": key, "prompt_cache_retention": "24h"}));
            let encoded = body_with_request(provider.clone(), untouched, request);
            assert_eq!(
                encoded.get("prompt_cache_key"),
                Some(&serde_json::json!(key))
            );
            assert_eq!(
                encoded.get("prompt_cache_retention"),
                Some(&serde_json::json!("24h"))
            );
        }
    }
}

#[test]
fn openrouter_cache_controls_do_not_change_openai_requests() {
    for provider in [chat(), responses()] {
        assert_eq!(
            body(provider.clone(), OpenAiWire::with_prompt_caching),
            body(provider, untouched),
        );
    }
}

#[test]
fn map_wire_reaches_a_wire_level_tool_on_the_responses_route_only() {
    only_on(responses(), chat(), |wire| wire.with_tool(tool()));
}

#[test]
fn map_wire_reaches_wire_level_tools_on_the_responses_route_only() {
    only_on(responses(), chat(), |wire| wire.with_tools([tool()]));
}

#[test]
fn map_wire_reaches_the_system_instructions_placement_on_the_responses_route_only() {
    only_on(responses(), chat(), as_messages);
}

/// The sugar is the placement, which is what the encoded body shows.
#[test]
fn map_wire_reaches_system_instructions_as_messages_on_the_responses_route_only() {
    only_on(
        responses(),
        chat(),
        OpenAiWire::with_system_instructions_as_messages,
    );
    assert_eq!(
        body(
            responses(),
            OpenAiWire::with_system_instructions_as_messages
        ),
        body(responses(), as_messages),
    );
}

/// Synthetic hooks test extension dispatch and precedence, which recorded provider traffic cannot exercise.
#[test]
fn dialect_hooks_apply_to_both_routes_without_provider_identity() {
    use super::super::{Dialect, DialectHooks, Quirks};
    use std::sync::atomic::{AtomicUsize, Ordering};

    static ENVELOPES: AtomicUsize = AtomicUsize::new(0);
    static HOOKS: DialectHooks = DialectHooks {
        default_endpoint: Some(|key| (key == "regional").then(|| "https://region.invalid".into())),
        model_route: Some(|model| {
            if model == "responses-model" {
                Route::Responses
            } else {
                Route::Chat
            }
        }),
        completion_envelope: Some(|provider, request, mut builder| {
            ENVELOPES.fetch_add(1, Ordering::SeqCst);
            assert_eq!(provider.api_key.expose(), "regional");
            assert!(!request.chat_history.is_empty());
            assert_eq!(
                builder.headers_ref().unwrap()[http::header::AUTHORIZATION],
                "Bearer regional"
            );
            builder
                .headers_mut()
                .unwrap()
                .remove(http::header::AUTHORIZATION);
            builder
                .header(http::header::AUTHORIZATION, "custom credential")
                .header("x-custom-envelope", "applied")
        }),
        modality_envelope: None,
    };
    let dialect = Dialect {
        quirks: Quirks {
            hooks: Some(&HOOKS),
            ..Quirks::openai()
        },
        ..Dialect::gateway("custom", "https://default.invalid", "UNUSED_KEY")
    };
    assert_eq!(
        OpenAI::with_key(&dialect, "other").base_url,
        "https://default.invalid"
    );
    let provider = OpenAI::with_key(&dialect, "regional");
    assert_eq!(provider.base_url, "https://region.invalid");
    assert!(matches!(
        provider.completion("responses-model"),
        OpenAiWire::Responses(_)
    ));
    assert!(matches!(
        provider.completion("chat-model"),
        OpenAiWire::Chat(_)
    ));
    let mut calls = 0;
    for model in ["responses-model", "chat-model"] {
        for route in [Route::Chat, Route::Responses] {
            let overridden = provider
                .clone()
                .with_route(route)
                .with_base_url("https://explicit.invalid");
            for wire in [
                overridden.completion(model),
                match route {
                    Route::Chat => overridden.chat(model).into(),
                    Route::Responses => overridden.responses(model).into(),
                },
            ] {
                for mode in [Mode::Unary, Mode::Streaming] {
                    let encoded = wire.encode(request(), mode).unwrap();
                    calls += 1;
                    assert_eq!(ENVELOPES.load(Ordering::SeqCst), calls);
                    let [request] = encoded.requests.as_slice() else {
                        panic!("one request")
                    };
                    assert_eq!(request.uri().host(), Some("explicit.invalid"));
                    assert_eq!(
                        request.uri().path(),
                        match route {
                            Route::Chat => "/chat/completions",
                            Route::Responses => "/responses",
                        }
                    );
                    assert_eq!(request.headers()["x-custom-envelope"], "applied");
                    assert_eq!(
                        request.headers()[http::header::AUTHORIZATION],
                        "custom credential"
                    );
                    assert_eq!(
                        request
                            .headers()
                            .get_all(http::header::AUTHORIZATION)
                            .iter()
                            .count(),
                        1
                    );
                }
            }
        }
    }
    assert!(
        serde_json::to_value(&provider).is_err(),
        "custom callbacks cannot round-trip by name"
    );
}

/// A synthetic dialect isolates the default capability from provider identity; no server is involved.
#[test]
fn responses_strict_tools_default_is_an_independent_capability() {
    use super::super::{Dialect, OPENAI, Quirks, ResponsesQuirks};
    let dialect = Dialect {
        quirks: Quirks {
            responses: ResponsesQuirks {
                strict_tools_by_default: true,
                ..ResponsesQuirks::openai()
            },
            ..OPENAI.quirks
        },
        ..OPENAI
    };
    let provider = OpenAI::with_key(&dialect, "test");
    assert!(provider.responses("model").strict_tools);
    assert!(!provider.chat("model").strict_tools);
    let strict = body(provider, untouched);
    let ordinary = body(OpenAI::new("test"), untouched);
    assert_eq!(strict["tools"][0]["strict"], true);
    assert_eq!(
        strict["tools"][0]["parameters"]["additionalProperties"],
        false
    );
    assert_ne!(ordinary["tools"][0]["strict"], true);
    assert!(!OpenAI::new("test").responses("model").strict_tools);
}

/// Invalid local headers fail before transport, so there is no cassette interaction.
#[test]
fn completion_envelope_builder_errors_are_returned_on_both_routes() {
    use super::super::{Dialect, DialectHooks, OPENAI, Quirks};
    static HOOKS: DialectHooks = DialectHooks {
        default_endpoint: None,
        model_route: None,
        completion_envelope: Some(|_, _, builder| builder.header("invalid\nname", "value")),
        modality_envelope: None,
    };
    let provider = OpenAI::with_key(
        &Dialect {
            quirks: Quirks {
                hooks: Some(&HOOKS),
                ..OPENAI.quirks
            },
            ..OPENAI
        },
        "test",
    );
    for wire in [
        OpenAiWire::from(provider.chat("model")),
        provider.responses("model").into(),
    ] {
        assert!(wire.encode(request(), Mode::Unary).is_err());
    }
    assert!(
        serde_json::to_value(&provider).is_err(),
        "a registered name cannot hide changed hooks"
    );
}
