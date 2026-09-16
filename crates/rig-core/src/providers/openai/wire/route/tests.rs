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
    let bound = Bound::new(provider, ())
        .completion("gpt-5.2")
        .map_wire(option);
    let encoded = bound
        .wire
        .encode(request(), Mode::Unary)
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
