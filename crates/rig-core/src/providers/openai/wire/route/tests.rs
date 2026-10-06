//! The route's options, reached the way a caller reaches them.
//!
//! `OpenAI::completion` yields an `OpenAiWire`, so every per-route option
//! has to be reachable on that wire or it is not reachable at all. Each test below asserts on the *encoded request*:
//! an option that only sets a field the encoder ignores is not forwarded.

use super::*;
use crate::completion::ToolDefinition;
use crate::message::{AssistantContent, Message, ToolResultContent, UserContent};
use crate::test_utils::json_body;

/// A turn with a system prompt, a tool and a tool result, so each option
/// under test has something in the body it could change.
fn request() -> CompletionRequest {
    CompletionRequest::from(vec![
        Message::system("be brief"),
        "probe".into(),
        Message::Assistant(crate::message::AssistantMessage::new(vec![
            AssistantContent::tool_call(
                "call_1",
                crate::message::ToolName::new("lookup").expect("tool name"),
                serde_json::json!({"q": "x"}),
            ),
        ])),
        Message::User {
            content: vec![UserContent::tool_result(
                crate::message::CallId::from_wire("call_1"),
                crate::message::ToolName::new("lookup").expect("tool name"),
                vec![ToolResultContent::text("the answer")],
            )],
        },
    ])
    .tools(vec![ToolDefinition {
        name: crate::message::ToolName::new("lookup").expect("tool name"),
        description: "look something up".to_owned(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"q": {"type": "string"}}
        }),
    }])
}

/// What `provider`'s route sends after `option` rewrote its wire.
fn body(
    provider: OpenAIConfig,
    option: impl FnOnce(OpenAiWire) -> OpenAiWire,
) -> serde_json::Value {
    body_with_request(provider, option, request())
}

fn body_with_request(
    provider: OpenAIConfig,
    option: impl FnOnce(OpenAiWire) -> OpenAiWire,
    request: CompletionRequest,
) -> serde_json::Value {
    let encoded = option(provider.completion("gpt-5.2"))
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    json_body(&encoded.request)
}

/// The wire unchanged, as the baseline every assertion compares against.
fn untouched(wire: OpenAiWire) -> OpenAiWire {
    wire
}

/// `option` changes what `changes` sends and leaves `unchanged` as it was —
/// a route without the option is a no-op, not a panic and not a type error.
fn only_on(
    changes: OpenAIConfig,
    unchanged: OpenAIConfig,
    option: impl Fn(OpenAiWire) -> OpenAiWire + Copy,
) {
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

fn chat() -> OpenAIConfig {
    OpenAIConfig::new("sk-test").with_route(Route::Chat)
}

fn responses() -> OpenAIConfig {
    OpenAIConfig::new("sk-test").with_route(Route::Responses)
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

/// The document `frames` rebuild on `reassembler`.
fn rebuilt<R: Reassemble<WireFrame>>(
    mut reassembler: R,
    frames: &[serde_json::Value],
) -> serde_json::Value {
    for frame in frames {
        reassembler.absorb(&WireFrame::Text(frame.to_string()));
    }
    reassembler.finish()
}

/// Each route rebuilds a streamed reply with its own wire's reassembler,
/// and a default one is the chat route's, as the default route is.
#[test]
fn each_route_rebuilds_its_own_document() {
    let chunk = serde_json::json!({
        "id": "chatcmpl-1",
        "object": "chat.completion.chunk",
        "created": 1,
        "model": "model",
        "choices": [{"index": 0, "delta": {"role": "assistant", "content": "hi"}, "finish_reason": "stop"}]
    });
    let completed = serde_json::json!({
        "type": "response.completed",
        "sequence_number": 0,
        "response": {"id": "resp_1", "object": "response", "status": "completed", "output": []}
    });
    for (provider, frame) in [(chat(), &chunk), (responses(), &completed)] {
        let wire = OpenAiWire::new(provider, "model");
        let own = match &wire {
            OpenAiWire::Chat(chat) => rebuilt(chat.reassembler(), std::slice::from_ref(frame)),
            OpenAiWire::Responses(responses) => {
                rebuilt(responses.reassembler(), std::slice::from_ref(frame))
            }
        };
        let routed = rebuilt(wire.reassembler(), std::slice::from_ref(frame));
        assert!(!routed.is_null(), "{routed}");
        assert_eq!(routed, own);
    }
    assert!(matches!(
        OpenAiReassembler::default(),
        OpenAiReassembler::Chat(_)
    ));
    assert_eq!(
        rebuilt(OpenAiReassembler::default(), std::slice::from_ref(&chunk)),
        rebuilt(<Chat as Wire>::Reassembler::default(), &[chunk]),
    );
}
