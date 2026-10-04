//! Live facts behind Rig's handling of Claude thinking bound to the request's
//! tools and system prompt (#2703).
//!
//! Claude Opus 5.5 binds a signed thinking block to the tools and system
//! prompt it was made under. In adaptive thinking Rig replays such a turn
//! verbatim and asks Anthropic, under the
//! `thinking-binding-controls-2026-08-01` beta, to drop a block whose
//! binding no longer matches (`block_binding.prefix_mismatch_behavior:
//! "drop_block"`). Thinking that takes no binding (Sonnet 5.5's
//! `between_tools`) gets no `drop_block`, so Rig replays a turn made under
//! other tools as another model's. These cells pin what that rests on:
//!
//! | # | Cell | Fact |
//! |---|------|------|
//! | 1 | `changed_tools_with_drop_block` | After the tools change, the first reply's content goes back exactly as received, with `drop_block` and the beta, and Anthropic accepts it |
//! | 2 | `changed_tools_without_drop_block` | Rig's same request with the binding taken out is refused with a 400 that names the tools list, the beta and `drop_block` |
//! | 3 | `changed_system_with_drop_block` | The same as 1 when only the system prompt changes |
//! | 4 | `changed_system_without_drop_block` | The same as 2 when only the system prompt changes: the refusal names the system prompt |
//! | 5 | `between_tools_after_tool_change` | Under `between_tools`, which takes no binding, Rig's request (the turn as text) is accepted, and the same request with the signed thinking verbatim is refused |
//! | 6 | `thinkingless_tool_loop` | A tool loop opened by a turn with text and a call but no thinking block is accepted under adaptive thinking |
//!
//! The counterfactual requests (2, 4 and 5) are Rig's own encoded requests,
//! changed only where the fact needs, and sent raw through the session's
//! client. Every check runs on the recorded exchanges before a recording is
//! written, so a re-record that loses a fact never replaces its fixture.

use std::path::Path;

use rig::completion::{CompletionRequest, CompletionResponse, Message, ToolDefinition};
use rig::message::{
    AssistantContent, AssistantMessage, CallId, ToolCall, ToolFunction, ToolName,
    ToolResultContent, UserContent,
};
use rig::providers::anthropic::completion::{CLAUDE_OPUS_5_5, CLAUDE_SONNET_5_5};
use rig::wire::{Body, Mode, Operation, Wire};
use rig_core::operation::Completion;
use rig_test_support::cassette_models::AnthropicModels;
use serde_json::{Value, json};

use super::super::support::with_anthropic_checked_cassette;

const BETA: &str = "thinking-binding-controls-2026-08-01";

/// A question the model reasons about before verifying with the tool.
const PROMPT: &str = "A train leaves at 3:17pm going 83 km/h; another leaves 41 minutes later \
     going 112 km/h on the same track. Reason through when the second catches up, then call \
     calc once to verify your final expression.";

const FIRST_SYSTEM: &str = "You are a careful assistant. Show your work.";
const SECOND_SYSTEM: &str = "You are a careful assistant. Answer in one line.";

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition::new(
        ToolName::new(name).expect("a tool name"),
        "Evaluates an arithmetic expression.",
        json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
    )
}

/// Adaptive thinking with its summary shown, at high effort, so the first
/// turn thinks before it calls the tool.
fn thinking() -> Value {
    json!({
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": "high"}
    })
}

/// The first turn on `model`: it thinks, then calls `calc`.
async fn thinking_turn(
    models: &AnthropicModels,
    model: &str,
    system: Option<&str>,
) -> CompletionResponse {
    let mut request = CompletionRequest::new(PROMPT)
        .max_tokens(8000)
        .tool(tool("calc"))
        .additional_params(thinking());
    if let Some(system) = system {
        request = request.preamble(system.to_owned());
    }
    let response = models
        .completion(model)
        .call(request)
        .await
        .expect("the first turn succeeds");
    let kinds: Vec<&str> = response.raw["content"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|block| block["type"].as_str())
        .collect();
    assert!(
        kinds.contains(&"thinking") && kinds.contains(&"tool_use"),
        "the premise is a turn that thinks and calls a tool: {kinds:?}"
    );
    response
}

/// The second request: the first turn and its results, under `tools`,
/// `system` and `params`.
fn second_request(
    first: &CompletionResponse,
    tools: Vec<ToolDefinition>,
    system: Option<&str>,
    params: Value,
) -> CompletionRequest {
    let Some(Message::Assistant(turn)) = first.message() else {
        panic!("a turn");
    };
    let answers = Message::User {
        content: turn
            .tool_calls()
            .map(|call| UserContent::ToolResult(call.result(vec![ToolResultContent::text("1.07")])))
            .collect(),
    };
    let mut request = CompletionRequest::new(answers)
        .messages([Message::user(PROMPT), Message::Assistant(turn)])
        .max_tokens(8000)
        .tools(tools)
        .additional_params(params);
    if let Some(system) = system {
        request = request.preamble(system.to_owned());
    }
    request
}

/// Rig's encoded `request` to `model`: its URI, headers and JSON body.
fn encoded(
    models: &AnthropicModels,
    model: &str,
    request: CompletionRequest,
) -> (String, Vec<(String, String)>, Value) {
    let wire = models.completion(model).wire;
    let request = Completion::prepare(request, &wire.describe()).expect("the request prepares");
    let encoded = wire
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let uri = encoded.request.uri().to_string();
    let headers = encoded
        .request
        .headers()
        .iter()
        .filter_map(|(name, value)| Some((name.to_string(), value.to_str().ok()?.to_owned())))
        .collect();
    let Body::Bytes(bytes) = encoded.request.body() else {
        panic!("a serialized body");
    };
    let body = serde_json::from_slice(&bytes[..]).expect("a JSON body");
    (uri, headers, body)
}

/// Send `body` to `uri` with `headers` through the session's client, and
/// return the status and the JSON reply.
async fn send_raw(uri: &str, headers: &[(String, String)], body: &Value) -> (u16, Value) {
    let client = rig_test_support::cassettes::local_reqwest();
    let http = client.inner().expect("the session's reqwest client");
    let mut request = http.post(uri).body(body.to_string());
    for (name, value) in headers {
        request = request.header(name, value);
    }
    let response = request.send().await.expect("the raw request is sent");
    let status = response.status().as_u16();
    let reply = response.json().await.expect("a JSON reply");
    (status, reply)
}

/// `headers` without the binding beta.
fn without_beta(headers: Vec<(String, String)>) -> Vec<(String, String)> {
    headers
        .into_iter()
        .filter(|(name, _)| name != "anthropic-beta")
        .collect()
}

/// Rig's second request with the binding taken out: no `block_binding`,
/// no beta.
async fn send_without_binding(
    models: &AnthropicModels,
    model: &str,
    request: CompletionRequest,
) -> (u16, Value) {
    let (uri, headers, mut body) = encoded(models, model, request);
    if let Some(thinking) = body["thinking"].as_object_mut() {
        thinking.shift_remove("block_binding");
    }
    send_raw(&uri, &without_beta(headers), &body).await
}

/// The recorded request bodies, statuses and replies of `scenario` under
/// `root`.
fn recorded(root: &Path, scenario: &str) -> (Vec<Value>, Vec<(u16, Value)>) {
    let bodies = rig_cassette::http::recorded_interaction_bodies(root, "anthropic", scenario)
        .into_iter()
        .map(|(request, _)| serde_json::from_str(&request).expect("a JSON request body"))
        .collect();
    let replies = rig_cassette::http::recorded_statuses_and_bodies(root, "anthropic", scenario)
        .into_iter()
        .map(|(status, body)| (status, serde_json::from_str(&body).unwrap_or(Value::Null)))
        .collect();
    (bodies, replies)
}

fn tool_names(body: &Value) -> Vec<&str> {
    body["tools"]
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|tool| tool["name"].as_str())
        .collect()
}

/// The assertions for a drop_block cell: the context changes as `changed`
/// says, the second request carries the binding and its beta and sends the
/// first reply's content back exactly, and Anthropic accepts it.
fn assert_dropped(root: &Path, scenario: &str, changed: impl Fn(&Value, &Value) -> bool) {
    let (bodies, replies) = recorded(root, scenario);
    assert_eq!(bodies.len(), 2, "{scenario}: two turns are recorded");
    assert!(
        changed(&bodies[0], &bodies[1]),
        "{scenario}: the context changes"
    );
    assert_eq!(
        bodies[1]["thinking"]["block_binding"]["prefix_mismatch_behavior"], "drop_block",
        "{scenario}: the second request asks Anthropic to drop a mismatched block"
    );
    assert_eq!(
        bodies[1]["messages"][1]["content"], replies[0].1["content"],
        "{scenario}: the first reply's content goes back exactly as received"
    );
    let headers = rig_cassette::http::recorded_request_header_pairs(root, "anthropic", scenario);
    assert!(
        headers[1]
            .iter()
            .any(|(name, value)| name == "anthropic-beta" && value.contains(BETA)),
        "{scenario}: the second request carries the binding beta"
    );
    assert_eq!(replies[1].0, 200, "{scenario}: Anthropic accepts it");
}

/// The assertions for a counterfactual cell: the second request is the
/// first reply verbatim with no binding, and Anthropic refuses it naming
/// `reason`.
fn assert_refused(root: &Path, scenario: &str, reason: &str) {
    let (bodies, replies) = recorded(root, scenario);
    assert_eq!(bodies.len(), 2, "{scenario}: two requests are recorded");
    assert!(
        bodies[1]["thinking"].get("block_binding").is_none(),
        "{scenario}: the second request carries no binding"
    );
    assert_eq!(
        bodies[1]["messages"][1]["content"], replies[0].1["content"],
        "{scenario}: the first reply's content goes back exactly as received"
    );
    let headers = rig_cassette::http::recorded_request_header_pairs(root, "anthropic", scenario);
    assert!(
        !headers[1]
            .iter()
            .any(|(name, value)| name == "anthropic-beta" && value.contains(BETA)),
        "{scenario}: the second request carries no binding beta"
    );
    assert_eq!(
        replies[1].0, 400,
        "{scenario}: the mismatched binding is refused"
    );
    let message = replies[1].1["error"]["message"]
        .as_str()
        .unwrap_or_default();
    assert!(
        message.contains(reason) && message.contains(BETA) && message.contains("drop_block"),
        "{scenario}: the refusal names {reason:?}, the beta and drop_block: {message}"
    );
}

fn tools_differ(first: &Value, second: &Value) -> bool {
    tool_names(first) != tool_names(second) && first["system"] == second["system"]
}

fn system_differs(first: &Value, second: &Value) -> bool {
    tool_names(first) == tool_names(second) && first["system"] != second["system"]
}

#[tokio::test]
async fn changed_tools_with_drop_block() {
    with_anthropic_checked_cassette(
        "context_binding/changed_tools_with_drop_block",
        |models| async move {
            let first = thinking_turn(&models, CLAUDE_OPUS_5_5, None).await;
            let request =
                second_request(&first, vec![tool("calc"), tool("plot")], None, thinking());
            let reply = models
                .completion(CLAUDE_OPUS_5_5)
                .call(request)
                .await
                .expect("Anthropic accepts thinking bound to the old tools under drop_block");
            assert!(!reply.choice.is_empty(), "the second turn answers");
        },
        |root, scenario| assert_dropped(root, scenario, tools_differ),
    )
    .await;
}

#[tokio::test]
async fn changed_tools_without_drop_block() {
    with_anthropic_checked_cassette(
        "context_binding/changed_tools_without_drop_block",
        |models| async move {
            let first = thinking_turn(&models, CLAUDE_OPUS_5_5, None).await;
            let request =
                second_request(&first, vec![tool("calc"), tool("plot")], None, thinking());
            send_without_binding(&models, CLAUDE_OPUS_5_5, request).await;
        },
        |root, scenario| {
            assert_refused(root, scenario, "`tools` list");
            let (bodies, _) = recorded(root, scenario);
            assert!(
                tools_differ(&bodies[0], &bodies[1]),
                "{scenario}: the tools change"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn changed_system_with_drop_block() {
    with_anthropic_checked_cassette(
        "context_binding/changed_system_with_drop_block",
        |models| async move {
            let first = thinking_turn(&models, CLAUDE_OPUS_5_5, Some(FIRST_SYSTEM)).await;
            let request =
                second_request(&first, vec![tool("calc")], Some(SECOND_SYSTEM), thinking());
            let reply = models
                .completion(CLAUDE_OPUS_5_5)
                .call(request)
                .await
                .expect(
                    "Anthropic accepts thinking bound to the old system prompt under drop_block",
                );
            assert!(!reply.choice.is_empty(), "the second turn answers");
        },
        |root, scenario| assert_dropped(root, scenario, system_differs),
    )
    .await;
}

#[tokio::test]
async fn changed_system_without_drop_block() {
    with_anthropic_checked_cassette(
        "context_binding/changed_system_without_drop_block",
        |models| async move {
            let first = thinking_turn(&models, CLAUDE_OPUS_5_5, Some(FIRST_SYSTEM)).await;
            let request =
                second_request(&first, vec![tool("calc")], Some(SECOND_SYSTEM), thinking());
            send_without_binding(&models, CLAUDE_OPUS_5_5, request).await;
        },
        |root, scenario| {
            assert_refused(root, scenario, "system");
            let (bodies, _) = recorded(root, scenario);
            assert!(
                system_differs(&bodies[0], &bodies[1]),
                "{scenario}: the system prompt changes"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn between_tools_after_tool_change() {
    with_anthropic_checked_cassette(
        "context_binding/between_tools_after_tool_change",
        |models| async move {
            let first = thinking_turn(&models, CLAUDE_SONNET_5_5, None).await;
            let off = json!({"thinking": {"type": "between_tools"}});
            let request = second_request(&first, vec![tool("calc"), tool("plot")], None, off);
            let (uri, headers, body) = encoded(&models, CLAUDE_SONNET_5_5, request.clone());
            let reply = models
                .completion(CLAUDE_SONNET_5_5)
                .call(request)
                .await
                .expect("Rig's request, with the turn as text, is accepted");
            assert!(!reply.choice.is_empty(), "the second turn answers");
            // The same request with the signed thinking verbatim.
            let mut verbatim = body;
            verbatim["messages"][1]["content"] = first.raw["content"].clone();
            send_raw(&uri, &headers, &verbatim).await;
        },
        |root, scenario| {
            let (bodies, replies) = recorded(root, scenario);
            assert_eq!(bodies.len(), 3, "{scenario}: three requests are recorded");
            assert!(
                tools_differ(&bodies[0], &bodies[1]),
                "{scenario}: the tools change"
            );
            assert_eq!(bodies[1]["thinking"], json!({"type": "between_tools"}));
            let rig_turn = &bodies[1]["messages"][1]["content"];
            assert!(
                rig_turn
                    .as_array()
                    .into_iter()
                    .flatten()
                    .all(|block| block["type"] != "thinking"),
                "{scenario}: Rig sends the turn made under other tools as text: {rig_turn}"
            );
            assert_eq!(replies[1].0, 200, "{scenario}: Rig's request is accepted");
            assert_eq!(bodies[2]["messages"][1]["content"], replies[0].1["content"]);
            assert_eq!(
                replies[2].0, 400,
                "{scenario}: the signed thinking verbatim is refused"
            );
            let message = replies[2].1["error"]["message"]
                .as_str()
                .unwrap_or_default();
            assert!(
                message.contains("bound to a different conversation"),
                "{scenario}: the refusal is the binding's: {message}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn thinkingless_tool_loop() {
    with_anthropic_checked_cassette(
        "thinking/thinkingless_tool_loop",
        |models| async move {
            // A turn another model made: text, then a call, and no thinking.
            let call = ToolCall::new(
                CallId::from_wire("toolu_x1"),
                ToolFunction::new(
                    ToolName::new("calc").expect("a tool name"),
                    json!({"q": "17*23"}),
                ),
            );
            let turn = AssistantMessage::new(vec![
                AssistantContent::text("Let me compute."),
                AssistantContent::ToolCall(call.clone()),
            ]);
            let request = CompletionRequest::new(Message::User {
                content: vec![UserContent::ToolResult(
                    call.result(vec![ToolResultContent::text("391")]),
                )],
            })
            .messages([
                Message::user("What is 17*23? Use calc."),
                Message::Assistant(turn),
            ])
            .max_tokens(3000)
            .tool(tool("calc"))
            .additional_params(json!({"thinking": {"type": "adaptive"}}));
            let reply = models
                .completion(CLAUDE_OPUS_5_5)
                .call(request)
                .await
                .expect("a tool loop opened without thinking is accepted under adaptive thinking");
            assert!(!reply.choice.is_empty(), "the model answers");
        },
        |root, scenario| {
            let (bodies, replies) = recorded(root, scenario);
            assert_eq!(bodies.len(), 1, "{scenario}: one request is recorded");
            let body = &bodies[0];
            assert_eq!(body["thinking"]["type"], "adaptive", "thinking is on");
            let turn = body["messages"][1]["content"]
                .as_array()
                .cloned()
                .unwrap_or_default();
            let kinds: Vec<&str> = turn
                .iter()
                .filter_map(|block| block["type"].as_str())
                .collect();
            assert_eq!(
                kinds,
                ["text", "tool_use"],
                "{scenario}: the open loop's turn is text and a call, with no thinking"
            );
            assert_eq!(turn[1]["id"], "toolu_x1");
            assert_eq!(
                body["messages"][2]["content"][0]["type"], "tool_result",
                "{scenario}: the loop is open: the next message answers the call"
            );
            assert_eq!(body["messages"][2]["content"][0]["tool_use_id"], "toolu_x1");
            assert_eq!(replies[0].0, 200, "{scenario}: Anthropic accepts it");
        },
    )
    .await;
}
