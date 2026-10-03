//! Live fact behind Rig replaying OpenRouter Claude turns verbatim: on each
//! route OpenRouter serves Claude Opus 5.5 from, a turn's signed
//! `reasoning_details` go back after the tool list changes, and the request
//! succeeds. So Chat does not bind context on OpenRouter.
//!
//! | # | Cell | Route |
//! |---|------|-------|
//! | 1 | `anthropic` | Anthropic |
//! | 2 | `amazon_bedrock` | Amazon Bedrock |
//! | 3 | `google_vertex` | Google Vertex |

use rig::completion::{CompletionRequest, Message, ToolDefinition};
use rig::message::{ToolName, ToolResultContent, UserContent};
use serde_json::{Value, json};

use rig_test_support::cassette_models::OpenAiModels;

use super::super::support::with_openrouter_cassette;

const MODEL: &str = "anthropic/claude-opus-5.5";

/// A question the model reasons about before verifying with the tool.
const PROMPT: &str = "A train leaves at 3:17pm going 83 km/h; another leaves 41 minutes later \
     going 112 km/h on the same track. Reason through when the second catches up, then call \
     calc once to verify your final expression.";

fn tool(name: &str) -> ToolDefinition {
    ToolDefinition::new(
        ToolName::new(name).expect("a tool name"),
        "Evaluates an arithmetic expression.",
        json!({"type": "object", "properties": {"q": {"type": "string"}}, "required": ["q"]}),
    )
}

/// Reasoning on, pinned to one route with no fallback.
fn params(route: &str) -> Value {
    json!({
        "reasoning": {"effort": "high"},
        "provider": {"only": [route], "allow_fallbacks": false}
    })
}

async fn two_turns(client: OpenAiModels, route: &'static str) {
    let model = client.completion(MODEL);
    let first = model
        .call(
            CompletionRequest::new(PROMPT)
                .max_tokens(6000)
                .tool(tool("calc"))
                .additional_params(params(route)),
        )
        .await
        .expect("the first turn succeeds");
    let message = &first.raw["choices"][0]["message"];
    assert!(
        message["reasoning_details"]
            .as_array()
            .is_some_and(|details| !details.is_empty())
            && message["tool_calls"]
                .as_array()
                .is_some_and(|calls| !calls.is_empty()),
        "the premise is a turn that reasons and calls a tool: {message}"
    );
    let Some(Message::Assistant(turn)) = first.message() else {
        panic!("a turn");
    };
    let answers = Message::User {
        content: turn
            .tool_calls()
            .map(|call| UserContent::ToolResult(call.result(vec![ToolResultContent::text("1.07")])))
            .collect(),
    };
    let second = model
        .call(
            CompletionRequest::new(answers)
                .messages([Message::user(PROMPT), Message::Assistant(turn)])
                .max_tokens(6000)
                .tools(vec![tool("calc"), tool("plot")])
                .additional_params(params(route)),
        )
        .await
        .expect("OpenRouter accepts the reasoning made under the old tools");
    assert!(!second.choice.is_empty(), "the second turn answers");
}

/// The recorded pair: the tools change, the signed reasoning goes back
/// verbatim, and OpenRouter accepts it.
fn assert_recorded(scenario: &str) {
    let bodies: Vec<Value> = crate::cassettes::recorded_interaction_bodies("openrouter", scenario)
        .into_iter()
        .map(|(request, _)| serde_json::from_str(&request).expect("a JSON request body"))
        .collect();
    assert_eq!(bodies.len(), 2, "{scenario}: two turns are recorded");
    let names = |body: &Value| -> Vec<String> {
        body["tools"]
            .as_array()
            .into_iter()
            .flatten()
            .filter_map(|tool| tool["function"]["name"].as_str().map(str::to_owned))
            .collect()
    };
    assert_ne!(
        names(&bodies[0]),
        names(&bodies[1]),
        "{scenario}: the tools change"
    );
    let assistant = bodies[1]["messages"]
        .as_array()
        .into_iter()
        .flatten()
        .find(|message| message["role"] == "assistant")
        .expect("the replayed assistant turn");
    let signed = assistant["reasoning_details"]
        .as_array()
        .into_iter()
        .flatten()
        .any(|detail| {
            detail["signature"]
                .as_str()
                .is_some_and(|sig| !sig.is_empty())
        });
    assert!(
        signed,
        "{scenario}: the signed reasoning goes back verbatim: {assistant}"
    );
    let statuses = crate::cassettes::recorded_statuses_and_bodies("openrouter", scenario);
    assert_eq!(statuses[1].0, 200, "{scenario}: OpenRouter accepts it");
}

#[tokio::test]
async fn anthropic() {
    with_openrouter_cassette("context_binding/anthropic", |client| {
        two_turns(client, "anthropic")
    })
    .await;
    assert_recorded("context_binding/anthropic");
}

#[tokio::test]
async fn amazon_bedrock() {
    with_openrouter_cassette("context_binding/amazon_bedrock", |client| {
        two_turns(client, "amazon-bedrock")
    })
    .await;
    assert_recorded("context_binding/amazon_bedrock");
}

#[tokio::test]
async fn google_vertex() {
    with_openrouter_cassette("context_binding/google_vertex", |client| {
        two_turns(client, "google-vertex")
    })
    .await;
    assert_recorded("context_binding/google_vertex");
}
