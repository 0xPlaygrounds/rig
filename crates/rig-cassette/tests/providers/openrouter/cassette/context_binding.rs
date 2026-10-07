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

use rig::completion::{CompletionRequest, Effort, GenerationOptions, Message, ToolDefinition};
use rig::message::{ToolName, ToolResultContent, UserContent};
use rig::providers::openrouter::extension::{OpenRouterOptions, ProviderPreferences};
use serde_json::{Value, json};

use rig_test_support::cassette_models::OpenAiModels;

use super::super::support::{openrouter_options, with_openrouter_checked_cassette};

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
fn with_params(request: CompletionRequest, route: &str) -> CompletionRequest {
    request
        .options(GenerationOptions::default().reasoning(Effort::High))
        .provider_options(openrouter_options(
            OpenRouterOptions::new().provider(
                ProviderPreferences::new()
                    .only([route])
                    .allow_fallbacks(false),
            ),
        ))
}

async fn two_turns(client: OpenAiModels, route: &'static str) {
    let model = client.completion(MODEL);
    let first = model
        .call(with_params(
            CompletionRequest::new(PROMPT)
                .max_tokens(6000)
                .tool(tool("calc")),
            route,
        ))
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
        .call(with_params(
            CompletionRequest::new(answers)
                .messages([Message::user(PROMPT), Message::Assistant(turn)])
                .max_tokens(6000)
                .tools(vec![tool("calc"), tool("plot")]),
            route,
        ))
        .await
        .expect("OpenRouter accepts the reasoning made under the old tools");
    assert!(!second.choice.is_empty(), "the second turn answers");
}

/// The recorded pair: the tools change, the first reply's signed reasoning
/// goes back exactly as received, both replies are served by `served`, the
/// pinned route, and OpenRouter accepts the second.
fn assert_recorded(root: &std::path::Path, scenario: &str, served: &str) {
    let bodies: Vec<Value> =
        rig_cassette::http::recorded_interaction_bodies(root, "openrouter", scenario)
            .into_iter()
            .map(|(request, _)| serde_json::from_str(&request).expect("a JSON request body"))
            .collect();
    let replies: Vec<(u16, Value)> =
        rig_cassette::http::recorded_statuses_and_bodies(root, "openrouter", scenario)
            .into_iter()
            .map(|(status, body)| (status, serde_json::from_str(&body).unwrap_or(Value::Null)))
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
    let first = &replies[0].1["choices"][0]["message"]["reasoning_details"];
    assert!(
        first
            .as_array()
            .is_some_and(|details| details.iter().any(|detail| detail["signature"]
                .as_str()
                .is_some_and(|sig| !sig.is_empty()))),
        "{scenario}: the first reply's reasoning is signed: {first}"
    );
    assert_eq!(
        &assistant["reasoning_details"], first,
        "{scenario}: the signed reasoning goes back exactly as received"
    );
    for (status, reply) in &replies {
        assert_eq!(*status, 200, "{scenario}: OpenRouter accepts every turn");
        assert_eq!(
            reply["provider"], served,
            "{scenario}: the pinned route serves it"
        );
    }
}

#[tokio::test]
async fn anthropic() {
    with_openrouter_checked_cassette(
        "context_binding/anthropic",
        |client| two_turns(client, "anthropic"),
        |root, scenario| assert_recorded(root, scenario, "Anthropic"),
    )
    .await;
}

#[tokio::test]
async fn amazon_bedrock() {
    with_openrouter_checked_cassette(
        "context_binding/amazon_bedrock",
        |client| two_turns(client, "amazon-bedrock"),
        |root, scenario| assert_recorded(root, scenario, "Amazon Bedrock"),
    )
    .await;
}

#[tokio::test]
async fn google_vertex() {
    with_openrouter_checked_cassette(
        "context_binding/google_vertex",
        |client| two_turns(client, "google-vertex"),
        |root, scenario| assert_recorded(root, scenario, "Google"),
    )
    .await;
}
