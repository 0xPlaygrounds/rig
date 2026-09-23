//! Reasoning through OpenRouter belongs to the upstream family that produced
//! it. A conversation that moves between families inside OpenRouter must not
//! forward one family's signatures to another, and the home family's
//! reasoning must return intact when the conversation comes back.
//!
//! The switch cells run a Claude tool turn, continue on Gemini, then return
//! to Claude, all through OpenRouter. The same-family cells continue a Claude
//! tool turn on another Claude model. Each runs on the chat route, and the
//! `responses_` cells repeat them on OpenRouter's Responses route, where
//! Claude's signature rides beside the reasoning item. Each is checked from
//! the recorded bytes: which signatures and reasoning texts every request
//! carries.

use futures::StreamExt;
use rig::completion::{CompletionModel, CompletionRequest, ToolDefinition};
use rig::message::{AssistantContent, Message, ToolResultContent, UserContent};
use serde_json::{Value, json};

use super::super::support::{BoundOpenRouter, with_openrouter_cassette};

const CLAUDE: &str = "anthropic/claude-haiku-4.5";
const CLAUDE_SONNET: &str = "anthropic/claude-sonnet-4.6";
const GEMINI: &str = "google/gemini-3-flash-preview";
const CODE: &str = "amber-5521";

fn lookup() -> ToolDefinition {
    ToolDefinition {
        name: "lookup_code".to_owned(),
        description: "Return the code stored for a record. Always call it before answering."
            .to_owned(),
        parameters: json!({
            "type": "object",
            "properties": { "record": { "type": "string" } },
            "required": ["record"]
        }),
    }
}

fn request(route: Route, history: Vec<Message>) -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: history,
        documents: vec![],
        tools: vec![lookup()],
        temperature: None,
        max_tokens: Some(4096),
        tool_choice: None,
        // The Responses route takes an effort, not a token budget.
        additional_params: Some(match route {
            Route::Chat => json!({ "reasoning": { "max_tokens": 1024 } }),
            Route::Responses => json!({ "reasoning": { "effort": "low" } }),
        }),
        output_schema: None,
        record_telemetry_content: false,
    }
}

/// Which OpenRouter endpoint a cell drives.
#[derive(Clone, Copy)]
enum Route {
    Chat,
    Responses,
}

async fn turn(
    client: &BoundOpenRouter,
    route: Route,
    model: &str,
    history: Vec<Message>,
    streamed: bool,
) -> Vec<AssistantContent> {
    let request = request(route, history);
    match route {
        Route::Chat => run(client.completion(model), request, streamed).await,
        Route::Responses => run(client.responses(model), request, streamed).await,
    }
}

async fn run<M: CompletionModel>(
    model: M,
    request: CompletionRequest,
    streamed: bool,
) -> Vec<AssistantContent> {
    if !streamed {
        return model
            .completion(request)
            .await
            .expect("the turn completes")
            .choice
            .to_vec();
    }
    let mut stream = model.stream(request).await.expect("the stream opens");
    while let Some(item) = stream.next().await {
        item.expect("a stream item");
    }
    stream.finish().expect("a terminal record").choice.to_vec()
}

fn text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

/// A Claude tool turn answered: the prompt, the reply, and the result.
async fn claude_tool_turn(client: &BoundOpenRouter, route: Route, streamed: bool) -> Vec<Message> {
    let prompt = Message::user("Think it through, then call lookup_code for record alpha.");
    let first = turn(client, route, CLAUDE, vec![prompt.clone()], streamed).await;
    let call = first
        .iter()
        .find_map(|content| match content {
            AssistantContent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .unwrap_or_else(|| panic!("Claude calls lookup_code: {first:?}"));
    vec![
        prompt,
        Message::Assistant {
            id: None,
            content: first.into_iter().collect(),
        },
        Message::User {
            content: vec![UserContent::tool_result_for(
                call.id.clone(),
                call.provider.clone(),
                call.function.name.clone(),
                vec![ToolResultContent::text(format!(
                    "record alpha: code {CODE}"
                ))],
            )]
            .into_iter()
            .collect(),
        },
    ]
}

async fn switch(client: BoundOpenRouter, route: Route, streamed: bool) {
    let mut history = claude_tool_turn(&client, route, streamed).await;
    history.push(Message::user(
        "Without calling any tool, repeat the code you were given, exactly.",
    ));
    let on_gemini = turn(&client, route, GEMINI, history.clone(), streamed).await;
    assert!(text(&on_gemini).contains(CODE), "{on_gemini:?}");
    history.push(Message::Assistant {
        id: None,
        content: on_gemini.into_iter().collect(),
    });
    history.push(Message::user(
        "Without calling any tool, say the code once more, in uppercase.",
    ));
    let home = turn(&client, route, CLAUDE, history, streamed).await;
    assert!(
        text(&home).to_uppercase().contains(&CODE.to_uppercase()),
        "{home:?}"
    );
}

async fn same_family(client: BoundOpenRouter, route: Route, streamed: bool) {
    let mut history = claude_tool_turn(&client, route, streamed).await;
    history.push(Message::user(
        "Now report the code, without calling any tool.",
    ));
    let other = turn(&client, route, CLAUDE_SONNET, history, streamed).await;
    assert!(text(&other).contains(CODE), "{other:?}");
}

/// Every reasoning value in `value`: `opaque` collects signatures and
/// ciphertext, which arrive whole; `all` adds reasoning text, which a stream
/// delivers in fragments. Chat carries `reasoning_details` entries
/// (`reasoning.text`, `reasoning.encrypted`); Responses carries `reasoning`
/// items with the signature beside their text parts.
fn reasoning_values(value: &Value, opaque: &mut Vec<String>, all: &mut Vec<String>) {
    match value {
        Value::Object(object) => {
            if object.get("type").and_then(Value::as_str) == Some("reasoning") {
                for key in ["signature", "encrypted_content"] {
                    if let Some(value) = object.get(key).and_then(Value::as_str)
                        && value.len() > 16
                    {
                        opaque.push(value.to_owned());
                        all.push(value.to_owned());
                    }
                }
                for part in ["content", "summary"]
                    .iter()
                    .filter_map(|key| object.get(*key).and_then(Value::as_array))
                    .flatten()
                {
                    if let Some(text) = part.get("text").and_then(Value::as_str)
                        && text.len() > 16
                    {
                        all.push(text.to_owned());
                    }
                }
            }
            if object
                .get("type")
                .and_then(Value::as_str)
                .is_some_and(|kind| kind.starts_with("reasoning."))
            {
                for key in ["signature", "data", "text", "summary"] {
                    if let Some(value) = object.get(key).and_then(Value::as_str)
                        && value.len() > 16
                    {
                        if matches!(key, "signature" | "data") {
                            opaque.push(value.to_owned());
                        }
                        all.push(value.to_owned());
                    }
                }
            }
            object
                .values()
                .for_each(|child| reasoning_values(child, opaque, all));
        }
        Value::Array(items) => items
            .iter()
            .for_each(|item| reasoning_values(item, opaque, all)),
        _ => {}
    }
}

/// Each exchange's request, as sent, and the opaque and all reasoning values
/// its reply delivered.
struct Exchange {
    request: Value,
    sent: String,
    opaque: Vec<String>,
    all: Vec<String>,
}

fn exchanges(scenario: &str) -> Vec<Exchange> {
    crate::cassettes::recorded_interaction_bodies("openrouter", scenario)
        .into_iter()
        .map(|(sent, reply)| {
            let request: Value = serde_json::from_str(&sent).expect("request JSON");
            let (mut opaque, mut all) = (Vec::new(), Vec::new());
            for document in crate::history_survival::response_documents(&reply) {
                reasoning_values(&document, &mut opaque, &mut all);
            }
            Exchange {
                request,
                sent,
                opaque,
                all,
            }
        })
        .collect()
}

/// The opaque reasoning values a request carries, each whole.
fn carried(request: &Value) -> Vec<String> {
    let (mut opaque, mut all) = (Vec::new(), Vec::new());
    reasoning_values(&request["messages"], &mut opaque, &mut all);
    reasoning_values(&request["input"], &mut opaque, &mut all);
    opaque
}

fn assert_switch_recorded(scenario: &str) {
    let turns = exchanges(scenario);
    assert_eq!(turns.len(), 3, "Claude, Gemini, Claude");
    assert!(
        !turns[0].opaque.is_empty(),
        "Claude delivered signed reasoning"
    );
    assert!(
        turns[0]
            .all
            .iter()
            .all(|value| !turns[1].sent.contains(value.as_str())),
        "no Claude reasoning reaches Gemini"
    );
    let back_home = carried(&turns[2].request);
    assert!(
        turns[0]
            .opaque
            .iter()
            .all(|value| back_home.contains(value)),
        "Claude's signed reasoning returns to Claude intact"
    );
    assert!(
        turns[1]
            .all
            .iter()
            .all(|value| !turns[2].sent.contains(value.as_str())),
        "no Gemini reasoning reaches Claude"
    );
}

fn assert_same_family_recorded(scenario: &str) {
    let turns = exchanges(scenario);
    assert_eq!(turns.len(), 2, "a Claude turn and its continuation");
    assert!(
        !turns[0].opaque.is_empty(),
        "Claude delivered signed reasoning"
    );
    let next = carried(&turns[1].request);
    assert!(
        turns[0].opaque.iter().all(|value| next.contains(value)),
        "the family's signed reasoning continues intact on another Claude model"
    );
}

#[tokio::test]
async fn switch_unary() {
    with_openrouter_cassette("upstream_switch_matrix/switch_unary", |client| {
        switch(client, Route::Chat, false)
    })
    .await;
    assert_switch_recorded("upstream_switch_matrix/switch_unary");
}

#[tokio::test]
async fn switch_streamed() {
    with_openrouter_cassette("upstream_switch_matrix/switch_streamed", |client| {
        switch(client, Route::Chat, true)
    })
    .await;
    assert_switch_recorded("upstream_switch_matrix/switch_streamed");
}

#[tokio::test]
async fn same_family_unary() {
    with_openrouter_cassette("upstream_switch_matrix/same_family_unary", |client| {
        same_family(client, Route::Chat, false)
    })
    .await;
    assert_same_family_recorded("upstream_switch_matrix/same_family_unary");
}

#[tokio::test]
async fn same_family_streamed() {
    with_openrouter_cassette("upstream_switch_matrix/same_family_streamed", |client| {
        same_family(client, Route::Chat, true)
    })
    .await;
    assert_same_family_recorded("upstream_switch_matrix/same_family_streamed");
}

#[tokio::test]
async fn responses_switch_unary() {
    with_openrouter_cassette("upstream_switch_matrix/responses_switch_unary", |client| {
        switch(client, Route::Responses, false)
    })
    .await;
    assert_switch_recorded("upstream_switch_matrix/responses_switch_unary");
}

#[tokio::test]
async fn responses_switch_streamed() {
    with_openrouter_cassette(
        "upstream_switch_matrix/responses_switch_streamed",
        |client| switch(client, Route::Responses, true),
    )
    .await;
    assert_switch_recorded("upstream_switch_matrix/responses_switch_streamed");
}

#[tokio::test]
async fn responses_same_family_unary() {
    with_openrouter_cassette(
        "upstream_switch_matrix/responses_same_family_unary",
        |client| same_family(client, Route::Responses, false),
    )
    .await;
    assert_same_family_recorded("upstream_switch_matrix/responses_same_family_unary");
}

#[tokio::test]
async fn responses_same_family_streamed() {
    with_openrouter_cassette(
        "upstream_switch_matrix/responses_same_family_streamed",
        |client| same_family(client, Route::Responses, true),
    )
    .await;
    assert_same_family_recorded("upstream_switch_matrix/responses_same_family_streamed");
}
