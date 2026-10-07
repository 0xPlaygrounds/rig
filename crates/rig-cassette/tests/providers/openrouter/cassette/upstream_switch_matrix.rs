//! Reasoning through OpenRouter replays only to the model that produced it.
//! A conversation that moves to another model, of any family, sends that
//! model the reasoning as plain text and no signature or ciphertext, and the
//! home model's signed reasoning returns intact when the conversation comes
//! back.
//!
//! The switch cells run a Claude tool turn, continue on Gemini, then return
//! to Claude, all through OpenRouter. The same-family cells continue a Claude
//! tool turn on another Claude model, which is another model too. Each runs on the chat route, and the
//! `responses_` cells repeat them on OpenRouter's Responses route, where
//! Claude's signature rides beside the reasoning item. Each is checked from
//! the recorded bytes: which signatures and reasoning texts every request
//! carries.

use futures::StreamExt;
use rig::completion::{CompletionRequest, ToolDefinition};
use rig::message::{AssistantContent, AssistantMessage, Message, ToolResultContent, UserContent};
use rig_test_support::cassette_models::OpenAiModels;
use serde_json::{Value, json};

use super::super::support::with_openrouter_cassette;

const CLAUDE: &str = "anthropic/claude-haiku-4.5";
const GEMINI: &str = "google/gemini-3-flash-preview";
const CODE: &str = "amber-5521";

fn lookup() -> ToolDefinition {
    ToolDefinition {
        name: rig_core::message::ToolName::new("lookup_code").expect("tool name"),
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
    let mut request = CompletionRequest::from(history);
    request.tools = vec![lookup()];
    request.max_tokens = Some(4096);
    // The Responses route takes an effort, not a token budget.
    request.options = rig::completion::GenerationOptions::default().reasoning(match route {
        Route::Chat => rig::completion::Reasoning::Budget { tokens: 1024 },
        Route::Responses => rig::completion::Reasoning::Effort(rig::completion::Effort::Low),
    });
    request
}

/// Which OpenRouter endpoint a cell drives.
#[derive(Clone, Copy)]
enum Route {
    Chat,
    Responses,
}

async fn turn(
    client: &OpenAiModels,
    route: Route,
    model: &str,
    history: Vec<Message>,
    streamed: bool,
) -> AssistantMessage {
    let request = request(route, history);
    match route {
        Route::Chat => run(client.completion(model), request, streamed).await,
        Route::Responses => run(client.responses(model), request, streamed).await,
    }
}

/// The turn the model answered, with who answered it.
async fn run<W, T>(
    model: rig::driver::Model<W, T>,
    request: CompletionRequest,
    streamed: bool,
) -> AssistantMessage
where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    T: rig::driver::Transport<W>,
{
    let response = if streamed {
        let mut stream = model.stream(request).expect("the stream opens");
        while let Some(item) = stream.next().await {
            item.expect("a stream item");
        }
        stream.finish().await.expect("a terminal record")
    } else {
        model.call(request).await.expect("the turn completes")
    };
    response.head().with_content(response.choice.clone())
}

fn text(turn: &AssistantMessage) -> String {
    turn.content
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

/// A Claude tool turn answered: the prompt, the reply, and the result.
async fn claude_tool_turn(client: &OpenAiModels, route: Route, streamed: bool) -> Vec<Message> {
    let prompt = Message::user("Think it through, then call lookup_code for record alpha.");
    let first = turn(client, route, CLAUDE, vec![prompt.clone()], streamed).await;
    let call = first
        .tool_calls()
        .next()
        .cloned()
        .unwrap_or_else(|| panic!("Claude calls lookup_code: {first:?}"));
    vec![
        prompt,
        Message::Assistant(first),
        Message::User {
            content: vec![UserContent::tool_result(
                call.id.clone(),
                call.function.name.clone(),
                vec![ToolResultContent::text(format!(
                    "record alpha: code {CODE}"
                ))],
            )],
        },
    ]
}

async fn switch(client: OpenAiModels, route: Route, streamed: bool) {
    let mut history = claude_tool_turn(&client, route, streamed).await;
    history.push(Message::user(
        "Without calling any tool, repeat the code you were given, exactly.",
    ));
    let on_gemini = turn(&client, route, GEMINI, history.clone(), streamed).await;
    assert!(text(&on_gemini).contains(CODE), "{on_gemini:?}");
    history.push(Message::Assistant(on_gemini));
    history.push(Message::user(
        "Without calling any tool, say the code once more, in uppercase.",
    ));
    let home = turn(&client, route, CLAUDE, history, streamed).await;
    assert!(
        text(&home).to_uppercase().contains(&CODE.to_uppercase()),
        "{home:?}"
    );
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

/// Each exchange's request, as sent, and the opaque reasoning values its
/// reply delivered.
struct Exchange {
    request: Value,
    sent: String,
    opaque: Vec<String>,
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
            .opaque
            .iter()
            .all(|value| !turns[1].sent.contains(value.as_str())),
        "no Claude signature reaches Gemini"
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
            .opaque
            .iter()
            .all(|value| !turns[2].sent.contains(value.as_str())),
        "no Gemini signature reaches Claude"
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
