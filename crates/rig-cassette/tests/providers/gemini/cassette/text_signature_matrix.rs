//! Answer-text thought signatures, provoked on purpose and replayed.
//!
//! Gemini may sign an answer part that carries no `thought` flag, and a
//! signature must return inside the part that carried it, never merged into
//! another part. Each cell asks a thinking model a question with no tools,
//! then continues the conversation from the full history, and checks from
//! the recorded bytes that every signature the first reply delivered returns
//! in its own slot: an answer part's on an answer part, a thought's on a
//! thought. Interactions keeps signatures on thought steps only, and the
//! cells check that too.

use futures::StreamExt;
use rig::completion::CompletionRequest;
use rig::message::{AssistantContent, Message};
use rig::providers::gemini;
use rig_test_support::cassette_models::GeminiModels;
use serde_json::{Value, json};

use super::super::support::with_gemini_cassette;
use crate::history_survival::{Dialect, lost_tokens};

const QUESTION: &str = "What is 17 squared? Answer with the number only.";
const FOLLOW_UP: &str = "Add one to that number. Answer with the number only.";

#[derive(Clone, Copy)]
enum Api {
    Interactions,
}

#[derive(Clone, Copy)]
struct Cell {
    model: &'static str,
    api: Api,
    streamed: bool,
}

fn params(cell: Cell) -> Value {
    match (cell.api, cell.model) {
        (Api::Interactions, _) => json!({
            "store": false,
            "generation_config": { "thinking_level": "low", "thinking_summaries": "auto" }
        }),
    }
}

fn request(cell: Cell, history: Vec<Message>) -> CompletionRequest {
    let mut request = CompletionRequest::from(history);
    request.max_tokens = Some(2048);
    request.additional_params = Some(params(cell));
    request
}

async fn turn<W, T>(
    model: &rig::driver::Model<W, T>,
    cell: Cell,
    history: Vec<Message>,
) -> rig::message::AssistantMessage
where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    T: rig::driver::Transport<W>,
{
    let request = request(cell, history);
    let response = if cell.streamed {
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

fn answer(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

async fn conversation<W, T>(model: rig::driver::Model<W, T>, cell: Cell)
where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    T: rig::driver::Transport<W>,
{
    let first = turn(&model, cell, vec![Message::user(QUESTION)]).await;
    assert!(answer(&first.content).contains("289"), "{first:?}");
    let history = vec![
        Message::user(QUESTION),
        Message::Assistant(first),
        Message::user(FOLLOW_UP),
    ];
    let second = turn(&model, cell, history).await;
    assert!(answer(&second.content).contains("290"), "{second:?}");
}

fn assert_recorded(cell: Cell, scenario: &str) {
    let bodies = crate::cassettes::recorded_interaction_bodies("gemini", scenario);
    assert_eq!(bodies.len(), 2, "a question and its continuation");
    let dialect = match cell.api {
        Api::Interactions => Dialect::GeminiInteractions,
    };
    let next: Value = serde_json::from_str(&bodies[1].0).expect("the continuation is JSON");
    let lost = lost_tokens(dialect, &bodies[0].1, &next);
    assert!(
        lost.is_empty(),
        "lost or misplaced in the continuation: {lost:?}"
    );

    let reply = &bodies[0].1;
    match cell.api {
        Api::Interactions => {
            // Signatures live on thought steps only, and return there. A
            // whole reply lists its steps; a stream opens each step with its
            // type (`step.start`) and delivers its signature in `step.delta`.
            let outputs = crate::history_survival::response_documents(reply);
            let mut step_types = std::collections::BTreeMap::new();
            let mut signed_types = Vec::new();
            for document in &outputs {
                for step in document["steps"].as_array().into_iter().flatten() {
                    if step.get("signature").is_some() {
                        signed_types.push(step["type"].clone());
                    }
                }
                match document["event_type"].as_str() {
                    Some("step.start") => {
                        step_types
                            .insert(document["index"].as_u64(), document["step"]["type"].clone());
                    }
                    Some("step.delta") if document["delta"].get("signature").is_some() => {
                        signed_types.push(
                            step_types
                                .get(&document["index"].as_u64())
                                .cloned()
                                .unwrap_or_default(),
                        );
                    }
                    _ => {}
                }
            }
            assert!(
                !signed_types.is_empty(),
                "{scenario}: the reply signed a step"
            );
            assert!(
                signed_types.iter().all(|kind| kind == "thought"),
                "{scenario}: Interactions signs thought steps only: {signed_types:?}"
            );
        }
    }
}

async fn run(client: GeminiModels, cell: Cell) {
    match cell.api {
        Api::Interactions => conversation(client.interactions(cell.model), cell).await,
    }
}

#[tokio::test]
async fn interactions_unary() {
    const CELL: Cell = Cell {
        model: gemini::completion::GEMINI_3_FLASH_PREVIEW,
        api: Api::Interactions,
        streamed: false,
    };
    with_gemini_cassette("text_signature_matrix/interactions_unary", |client| {
        run(client, CELL)
    })
    .await;
    assert_recorded(CELL, "text_signature_matrix/interactions_unary");
}
