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
use rig::completion::{CompletionModel, CompletionRequest};
use rig::message::{AssistantContent, Message};
use rig::providers::gemini;
use serde_json::{Value, json};

use super::super::support::{BoundGemini, with_gemini_cassette};
use crate::history_survival::{Dialect, lost_tokens};

const QUESTION: &str = "What is 17 squared? Answer with the number only.";
const FOLLOW_UP: &str = "Add one to that number. Answer with the number only.";

#[derive(Clone, Copy)]
enum Api {
    GenerateContent,
    Interactions,
}

#[derive(Clone, Copy)]
struct Cell {
    model: &'static str,
    api: Api,
    streamed: bool,
    /// The first reply must sign an answer part (Gemini 3 does; Gemini 2.5
    /// signs only function calls).
    answer_signed: bool,
}

fn params(cell: Cell) -> Value {
    match (cell.api, cell.model) {
        (Api::Interactions, _) => json!({
            "store": false,
            "generation_config": { "thinking_level": "low", "thinking_summaries": "auto" }
        }),
        (Api::GenerateContent, gemini::completion::GEMINI_2_5_FLASH) => json!({
            "generationConfig": { "thinkingConfig": { "includeThoughts": true, "thinkingBudget": 512 } }
        }),
        (Api::GenerateContent, _) => json!({
            "generationConfig": { "thinkingConfig": { "includeThoughts": true, "thinkingLevel": "low" } }
        }),
    }
}

fn request(cell: Cell, history: Vec<Message>) -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: history,
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: Some(2048),
        tool_choice: None,
        additional_params: Some(params(cell)),
        output_schema: None,
        record_telemetry_content: false,
    }
}

async fn turn<M: CompletionModel>(
    model: &M,
    cell: Cell,
    history: Vec<Message>,
) -> Vec<AssistantContent> {
    let request = request(cell, history);
    if !cell.streamed {
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

fn answer(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

async fn conversation<M: CompletionModel>(model: M, cell: Cell) {
    let first = turn(&model, cell, vec![Message::user(QUESTION)]).await;
    assert!(answer(&first).contains("289"), "{first:?}");
    let history = vec![
        Message::user(QUESTION),
        Message::Assistant {
            id: None,
            content: first.into_iter().collect(),
        },
        Message::user(FOLLOW_UP),
    ];
    let second = turn(&model, cell, history).await;
    assert!(answer(&second).contains("290"), "{second:?}");
}

/// Every model part of `request` carrying a signature, as (is a thought,
/// has text) pairs.
fn signed_model_parts(request: &Value) -> Vec<bool> {
    request["contents"]
        .as_array()
        .into_iter()
        .flatten()
        .filter(|content| content["role"] == "model")
        .flat_map(|content| content["parts"].as_array().into_iter().flatten())
        .filter(|part| part.get("thoughtSignature").is_some())
        .map(|part| part["thought"] == json!(true))
        .collect()
}

fn assert_recorded(cell: Cell, scenario: &str) {
    let bodies = crate::cassettes::recorded_interaction_bodies("gemini", scenario);
    assert_eq!(bodies.len(), 2, "a question and its continuation");
    let dialect = match cell.api {
        Api::GenerateContent => Dialect::GeminiGenerateContent,
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
        Api::GenerateContent => {
            // The premise, from the bytes: an answer part carried a signature.
            let answer_signed = crate::history_survival::response_documents(reply)
                .iter()
                .flat_map(|document| {
                    document["candidates"][0]["content"]["parts"]
                        .as_array()
                        .cloned()
                        .unwrap_or_default()
                })
                .any(|part| {
                    part.get("thoughtSignature").is_some() && part["thought"] != json!(true)
                });
            if cell.answer_signed {
                assert!(answer_signed, "{scenario}: the reply signed an answer part");
            }
            // Answer signatures return on answer parts, not on thoughts.
            let replayed = signed_model_parts(&next);
            if answer_signed {
                assert!(
                    replayed.iter().any(|thought| !thought),
                    "{scenario}: an answer part returns signed: {replayed:?}"
                );
            }
        }
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

async fn run(client: BoundGemini, cell: Cell) {
    match cell.api {
        Api::GenerateContent => conversation(client.completion(cell.model), cell).await,
        Api::Interactions => {
            conversation(
                client.map_wire(|config| config.interactions(cell.model)),
                cell,
            )
            .await
        }
    }
}

#[tokio::test]
async fn gemini_3_unary() {
    const CELL: Cell = Cell {
        model: gemini::completion::GEMINI_3_FLASH_PREVIEW,
        api: Api::GenerateContent,
        streamed: false,
        answer_signed: true,
    };
    with_gemini_cassette("text_signature_matrix/gemini_3_unary", |client| {
        run(client, CELL)
    })
    .await;
    assert_recorded(CELL, "text_signature_matrix/gemini_3_unary");
}

#[tokio::test]
async fn gemini_3_streamed() {
    const CELL: Cell = Cell {
        model: gemini::completion::GEMINI_3_FLASH_PREVIEW,
        api: Api::GenerateContent,
        streamed: true,
        answer_signed: true,
    };
    with_gemini_cassette("text_signature_matrix/gemini_3_streamed", |client| {
        run(client, CELL)
    })
    .await;
    assert_recorded(CELL, "text_signature_matrix/gemini_3_streamed");
}

#[tokio::test]
async fn gemini_2_5_unary() {
    const CELL: Cell = Cell {
        model: gemini::completion::GEMINI_2_5_FLASH,
        api: Api::GenerateContent,
        streamed: false,
        answer_signed: false,
    };
    with_gemini_cassette("text_signature_matrix/gemini_2_5_unary", |client| {
        run(client, CELL)
    })
    .await;
    assert_recorded(CELL, "text_signature_matrix/gemini_2_5_unary");
}

#[tokio::test]
async fn gemini_2_5_streamed() {
    const CELL: Cell = Cell {
        model: gemini::completion::GEMINI_2_5_FLASH,
        api: Api::GenerateContent,
        streamed: true,
        answer_signed: false,
    };
    with_gemini_cassette("text_signature_matrix/gemini_2_5_streamed", |client| {
        run(client, CELL)
    })
    .await;
    assert_recorded(CELL, "text_signature_matrix/gemini_2_5_streamed");
}

#[tokio::test]
async fn interactions_unary() {
    const CELL: Cell = Cell {
        model: gemini::completion::GEMINI_3_FLASH_PREVIEW,
        api: Api::Interactions,
        streamed: false,
        answer_signed: false,
    };
    with_gemini_cassette("text_signature_matrix/interactions_unary", |client| {
        run(client, CELL)
    })
    .await;
    assert_recorded(CELL, "text_signature_matrix/interactions_unary");
}

#[tokio::test]
async fn interactions_streamed() {
    const CELL: Cell = Cell {
        model: gemini::completion::GEMINI_3_FLASH_PREVIEW,
        api: Api::Interactions,
        streamed: true,
        answer_signed: false,
    };
    with_gemini_cassette("text_signature_matrix/interactions_streamed", |client| {
        run(client, CELL)
    })
    .await;
    assert_recorded(CELL, "text_signature_matrix/interactions_streamed");
}
