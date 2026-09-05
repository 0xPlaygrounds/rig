//! Cassette-backed coverage for Mistral's listing and embedding capabilities.
//!
//! Three defects live here, all found by comparing what the live API returns
//! and accepts against what rig models:
//!
//! - `EmbeddingsBuilder` chunked by the shared OpenAI cap of 1024 inputs;
//!   Mistral rejects anything over 256. The cap itself is pinned live here;
//!   that rig now chunks *below* it is a unit assertion, because recording a
//!   256-input success would commit ~5 MB of returned vectors to a fixture.
//! - Every Mistral embedding model reported `ndims() == 0`, because the
//!   default-dimension table consulted is OpenAI's and has no Mistral entry.
//! - Model listing dropped `description` and `max_context_length`, both of
//!   which `Model` has slots for.

use rig::streaming::Delta;

use anyhow::Result;
use futures::StreamExt;
use rig::client::{CompletionClient, EmbeddingsClient, ModelListingClient};
use rig::completion::CompletionModel as _;
use rig::embeddings::EmbeddingModel as _;
use rig::providers::mistral;

use super::support::with_mistral_capability_cassette;

/// One more than Mistral's real per-request cap, so a single un-chunked
/// request would be rejected and only correct chunking can succeed.
const OVER_ONE_BATCH: usize = 257;

#[tokio::test]
async fn one_request_over_mistrals_batch_cap_is_rejected() -> Result<()> {
    with_mistral_capability_cassette(
        "capability_edges/one_request_over_mistrals_batch_cap_is_rejected",
        |client| async move {
            let model = client.embedding_model(mistral::embedding::MISTRAL_EMBED);
            // Straight through the model, bypassing the builder's chunking, so
            // the cell pins Mistral's own cap rather than rig's arithmetic.
            let error = model
                .embed_texts((0..OVER_ONE_BATCH).map(|i| format!("document {i}")))
                .await
                // `Vec<Embedding>` is not `Debug`; map the Ok side away.
                .map(|_| ())
                .expect_err("Mistral must reject a batch over its cap");
            assert_batch_cap_rejection(&error);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

#[tokio::test]
async fn mistral_embed_reports_its_real_dimensions() -> Result<()> {
    with_mistral_capability_cassette(
        "capability_edges/mistral_embed_reports_its_real_dimensions",
        |client| async move {
            let model = client.embedding_model(mistral::embedding::MISTRAL_EMBED);
            // The claim under test is the *declared* dimension; the live call
            // is what proves the declaration matches the vectors Mistral
            // actually returns.
            let declared = model.ndims();
            let embedding = model.embed_text("dimension probe").await?;
            assert_declared_matches_returned(declared, embedding.vec.len());
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

#[tokio::test]
async fn list_models_keeps_description_and_context_length() -> Result<()> {
    with_mistral_capability_cassette(
        "capability_edges/list_models_keeps_description_and_context_length",
        |client| async move {
            let models = client.list_models().await?;
            assert_listing_carries_mistrals_fields(&models.data);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

fn assert_batch_cap_rejection(error: &rig::embeddings::EmbeddingError) {
    let rendered = error.to_string();
    assert!(
        rendered.contains("Too many inputs"),
        "the rejection must be Mistral's batch-cap error, not some other failure: {rendered}"
    );
}

fn assert_declared_matches_returned(declared: usize, returned: usize) {
    assert_ne!(
        declared, 0,
        "a model that declares 0 dimensions cannot size a vector store; \
         Mistral's models are absent from OpenAI's dimension table"
    );
    assert_eq!(
        declared, returned,
        "the declared dimension must match the vector Mistral actually returns"
    );
}

fn assert_listing_carries_mistrals_fields(models: &[rig::model::Model]) {
    assert!(!models.is_empty(), "the listing must not be empty");
    assert!(
        models.iter().any(|model| model.description.is_some()),
        "Mistral's listing carries a `description` and `Model` has a slot for it"
    );
    assert!(
        models
            .iter()
            .any(|model| model.context_length.is_some_and(|length| length > 0)),
        "Mistral reports `max_context_length`, which is `Model::context_length`"
    );
}

/// `n > 1` streams as interleaved chunks distinguished only by
/// `choices[].index`. The blocking path answers such a request from candidate
/// 0 alone; the streamed twin used to concatenate every candidate's deltas
/// into one answer that is not any candidate the model produced.
///
/// Recorded against `temperature: 1` so the two candidates genuinely differ —
/// with identical candidates the merge would be invisible.
#[tokio::test]
async fn streaming_with_two_candidates_answers_from_the_first() -> Result<()> {
    with_mistral_capability_cassette(
        "capability_edges/streaming_with_two_candidates_answers_from_the_first",
        |client| async move {
            let model = client.completion_model(mistral::MISTRAL_SMALL);
            let mut stream: rig::streaming::StreamingCompletionResponse = model
                .completion_request("Say one random word.")
                .temperature(1.0)
                .max_tokens(8)
                .additional_params(serde_json::json!({"n": 2}))
                .stream()
                .await?;

            let mut text = String::new();
            while let Some(item) = stream.next().await {
                if let rig::streaming::StreamEvent::BlockDelta {
                    delta: Delta::Text { text: chunk },
                    ..
                } = item?
                {
                    text.push_str(&chunk);
                }
            }
            assert_answers_from_candidate_zero(
                "capability_edges/streaming_with_two_candidates_answers_from_the_first",
                &text,
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// Assert the streamed text is candidate 0 exactly, derived from the
/// cassette's own bytes rather than hard-coded — the model picks different
/// words on every recording.
///
/// The cell's premise is checked too: if the recorded turn stopped carrying a
/// *second*, different candidate, a merge would be invisible and this cell
/// would pass while covering nothing.
fn assert_answers_from_candidate_zero(scenario: &str, streamed: &str) {
    let (first, second) = recorded_candidates(scenario);
    assert_ne!(
        first, second,
        "the recorded turn must carry two different candidates, or a merged stream would look \
         identical to a correct one"
    );
    assert_eq!(
        streamed.trim(),
        first,
        "the stream must deliver candidate 0 whole; interleaving the two yields neither"
    );
}

/// Concatenated `delta.content` per candidate index, read back out of the
/// recorded SSE body.
fn recorded_candidates(scenario: &str) -> (String, String) {
    let raw = std::fs::read_to_string(crate::cassettes::cassette_path("mistral", scenario))
        .expect("cassette should be readable");
    let mut candidates = [String::new(), String::new()];
    for line in raw.split("data: ").skip(1) {
        let payload = line.trim_start();
        let Some(end) = payload.rfind('}') else {
            continue;
        };
        let Ok(chunk) = serde_json::from_str::<serde_json::Value>(&payload[..=end]) else {
            continue;
        };
        let Some(choices) = chunk.get("choices").and_then(|c| c.as_array()) else {
            continue;
        };
        for choice in choices {
            let index = choice.get("index").and_then(serde_json::Value::as_u64);
            let text = choice
                .get("delta")
                .and_then(|delta| delta.get("content"))
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default();
            if let Some(slot) = index.and_then(|i| candidates.get_mut(i as usize)) {
                slot.push_str(text);
            }
        }
    }
    let [first, second] = candidates;
    (first, second)
}

#[derive(Debug, serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
struct SumReport {
    total: i64,
}

/// Mistral accepts a response format beside tools only under
/// `tool_choice: auto` — anything that *forces* a call is a 400:
/// "`json_schema` response type with tools is only compatible with
/// `tool_choice: auto`".
///
/// Rig builds exactly that combination by itself: a structured-output agent
/// defers `response_format` until a tool result exists, then emits it beside
/// the caller's standing `tool_choice`. So the turn *after* the first tool
/// call is rejected — after the tool has already run.
///
/// The cell drives that turn directly rather than through the agent loop: a
/// standing `ToolChoice::Required` forces a tool call on every turn by design,
/// so a loop with it never converges and the max-turns failure would mask the
/// wire error this is about.
#[tokio::test]
async fn a_forced_tool_choice_beside_a_response_format_is_accepted() -> Result<()> {
    with_mistral_capability_cassette(
        "capability_edges/a_forced_tool_choice_beside_a_response_format_is_accepted",
        |client| async move {
            let model = client.completion_model(mistral::MISTRAL_SMALL);
            let response = model
                .completion(
                    model
                        .completion_request("Add 2 and 3, then report the total.")
                        .preamble("Use the add tool, then report the total.".to_string())
                        .messages(turn_one_history())
                        .tools(vec![add_tool_definition()])
                        .tool_choice(rig::message::ToolChoice::Required)
                        .output_schema(schemars::schema_for!(SumReport))
                        .temperature(0.0)
                        .max_tokens(64)
                        .build(),
                )
                .await?;

            assert_turn_two_was_accepted(&response.choice);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The history rig itself would have accumulated after turn 1: the model's
/// tool call and the result of running it.
fn turn_one_history() -> Vec<rig::completion::Message> {
    vec![
        rig::completion::Message::Assistant {
            id: None,
            content: vec![rig::message::AssistantContent::tool_call(
                "call_REDACTED_1",
                "add",
                serde_json::json!({"x": 2, "y": 3}),
            )],
        },
        rig::completion::Message::User {
            content: vec![rig::message::UserContent::tool_result(
                "call_REDACTED_1",
                "add",
                vec![rig::message::ToolResultContent::text("5")],
            )],
        },
    ]
}

fn add_tool_definition() -> rig::completion::ToolDefinition {
    rig::completion::ToolDefinition {
        name: "add".to_string(),
        description: "Add two integers.".to_string(),
        parameters: serde_json::json!({
            "type": "object",
            "properties": {"x": {"type": "integer"}, "y": {"type": "integer"}},
            "required": ["x", "y"]
        }),
    }
}

/// The point is that the turn was *accepted*: before the fix it was a hard
/// 400 and there was no response at all.
fn assert_turn_two_was_accepted(choice: &[rig::completion::AssistantContent]) {
    assert!(
        !choice.is_empty(),
        "the turn after a tool result must reach Mistral and come back with content"
    );
}
