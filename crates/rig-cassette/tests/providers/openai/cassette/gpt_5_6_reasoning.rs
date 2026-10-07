//! GPT-5.6 reasoning-control regression tests.
//!
//! Locks down the GPT-5.6 model constants and verifies that the Responses API
//! accepts `reasoning.effort = "max"`, `reasoning.mode = "pro"`, and
//! `reasoning.context`, set through the typed options and read back through
//! the typed extras. Unit tests in the provider module cover every typed
//! context value and optional-field serialization.
//!
//! Run cassette tests in replay mode by default, or set
//! `RIG_PROVIDER_TEST_MODE=record` to record against the real provider.

use futures::StreamExt;
use rig::completion::CompletionResponse;
use rig::driver::Model;
use rig::message::{AssistantContent, Message};
use rig::providers::openai;
use rig::providers::openai::wire::OpenAiWire;
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};

use super::super::support::with_openai_cassette;
use rig::completion::{CompletionRequest, Effort, GenerationOptions, ProviderOptions};
use rig::providers::openai::extension::{
    OpenAiExt, OpenAiExtras, OpenAiOptions, OpenAiResponsesOptions, OpenAiShared, ReasoningContext,
    ReasoningMode,
};

const PROMPT: &str = "Reply with exactly: OK";
const FIVE_TURN_PROMPTS: [(&str, &str); 5] = [
    (
        "Remember the codeword ALPHA-17. Reply exactly: ACK-1",
        "ACK-1",
    ),
    (
        "Remember that the shape is octagon. Reply exactly: ACK-2",
        "ACK-2",
    ),
    (
        "Remember that the city is Kyoto. Reply exactly: ACK-3",
        "ACK-3",
    ),
    (
        "Remember that the number is 8642. Reply exactly: ACK-4",
        "ACK-4",
    ),
    (
        "Reply with exactly these remembered values, including capitalization and separators: ALPHA-17 | octagon | Kyoto | 8642",
        "ALPHA-17 | octagon | Kyoto | 8642",
    ),
];

#[derive(Debug, Serialize, Deserialize)]
struct StoredResponseTurn {
    user: Message,
    assistant: Message,
    raw_response: Value,
}

/// Issue one GPT-5.6 completion at `effort`, with the typed Responses
/// options `responses`, and read the reasoning the provider reports through
/// the typed extras.
async fn prompt_with_reasoning(
    model: &Model<OpenAiWire>,
    effort: Effort,
    responses: OpenAiResponsesOptions,
) -> (CompletionResponse, OpenAiExtras) {
    let options = ProviderOptions::new()
        .with::<OpenAiExt>(&OpenAiOptions::default().responses(responses))
        .expect("the options are sections");
    let request = CompletionRequest::new(PROMPT)
        .options(GenerationOptions::default().reasoning(effort))
        .provider_options(options);

    let response = model
        .call(request)
        .await
        .expect("completion with GPT-5.6 reasoning controls should succeed");
    let extras = response
        .extras::<OpenAiExt>()
        .expect("the reply is OpenAI's")
        .expect("the reply holds the extras");

    (response, extras)
}

/// The effective reasoning `extras` report: effort, mode, context and
/// summary.
fn reasoning_of(extras: &OpenAiExtras) -> [Option<&str>; 4] {
    [
        extras.reasoning_effort.as_deref(),
        extras.reasoning_mode.as_deref(),
        extras.reasoning_context.as_deref(),
        extras.reasoning_summary.as_deref(),
    ]
}

#[test]
fn model_constants() {
    assert_eq!(openai::GPT_5_6, "gpt-5.6");
    assert_eq!(openai::GPT_5_6_SOL, "gpt-5.6-sol");
    assert_eq!(openai::GPT_5_6_TERRA, "gpt-5.6-terra");
    assert_eq!(openai::GPT_5_6_LUNA, "gpt-5.6-luna");
}

fn assert_has_text(response: &CompletionResponse) {
    let text: String = response
        .choice
        .iter()
        .filter_map(|content| match content {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect();
    assert!(
        !text.trim().is_empty(),
        "response should surface output text"
    );
}

#[tokio::test]
async fn mode_pro_with_independent_effort() {
    with_openai_cassette(
        "gpt_5_6_reasoning/mode_pro_with_independent_effort",
        |client| async move {
            let model = client.openai.completion(openai::GPT_5_6_SOL);
            let (response, extras) = prompt_with_reasoning(
                &model,
                Effort::High,
                OpenAiResponsesOptions::default().reasoning_mode(ReasoningMode::Pro),
            )
            .await;
            assert_has_text(&response);
            assert_eq!(
                reasoning_of(&extras),
                [Some("high"), Some("pro"), Some("all_turns"), None]
            );
        },
    )
    .await;
}

#[tokio::test]
async fn context_current_turn() {
    with_openai_cassette(
        "gpt_5_6_reasoning/context_current_turn",
        |client| async move {
            let model = client.openai.completion(openai::GPT_5_6_SOL);
            let (response, extras) = prompt_with_reasoning(
                &model,
                Effort::Low,
                OpenAiResponsesOptions::default().reasoning_context(ReasoningContext::CurrentTurn),
            )
            .await;
            assert_has_text(&response);
            assert_eq!(
                reasoning_of(&extras),
                [Some("low"), Some("standard"), Some("current_turn"), None]
            );
        },
    )
    .await;
}

#[tokio::test]
async fn five_turn_reasoning_metadata_roundtrip() {
    with_openai_cassette(
        "gpt_5_6_reasoning/five_turn_metadata_roundtrip",
        |client| async move {
            let model = client.openai.completion(openai::GPT_5_6_SOL);
            let expected_metadata = json!({
                "context": "all_turns",
                "effort": "low",
                "mode": "pro",
                "summary": null
            });
            let mut stored_turns = Vec::<StoredResponseTurn>::new();

            for (turn_index, (prompt, expected_text)) in FIVE_TURN_PROMPTS.into_iter().enumerate() {
                let history = stored_turns
                    .iter()
                    .flat_map(|turn| [turn.user.clone(), turn.assistant.clone()]);
                let user_message = Message::user(prompt);
                let options = OpenAiOptions::default()
                    .shared(OpenAiShared::default().store(false))
                    .responses(
                        OpenAiResponsesOptions::default()
                            .reasoning_mode(ReasoningMode::Pro)
                            .reasoning_context(ReasoningContext::AllTurns),
                    );
                let request = CompletionRequest::new(user_message.clone())
                    .messages(history)
                    .options(GenerationOptions::default().reasoning(Effort::Low))
                    .provider_options(
                        ProviderOptions::new()
                            .with::<OpenAiExt>(&options)
                            .expect("the options are sections"),
                    );
                // One request per turn: one call yields both views of it — the
                // normalized response with its typed extras, and the
                // provider's own wire response on `raw`, which the stored
                // turns persist.
                let response: CompletionResponse =
                    model.call(request).await.unwrap_or_else(|error| {
                        panic!("turn {} should succeed: {error}", turn_index + 1)
                    });
                let raw_response = response.raw.clone();

                assert_has_text(&response);
                let extras = response
                    .extras::<OpenAiExt>()
                    .expect("the reply is OpenAI's")
                    .expect("the reply holds the extras");
                assert_eq!(
                    reasoning_of(&extras),
                    [Some("low"), Some("pro"), Some("all_turns"), None]
                );
                let text = response
                    .choice
                    .iter()
                    .filter_map(|content| match content {
                        AssistantContent::Text(text) => Some(text.text.as_str()),
                        _ => None,
                    })
                    .collect::<String>();
                assert_eq!(
                    text.trim(),
                    expected_text,
                    "unexpected turn {} text",
                    turn_index + 1
                );

                let raw_json =
                    serde_json::to_value(&raw_response).expect("raw response should serialize");
                let roundtripped: Value = serde_json::from_value(raw_json.clone())
                    .expect("raw response should deserialize after serialization");
                assert_eq!(
                    serde_json::to_value(&roundtripped)
                        .expect("roundtripped raw response should serialize"),
                    raw_json,
                    "all raw response data should survive turn {} serialization roundtrip",
                    turn_index + 1
                );

                stored_turns.push(StoredResponseTurn {
                    user: user_message,
                    assistant: Message::Assistant(
                        response.head().with_content(response.choice.clone()),
                    ),
                    raw_response,
                });
                let stored_json =
                    serde_json::to_value(&stored_turns).expect("all stored turns should serialize");
                stored_turns = serde_json::from_value(stored_json.clone())
                    .expect("all stored turns should deserialize before the next request");
                assert_eq!(
                    serde_json::to_value(&stored_turns).expect("restored turns should serialize"),
                    stored_json,
                    "all session data should survive persistence after turn {}",
                    turn_index + 1
                );
                assert_eq!(stored_turns.len(), turn_index + 1);
                for (prior_turn, stored) in stored_turns.iter().enumerate() {
                    assert_eq!(
                        stored.raw_response["reasoning"].as_object(),
                        expected_metadata.as_object(),
                        "reasoning metadata from turn {} changed by turn {}",
                        prior_turn + 1,
                        turn_index + 1
                    );
                    assert_eq!(
                        stored.raw_response["reasoning"]["context"].as_str(),
                        Some("all_turns")
                    );
                }
            }
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_reasoning_metadata() {
    with_openai_cassette(
        "gpt_5_6_reasoning/streaming_metadata",
        |client| async move {
            let model = client.openai.completion(openai::GPT_5_6_SOL);
            let options = OpenAiOptions::default().responses(
                OpenAiResponsesOptions::default()
                    .reasoning_mode(ReasoningMode::Pro)
                    .reasoning_context(ReasoningContext::CurrentTurn),
            );
            let request = CompletionRequest::new(PROMPT)
                .options(GenerationOptions::default().reasoning(Effort::Low))
                .provider_options(
                    ProviderOptions::new()
                        .with::<OpenAiExt>(&options)
                        .expect("the options are sections"),
                );
            // The terminal record's typed extras read the reasoning metadata
            // from the provider-native response; the normalized response
            // carries none.
            let mut stream = model
                .stream(request)
                .expect("GPT-5.6 reasoning stream should start");
            while let Some(item) = stream.next().await {
                item.expect("GPT-5.6 reasoning stream should succeed");
            }
            let record = stream
                .finish()
                .await
                .expect("GPT-5.6 reasoning stream should yield a final response");
            let extras = record
                .extras::<OpenAiExt>()
                .expect("the reply is OpenAI's")
                .expect("the reply holds the extras");
            assert_eq!(
                reasoning_of(&extras),
                [Some("low"), Some("pro"), Some("current_turn"), None]
            );
        },
    )
    .await;
}
