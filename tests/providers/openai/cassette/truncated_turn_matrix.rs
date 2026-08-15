//! Edge matrix for a **truncated** OpenAI turn that carried no content.
//!
//! **Bug.** A reasoning model whose output-token cap is spent entirely on
//! hidden reasoning answers with an empty message and the provider's own
//! diagnostic attached:
//!
//! ```json
//! {"message": {"role": "assistant", "content": ""}, "finish_reason": "length"}
//! ```
//!
//! The Chat Completions normalizer rejected that wholesale —
//! `Response contained no message or tool call (empty)` — throwing away both
//! the finish reason and the usage, so a caller could not tell "you hit the
//! cap" from "the provider misbehaved". The other three paths for the same
//! situation already got it right:
//!
//! * the Responses API deliberately lets a contentless `status: incomplete`
//!   turn through so its `Length` reaches the caller;
//! * the chat-completions *streaming* path yields a terminal record with the
//!   reason regardless of what the stream produced;
//! * and the same reasoning model driven through the Responses surface
//!   therefore behaved differently from the Chat Completions surface for one
//!   and the same rig-level request.
//!
//! The fix gives the unary path the same rule, named once on
//! [`FinishReason::truncated_output`]: a turn the provider *cut short* may be
//! empty, a turn that *ran to completion* may not.
//!
//! **How these cells fail on `origin/main`.** The `chat_*_budget_exhausted`
//! cells replay their recorded fixture into `main`'s normalizer and get
//! `CompletionError(ResponseError("Response contained no message or tool call
//! (empty)"))` where the cell expects `FinishReason::Length`.
//!
//! | # | cell | surface | transport | model | shape | status |
//! |---|------|---------|-----------|-------|-------|--------|
//! | 1 | `chat_blocking_reasoning_budget_exhausted` | chat | blocking | gpt-5-nano | empty + length | recorded |
//! | 2 | `chat_blocking_o4_mini_budget_exhausted` | chat | blocking | o4-mini | empty + length | recorded |
//! | 3 | `chat_blocking_gpt_5_1_budget_exhausted` | chat | blocking | gpt-5.1 (high effort) | empty + length | recorded |
//! | 4 | `chat_blocking_usage_survives_the_empty_turn` | chat | blocking | gpt-5-nano | usage | recorded |
//! | 5 | `chat_blocking_raw_and_normalized_agree` | chat | blocking | gpt-5-nano | raw vs normalized | recorded |
//! | 6 | `chat_blocking_tools_present_budget_exhausted` | chat | blocking | gpt-5-nano | tools in request | recorded |
//! | 7 | `chat_blocking_agent_reports_the_truncation` | chat | blocking | gpt-5-nano | agent-level diagnostic | recorded |
//! | 8 | `chat_blocking_partial_text_truncation` | chat | blocking | gpt-4o-mini | non-empty + length | recorded |
//! | 9 | `chat_streaming_reasoning_budget_exhausted` | chat | streaming | gpt-5-nano | empty + length | recorded |
//! | 10 | `chat_streaming_partial_text_truncation` | chat | streaming | gpt-4o-mini | non-empty + length | recorded |
//! | 11 | `chat_transports_agree_on_truncation` | chat | both | gpt-5-nano | cross-transport | recorded |
//! | 12 | `chat_blocking_completed_turn_is_unaffected` | chat | blocking | gpt-4o-mini | non-empty + stop | recorded |
//! | 13 | `responses_blocking_reasoning_budget_exhausted` | responses | blocking | gpt-5-nano | control | recorded |
//! | 14 | `responses_streaming_reasoning_budget_exhausted` | responses | streaming | gpt-5-nano | control | recorded |
//! | 15 | `responses_blocking_partial_text_truncation` | responses | blocking | gpt-4o-mini | control | recorded |
//! | 16 | `cross_surface_truncation_parity` | both | blocking | gpt-5-nano | cross-API | recorded |
//!
//! Unit cells — the accept/reject rule across the whole finish-reason
//! vocabulary, which live traffic cannot enumerate (no prompt reliably
//! produces an empty `content_filter` or an empty `tool_calls` turn) — live
//! beside the fix in
//! `crates/rig-core/src/providers/internal/openai_chat_completions_compatible.rs`
//! (`empty_choice_*`) and `crates/rig-core/src/completion/request.rs`
//! (`truncated_output_*`).
//!
//! Every cell re-reads its own fixture and fails if the recorded turn stopped
//! having the shape the cell is about.

use rig::client::completion::CompletionClient;
use rig::completion::{CompletionModel, FinishReason, NormalizeCompletionResponse};
use rig::prelude::*;
use rig::providers::openai;
use rig::telemetry::ProviderResponseExt;
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::with_openai_truncation_cassette;
use crate::cassettes;
use crate::support::{Adder, assistant_text_response, collect_text_and_terminal};

/// OpenAI's documented floor for the field, and far below what any reasoning
/// model needs to say anything — so the whole budget goes to hidden reasoning
/// and the turn comes back empty.
const TINY_CAP: u64 = 16;
const LONG_PROMPT: &str = "Write a 500 word essay about maple trees.";

// ---------------------------------------------------------------------------
// Chat Completions — the surface that threw the diagnostic away.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_blocking_reasoning_budget_exhausted() {
    const SCENARIO: &str = "truncated_turn_matrix/chat_blocking_reasoning_budget_exhausted";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/chat_blocking_reasoning_budget_exhausted",
        |client| async move {
            let model = client.completions_api().completion_model("gpt-5-nano");
            let request = model
                .completion_request(LONG_PROMPT)
                .max_tokens(TINY_CAP)
                .build();

            let response = model
                .completion(request)
                .await
                .expect("a truncated turn is a diagnostic, not a malformed response");

            assert_eq!(response.finish_reason(), Some(FinishReason::Length));
            assert!(response.choice.is_empty());
        },
    )
    .await;

    assert_recorded_empty_truncated_chat_turn(SCENARIO);
}

#[tokio::test]
async fn chat_blocking_o4_mini_budget_exhausted() {
    const SCENARIO: &str = "truncated_turn_matrix/chat_blocking_o4_mini_budget_exhausted";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/chat_blocking_o4_mini_budget_exhausted",
        |client| async move {
            let model = client.completions_api().completion_model("o4-mini");
            let request = model
                .completion_request(LONG_PROMPT)
                .max_tokens(TINY_CAP)
                .build();

            let response = model.completion(request).await.expect("truncated turn");

            assert_eq!(response.finish_reason(), Some(FinishReason::Length));
        },
    )
    .await;

    assert_recorded_empty_truncated_chat_turn(SCENARIO);
}

#[tokio::test]
async fn chat_blocking_gpt_5_1_budget_exhausted() {
    const SCENARIO: &str = "truncated_turn_matrix/chat_blocking_gpt_5_1_budget_exhausted";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/chat_blocking_gpt_5_1_budget_exhausted",
        |client| async move {
            let model = client.completions_api().completion_model("gpt-5.1");
            // gpt-5.1 reasons only as much as it judges necessary; at its
            // default effort this prompt yields partial *text* under the same
            // cap (cell 8's shape). Asking for high effort makes the reasoning
            // consume the whole budget, which is the shape this cell is about.
            let request = model
                .completion_request(LONG_PROMPT)
                .max_tokens(TINY_CAP)
                .additional_params(json!({ "reasoning_effort": "high" }))
                .build();

            let response = model.completion(request).await.expect("truncated turn");

            assert_eq!(response.finish_reason(), Some(FinishReason::Length));
        },
    )
    .await;

    assert_recorded_empty_truncated_chat_turn(SCENARIO);
}

/// The usage went out with the error too — including the reasoning-token count
/// that explains where the budget went.
#[tokio::test]
async fn chat_blocking_usage_survives_the_empty_turn() {
    const SCENARIO: &str = "truncated_turn_matrix/chat_blocking_usage_survives_the_empty_turn";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/chat_blocking_usage_survives_the_empty_turn",
        |client| async move {
            let model = client.completions_api().completion_model("gpt-5-nano");
            let request = model
                .completion_request(LONG_PROMPT)
                .max_tokens(TINY_CAP)
                .build();

            let response = model.completion(request).await.expect("truncated turn");

            assert!(response.usage.input_tokens > 0);
            assert_eq!(response.usage.output_tokens, TINY_CAP);
            assert!(
                response.usage.reasoning_tokens > 0,
                "the reasoning tokens are what consumed the cap: {:?}",
                response.usage
            );
        },
    )
    .await;

    assert_recorded_empty_truncated_chat_turn(SCENARIO);
}

#[tokio::test]
async fn chat_blocking_raw_and_normalized_agree() {
    const SCENARIO: &str = "truncated_turn_matrix/chat_blocking_raw_and_normalized_agree";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/chat_blocking_raw_and_normalized_agree",
        |client| async move {
            let model = client.completions_api().completion_model("gpt-5-nano");
            let request = model
                .completion_request(LONG_PROMPT)
                .max_tokens(TINY_CAP)
                .build();

            let raw = model.raw_completion(request).await.expect("raw turn");
            assert_eq!(
                raw.choices
                    .first()
                    .map(|choice| choice.finish_reason.as_str()),
                Some("length")
            );
            assert_eq!(raw.get_text_response(), None);

            let normalized: rig::completion::CompletionResponse =
                raw.normalize("openai").expect("normalization");
            assert_eq!(normalized.finish_reason(), Some(FinishReason::Length));
        },
    )
    .await;

    assert_recorded_empty_truncated_chat_turn(SCENARIO);
}

/// A turn cut short before it could call anything: the request carried tools,
/// so the empty choice is not "the model chose to say nothing".
#[tokio::test]
async fn chat_blocking_tools_present_budget_exhausted() {
    const SCENARIO: &str = "truncated_turn_matrix/chat_blocking_tools_present_budget_exhausted";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/chat_blocking_tools_present_budget_exhausted",
        |client| async move {
            let model = client.completions_api().completion_model("gpt-5-nano");
            let request = model
                .completion_request("What is 3 + 4? Use the add tool.")
                .max_tokens(TINY_CAP)
                .tool(rig::tool::tool_definition(&Adder))
                .build();

            let response = model.completion(request).await.expect("truncated turn");

            assert_eq!(response.finish_reason(), Some(FinishReason::Length));
            assert!(response.choice.is_empty());
        },
    )
    .await;

    assert_recorded_empty_truncated_chat_turn(SCENARIO);
}

/// At agent level the truncation must still be legible rather than an opaque
/// provider failure.
#[tokio::test]
async fn chat_blocking_agent_reports_the_truncation() {
    const SCENARIO: &str = "truncated_turn_matrix/chat_blocking_agent_reports_the_truncation";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/chat_blocking_agent_reports_the_truncation",
        |client| async move {
            let agent = client
                .completions_api()
                .agent("gpt-5-nano")
                .max_tokens(TINY_CAP)
                .build();

            // rig-agent already has a purpose-built message for this state;
            // before the fix it was unreachable, because the provider layer
            // failed first with the opaque `Response contained no message or
            // tool call (empty)`.
            let error = agent
                .prompt(LONG_PROMPT)
                .await
                .expect_err("an answerless turn is still an error at agent level");
            let message = error.to_string();

            assert!(
                message.contains("finish_reason=Length") && message.contains("max_tokens"),
                "the agent must name the budget as the cause: {message}"
            );
            assert!(
                !message.contains("no message or tool call"),
                "the opaque provider-layer error must no longer be what surfaces: {message}"
            );
        },
    )
    .await;

    assert_recorded_empty_truncated_chat_turn(SCENARIO);
}

/// Control: a *non-reasoning* model truncated mid-sentence keeps partial text,
/// which always worked — the fix must not disturb it.
#[tokio::test]
async fn chat_blocking_partial_text_truncation() {
    const SCENARIO: &str = "truncated_turn_matrix/chat_blocking_partial_text_truncation";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/chat_blocking_partial_text_truncation",
        |client| async move {
            let model = client
                .completions_api()
                .completion_model(openai::GPT_4O_MINI);
            let request = model
                .completion_request(LONG_PROMPT)
                .max_tokens(TINY_CAP)
                .build();

            let response = model.completion(request).await.expect("truncated turn");

            assert_eq!(response.finish_reason(), Some(FinishReason::Length));
            assert!(
                assistant_text_response(&response.choice)
                    .is_some_and(|text| !text.trim().is_empty())
            );
        },
    )
    .await;

    assert_recorded_nonempty_truncated_chat_turn(SCENARIO);
}

// ---------------------------------------------------------------------------
// Streaming: the transport that was already right, pinned as the parity anchor.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_streaming_reasoning_budget_exhausted() {
    const SCENARIO: &str = "truncated_turn_matrix/chat_streaming_reasoning_budget_exhausted";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/chat_streaming_reasoning_budget_exhausted",
        |client| async move {
            let model = client.completions_api().completion_model("gpt-5-nano");
            let request = model
                .completion_request(LONG_PROMPT)
                .max_tokens(TINY_CAP)
                .build();

            let stream = model.stream(request).await.expect("stream should connect");
            let (text, terminal) = collect_text_and_terminal(stream).await;

            assert!(text.is_empty());
            assert_eq!(
                terminal.expect("terminal record").finish_reason,
                Some(FinishReason::Length)
            );
        },
    )
    .await;

    assert_recorded_truncated_chat_stream(SCENARIO);
}

#[tokio::test]
async fn chat_streaming_partial_text_truncation() {
    const SCENARIO: &str = "truncated_turn_matrix/chat_streaming_partial_text_truncation";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/chat_streaming_partial_text_truncation",
        |client| async move {
            let model = client
                .completions_api()
                .completion_model(openai::GPT_4O_MINI);
            let request = model
                .completion_request(LONG_PROMPT)
                .max_tokens(TINY_CAP)
                .build();

            let stream = model.stream(request).await.expect("stream should connect");
            let (text, terminal) = collect_text_and_terminal(stream).await;

            assert!(!text.is_empty());
            assert_eq!(
                terminal.expect("terminal record").finish_reason,
                Some(FinishReason::Length)
            );
        },
    )
    .await;

    assert_recorded_truncated_chat_stream(SCENARIO);
}

/// The parity claim itself: one prompt, one cap, both transports, one
/// cassette — the caller must learn the same thing either way.
#[tokio::test]
async fn chat_transports_agree_on_truncation() {
    const SCENARIO: &str = "truncated_turn_matrix/chat_transports_agree_on_truncation";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/chat_transports_agree_on_truncation",
        |client| async move {
            let model = client.completions_api().completion_model("gpt-5-nano");

            let blocking = model
                .completion(
                    model
                        .completion_request(LONG_PROMPT)
                        .max_tokens(TINY_CAP)
                        .build(),
                )
                .await
                .expect("blocking truncated turn");

            let stream = model
                .stream(
                    model
                        .completion_request(LONG_PROMPT)
                        .max_tokens(TINY_CAP)
                        .build(),
                )
                .await
                .expect("streaming truncated turn");
            let (streamed_text, terminal) = collect_text_and_terminal(stream).await;

            assert_eq!(
                blocking.finish_reason(),
                terminal.and_then(|terminal| terminal.finish_reason)
            );
            assert_eq!(
                assistant_text_response(&blocking.choice).unwrap_or_default(),
                streamed_text
            );
        },
    )
    .await;

    assert_recorded_empty_truncated_chat_turn(SCENARIO);
    assert_recorded_truncated_chat_stream(SCENARIO);
}

/// Control: an ordinary completed turn still normalizes exactly as before.
#[tokio::test]
async fn chat_blocking_completed_turn_is_unaffected() {
    const SCENARIO: &str = "truncated_turn_matrix/chat_blocking_completed_turn_is_unaffected";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/chat_blocking_completed_turn_is_unaffected",
        |client| async move {
            let model = client
                .completions_api()
                .completion_model(openai::GPT_4O_MINI);
            let request = model.completion_request("Reply with the word OK.").build();

            let response = model.completion(request).await.expect("completed turn");

            assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
            assert!(!response.choice.is_empty());
        },
    )
    .await;

    assert_recorded_completed_chat_turn(SCENARIO);
}

// ---------------------------------------------------------------------------
// Responses API — the rule this fix copied, pinned so it stays the reference.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn responses_blocking_reasoning_budget_exhausted() {
    const SCENARIO: &str = "truncated_turn_matrix/responses_blocking_reasoning_budget_exhausted";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/responses_blocking_reasoning_budget_exhausted",
        |client| async move {
            let model = client.completion_model("gpt-5-nano");
            let request = model
                .completion_request(LONG_PROMPT)
                .max_tokens(TINY_CAP)
                .build();

            let response = model.completion(request).await.expect("truncated turn");

            assert_eq!(response.finish_reason(), Some(FinishReason::Length));
        },
    )
    .await;

    assert_recorded_incomplete_responses_turn(SCENARIO);
}

#[tokio::test]
async fn responses_streaming_reasoning_budget_exhausted() {
    const SCENARIO: &str = "truncated_turn_matrix/responses_streaming_reasoning_budget_exhausted";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/responses_streaming_reasoning_budget_exhausted",
        |client| async move {
            let model = client.completion_model("gpt-5-nano");
            let request = model
                .completion_request(LONG_PROMPT)
                .max_tokens(TINY_CAP)
                .build();

            let stream = model.stream(request).await.expect("stream should connect");
            let (_, terminal) = collect_text_and_terminal(stream).await;

            assert_eq!(
                terminal.expect("terminal record").finish_reason,
                Some(FinishReason::Length)
            );
        },
    )
    .await;

    assert_recorded_incomplete_responses_turn(SCENARIO);
}

#[tokio::test]
async fn responses_blocking_partial_text_truncation() {
    const SCENARIO: &str = "truncated_turn_matrix/responses_blocking_partial_text_truncation";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/responses_blocking_partial_text_truncation",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O_MINI);
            let request = model
                .completion_request(LONG_PROMPT)
                .max_tokens(TINY_CAP)
                .build();

            let response = model.completion(request).await.expect("truncated turn");

            assert_eq!(response.finish_reason(), Some(FinishReason::Length));
            assert!(
                assistant_text_response(&response.choice)
                    .is_some_and(|text| !text.trim().is_empty())
            );
        },
    )
    .await;

    assert_recorded_incomplete_responses_turn(SCENARIO);
}

/// The gap the bug actually was: one client, one prompt, one cap, both API
/// surfaces, one cassette.
#[tokio::test]
async fn cross_surface_truncation_parity() {
    const SCENARIO: &str = "truncated_turn_matrix/cross_surface_truncation_parity";

    with_openai_truncation_cassette(
        "truncated_turn_matrix/cross_surface_truncation_parity",
        |client| async move {
            let responses_model = client.clone().completion_model("gpt-5-nano");
            let responses = responses_model
                .completion(
                    responses_model
                        .completion_request(LONG_PROMPT)
                        .max_tokens(TINY_CAP)
                        .build(),
                )
                .await
                .expect("responses truncated turn");

            let chat_model = client.completions_api().completion_model("gpt-5-nano");
            let chat = chat_model
                .completion(
                    chat_model
                        .completion_request(LONG_PROMPT)
                        .max_tokens(TINY_CAP)
                        .build(),
                )
                .await
                .expect("chat truncated turn");

            assert_eq!(responses.finish_reason(), Some(FinishReason::Length));
            assert_eq!(chat.finish_reason(), responses.finish_reason());
        },
    )
    .await;

    assert_recorded_incomplete_responses_turn(SCENARIO);
    assert_recorded_empty_truncated_chat_turn(SCENARIO);
}

// ---------------------------------------------------------------------------
// Fixture-premise checks.
// ---------------------------------------------------------------------------

fn recorded_response_bodies(scenario: &str) -> Vec<String> {
    let path = cassettes::cassette_path("openai", scenario);
    let contents = std::fs::read_to_string(&path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable after recording: {error}",
            path.display()
        )
    });

    serde_yaml::Deserializer::from_str(&contents)
        .map(|document| serde_yaml::Value::deserialize(document).expect("cassette interaction"))
        .filter_map(|interaction| {
            interaction
                .get("then")
                .and_then(|then| then.get("body"))
                .and_then(serde_yaml::Value::as_str)
                .map(ToOwned::to_owned)
        })
        .collect()
}

/// `(finish_reason, message text)` for each recorded chat-completions choice.
fn recorded_chat_choices(scenario: &str) -> Vec<(String, String)> {
    recorded_response_bodies(scenario)
        .iter()
        .filter_map(|body| serde_json::from_str::<Value>(body).ok())
        .filter_map(|body| {
            Some(
                body.get("choices")?
                    .as_array()?
                    .iter()
                    .map(|choice| {
                        (
                            choice
                                .get("finish_reason")
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned(),
                            choice
                                .get("message")
                                .and_then(|message| message.get("content"))
                                .and_then(Value::as_str)
                                .unwrap_or_default()
                                .to_owned(),
                        )
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .flatten()
        .collect()
}

fn assert_recorded_empty_truncated_chat_turn(scenario: &str) {
    assert!(
        recorded_chat_choices(scenario)
            .iter()
            .any(|(reason, text)| reason == "length" && text.is_empty()),
        "cassette {scenario} no longer records an EMPTY `length` choice; this cell \
         would pass while covering nothing"
    );
}

fn assert_recorded_nonempty_truncated_chat_turn(scenario: &str) {
    assert!(
        recorded_chat_choices(scenario)
            .iter()
            .any(|(reason, text)| reason == "length" && !text.is_empty()),
        "cassette {scenario} no longer records a partial-text `length` choice"
    );
}

fn assert_recorded_completed_chat_turn(scenario: &str) {
    let choices = recorded_chat_choices(scenario);
    assert!(
        choices
            .iter()
            .any(|(reason, text)| reason == "stop" && !text.is_empty()),
        "control cassette {scenario} no longer records a completed, non-empty choice"
    );
    assert!(
        !choices.iter().any(|(reason, _)| reason == "length"),
        "control cassette {scenario} unexpectedly records a truncated choice"
    );
}

/// The streaming premise: a terminal chunk reporting `length`.
fn assert_recorded_truncated_chat_stream(scenario: &str) {
    let found = recorded_response_bodies(scenario)
        .iter()
        .flat_map(|body| body.lines())
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .any(|chunk| {
            chunk
                .get("choices")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .any(|choice| choice.get("finish_reason").and_then(Value::as_str) == Some("length"))
        });

    assert!(
        found,
        "cassette {scenario} no longer records a `length` finish reason on the stream"
    );
}

/// The Responses premise: `status: incomplete` with
/// `incomplete_details.reason == "max_output_tokens"`.
fn assert_recorded_incomplete_responses_turn(scenario: &str) {
    let found = recorded_response_bodies(scenario)
        .iter()
        .flat_map(|body| {
            // Both the unary body and each SSE frame's `response` object.
            let frames = body
                .lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim)
                .filter_map(|data| serde_json::from_str::<Value>(data).ok())
                .filter_map(|frame| frame.get("response").cloned())
                .collect::<Vec<_>>();
            serde_json::from_str::<Value>(body)
                .into_iter()
                .chain(frames)
                .collect::<Vec<_>>()
        })
        .any(|response| {
            response.get("status").and_then(Value::as_str) == Some("incomplete")
                && response
                    .get("incomplete_details")
                    .and_then(|details| details.get("reason"))
                    .and_then(Value::as_str)
                    == Some("max_output_tokens")
        });

    assert!(
        found,
        "cassette {scenario} no longer records an `incomplete` Responses turn"
    );
}
