//! Edge matrix for normalizing a turn that stopped on a stop sequence before
//! emitting anything.
//!
//! # The bug
//!
//! Anthropic strips the sequence it stopped on. When the match is the *first*
//! thing the model produces, the turn comes back 200 with:
//!
//! ```json
//! {"content":[],"stop_reason":"stop_sequence","stop_sequence":"alpha", ...}
//! ```
//!
//! `CompletionResponse::normalize` carved out exactly one legal empty case —
//! `end_turn` — and routed every other empty response through
//! `require_non_empty_response`. So a completed provider turn became
//! `CompletionError::ResponseError("Response contained no message or tool call
//! (empty)")`, destroying the usage, identity and finish-reason the same
//! response carried. The streamed twin of that request already finished
//! cleanly with an empty choice, so this was also a blocking/streaming
//! divergence — with the blocking side the broken one.
//!
//! # Matrix
//!
//! `expected` is what the *normalized* response must be. Recorded cells
//! re-derive their premise from their own fixture bytes: a cell whose provider
//! turn stopped coming back empty would otherwise pass while covering nothing.
//!
//! | # | Cell | Dimension | expected | Status |
//! |---|------|-----------|----------|--------|
//! | 1 | `raw_normalize_empty_stop_sequence` | `raw_completion` + `normalize` | empty choice | recorded |
//! | 2 | `completion_empty_stop_sequence` | `CompletionModel::completion` | empty choice | recorded |
//! | 3 | `agent_prompt_empty_stop_sequence` | agent `prompt` | empty text | recorded |
//! | 4 | `streaming_empty_stop_sequence` | streamed twin | empty choice | recorded |
//! | 5 | `agent_stream_empty_stop_sequence` | agent streamed twin | empty text | recorded |
//! | 6 | `nonempty_stop_sequence_control` | control: content precedes the match | non-empty | recorded |
//! | 7 | `unicode_empty_stop_sequence` | non-ASCII sequence | empty choice | recorded |
//! | 8 | `whitespace_empty_stop_sequence` | sequence with a space | empty choice | recorded |
//! | 9 | `punctuation_empty_stop_sequence` | punctuation sequence | empty choice | recorded |
//! | 10 | `two_sequences_empty_stop` | several sequences declared | empty choice | recorded |
//! | 11 | `with_preamble_empty_stop` | system prompt present | empty choice | recorded |
//! | 12 | `with_tools_empty_stop` | tools advertised | empty choice | recorded |
//! | 13 | `with_prompt_caching_empty_stop` | manual prompt caching | empty choice | recorded |
//! | 14 | `sonnet_empty_stop_sequence` | second model | empty choice | recorded |
//! | 15 | `identity_survives_empty_stop` | id / request-id / usage preserved | populated | recorded |
//! | 16 | `finish_reason_is_stop_on_empty_stop` | finish-reason mapping | `Stop` | recorded |
//! | 17 | `followup_after_empty_stop_turn` | the model stays usable afterwards | non-empty | recorded |
//! | 18 | `long_sequence_empty_stop` | multi-token sequence | empty choice | recorded |
//! | 19 | `unit_empty_end_turn_still_allowed` | the pre-existing legal empty | empty choice | unit |
//! | 20 | `unit_empty_max_tokens_still_rejected` | guard preserved | error | unit |
//! | 21 | `unit_empty_tool_use_still_rejected` | guard preserved | error | unit |
//! | 22 | `unit_empty_refusal_still_rejected` | guard preserved | error | unit |
//! | 23 | `unit_empty_pause_turn_still_rejected` | guard preserved | error | unit |
//! | 24 | `unit_empty_without_stop_reason_still_rejected` | guard preserved | error | unit |
//! | 25 | `unit_empty_stop_sequence_without_a_sequence_still_rejected` | carve-out stays inside its evidence | error | unit |
//! | 26 | `unit_empty_assistant_turn_cannot_be_replayed` | adjacent request boundary | error | unit |
//!
//! Cells 20–25 are unit tests because Anthropic will not produce those states
//! on demand: `max_tokens` truncation always emits at least one token (probed
//! live — a `max_tokens: 1` turn came back with a one-character text block); a
//! `tool_use` terminal by construction carries the tool-use block; `pause_turn`
//! requires a server tool to exhaust its iteration budget; `refusal` requires
//! tripping a safety classifier; a turn that reports *no* stop reason at all is
//! by definition not something a well-formed wire emits; and a `stop_sequence`
//! turn that names no sequence is the malformed shape the carve-out
//! deliberately stops short of. Each is exactly the case the guard must keep
//! rejecting, so pinning them by hand is the only way to prove the carve-out
//! did not widen past its evidence.
//!
//! Cell 19's *recorded* counterpart already exists as
//! `empty_end_turn/raw_followup_empty_end_turn_normalizes_to_empty_text_choice`;
//! it is repeated here as a unit cell so this matrix's own table is complete.

use rig::completion::{
    CompletionModel as _, CompletionResponse as RigCompletionResponse, FinishReason,
    NormalizeCompletionResponse, Prompt, ToolDefinition,
};
use rig::prelude::*;
use rig::providers::anthropic;
use rig::providers::anthropic::completion::{CompletionResponse, Content, Usage};
use rig::streaming::StreamingPrompt;
use serde_json::json;

use super::super::support::{recorded_response_body, with_anthropic_empty_stop_cassette};

const ANTHROPIC_PROVIDER: &str = "anthropic";

/// Asks for exactly one word so a stop sequence naming that word matches
/// before the model emits anything else.
const IMMEDIATE_PROMPT: &str = "Reply with exactly this one word and nothing else: alpha";
const IMMEDIATE_UNICODE_PROMPT: &str = "Reply with exactly this one character and nothing else: 🌊";
const IMMEDIATE_PHRASE_PROMPT: &str =
    "Reply with exactly this phrase and nothing else: alpha bravo charlie";
const IMMEDIATE_PUNCTUATION_PROMPT: &str = "Reply with exactly this and nothing else: ###";

type AnthropicModel = anthropic::completion::CompletionModel;

fn request(
    model: &AnthropicModel,
    prompt: &str,
    stop_sequences: &[&str],
    max_tokens: u64,
) -> rig::completion::CompletionRequest {
    model
        .completion_request(prompt)
        .max_tokens(max_tokens)
        .additional_params(json!({ "stop_sequences": stop_sequences }))
        .build()
}

fn weather_tool() -> ToolDefinition {
    ToolDefinition {
        name: "get_weather".to_string(),
        description: "Get the current weather for a city.".to_string(),
        parameters: json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        }),
    }
}

/// The premise every empty-stop cell rests on: the recorded blocking body had
/// **no** content blocks and stopped on a sequence. Read after the cassette
/// wrapper returns — record mode writes the fixture on the way out.
fn assert_recorded_empty_stop(scenario: &str) {
    let body = recorded_response_body(scenario);
    assert_eq!(
        body.get("stop_reason").and_then(serde_json::Value::as_str),
        Some("stop_sequence"),
        "{scenario}: the recorded turn must have stopped on a sequence"
    );
    assert_eq!(
        body.get("content").and_then(serde_json::Value::as_array),
        Some(&Vec::new()),
        "{scenario}: the recorded turn must carry no content blocks — without \
         that this cell asserts nothing about the empty case"
    );
}

/// Streaming premise: the recorded stream stopped on a sequence and never
/// emitted a text delta.
fn assert_recorded_streamed_empty_stop(scenario: &str) {
    let path = crate::cassettes::cassette_path("anthropic", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cassette {} should be readable: {err}", path.display()));

    assert!(
        contents.contains(r#""stop_reason":"stop_sequence""#),
        "{scenario}: the recorded stream must have stopped on a sequence"
    );
    assert!(
        !contents.contains(r#""type":"text_delta""#),
        "{scenario}: the recorded stream must not carry any text delta — \
         without that this cell asserts nothing about the empty case"
    );
}

fn empty_response(stop_reason: Option<&str>, content: Vec<Content>) -> CompletionResponse {
    CompletionResponse {
        content,
        id: "msg_matrix".to_string(),
        model: "claude-haiku-4-5".to_string(),
        role: "assistant".to_string(),
        stop_reason: stop_reason.map(str::to_string),
        stop_sequence: stop_reason
            .filter(|reason| *reason == "stop_sequence")
            .map(|_| "alpha".to_string()),
        usage: Usage {
            input_tokens: 18,
            cache_read_input_tokens: None,
            cache_creation_input_tokens: None,
            cache_creation: None,
            output_tokens: 1,
        },
        provider_request_id: None,
    }
}

// ---------------------------------------------------------------------------
// 1–5: the surfaces a caller reaches this through
// ---------------------------------------------------------------------------

#[tokio::test]
async fn raw_normalize_empty_stop_sequence() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/raw_normalize_empty_stop_sequence",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let raw = model
                .raw_completion(request(&model, IMMEDIATE_PROMPT, &["alpha"], 32))
                .await
                .expect("empty stop-sequence request should succeed");

            assert!(raw.content.is_empty(), "premise: the turn carried nothing");
            assert_eq!(raw.stop_sequence.as_deref(), Some("alpha"));

            let normalized: RigCompletionResponse = raw
                .normalize(ANTHROPIC_PROVIDER)
                .expect("a completed stop-sequence turn must not normalize into an error");
            assert!(normalized.choice.is_empty());
        },
    )
    .await;

    assert_recorded_empty_stop("empty_stop_sequence_matrix/raw_normalize_empty_stop_sequence");
}

#[tokio::test]
async fn completion_empty_stop_sequence() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/completion_empty_stop_sequence",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let response = rig::completion::CompletionModel::completion(
                &model,
                request(&model, IMMEDIATE_PROMPT, &["alpha"], 32),
            )
            .await
            .expect("`completion` must not turn a completed turn into an error");

            assert!(response.choice.is_empty());
            assert_eq!(response.finish_reason(), Some(FinishReason::Stop));
        },
    )
    .await;

    assert_recorded_empty_stop("empty_stop_sequence_matrix/completion_empty_stop_sequence");
}

#[tokio::test]
async fn agent_prompt_empty_stop_sequence() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/agent_prompt_empty_stop_sequence",
        |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
                .max_tokens(32)
                .additional_params(json!({ "stop_sequences": ["alpha"] }))
                .build();

            let response = agent
                .prompt(IMMEDIATE_PROMPT)
                .await
                .expect("agent prompt must not fail on a completed empty turn");
            assert!(
                response.trim().is_empty(),
                "the turn produced no text: {response:?}"
            );
        },
    )
    .await;

    assert_recorded_empty_stop("empty_stop_sequence_matrix/agent_prompt_empty_stop_sequence");
}

#[tokio::test]
async fn streaming_empty_stop_sequence() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/streaming_empty_stop_sequence",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let mut stream = rig::completion::CompletionModel::stream(
                &model,
                request(&model, IMMEDIATE_PROMPT, &["alpha"], 32),
            )
            .await
            .expect("stream should open");

            let mut errors = Vec::new();
            while let Some(item) = futures::StreamExt::next(&mut stream).await {
                if let Err(error) = item {
                    errors.push(error.to_string());
                }
            }

            assert!(
                errors.is_empty(),
                "streamed twin must not error: {errors:?}"
            );
            assert!(stream.choice.is_empty());
            assert_eq!(
                stream
                    .response
                    .as_ref()
                    .and_then(|final_| final_.finish_reason.clone()),
                Some(FinishReason::Stop)
            );
        },
    )
    .await;

    assert_recorded_streamed_empty_stop("empty_stop_sequence_matrix/streaming_empty_stop_sequence");
}

#[tokio::test]
async fn agent_stream_empty_stop_sequence() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/agent_stream_empty_stop_sequence",
        |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_HAIKU_4_5)
                .max_tokens(32)
                .additional_params(json!({ "stop_sequences": ["alpha"] }))
                .build();

            let mut stream = agent.stream_prompt(IMMEDIATE_PROMPT).await;
            let mut errors = Vec::new();
            let mut final_output = None;
            let mut completion_finish = None;
            while let Some(item) = futures::StreamExt::next(&mut stream).await {
                match item {
                    Ok(rig::agent::MultiTurnStreamItem::FinalResponse(response)) => {
                        final_output = Some(response.output().to_owned());
                    }
                    Ok(rig::agent::MultiTurnStreamItem::CompletionCall(call)) => {
                        completion_finish = call.finish_reason.clone();
                    }
                    Ok(_) => {}
                    Err(error) => errors.push(error.to_string()),
                }
            }

            assert!(
                errors.is_empty(),
                "the agent stream must not fail on a completed empty turn: {errors:?}"
            );
            assert_eq!(
                final_output.as_deref(),
                Some(""),
                "the run finishes with empty output rather than an error"
            );
            // An empty turn yields no `StreamedAssistantContent::Final` because
            // it produced no assistant content at all; the provider terminal's
            // metadata still reaches the consumer on the `CompletionCall` item,
            // so nothing is lost. Pinned so that stays true.
            assert_eq!(completion_finish, Some(FinishReason::Stop));
        },
    )
    .await;

    assert_recorded_streamed_empty_stop(
        "empty_stop_sequence_matrix/agent_stream_empty_stop_sequence",
    );
}

// ---------------------------------------------------------------------------
// 6: control — the same stop reason with content must still produce content
// ---------------------------------------------------------------------------

#[tokio::test]
async fn nonempty_stop_sequence_control() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/nonempty_stop_sequence_control",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let response = rig::completion::CompletionModel::completion(
                &model,
                request(&model, IMMEDIATE_PHRASE_PROMPT, &["charlie"], 64),
            )
            .await
            .expect("stop-sequence turn with content should succeed");

            assert!(
                !response.choice.is_empty(),
                "the carve-out must not swallow content that was actually produced"
            );
        },
    )
    .await;

    let body = recorded_response_body("empty_stop_sequence_matrix/nonempty_stop_sequence_control");
    assert_eq!(
        body.get("stop_reason").and_then(serde_json::Value::as_str),
        Some("stop_sequence")
    );
    assert!(
        !body
            .get("content")
            .and_then(serde_json::Value::as_array)
            .expect("content array")
            .is_empty(),
        "control premise: this recorded turn must carry content"
    );
}

// ---------------------------------------------------------------------------
// 7–10: sequence shapes that can match at position zero
// ---------------------------------------------------------------------------

#[tokio::test]
async fn unicode_empty_stop_sequence() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/unicode_empty_stop_sequence",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let response = rig::completion::CompletionModel::completion(
                &model,
                request(&model, IMMEDIATE_UNICODE_PROMPT, &["🌊"], 32),
            )
            .await
            .expect("unicode empty stop-sequence turn should succeed");
            assert!(response.choice.is_empty());
        },
    )
    .await;

    assert_recorded_empty_stop("empty_stop_sequence_matrix/unicode_empty_stop_sequence");
}

#[tokio::test]
async fn whitespace_empty_stop_sequence() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/whitespace_empty_stop_sequence",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let response = rig::completion::CompletionModel::completion(
                &model,
                request(&model, IMMEDIATE_PHRASE_PROMPT, &["alpha bravo"], 32),
            )
            .await
            .expect("whitespace empty stop-sequence turn should succeed");
            assert!(response.choice.is_empty());
        },
    )
    .await;

    assert_recorded_empty_stop("empty_stop_sequence_matrix/whitespace_empty_stop_sequence");
}

#[tokio::test]
async fn punctuation_empty_stop_sequence() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/punctuation_empty_stop_sequence",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let response = rig::completion::CompletionModel::completion(
                &model,
                request(&model, IMMEDIATE_PUNCTUATION_PROMPT, &["###"], 32),
            )
            .await
            .expect("punctuation empty stop-sequence turn should succeed");
            assert!(response.choice.is_empty());
        },
    )
    .await;

    assert_recorded_empty_stop("empty_stop_sequence_matrix/punctuation_empty_stop_sequence");
}

#[tokio::test]
async fn two_sequences_empty_stop() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/two_sequences_empty_stop",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let response = rig::completion::CompletionModel::completion(
                &model,
                request(&model, IMMEDIATE_PROMPT, &["zulu", "alpha"], 32),
            )
            .await
            .expect("multi-sequence empty stop turn should succeed");
            assert!(response.choice.is_empty());
        },
    )
    .await;

    assert_recorded_empty_stop("empty_stop_sequence_matrix/two_sequences_empty_stop");
}

// ---------------------------------------------------------------------------
// 11–14: the rest of the request surface
// ---------------------------------------------------------------------------

#[tokio::test]
async fn with_preamble_empty_stop() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/with_preamble_empty_stop",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let request = model
                .completion_request(IMMEDIATE_PROMPT)
                .preamble("You follow formatting instructions exactly.".to_string())
                .max_tokens(32)
                .additional_params(json!({ "stop_sequences": ["alpha"] }))
                .build();
            let response = rig::completion::CompletionModel::completion(&model, request)
                .await
                .expect("empty stop turn with a system prompt should succeed");
            assert!(response.choice.is_empty());
        },
    )
    .await;

    assert_recorded_empty_stop("empty_stop_sequence_matrix/with_preamble_empty_stop");
}

#[tokio::test]
async fn with_tools_empty_stop() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/with_tools_empty_stop",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let request = model
                .completion_request(IMMEDIATE_PROMPT)
                .max_tokens(32)
                .tool(weather_tool())
                .additional_params(json!({ "stop_sequences": ["alpha"] }))
                .build();
            let response = rig::completion::CompletionModel::completion(&model, request)
                .await
                .expect("empty stop turn with tools advertised should succeed");
            assert!(response.choice.is_empty());
        },
    )
    .await;

    assert_recorded_empty_stop("empty_stop_sequence_matrix/with_tools_empty_stop");
}

#[tokio::test]
async fn with_prompt_caching_empty_stop() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/with_prompt_caching_empty_stop",
        |client| async move {
            let model = client
                .completion_model(anthropic::completion::CLAUDE_HAIKU_4_5)
                .with_prompt_caching();
            let response = rig::completion::CompletionModel::completion(
                &model,
                request(&model, IMMEDIATE_PROMPT, &["alpha"], 32),
            )
            .await
            .expect("empty stop turn with prompt caching should succeed");
            assert!(response.choice.is_empty());
        },
    )
    .await;

    assert_recorded_empty_stop("empty_stop_sequence_matrix/with_prompt_caching_empty_stop");
}

#[tokio::test]
async fn sonnet_empty_stop_sequence() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/sonnet_empty_stop_sequence",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let response = rig::completion::CompletionModel::completion(
                &model,
                request(&model, IMMEDIATE_PROMPT, &["alpha"], 32),
            )
            .await
            .expect("empty stop turn on a second model should succeed");
            assert!(response.choice.is_empty());
        },
    )
    .await;

    assert_recorded_empty_stop("empty_stop_sequence_matrix/sonnet_empty_stop_sequence");
}

// ---------------------------------------------------------------------------
// 15–18: what the response must still carry, and life after it
// ---------------------------------------------------------------------------

#[tokio::test]
async fn identity_survives_empty_stop() {
    let observed_model: std::sync::Arc<std::sync::Mutex<Option<String>>> = Default::default();
    let sink = observed_model.clone();

    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/identity_survives_empty_stop",
        move |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let response = rig::completion::CompletionModel::completion(
                &model,
                request(&model, IMMEDIATE_PROMPT, &["alpha"], 32),
            )
            .await
            .expect("empty stop turn should succeed");

            // Everything the discarded error used to take with it.
            assert!(response.message_id.is_some(), "message id must survive");
            assert!(
                response.provider_request_id.is_some(),
                "transport request id must survive — it is what Anthropic support asks for"
            );
            assert!(response.usage.input_tokens > 0, "usage must survive");
            *sink.lock().expect("model sink should not be poisoned") = response.model.clone();
        },
    )
    .await;

    let scenario = "empty_stop_sequence_matrix/identity_survives_empty_stop";
    assert_recorded_empty_stop(scenario);

    // The responding model must reach the normalized response *verbatim*. Read
    // from the fixture rather than hardcoded, so a re-record on a newer dated
    // snapshot stays valid — but compared against what `normalize` produced,
    // because asserting the two independently would pin neither: the wire's
    // dated id (`claude-haiku-4-5-20251001`) differs from the requested alias,
    // and only this comparison catches the response reporting the alias back.
    let recorded_model = recorded_response_body(scenario)
        .get("model")
        .and_then(serde_json::Value::as_str)
        .map(str::to_string)
        .expect("the recorded turn should name its model");
    assert_eq!(
        observed_model
            .lock()
            .expect("model sink should not be poisoned")
            .as_deref(),
        Some(recorded_model.as_str()),
        "the model the wire reported must reach the normalized response unchanged"
    );
}

#[tokio::test]
async fn finish_reason_is_stop_on_empty_stop() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/finish_reason_is_stop_on_empty_stop",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let response = rig::completion::CompletionModel::completion(
                &model,
                request(&model, IMMEDIATE_PROMPT, &["alpha"], 32),
            )
            .await
            .expect("empty stop turn should succeed");

            assert_eq!(
                response.finish_reason(),
                Some(FinishReason::Stop),
                "a stop-sequence stop is a natural termination, not a failure"
            );
        },
    )
    .await;

    assert_recorded_empty_stop("empty_stop_sequence_matrix/finish_reason_is_stop_on_empty_stop");
}

#[tokio::test]
async fn followup_after_empty_stop_turn() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/followup_after_empty_stop_turn",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);

            let first = rig::completion::CompletionModel::completion(
                &model,
                request(&model, IMMEDIATE_PROMPT, &["alpha"], 32),
            )
            .await
            .expect("first (empty) turn should succeed");
            assert!(first.choice.is_empty());

            let second = rig::completion::CompletionModel::completion(
                &model,
                request(&model, IMMEDIATE_PROMPT, &["zulu"], 32),
            )
            .await
            .expect("second turn should succeed");
            assert!(
                !second.choice.is_empty(),
                "an empty turn must not leave the model unusable"
            );
        },
    )
    .await;

    // Reads the first interaction of the two-interaction cassette.
    assert_recorded_empty_stop("empty_stop_sequence_matrix/followup_after_empty_stop_turn");
}

#[tokio::test]
async fn long_sequence_empty_stop() {
    with_anthropic_empty_stop_cassette(
        "empty_stop_sequence_matrix/long_sequence_empty_stop",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_HAIKU_4_5);
            let response = rig::completion::CompletionModel::completion(
                &model,
                request(
                    &model,
                    IMMEDIATE_PHRASE_PROMPT,
                    &["alpha bravo charlie"],
                    32,
                ),
            )
            .await
            .expect("long-sequence empty stop turn should succeed");
            assert!(response.choice.is_empty());
        },
    )
    .await;

    assert_recorded_empty_stop("empty_stop_sequence_matrix/long_sequence_empty_stop");
}

// ---------------------------------------------------------------------------
// 19–26: states no provider turn will produce on demand
// ---------------------------------------------------------------------------

#[test]
fn unit_empty_end_turn_still_allowed() {
    let normalized = empty_response(Some("end_turn"), Vec::new()).normalize(ANTHROPIC_PROVIDER);
    assert!(
        normalized.is_ok_and(|response| response.choice.is_empty()),
        "the pre-existing legal empty case must keep working"
    );
}

#[test]
fn unit_empty_max_tokens_still_rejected() {
    assert!(
        empty_response(Some("max_tokens"), Vec::new())
            .normalize(ANTHROPIC_PROVIDER)
            .is_err(),
        "a truncated turn that produced nothing is still the shared provider defect"
    );
}

#[test]
fn unit_empty_tool_use_still_rejected() {
    assert!(
        empty_response(Some("tool_use"), Vec::new())
            .normalize(ANTHROPIC_PROVIDER)
            .is_err(),
        "a tool-use terminal with no tool-use block is malformed"
    );
}

#[test]
fn unit_empty_refusal_still_rejected() {
    assert!(
        empty_response(Some("refusal"), Vec::new())
            .normalize(ANTHROPIC_PROVIDER)
            .is_err(),
        "widening the carve-out past its evidence would swallow refusals silently"
    );
}

#[test]
fn unit_empty_pause_turn_still_rejected() {
    assert!(
        empty_response(Some("pause_turn"), Vec::new())
            .normalize(ANTHROPIC_PROVIDER)
            .is_err(),
        "a paused turn carries the server-tool blocks it paused on"
    );
}

#[test]
fn unit_empty_without_stop_reason_still_rejected() {
    assert!(
        empty_response(None, Vec::new())
            .normalize(ANTHROPIC_PROVIDER)
            .is_err(),
        "an empty response that never said why it ended is the case the guard exists for"
    );
}

#[test]
fn unit_empty_stop_sequence_without_a_sequence_still_rejected() {
    // The carve-out is keyed on the sequence being *named*, not merely on the
    // stop reason. Every recorded stop-sequence turn names it, so a turn that
    // claims the reason and names nothing is outside the evidence and stays
    // the malformed shape the guard exists for. This matters most for the
    // Anthropic-compatible gateways sharing this mapping.
    let mut response = empty_response(Some("stop_sequence"), Vec::new());
    response.stop_sequence = None;

    assert!(
        response.normalize(ANTHROPIC_PROVIDER).is_err(),
        "a stop-sequence terminal that names no sequence must stay rejected"
    );
}

#[test]
fn unit_empty_assistant_turn_cannot_be_replayed() {
    // The adjacent boundary: an empty assistant turn normalizes fine, but the
    // Anthropic wire rejects empty content, so rig refuses to send one back.
    // Nothing here changed — pinned so widening the response-side carve-out is
    // never mistaken for widening the request side too.
    let empty_assistant = rig::message::Message::Assistant {
        id: None,
        content: Vec::new(),
    };
    assert!(
        anthropic::completion::Message::try_from(empty_assistant).is_err(),
        "an assistant turn with no content must not reach the wire"
    );
}
