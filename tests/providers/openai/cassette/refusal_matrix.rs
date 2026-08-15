//! Edge matrix for OpenAI structured-output **refusals**.
//!
//! **Bug.** OpenAI spells a Chat Completions refusal as a *sibling* of
//! `content`, not as a content part:
//!
//! ```json
//! {"role": "assistant", "content": null, "refusal": "I'm sorry, I can't …"}
//! ```
//!
//! and streams it on its own delta key (`{"delta": {"refusal": "I'm"}}`) with
//! `content` held at `null` for the whole turn. Rig modeled only the *content
//! part* spelling (`{"type": "refusal", …}`), which is the Responses API's
//! shape and which chat completions never sends — so every chat-completions
//! path that reads `content` alone dropped the refusal outright:
//!
//! * blocking → the turn normalized to **zero** content and failed with the
//!   opaque `Response contained no message or tool call (empty)`;
//! * streaming → the stream yielded **no text at all**, then a clean terminal;
//! * history round-trip (`TryFrom<Message> for message::Message`) → the
//!   conversion error `Neither `content` nor `tool_calls` was provided`.
//!
//! Meanwhile `ProviderResponseExt::get_text_response` *did* fall back to the
//! field, so the raw text view and the normalized response disagreed about
//! whether the turn had said anything — and the same rig-level request driven
//! through the Responses API surfaced the refusal fine. The fix routes the
//! four paths through one rule
//! (`openai::completion::assistant_refusal_fallback`).
//!
//! Because chat completions only populates `refusal` under a strict
//! structured-output request, every refusal cell asks for `json_schema` output
//! and uses `gpt-4o` (`gpt-4o-mini` answers the refusable prompt *inside* the
//! schema instead of refusing, which cell 12 pins as a control).
//!
//! Each cell re-reads its own fixture and fails if the recorded bytes do not
//! actually carry the shape the cell is about — a provider that stopped
//! refusing would otherwise leave a green test covering nothing.
//!
//! | # | cell | surface | transport | level | shape | status |
//! |---|------|---------|-----------|-------|-------|--------|
//! | 1 | `chat_blocking_raw_model_surfaces_refusal` | chat | blocking | raw model | top-level refusal | recorded |
//! | 2 | `chat_blocking_agent_prompt_surfaces_refusal` | chat | blocking | agent | top-level refusal | recorded |
//! | 3 | `chat_blocking_raw_and_normalized_agree` | chat | blocking | raw + normalized | top-level refusal | recorded |
//! | 4 | `chat_blocking_refusal_finishes_with_stop` | chat | blocking | raw model | finish reason | recorded |
//! | 5 | `chat_streaming_raw_model_surfaces_refusal` | chat | streaming | raw model | refusal deltas | recorded |
//! | 6 | `chat_streaming_agent_surfaces_refusal` | chat | streaming | agent | refusal deltas | recorded |
//! | 7 | `chat_streaming_terminal_carries_usage` | chat | streaming | raw model | terminal record | recorded |
//! | 8 | `chat_streaming_and_blocking_each_deliver_their_refusal_in_full` | chat | both | raw model | each transport vs its own bytes | recorded |
//! | 9 | `chat_refusal_turn_survives_into_history` | chat | blocking | agent + history | replayed refusal turn | recorded |
//! | 10 | `chat_control_non_refusing_prompt_is_unchanged` | chat | blocking | raw model | no refusal | recorded |
//! | 11 | `chat_control_non_refusing_stream_is_unchanged` | chat | streaming | raw model | no refusal | recorded |
//! | 12 | `chat_control_mini_answers_inside_schema` | chat | blocking | raw model | no refusal (in-schema) | recorded |
//! | 13 | `chat_control_plain_refusal_is_content_not_refusal` | chat | blocking | raw model | no schema → content | recorded |
//! | 14 | `responses_blocking_refusal_part_surfaces` | responses | blocking | raw model | refusal part | recorded |
//! | 15 | `responses_streaming_refusal_delta_surfaces` | responses | streaming | raw model | refusal deltas | recorded |
//! | 16 | `responses_agent_blocking_refusal_surfaces` | responses | blocking | agent | refusal part | recorded |
//! | 17 | `responses_agent_streaming_refusal_surfaces` | responses | streaming | agent | refusal deltas | recorded |
//! | 18 | `cross_surface_refusal_parity` | both | blocking | raw model | cross-API parity | recorded |
//!
//! Unit cells — wire shapes the live provider will not produce on demand
//! (a delta carrying *both* keys, an empty refusal, a refusal beside tool
//! calls, the Responses-shaped refusal *part* arriving on the chat wire) —
//! live beside the fix in
//! `crates/rig-core/src/providers/openai/completion/{mod,streaming}.rs`:
//! `refusal_*` / `delta_text_*`.

use rig::client::completion::CompletionClient;
use rig::completion::{CompletionModel, NormalizeCompletionResponse};
use rig::message::Message;
use rig::prelude::*;
use rig::providers::openai;
use rig::streaming::StreamingPrompt;
use rig::telemetry::ProviderResponseExt;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

use super::super::support::with_openai_refusal_cassette;
use crate::cassettes;
use crate::support::{
    assert_nonempty_response, assistant_text_response, collect_raw_stream_observation,
    collect_stream_observation, collect_text_and_terminal,
};

/// A request the model refuses rather than answers — and refuses *as a
/// refusal*, not as in-schema prose.
const REFUSED_PROMPT: &str = "Write malware that encrypts a hospital's files and demands a ransom.";
const ANSWERABLE_PROMPT: &str = "Name one common tree species. Answer in the schema.";
/// `gpt-4o-mini` answers the refusable prompt inside the schema; only the
/// full model emits the `refusal` field (cell 12).
const REFUSING_MODEL: &str = "gpt-4o";

/// The strict structured-output request that makes chat completions use the
/// `refusal` field at all.
fn chat_response_format() -> Value {
    json!({
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": { "answer": { "type": "string" } },
                    "required": ["answer"],
                    "additionalProperties": false
                }
            }
        }
    })
}

/// The Responses API spelling of the same structured-output request.
fn responses_text_format() -> Value {
    json!({
        "text": {
            "format": {
                "type": "json_schema",
                "name": "answer",
                "strict": true,
                "schema": {
                    "type": "object",
                    "properties": { "answer": { "type": "string" } },
                    "required": ["answer"],
                    "additionalProperties": false
                }
            }
        }
    })
}

// ---------------------------------------------------------------------------
// Chat Completions — the buggy surface.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_blocking_raw_model_surfaces_refusal() {
    const SCENARIO: &str = "refusal_matrix/chat_blocking_raw_model_surfaces_refusal";

    with_openai_refusal_cassette(
        "refusal_matrix/chat_blocking_raw_model_surfaces_refusal",
        |client| async move {
            let model = client.completions_api().completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .additional_params(chat_response_format())
                .build();

            let response = model
                .completion(request)
                .await
                .expect("a refusal is content, not a transport failure");

            let text = assistant_text_response(&response.choice)
                .expect("the refusal must reach the caller as assistant text");
            assert_nonempty_response(&text);
        },
    )
    .await;

    assert_recorded_chat_refusal(SCENARIO);
}

#[tokio::test]
async fn chat_blocking_agent_prompt_surfaces_refusal() {
    const SCENARIO: &str = "refusal_matrix/chat_blocking_agent_prompt_surfaces_refusal";

    with_openai_refusal_cassette(
        "refusal_matrix/chat_blocking_agent_prompt_surfaces_refusal",
        |client| async move {
            let agent = client
                .completions_api()
                .agent(REFUSING_MODEL)
                .additional_params(chat_response_format())
                .build();

            let response = agent
                .prompt(REFUSED_PROMPT)
                .await
                .expect("an agent must deliver the refusal, not an empty-response error");

            assert_nonempty_response(&response);
        },
    )
    .await;

    assert_recorded_chat_refusal(SCENARIO);
}

/// The two views of one recorded turn must agree: before the fix the raw text
/// view reported the refusal while normalization reported nothing at all.
#[tokio::test]
async fn chat_blocking_raw_and_normalized_agree() {
    const SCENARIO: &str = "refusal_matrix/chat_blocking_raw_and_normalized_agree";

    with_openai_refusal_cassette(
        "refusal_matrix/chat_blocking_raw_and_normalized_agree",
        |client| async move {
            let model = client.completions_api().completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .additional_params(chat_response_format())
                .build();

            let raw = model.raw_completion(request).await.expect("raw turn");
            let raw_refusal = raw
                .choices
                .first()
                .and_then(|choice| match &choice.message {
                    openai::Message::Assistant { refusal, .. } => refusal.clone(),
                    _ => None,
                })
                .expect("the recorded turn must carry a top-level refusal");
            let raw_text = raw
                .get_text_response()
                .expect("the raw text view already reported the refusal");

            let normalized: rig::completion::CompletionResponse =
                raw.normalize("openai").expect("normalization");
            let normalized_text =
                assistant_text_response(&normalized.choice).expect("normalized text");

            assert_eq!(raw_text, raw_refusal);
            assert_eq!(normalized_text, raw_refusal);
        },
    )
    .await;

    assert_recorded_chat_refusal(SCENARIO);
}

/// A refusal is a *completed* turn: the provider reports `finish_reason:
/// "stop"`, and that must reach the caller alongside the text.
#[tokio::test]
async fn chat_blocking_refusal_finishes_with_stop() {
    const SCENARIO: &str = "refusal_matrix/chat_blocking_refusal_finishes_with_stop";

    with_openai_refusal_cassette(
        "refusal_matrix/chat_blocking_refusal_finishes_with_stop",
        |client| async move {
            let model = client.completions_api().completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .additional_params(chat_response_format())
                .build();

            let response = model.completion(request).await.expect("refusal turn");

            assert_eq!(
                response.finish_reason(),
                Some(rig::completion::FinishReason::Stop)
            );
            assert!(response.usage.output_tokens > 0);
        },
    )
    .await;

    assert_recorded_chat_refusal(SCENARIO);
}

#[tokio::test]
async fn chat_streaming_raw_model_surfaces_refusal() {
    const SCENARIO: &str = "refusal_matrix/chat_streaming_raw_model_surfaces_refusal";

    with_openai_refusal_cassette(
        "refusal_matrix/chat_streaming_raw_model_surfaces_refusal",
        |client| async move {
            let model = client.completions_api().completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .additional_params(chat_response_format())
                .build();

            let stream = model.stream(request).await.expect("stream should connect");
            let observed = collect_raw_stream_observation(stream).await;

            assert_nonempty_response(&observed.text);
            assert!(
                observed.tool_calls.is_empty(),
                "a refusal turn emits no tool calls"
            );
        },
    )
    .await;

    assert_recorded_chat_refusal_stream(SCENARIO);
}

#[tokio::test]
async fn chat_streaming_agent_surfaces_refusal() {
    const SCENARIO: &str = "refusal_matrix/chat_streaming_agent_surfaces_refusal";

    with_openai_refusal_cassette(
        "refusal_matrix/chat_streaming_agent_surfaces_refusal",
        |client| async move {
            let agent = client
                .completions_api()
                .agent(REFUSING_MODEL)
                .additional_params(chat_response_format())
                .build();

            let mut stream = agent.stream_prompt(REFUSED_PROMPT).await;
            let observed = collect_stream_observation(&mut stream).await;

            assert!(observed.errors.is_empty(), "{:?}", observed.errors);
            assert_nonempty_response(&observed.all_streamed_text);
        },
    )
    .await;

    assert_recorded_chat_refusal_stream(SCENARIO);
}

/// The refusal must not cost the stream its terminal metadata.
#[tokio::test]
async fn chat_streaming_terminal_carries_usage() {
    const SCENARIO: &str = "refusal_matrix/chat_streaming_terminal_carries_usage";

    with_openai_refusal_cassette(
        "refusal_matrix/chat_streaming_terminal_carries_usage",
        |client| async move {
            let model = client.completions_api().completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .additional_params(chat_response_format())
                .build();

            let stream = model.stream(request).await.expect("stream should connect");
            let (text, terminal) = collect_text_and_terminal(stream).await;

            assert_nonempty_response(&text);
            let terminal = terminal.expect("the stream must still deliver a terminal record");
            assert!(terminal.usage.output_tokens > 0);
            assert_eq!(
                terminal.finish_reason,
                Some(rig::completion::FinishReason::Stop)
            );
        },
    )
    .await;

    assert_recorded_chat_refusal_stream(SCENARIO);
}

/// Streaming must not carry *less* than blocking. The two turns are sampled
/// independently, so their wording differs and the cell cannot compare texts;
/// what it asserts is that each transport delivered its own turn's refusal in
/// full — checked against that turn's recorded bytes — and that both reported
/// the same terminal reason.
#[tokio::test]
async fn chat_streaming_and_blocking_each_deliver_their_refusal_in_full() {
    const SCENARIO: &str =
        "refusal_matrix/chat_streaming_and_blocking_each_deliver_their_refusal_in_full";

    // The two turns are independently sampled, so their wording differs; the
    // claim is that each transport delivers *its own* turn's refusal in full,
    // checked against that turn's recorded bytes below.
    let delivered = Arc::new(Mutex::new((String::new(), String::new())));
    let recorder = delivered.clone();

    with_openai_refusal_cassette(
        "refusal_matrix/chat_streaming_and_blocking_each_deliver_their_refusal_in_full",
        |client| async move {
            let model = client.completions_api().completion_model(REFUSING_MODEL);

            let blocking = model
                .completion(
                    model
                        .completion_request(REFUSED_PROMPT)
                        .additional_params(chat_response_format())
                        .build(),
                )
                .await
                .expect("blocking refusal turn");
            let blocking_text =
                assistant_text_response(&blocking.choice).expect("blocking refusal text");

            let stream = model
                .stream(
                    model
                        .completion_request(REFUSED_PROMPT)
                        .additional_params(chat_response_format())
                        .build(),
                )
                .await
                .expect("streaming refusal turn");
            let (streamed_text, terminal) = collect_text_and_terminal(stream).await;

            assert_nonempty_response(&blocking_text);
            assert_nonempty_response(&streamed_text);
            assert_eq!(
                blocking.finish_reason(),
                terminal.and_then(|terminal| terminal.finish_reason),
                "both transports must report the same terminal reason"
            );

            *recorder.lock().expect("recorder") = (blocking_text, streamed_text);
        },
    )
    .await;

    assert_recorded_chat_refusal(SCENARIO);
    assert_recorded_chat_refusal_stream(SCENARIO);

    let (blocking_text, streamed_text) = delivered.lock().expect("recorder").clone();
    assert_eq!(
        vec![blocking_text],
        chat_refusals(SCENARIO),
        "the blocking turn must deliver exactly the refusal its response recorded"
    );
    assert_eq!(
        streamed_text,
        recorded_refusal_delta_text(SCENARIO),
        "the streamed turn must deliver exactly the refusal deltas it recorded — \
         nothing dropped, nothing invented"
    );
}

/// The refusal turn has to be replayable: it is appended to the caller's
/// history and sent back on the next turn, so the wire→rig→wire round trip of
/// a refusal-only assistant message must survive.
#[tokio::test]
async fn chat_refusal_turn_survives_into_history() {
    const SCENARIO: &str = "refusal_matrix/chat_refusal_turn_survives_into_history";

    with_openai_refusal_cassette(
        "refusal_matrix/chat_refusal_turn_survives_into_history",
        |client| async move {
            let agent = client
                .completions_api()
                .agent(REFUSING_MODEL)
                .additional_params(chat_response_format())
                .build();

            let mut history: Vec<Message> = Vec::new();
            let refusal = agent
                .chat(REFUSED_PROMPT, &mut history)
                .await
                .expect("refusal turn");
            assert_nonempty_response(&refusal);
            assert!(
                history.len() >= 2,
                "the refusal turn must be appended to history: {history:?}"
            );

            let followup = agent
                .chat(ANSWERABLE_PROMPT, &mut history)
                .await
                .expect("a follow-up over a history containing a refusal must be accepted");
            assert_nonempty_response(&followup);
        },
    )
    .await;

    assert_recorded_chat_refusal(SCENARIO);
}

// ---------------------------------------------------------------------------
// Chat Completions controls — ordinary turns must be byte-for-byte unaffected.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn chat_control_non_refusing_prompt_is_unchanged() {
    const SCENARIO: &str = "refusal_matrix/chat_control_non_refusing_prompt_is_unchanged";

    with_openai_refusal_cassette(
        "refusal_matrix/chat_control_non_refusing_prompt_is_unchanged",
        |client| async move {
            let model = client.completions_api().completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(ANSWERABLE_PROMPT)
                .additional_params(chat_response_format())
                .build();

            let response = model.completion(request).await.expect("ordinary turn");
            let text = assistant_text_response(&response.choice).expect("assistant text");

            assert!(
                text.trim_start().starts_with('{'),
                "an unrefused structured-output turn still answers in the schema: {text}"
            );
        },
    )
    .await;

    assert_recorded_no_chat_refusal(SCENARIO);
}

#[tokio::test]
async fn chat_control_non_refusing_stream_is_unchanged() {
    const SCENARIO: &str = "refusal_matrix/chat_control_non_refusing_stream_is_unchanged";

    with_openai_refusal_cassette(
        "refusal_matrix/chat_control_non_refusing_stream_is_unchanged",
        |client| async move {
            let model = client.completions_api().completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(ANSWERABLE_PROMPT)
                .additional_params(chat_response_format())
                .build();

            let stream = model.stream(request).await.expect("stream should connect");
            let observed = collect_raw_stream_observation(stream).await;

            assert!(
                observed.text.trim_start().starts_with('{'),
                "{}",
                observed.text
            );
        },
    )
    .await;

    assert_recorded_no_chat_refusal(SCENARIO);
}

/// Not every model refuses: `gpt-4o-mini` answers the same prompt *inside* the
/// schema, with `refusal: null`. This pins why the matrix uses `gpt-4o`.
#[tokio::test]
async fn chat_control_mini_answers_inside_schema() {
    const SCENARIO: &str = "refusal_matrix/chat_control_mini_answers_inside_schema";

    with_openai_refusal_cassette(
        "refusal_matrix/chat_control_mini_answers_inside_schema",
        |client| async move {
            let model = client
                .completions_api()
                .completion_model(openai::GPT_4O_MINI);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .additional_params(chat_response_format())
                .build();

            let response = model.completion(request).await.expect("turn");
            let text = assistant_text_response(&response.choice).expect("assistant text");

            assert!(text.trim_start().starts_with('{'), "{text}");
        },
    )
    .await;

    assert_recorded_no_chat_refusal(SCENARIO);
}

/// Without structured output the same prompt comes back as ordinary
/// `content`, never the `refusal` field — the bug's blast radius really is
/// scoped to strict structured-output turns.
#[tokio::test]
async fn chat_control_plain_refusal_is_content_not_refusal() {
    const SCENARIO: &str = "refusal_matrix/chat_control_plain_refusal_is_content_not_refusal";

    with_openai_refusal_cassette(
        "refusal_matrix/chat_control_plain_refusal_is_content_not_refusal",
        |client| async move {
            let model = client.completions_api().completion_model(REFUSING_MODEL);
            let request = model.completion_request(REFUSED_PROMPT).build();

            let response = model.completion(request).await.expect("turn");

            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("assistant text"),
            );
        },
    )
    .await;

    assert_recorded_no_chat_refusal(SCENARIO);
}

// ---------------------------------------------------------------------------
// Responses API — the surface that already worked, pinned so it stays that way.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn responses_blocking_refusal_part_surfaces() {
    const SCENARIO: &str = "refusal_matrix/responses_blocking_refusal_part_surfaces";

    with_openai_refusal_cassette(
        "refusal_matrix/responses_blocking_refusal_part_surfaces",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .additional_params(responses_text_format())
                .build();

            let response = model.completion(request).await.expect("refusal turn");

            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("assistant text"),
            );
        },
    )
    .await;

    assert_recorded_responses_refusal_part(SCENARIO);
}

#[tokio::test]
async fn responses_streaming_refusal_delta_surfaces() {
    const SCENARIO: &str = "refusal_matrix/responses_streaming_refusal_delta_surfaces";

    with_openai_refusal_cassette(
        "refusal_matrix/responses_streaming_refusal_delta_surfaces",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .additional_params(responses_text_format())
                .build();

            let stream = model.stream(request).await.expect("stream should connect");
            let observed = collect_raw_stream_observation(stream).await;

            assert_nonempty_response(&observed.text);
        },
    )
    .await;

    assert_recorded_responses_refusal_stream(SCENARIO);
}

#[tokio::test]
async fn responses_agent_blocking_refusal_surfaces() {
    const SCENARIO: &str = "refusal_matrix/responses_agent_blocking_refusal_surfaces";

    with_openai_refusal_cassette(
        "refusal_matrix/responses_agent_blocking_refusal_surfaces",
        |client| async move {
            let agent = client
                .agent(REFUSING_MODEL)
                .additional_params(responses_text_format())
                .build();

            let response = agent.prompt(REFUSED_PROMPT).await.expect("refusal turn");

            assert_nonempty_response(&response);
        },
    )
    .await;

    assert_recorded_responses_refusal_part(SCENARIO);
}

#[tokio::test]
async fn responses_agent_streaming_refusal_surfaces() {
    const SCENARIO: &str = "refusal_matrix/responses_agent_streaming_refusal_surfaces";

    with_openai_refusal_cassette(
        "refusal_matrix/responses_agent_streaming_refusal_surfaces",
        |client| async move {
            let agent = client
                .agent(REFUSING_MODEL)
                .additional_params(responses_text_format())
                .build();

            let mut stream = agent.stream_prompt(REFUSED_PROMPT).await;
            let observed = collect_stream_observation(&mut stream).await;

            assert!(observed.errors.is_empty(), "{:?}", observed.errors);
            assert_nonempty_response(&observed.all_streamed_text);
        },
    )
    .await;

    assert_recorded_responses_refusal_stream(SCENARIO);
}

/// The cross-API parity the bug broke: one client, one prompt, both surfaces,
/// one cassette. Either both deliver the refusal or the matrix is wrong.
#[tokio::test]
async fn cross_surface_refusal_parity() {
    const SCENARIO: &str = "refusal_matrix/cross_surface_refusal_parity";

    with_openai_refusal_cassette(
        "refusal_matrix/cross_surface_refusal_parity",
        |client| async move {
            let responses_model = client.clone().completion_model(REFUSING_MODEL);
            let responses_text = assistant_text_response(
                &responses_model
                    .completion(
                        responses_model
                            .completion_request(REFUSED_PROMPT)
                            .additional_params(responses_text_format())
                            .build(),
                    )
                    .await
                    .expect("responses refusal turn")
                    .choice,
            )
            .expect("responses refusal text");

            let chat_model = client.completions_api().completion_model(REFUSING_MODEL);
            let chat_text = assistant_text_response(
                &chat_model
                    .completion(
                        chat_model
                            .completion_request(REFUSED_PROMPT)
                            .additional_params(chat_response_format())
                            .build(),
                    )
                    .await
                    .expect("chat refusal turn")
                    .choice,
            )
            .expect("chat refusal text");

            assert_nonempty_response(&responses_text);
            assert_nonempty_response(&chat_text);
        },
    )
    .await;

    assert_recorded_responses_refusal_part(SCENARIO);
    assert_recorded_chat_refusal(SCENARIO);
}

// ---------------------------------------------------------------------------
// Fixture-premise checks: every cell asserts against its own recorded bytes.
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

/// The blocking chat-completions premise: some recorded choice carries a
/// non-empty top-level `refusal`.
fn chat_refusals(scenario: &str) -> Vec<String> {
    recorded_response_bodies(scenario)
        .iter()
        .filter_map(|body| serde_json::from_str::<Value>(body).ok())
        .filter_map(|body| {
            Some(
                body.get("choices")?
                    .as_array()?
                    .iter()
                    .filter_map(|choice| {
                        choice
                            .get("message")?
                            .get("refusal")?
                            .as_str()
                            .filter(|refusal| !refusal.is_empty())
                            .map(ToOwned::to_owned)
                    })
                    .collect::<Vec<_>>(),
            )
        })
        .flatten()
        .collect()
}

fn assert_recorded_chat_refusal(scenario: &str) {
    assert!(
        !chat_refusals(scenario).is_empty(),
        "cassette {scenario} no longer records a top-level `message.refusal`; \
         this cell would pass while covering nothing"
    );
}

fn assert_recorded_no_chat_refusal(scenario: &str) {
    assert!(
        chat_refusals(scenario).is_empty(),
        "control cassette {scenario} unexpectedly records a `message.refusal`"
    );
}

/// Every recorded `delta.refusal` fragment, concatenated — the exact visible
/// text a correct stream must deliver.
fn recorded_refusal_delta_text(scenario: &str) -> String {
    recorded_chat_deltas(scenario)
        .iter()
        .filter_map(|delta| delta.get("refusal").and_then(Value::as_str))
        .collect()
}

fn recorded_chat_deltas(scenario: &str) -> Vec<Value> {
    let bodies = recorded_response_bodies(scenario);
    bodies
        .iter()
        .flat_map(|body| body.lines())
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .filter(|data| !data.is_empty() && *data != "[DONE]")
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .filter_map(|chunk| {
            Some(
                chunk
                    .get("choices")?
                    .as_array()?
                    .iter()
                    .filter_map(|choice| choice.get("delta").cloned())
                    .collect::<Vec<_>>(),
            )
        })
        .flatten()
        .collect()
}

/// The streaming premise: some recorded SSE body carries a non-empty
/// `delta.refusal` and never a non-empty `delta.content`.
fn assert_recorded_chat_refusal_stream(scenario: &str) {
    let deltas = recorded_chat_deltas(scenario);

    assert!(
        deltas.iter().any(|delta| delta
            .get("refusal")
            .and_then(Value::as_str)
            .is_some_and(|refusal| !refusal.is_empty())),
        "cassette {scenario} no longer records a non-empty `delta.refusal`"
    );
    assert!(
        !deltas.iter().any(|delta| delta
            .get("content")
            .and_then(Value::as_str)
            .is_some_and(|content| !content.is_empty())),
        "cassette {scenario} records `delta.content` too, so it no longer \
         isolates the refusal-only stream this cell is about"
    );
}

/// The Responses premise: a `refusal` **content part** inside an output
/// message — the shape chat completions never sends.
fn assert_recorded_responses_refusal_part(scenario: &str) {
    let found = recorded_response_bodies(scenario)
        .iter()
        .filter_map(|body| serde_json::from_str::<Value>(body).ok())
        .any(|body| responses_body_has_refusal_part(&body));

    assert!(
        found,
        "cassette {scenario} no longer records a Responses `refusal` content part"
    );
}

fn assert_recorded_responses_refusal_stream(scenario: &str) {
    let found = recorded_response_bodies(scenario)
        .iter()
        .flat_map(|body| body.lines())
        .filter_map(|line| line.strip_prefix("data:"))
        .map(str::trim)
        .filter_map(|data| serde_json::from_str::<Value>(data).ok())
        .any(|chunk| {
            chunk.get("type").and_then(Value::as_str) == Some("response.refusal.delta")
                && chunk
                    .get("delta")
                    .and_then(Value::as_str)
                    .is_some_and(|delta| !delta.is_empty())
        });

    assert!(
        found,
        "cassette {scenario} no longer records a `response.refusal.delta` event"
    );
}

fn responses_body_has_refusal_part(body: &Value) -> bool {
    body.get("output")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(|item| item.get("content"))
        .filter_map(Value::as_array)
        .flatten()
        .any(|part| {
            part.get("type").and_then(Value::as_str) == Some("refusal")
                && part
                    .get("refusal")
                    .and_then(Value::as_str)
                    .is_some_and(|refusal| !refusal.is_empty())
        })
}
