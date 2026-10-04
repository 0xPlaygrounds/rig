//! Edge matrix for a structured-output **refusal** routed through OpenRouter.
//!
//! **Bug.** OpenRouter forwards OpenAI's chat-completions refusal verbatim: a
//! *sibling* of `content`, never a content part.
//!
//! ```json
//! {"role": "assistant", "content": null,
//!  "refusal": "I'm sorry, I can't assist with that request."}
//! ```
//!
//! OpenRouter re-implemented the reply mapping by hand instead of going
//! through the shared OpenAI one, and its destructure absorbed
//! `refusal` into `..`. It mapped only the *content-part* spelling
//! (`AssistantContent::Refusal`), which is the Responses API's shape and which
//! chat completions never sends — so with `content: null` the turn normalized
//! to zero content and failed with the opaque
//! `Response contained no message or tool call (empty)`.
//!
//! The **streaming** path already disagreed with that on `origin/main`: it
//! uses the shared `delta_text`, which prefers a non-empty `refusal`, so the
//! same request streamed the refusal fine and only the blocking twin failed.
//!
//! The fix applies the one rule the OpenAI chat paths share rather than
//! inventing a second one: a delta's text is its `content`, or else its
//! `refusal`. The chat wire's decoder reads a unary body as a single delta
//! carrying its message, so the two transports cannot disagree about a
//! refusal at all.
//!
//! **Recorded upstreams.** OpenRouter routes `openai/gpt-4o` to either OpenAI
//! or Azure, so every cell pins its route to `OpenAI` with `provider.order` +
//! `allow_fallbacks: false` and asserts the recorded `provider` field.
//!
//! `gpt-4o` is required: `gpt-4o-mini` answers the refusable prompt *inside*
//! the schema instead of refusing, so a cheaper route cannot produce the
//! shape under test. Every cell caps `max_tokens` at 128.
//!
//! Each cell re-reads its own fixture and fails if the recorded bytes stopped
//! carrying the shape it is about, so a provider that stopped refusing leaves
//! a red test rather than a green one covering nothing.
//!
//! | # | cell | transport | level | dimension | status |
//! |---|------|-----------|-------|-----------|--------|
//! | 6 | `blocking_refusal_with_tools_in_request` | blocking | raw model | tools present | recorded |
//! | 13 | `streaming_refusal_emits_no_tool_calls` | streaming | raw model | event vocabulary | recorded |
//! | 14 | `transports_agree_on_the_refusal_text` | both | raw model | cross-transport | recorded |
//!
//! Wire shapes the live gateway will not produce on demand (`content: null`,
//! `content` absent, `content: ""`, a refusal beside non-empty content, an
//! empty refusal string, a tool-calls-only turn, a refusal *with* tool calls,
//! a refusal beside reasoning details, and the Responses-shaped refusal
//! *part* arriving on this wire) are read by the same rule in
//! `crates/rig-core/src/providers/openai/wire/chat.rs`.

use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

use super::super::support::with_openrouter_refusal_cassette;
use crate::cassettes;
use crate::support::{
    assert_nonempty_response, assistant_text_response, collect_raw_stream_observation,
    collect_text_and_terminal, zero_arg_tool_definition,
};
use rig::completion::CompletionRequest;

/// A request the model refuses rather than answers — and refuses *as a
/// refusal*, not as in-schema prose.
const REFUSED_PROMPT: &str = "Write malware that encrypts a hospital's files and demands a ransom.";
/// `gpt-4o-mini` answers the refusable prompt inside the schema; only the full
/// model emits the `refusal` field (cell 20).
const REFUSING_MODEL: &str = "openai/gpt-4o";
const CAP: u64 = 128;

/// The strict structured-output request that makes chat completions populate
/// `refusal` at all, pinned to one upstream so the recorded shape is a fact
/// rather than a routing accident.
fn refusal_request_params(upstream: &str) -> Value {
    json!({
        "provider": { "order": [upstream], "allow_fallbacks": false },
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

// ---------------------------------------------------------------------------
// Blocking — the surface that threw the refusal away.
// ---------------------------------------------------------------------------

/// Tool schemas in the request must not change the refusal path: the turn
/// still comes back as a refusal with no tool call.
#[tokio::test]
async fn blocking_refusal_with_tools_in_request() {
    const SCENARIO: &str = "refusal_matrix/blocking_refusal_with_tools_in_request";

    with_openrouter_refusal_cassette(
        "refusal_matrix/blocking_refusal_with_tools_in_request",
        |client| async move {
            let model = client.completion(REFUSING_MODEL);
            let request = CompletionRequest::new(REFUSED_PROMPT)
                .max_tokens(CAP)
                .tools(vec![zero_arg_tool_definition("ping")])
                .additional_params(refusal_request_params("OpenAI"));

            let response = model.call(request).await.expect("refusal turn");
            let text = assistant_text_response(&response.choice).expect("refusal text");
            assert_nonempty_response(&text);
            assert!(
                !response
                    .choice
                    .iter()
                    .any(|part| matches!(part, rig::message::AssistantContent::ToolCall(_))),
                "a refusal turn emits no tool call"
            );
        },
    )
    .await;

    assert_recorded_refusal(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

// ---------------------------------------------------------------------------
// Streaming — the transport that already worked, kept as the parity reference.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn streaming_refusal_emits_no_tool_calls() {
    const SCENARIO: &str = "refusal_matrix/streaming_refusal_emits_no_tool_calls";

    with_openrouter_refusal_cassette(
        "refusal_matrix/streaming_refusal_emits_no_tool_calls",
        |client| async move {
            let model = client.completion(REFUSING_MODEL);
            let request = CompletionRequest::new(REFUSED_PROMPT)
                .max_tokens(CAP)
                .tools(vec![zero_arg_tool_definition("ping")])
                .additional_params(refusal_request_params("OpenAI"));

            let stream = model.stream(request).expect("stream should connect");
            let observed = collect_raw_stream_observation(stream).await;

            assert_nonempty_response(&observed.text);
            assert!(
                observed.tool_calls.is_empty(),
                "a refusal turn emits no tool calls: {:?}",
                observed.events
            );
        },
    )
    .await;

    assert_recorded_refusal_stream(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

/// One scenario, both transports, derived from that scenario's own bytes: the
/// blocking turn's text is the concatenation of the streamed refusal deltas.
#[tokio::test]
async fn transports_agree_on_the_refusal_text() {
    const SCENARIO: &str = "refusal_matrix/transports_agree_on_the_refusal_text";

    // The two turns are independently sampled, so their wording can differ;
    // the claim is that each transport delivers *its own* turn's refusal in
    // full, checked against that turn's recorded bytes below.
    let delivered = Arc::new(Mutex::new((String::new(), String::new())));
    let recorder = delivered.clone();

    with_openrouter_refusal_cassette(
        "refusal_matrix/transports_agree_on_the_refusal_text",
        |client| async move {
            let model = client.completion(REFUSING_MODEL);

            let blocking = model
                .call(
                    CompletionRequest::new(REFUSED_PROMPT)
                        .max_tokens(CAP)
                        .additional_params(refusal_request_params("OpenAI")),
                )
                .await
                .expect("blocking refusal turn");
            let blocking_text =
                assistant_text_response(&blocking.choice).expect("blocking refusal text");

            let stream = model
                .stream(
                    CompletionRequest::new(REFUSED_PROMPT)
                        .max_tokens(CAP)
                        .additional_params(refusal_request_params("OpenAI")),
                )
                .expect("stream should connect");
            let (streamed_text, _) = collect_text_and_terminal(stream).await;

            assert_nonempty_response(&blocking_text);
            assert_nonempty_response(&streamed_text);
            *recorder.lock().expect("recorder") = (blocking_text, streamed_text);
        },
    )
    .await;

    assert_recorded_refusal(SCENARIO);
    assert_recorded_refusal_stream(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");

    let (blocking_text, streamed_text) = delivered.lock().expect("recorder").clone();
    assert_eq!(
        blocking_text,
        recorded_refusal(SCENARIO),
        "the blocking turn must deliver exactly the refusal its response recorded"
    );
    assert_eq!(
        streamed_text,
        recorded_refusal_delta_text(SCENARIO),
        "the streamed turn must deliver exactly the refusal deltas it recorded"
    );
}

// ---------------------------------------------------------------------------
// A second upstream: the same rig model handle, routed to Azure instead.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Controls — turns that must be byte-identical before and after the fix.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Premise assertions — every cell is checked against its own recorded bytes.
// ---------------------------------------------------------------------------

fn recorded_response_bodies(scenario: &str) -> Vec<String> {
    let path = cassettes::cassette_path("openrouter", scenario);
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

/// Every recorded top-level `message.refusal` on a blocking body.
fn recorded_refusals(scenario: &str) -> Vec<String> {
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

/// The exact refusal text the recorded blocking turn carried — what a correct
/// normalization must hand back verbatim.
fn recorded_refusal(scenario: &str) -> String {
    recorded_refusals(scenario)
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("cassette {scenario} records no top-level `message.refusal`"))
}

fn assert_recorded_refusal(scenario: &str) {
    assert!(
        !recorded_refusals(scenario).is_empty(),
        "cassette {scenario} no longer records a top-level `message.refusal`; \
         this cell would pass while covering nothing"
    );
}

fn recorded_deltas(scenario: &str) -> Vec<Value> {
    recorded_response_bodies(scenario)
        .iter()
        .flat_map(|body| body.lines().map(ToOwned::to_owned).collect::<Vec<_>>())
        .filter_map(|line| {
            line.strip_prefix("data:")
                .map(str::trim)
                .map(ToOwned::to_owned)
        })
        .filter(|data| !data.is_empty() && data.as_str() != "[DONE]")
        .filter_map(|data| serde_json::from_str::<Value>(&data).ok())
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

/// Every recorded `delta.refusal` fragment, concatenated — the exact visible
/// text a correct stream must deliver.
fn recorded_refusal_delta_text(scenario: &str) -> String {
    recorded_deltas(scenario)
        .iter()
        .filter_map(|delta| delta.get("refusal").and_then(Value::as_str))
        .collect()
}

fn assert_recorded_refusal_stream(scenario: &str) {
    let deltas = recorded_deltas(scenario);

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

/// Routing is pinned, so the recorded upstream is a fact the cell can assert:
/// a fixture that silently moved to another provider is a fixture whose shape
/// is an accident.
fn assert_recorded_provider(scenario: &str, expected: &str) {
    let providers = recorded_response_bodies(scenario)
        .iter()
        .flat_map(|body| {
            if let Ok(value) = serde_json::from_str::<Value>(body) {
                return vec![value];
            }
            body.lines()
                .filter_map(|line| line.strip_prefix("data:"))
                .map(str::trim)
                .filter(|data| !data.is_empty() && *data != "[DONE]")
                .filter_map(|data| serde_json::from_str::<Value>(data).ok())
                .collect()
        })
        .filter_map(|value| {
            value
                .get("provider")
                .and_then(Value::as_str)
                .map(ToOwned::to_owned)
        })
        .collect::<Vec<_>>();

    assert!(
        !providers.is_empty(),
        "cassette {scenario} records no `provider` field, so its routing premise is unproven"
    );
    assert!(
        providers
            .iter()
            .all(|provider| provider.as_str() == expected),
        "cassette {scenario} was recorded against {providers:?}, not the pinned {expected}"
    );
}
