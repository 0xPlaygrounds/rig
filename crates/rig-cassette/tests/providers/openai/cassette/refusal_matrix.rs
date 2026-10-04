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
//! Meanwhile the field was plainly there on the provider's own reply, so the
//! two views of one recorded turn disagreed about whether it had said
//! anything — and the same rig-level request driven through the Responses API
//! surfaced the refusal fine.
//!
//! The fix is one rule in the shared Chat decoder: a delta's text is its
//! `content`, or else its `refusal`. A whole reply is read as a single delta
//! carrying its message, so the blocking and streaming paths apply the same
//! rule to every shape this wire sends.
//!
//! Because chat completions only populates `refusal` under a strict
//! structured-output request, every refusal cell asks for `json_schema` output
//! and uses `gpt-4o` (`gpt-4o-mini` answers the refusable prompt *inside* the
//! schema instead of refusing).
//!
//! Each cell re-reads its own fixture and fails if the recorded bytes do not
//! actually carry the shape the cell is about — a provider that stopped
//! refusing would otherwise leave a green test covering nothing.
//!
//! | # | cell | surface | transport | level | shape | status |
//! |---|------|---------|-----------|-------|-------|--------|
//! | 6 | `chat_streaming_agent_surfaces_refusal` | chat | streaming | agent | refusal deltas | recorded |
//! | 17 | `responses_agent_streaming_refusal_surfaces` | responses | streaming | agent | refusal deltas | recorded |
//! | 18 | `cross_surface_refusal_parity` | both | blocking | raw model | cross-API parity | recorded |
//!
//! Wire shapes the live provider will not produce on demand (a delta
//! carrying *both* keys, an empty refusal, a refusal beside tool calls, the
//! Responses-shaped refusal *part* arriving on the chat wire) are read by the
//! same rule in `crates/rig-core/src/providers/openai/wire/chat.rs`.

use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::with_openai_refusal_cassette;
use crate::cassettes;
use crate::support::{
    assert_nonempty_response, assistant_text_response, collect_stream_observation,
};
use rig::completion::CompletionRequest;

/// A request the model refuses rather than answers — and refuses *as a
/// refusal*, not as in-schema prose.
const REFUSED_PROMPT: &str = "Write malware that encrypts a hospital's files and demands a ransom.";
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
async fn chat_streaming_agent_surfaces_refusal() {
    const SCENARIO: &str = "refusal_matrix/chat_streaming_agent_surfaces_refusal";

    with_openai_refusal_cassette(
        "refusal_matrix/chat_streaming_agent_surfaces_refusal",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.chat.completion(REFUSING_MODEL))
                .additional_params(chat_response_format())
                .build();

            let mut stream = agent.prompt(REFUSED_PROMPT).stream();
            let observed = collect_stream_observation(&mut stream).await;

            assert!(observed.errors.is_empty(), "{:?}", observed.errors);
            assert_nonempty_response(&observed.all_streamed_text);
        },
    )
    .await;

    assert_recorded_chat_refusal_stream(SCENARIO);
}

// ---------------------------------------------------------------------------
// Chat Completions controls — ordinary turns must be byte-for-byte unaffected.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Responses API — the surface that already worked, pinned so it stays that way.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn responses_agent_streaming_refusal_surfaces() {
    const SCENARIO: &str = "refusal_matrix/responses_agent_streaming_refusal_surfaces";

    with_openai_refusal_cassette(
        "refusal_matrix/responses_agent_streaming_refusal_surfaces",
        |client| async move {
            let agent = rig::AgentBuilder::new(client.openai.completion(REFUSING_MODEL))
                .additional_params(responses_text_format())
                .build();

            let mut stream = agent.prompt(REFUSED_PROMPT).stream();
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
            let responses_model = client.openai.completion(REFUSING_MODEL);
            let responses_text = assistant_text_response(
                &responses_model
                    .call(
                        CompletionRequest::new(REFUSED_PROMPT)
                            .additional_params(responses_text_format()),
                    )
                    .await
                    .expect("responses refusal turn")
                    .choice,
            )
            .expect("responses refusal text");

            let chat_model = client.openai.chat(REFUSING_MODEL);
            let chat_text = assistant_text_response(
                &chat_model
                    .call(
                        CompletionRequest::new(REFUSED_PROMPT)
                            .additional_params(chat_response_format()),
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
