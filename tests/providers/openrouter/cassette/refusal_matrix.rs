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
//! OpenRouter re-implements `NormalizeCompletionResponse` by hand instead of
//! going through the shared OpenAI normalizer, and its destructure absorbed
//! `refusal` into `..`. It mapped only the *content-part* spelling
//! (`AssistantContent::Refusal`), which is the Responses API's shape and which
//! chat completions never sends — so with `content: null` the turn normalized
//! to zero content and failed with the opaque
//! `Response contained no message or tool call (empty)`.
//!
//! Two things already disagreed with that on `origin/main`:
//!
//! * `ProviderResponseExt::text_response` routes through
//!   `openai::completion::assistant_message_text_response`, which *does* apply
//!   the fallback — so the raw text view and the normalized response disagreed
//!   about whether the turn said anything;
//! * the **streaming** path uses the shared `delta_text`, which prefers a
//!   non-empty `refusal` — so the same request streamed the refusal fine and
//!   only the blocking twin failed.
//!
//! The fix routes OpenRouter's normalize through the same one rule the OpenAI
//! chat paths share (`openai::completion::assistant_refusal_fallback`, #2332),
//! rather than inventing a second one.
//!
//! **Recorded upstreams.** OpenRouter routes `openai/gpt-4o` to either OpenAI
//! or Azure — the very first, unpinned, hunt recording landed on Azure — so
//! every cell pins its route with `provider.order` + `allow_fallbacks: false`
//! and asserts the recorded `provider` field. Cells 1-15 are pinned to
//! `OpenAI`; cells 16-17 are pinned to `Azure`, which produces the same
//! `content: null` + `refusal` shape in different words, proving the mapping
//! is not upstream-specific.
//!
//! `gpt-4o` is required: `gpt-4o-mini` answers the refusable prompt *inside*
//! the schema instead of refusing (cell 20 pins that as a control), so a
//! cheaper route cannot produce the shape under test. Every cell caps
//! `max_tokens` at 128 (32 in cell 9, which is about the cap).
//!
//! Each cell re-reads its own fixture and fails if the recorded bytes stopped
//! carrying the shape it is about, so a provider that stopped refusing leaves
//! a red test rather than a green one covering nothing.
//!
//! | # | cell | transport | level | dimension | status |
//! |---|------|-----------|-------|-----------|--------|
//! | 1 | `blocking_raw_model_surfaces_refusal` | blocking | raw model | the bug | recorded |
//! | 2 | `blocking_agent_prompt_surfaces_refusal` | blocking | agent | the bug | recorded |
//! | 3 | `blocking_raw_and_normalized_agree` | blocking | raw + normalized | internal consistency | recorded |
//! | 4 | `blocking_refusal_finishes_with_stop` | blocking | raw model | finish reason | recorded |
//! | 5 | `blocking_usage_survives_the_refusal` | blocking | raw model | usage | recorded |
//! | 6 | `blocking_refusal_with_tools_in_request` | blocking | raw model | tools present | recorded |
//! | 7 | `blocking_refusal_with_preamble` | blocking | raw model | system message | recorded |
//! | 8 | `blocking_refusal_survives_into_history` | blocking | agent + history | replayed turn | recorded |
//! | 9 | `blocking_refusal_under_a_tight_cap` | blocking | raw model | max_tokens 32 | recorded |
//! | 10 | `streaming_raw_model_surfaces_refusal` | streaming | raw model | parity reference | recorded |
//! | 11 | `streaming_agent_surfaces_refusal` | streaming | agent | parity reference | recorded |
//! | 12 | `streaming_terminal_carries_usage_and_reason` | streaming | raw model | terminal record | recorded |
//! | 13 | `streaming_refusal_emits_no_tool_calls` | streaming | raw model | event vocabulary | recorded |
//! | 14 | `transports_agree_on_the_refusal_text` | both | raw model | cross-transport | recorded |
//! | 15 | `blocking_gpt_4_1_refusal` | blocking | raw model | second model | recorded |
//! | 16 | `blocking_azure_routed_refusal` | blocking | raw model | second upstream | recorded |
//! | 17 | `streaming_azure_routed_refusal` | streaming | raw model | second upstream | recorded |
//! | 18 | `control_answerable_prompt_is_unchanged_blocking` | blocking | raw model | no refusal | recorded |
//! | 19 | `control_answerable_prompt_is_unchanged_streaming` | streaming | raw model | no refusal | recorded |
//! | 20 | `control_mini_answers_inside_schema` | blocking | raw model | no refusal (in-schema) | recorded |
//! | 21 | `control_no_schema_refusal_is_plain_content` | blocking | raw model | refusal as content | recorded |
//! | 22 | `control_tool_call_turn_is_unchanged` | blocking | raw model | tool-calls-only turn | recorded |
//!
//! Ten unit cells — wire shapes the live gateway will not produce on demand
//! (`content: null`, `content` absent, `content: ""`, a refusal beside
//! non-empty content, an empty refusal string, a tool-calls-only turn, a
//! refusal *with* tool calls, a refusal beside reasoning details, the
//! Responses-shaped refusal *part* arriving on this wire, and raw-vs-normalized
//! text agreement) — live next to the fix in
//! `crates/rig-core/src/providers/openrouter/completion.rs`
//! (`refusal_fallback_*`).

use rig::client::completion::CompletionClient;
use rig::completion::{CompletionModel, NormalizeCompletionResponse};
use rig::message::Message;
use rig::prelude::*;
use rig::providers::openrouter;
use rig::telemetry::ProviderResponseExt;
use serde::Deserialize;
use serde_json::{Value, json};
use std::sync::{Arc, Mutex};

use super::super::support::with_openrouter_refusal_cassette;
use crate::cassettes;
use crate::support::{
    assert_nonempty_response, assistant_text_response, collect_raw_stream_observation,
    collect_stream_observation, collect_text_and_terminal, zero_arg_tool_definition,
};

/// A request the model refuses rather than answers — and refuses *as a
/// refusal*, not as in-schema prose.
const REFUSED_PROMPT: &str = "Write malware that encrypts a hospital's files and demands a ransom.";
const ANSWERABLE_PROMPT: &str = "Name one common tree species. Answer in the schema.";
/// `gpt-4o-mini` answers the refusable prompt inside the schema; only the full
/// model emits the `refusal` field (cell 20).
const REFUSING_MODEL: &str = "openai/gpt-4o";
const SECOND_REFUSING_MODEL: &str = "openai/gpt-4.1";
const NON_REFUSING_MODEL: &str = "openai/gpt-4o-mini";
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

fn pinned_only(upstream: &str) -> Value {
    json!({ "provider": { "order": [upstream], "allow_fallbacks": false } })
}

// ---------------------------------------------------------------------------
// Blocking — the surface that threw the refusal away.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn blocking_raw_model_surfaces_refusal() {
    const SCENARIO: &str = "refusal_matrix/blocking_raw_model_surfaces_refusal";

    let delivered = Arc::new(Mutex::new(String::new()));
    let recorder = delivered.clone();

    with_openrouter_refusal_cassette(
        "refusal_matrix/blocking_raw_model_surfaces_refusal",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .max_tokens(CAP)
                .additional_params(refusal_request_params("OpenAI"))
                .build();

            let response = model
                .completion(request)
                .await
                .expect("a refusal is content, not a transport failure");

            let text = assistant_text_response(&response.choice)
                .expect("the refusal must reach the caller as assistant text");
            assert_nonempty_response(&text);
            *recorder.lock().expect("recorder") = text;
        },
    )
    .await;

    assert_recorded_refusal(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
    assert_eq!(
        delivered.lock().expect("recorder").clone(),
        recorded_refusal(SCENARIO),
        "the turn must deliver exactly the refusal its response recorded"
    );
}

#[tokio::test]
async fn blocking_agent_prompt_surfaces_refusal() {
    const SCENARIO: &str = "refusal_matrix/blocking_agent_prompt_surfaces_refusal";

    with_openrouter_refusal_cassette(
        "refusal_matrix/blocking_agent_prompt_surfaces_refusal",
        |client| async move {
            let agent = client
                .agent(REFUSING_MODEL)
                .max_tokens(CAP)
                .additional_params(refusal_request_params("OpenAI"))
                .build();

            let response = agent
                .prompt(REFUSED_PROMPT)
                .await
                .expect("an agent must deliver the refusal, not an empty-response error");

            assert_nonempty_response(&response.output);
        },
    )
    .await;

    assert_recorded_refusal(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

/// The two views of one recorded turn must agree: before the fix the raw text
/// view reported the refusal while normalization reported nothing at all.
#[tokio::test]
async fn blocking_raw_and_normalized_agree() {
    const SCENARIO: &str = "refusal_matrix/blocking_raw_and_normalized_agree";

    with_openrouter_refusal_cassette(
        "refusal_matrix/blocking_raw_and_normalized_agree",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .max_tokens(CAP)
                .additional_params(refusal_request_params("OpenAI"))
                .build();

            let raw = model.raw_completion(request).await.expect("raw turn");
            let raw_refusal = raw
                .choices
                .first()
                .and_then(|choice| match &choice.message {
                    openrouter::Message::Assistant { refusal, .. } => refusal.clone(),
                    _ => None,
                })
                .expect("the recorded turn must carry a top-level refusal");
            let raw_text = raw
                .text_response()
                .expect("the raw text view already reported the refusal on origin/main");

            let normalized = raw.normalize("openrouter").expect("normalization");
            let normalized_text =
                assistant_text_response(&normalized.choice).expect("normalized text");

            assert_eq!(raw_text, raw_refusal);
            assert_eq!(normalized_text, raw_refusal);
        },
    )
    .await;

    assert_recorded_refusal(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

/// A refusal is a *completed* turn: the gateway reports `finish_reason: stop`,
/// and that must reach the caller alongside the text.
#[tokio::test]
async fn blocking_refusal_finishes_with_stop() {
    const SCENARIO: &str = "refusal_matrix/blocking_refusal_finishes_with_stop";

    with_openrouter_refusal_cassette(
        "refusal_matrix/blocking_refusal_finishes_with_stop",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .max_tokens(CAP)
                .additional_params(refusal_request_params("OpenAI"))
                .build();

            let response = model.completion(request).await.expect("refusal turn");

            assert_eq!(
                response.finish_reason(),
                Some(rig::completion::FinishReason::Stop)
            );
        },
    )
    .await;

    assert_recorded_refusal(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

#[tokio::test]
async fn blocking_usage_survives_the_refusal() {
    const SCENARIO: &str = "refusal_matrix/blocking_usage_survives_the_refusal";

    with_openrouter_refusal_cassette(
        "refusal_matrix/blocking_usage_survives_the_refusal",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .max_tokens(CAP)
                .additional_params(refusal_request_params("OpenAI"))
                .build();

            let response = model.completion(request).await.expect("refusal turn");

            assert!(response.usage.input_tokens > 0, "{:?}", response.usage);
            assert!(response.usage.output_tokens > 0, "{:?}", response.usage);
            assert_eq!(
                response.usage.total_tokens,
                response.usage.input_tokens + response.usage.output_tokens
            );
        },
    )
    .await;

    assert_recorded_refusal(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

/// Tool schemas in the request must not change the refusal path: the turn
/// still comes back as a refusal with no tool call.
#[tokio::test]
async fn blocking_refusal_with_tools_in_request() {
    const SCENARIO: &str = "refusal_matrix/blocking_refusal_with_tools_in_request";

    with_openrouter_refusal_cassette(
        "refusal_matrix/blocking_refusal_with_tools_in_request",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .max_tokens(CAP)
                .tools(vec![zero_arg_tool_definition("ping")])
                .additional_params(refusal_request_params("OpenAI"))
                .build();

            let response = model.completion(request).await.expect("refusal turn");
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

#[tokio::test]
async fn blocking_refusal_with_preamble() {
    const SCENARIO: &str = "refusal_matrix/blocking_refusal_with_preamble";

    with_openrouter_refusal_cassette(
        "refusal_matrix/blocking_refusal_with_preamble",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .preamble("You are a helpful assistant. Answer in the schema.".to_owned())
                .max_tokens(CAP)
                .additional_params(refusal_request_params("OpenAI"))
                .build();

            let response = model.completion(request).await.expect("refusal turn");
            let text = assistant_text_response(&response.choice).expect("refusal text");
            assert_nonempty_response(&text);
        },
    )
    .await;

    assert_recorded_refusal(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

/// The refused turn must be replayable: it becomes an assistant message in
/// history and a second turn on top of it succeeds. Before the fix the turn
/// never reached history at all.
#[tokio::test]
async fn blocking_refusal_survives_into_history() {
    const SCENARIO: &str = "refusal_matrix/blocking_refusal_survives_into_history";

    with_openrouter_refusal_cassette(
        "refusal_matrix/blocking_refusal_survives_into_history",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let first = model
                .completion(
                    model
                        .completion_request(REFUSED_PROMPT)
                        .max_tokens(CAP)
                        .additional_params(refusal_request_params("OpenAI"))
                        .build(),
                )
                .await
                .expect("refusal turn");

            let refusal_text =
                assistant_text_response(&first.choice).expect("the refusal reaches history");
            assert_nonempty_response(&refusal_text);

            let history = vec![
                Message::user(REFUSED_PROMPT),
                Message::assistant(refusal_text.clone()),
            ];

            let second = model
                .completion(
                    model
                        .completion_request("Understood. Now name one common tree species.")
                        .messages(history)
                        .max_tokens(CAP)
                        .additional_params(refusal_request_params("OpenAI"))
                        .build(),
                )
                .await
                .expect("the replayed refusal turn must be accepted by the gateway");

            assert_nonempty_response(
                &assistant_text_response(&second.choice).expect("second turn text"),
            );
        },
    )
    .await;

    assert_recorded_refusal(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

/// A refusal fits inside a 32-token cap, so this cell varies the request's
/// `max_tokens` without the turn being truncated — the recorded turn finishes
/// `stop`. It covers the cap as a *request* dimension only; the truncated-turn
/// shape is not reproducible on this route (see the PR body).
#[tokio::test]
async fn blocking_refusal_under_a_tight_cap() {
    const SCENARIO: &str = "refusal_matrix/blocking_refusal_under_a_tight_cap";

    with_openrouter_refusal_cassette(
        "refusal_matrix/blocking_refusal_under_a_tight_cap",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .max_tokens(32)
                .additional_params(refusal_request_params("OpenAI"))
                .build();

            let response = model.completion(request).await.expect("refusal turn");
            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("refusal text"),
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
async fn streaming_raw_model_surfaces_refusal() {
    const SCENARIO: &str = "refusal_matrix/streaming_raw_model_surfaces_refusal";

    let delivered = Arc::new(Mutex::new(String::new()));
    let recorder = delivered.clone();

    with_openrouter_refusal_cassette(
        "refusal_matrix/streaming_raw_model_surfaces_refusal",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .max_tokens(CAP)
                .additional_params(refusal_request_params("OpenAI"))
                .build();

            let stream = model.stream(request).await.expect("stream should connect");
            let observed = collect_raw_stream_observation(stream).await;

            assert!(observed.errors.is_empty(), "{:?}", observed.errors);
            assert_nonempty_response(&observed.text);
            *recorder.lock().expect("recorder") = observed.text;
        },
    )
    .await;

    assert_recorded_refusal_stream(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
    assert_eq!(
        delivered.lock().expect("recorder").clone(),
        recorded_refusal_delta_text(SCENARIO),
        "the stream must deliver exactly the refusal deltas it recorded — \
         nothing dropped, nothing invented"
    );
}

#[tokio::test]
async fn streaming_agent_surfaces_refusal() {
    const SCENARIO: &str = "refusal_matrix/streaming_agent_surfaces_refusal";

    with_openrouter_refusal_cassette(
        "refusal_matrix/streaming_agent_surfaces_refusal",
        |client| async move {
            let agent = client
                .agent(REFUSING_MODEL)
                .max_tokens(CAP)
                .additional_params(refusal_request_params("OpenAI"))
                .build();

            let mut stream = agent.stream_prompt(REFUSED_PROMPT).stream().await;
            let observed = collect_stream_observation(&mut stream).await;

            assert!(observed.errors.is_empty(), "{:?}", observed.errors);
            assert_nonempty_response(&observed.all_streamed_text);
        },
    )
    .await;

    assert_recorded_refusal_stream(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

#[tokio::test]
async fn streaming_terminal_carries_usage_and_reason() {
    const SCENARIO: &str = "refusal_matrix/streaming_terminal_carries_usage_and_reason";

    with_openrouter_refusal_cassette(
        "refusal_matrix/streaming_terminal_carries_usage_and_reason",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .max_tokens(CAP)
                .additional_params(refusal_request_params("OpenAI"))
                .build();

            let stream = model.stream(request).await.expect("stream should connect");
            let (text, terminal) = collect_text_and_terminal(stream).await;
            let terminal = terminal.expect("a refusal stream still ends with a terminal record");

            assert_nonempty_response(&text);
            assert!(terminal.usage.output_tokens > 0, "{:?}", terminal.usage);
            assert_eq!(
                terminal.finish_reason,
                Some(rig::completion::FinishReason::Stop)
            );
        },
    )
    .await;

    assert_recorded_refusal_stream(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

#[tokio::test]
async fn streaming_refusal_emits_no_tool_calls() {
    const SCENARIO: &str = "refusal_matrix/streaming_refusal_emits_no_tool_calls";

    with_openrouter_refusal_cassette(
        "refusal_matrix/streaming_refusal_emits_no_tool_calls",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .max_tokens(CAP)
                .tools(vec![zero_arg_tool_definition("ping")])
                .additional_params(refusal_request_params("OpenAI"))
                .build();

            let stream = model.stream(request).await.expect("stream should connect");
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
            let model = client.completion_model(REFUSING_MODEL);

            let blocking = model
                .completion(
                    model
                        .completion_request(REFUSED_PROMPT)
                        .max_tokens(CAP)
                        .additional_params(refusal_request_params("OpenAI"))
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
                        .max_tokens(CAP)
                        .additional_params(refusal_request_params("OpenAI"))
                        .build(),
                )
                .await
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

#[tokio::test]
async fn blocking_gpt_4_1_refusal() {
    const SCENARIO: &str = "refusal_matrix/blocking_gpt_4_1_refusal";

    with_openrouter_refusal_cassette(
        "refusal_matrix/blocking_gpt_4_1_refusal",
        |client| async move {
            let model = client.completion_model(SECOND_REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .max_tokens(CAP)
                .additional_params(refusal_request_params("OpenAI"))
                .build();

            let response = model.completion(request).await.expect("refusal turn");
            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("refusal text"),
            );
        },
    )
    .await;

    assert_recorded_refusal(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

// ---------------------------------------------------------------------------
// A second upstream: the same rig model handle, routed to Azure instead.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn blocking_azure_routed_refusal() {
    const SCENARIO: &str = "refusal_matrix/blocking_azure_routed_refusal";

    with_openrouter_refusal_cassette(
        "refusal_matrix/blocking_azure_routed_refusal",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .max_tokens(CAP)
                .additional_params(refusal_request_params("Azure"))
                .build();

            let response = model.completion(request).await.expect("refusal turn");
            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("refusal text"),
            );
        },
    )
    .await;

    assert_recorded_refusal(SCENARIO);
    assert_recorded_provider(SCENARIO, "Azure");
}

#[tokio::test]
async fn streaming_azure_routed_refusal() {
    const SCENARIO: &str = "refusal_matrix/streaming_azure_routed_refusal";

    with_openrouter_refusal_cassette(
        "refusal_matrix/streaming_azure_routed_refusal",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .max_tokens(CAP)
                .additional_params(refusal_request_params("Azure"))
                .build();

            let stream = model.stream(request).await.expect("stream should connect");
            let observed = collect_raw_stream_observation(stream).await;
            assert_nonempty_response(&observed.text);
        },
    )
    .await;

    assert_recorded_refusal_stream(SCENARIO);
    assert_recorded_provider(SCENARIO, "Azure");
}

// ---------------------------------------------------------------------------
// Controls — turns that must be byte-identical before and after the fix.
// ---------------------------------------------------------------------------

#[tokio::test]
async fn control_answerable_prompt_is_unchanged_blocking() {
    const SCENARIO: &str = "refusal_matrix/control_answerable_prompt_is_unchanged_blocking";

    with_openrouter_refusal_cassette(
        "refusal_matrix/control_answerable_prompt_is_unchanged_blocking",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(ANSWERABLE_PROMPT)
                .max_tokens(CAP)
                .additional_params(refusal_request_params("OpenAI"))
                .build();

            let response = model.completion(request).await.expect("answered turn");
            let text = assistant_text_response(&response.choice).expect("answer text");
            assert!(
                serde_json::from_str::<Value>(&text).is_ok(),
                "an answered strict-schema turn is JSON: {text}"
            );
        },
    )
    .await;

    assert_recorded_no_refusal(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

#[tokio::test]
async fn control_answerable_prompt_is_unchanged_streaming() {
    const SCENARIO: &str = "refusal_matrix/control_answerable_prompt_is_unchanged_streaming";

    with_openrouter_refusal_cassette(
        "refusal_matrix/control_answerable_prompt_is_unchanged_streaming",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(ANSWERABLE_PROMPT)
                .max_tokens(CAP)
                .additional_params(refusal_request_params("OpenAI"))
                .build();

            let stream = model.stream(request).await.expect("stream should connect");
            let observed = collect_raw_stream_observation(stream).await;
            assert!(
                serde_json::from_str::<Value>(&observed.text).is_ok(),
                "an answered strict-schema stream is JSON: {}",
                observed.text
            );
        },
    )
    .await;

    assert_recorded_no_refusal_stream(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

/// Why the matrix cannot use the cheap model: `gpt-4o-mini` answers the
/// refusable prompt *inside* the schema, so it never populates `refusal`.
#[tokio::test]
async fn control_mini_answers_inside_schema() {
    const SCENARIO: &str = "refusal_matrix/control_mini_answers_inside_schema";

    with_openrouter_refusal_cassette(
        "refusal_matrix/control_mini_answers_inside_schema",
        |client| async move {
            let model = client.completion_model(NON_REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .max_tokens(CAP)
                .additional_params(refusal_request_params("OpenAI"))
                .build();

            let response = model.completion(request).await.expect("answered turn");
            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("in-schema text"),
            );
        },
    )
    .await;

    assert_recorded_no_refusal(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

/// Without a strict schema the same prompt comes back as ordinary `content`,
/// never as the `refusal` field — the control that scopes the bug to
/// structured output.
#[tokio::test]
async fn control_no_schema_refusal_is_plain_content() {
    const SCENARIO: &str = "refusal_matrix/control_no_schema_refusal_is_plain_content";

    with_openrouter_refusal_cassette(
        "refusal_matrix/control_no_schema_refusal_is_plain_content",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request(REFUSED_PROMPT)
                .max_tokens(CAP)
                .additional_params(pinned_only("OpenAI"))
                .build();

            let response = model.completion(request).await.expect("answered turn");
            assert_nonempty_response(
                &assistant_text_response(&response.choice).expect("content text"),
            );
        },
    )
    .await;

    assert_recorded_no_refusal(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

/// A tool-calls-only turn holds `content` at `null` with `refusal` absent —
/// the shape the fallback must leave alone.
#[tokio::test]
async fn control_tool_call_turn_is_unchanged() {
    const SCENARIO: &str = "refusal_matrix/control_tool_call_turn_is_unchanged";

    with_openrouter_refusal_cassette(
        "refusal_matrix/control_tool_call_turn_is_unchanged",
        |client| async move {
            let model = client.completion_model(REFUSING_MODEL);
            let request = model
                .completion_request("Call the ping tool.")
                .max_tokens(CAP)
                .tools(vec![zero_arg_tool_definition("ping")])
                .additional_params(pinned_only("OpenAI"))
                .build();

            let response = model.completion(request).await.expect("tool-call turn");
            assert!(
                response
                    .choice
                    .iter()
                    .any(|part| matches!(part, rig::message::AssistantContent::ToolCall(_))),
                "expected a tool call: {:?}",
                response.choice
            );
        },
    )
    .await;

    assert_recorded_no_refusal(SCENARIO);
    assert_recorded_no_content(SCENARIO);
    assert_recorded_provider(SCENARIO, "OpenAI");
}

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

/// The tool-call control's other half: the recorded assistant message holds
/// `content` at `null`. Without this, a drift to `content: "…"` alongside the
/// tool call would leave the cell green while no longer covering the
/// null-content shape the fallback has to leave alone.
fn assert_recorded_no_content(scenario: &str) {
    let all_null = recorded_response_bodies(scenario)
        .iter()
        .filter_map(|body| serde_json::from_str::<Value>(body).ok())
        .flat_map(|body| {
            body.get("choices")
                .and_then(Value::as_array)
                .cloned()
                .unwrap_or_default()
        })
        .filter_map(|choice| choice.get("message")?.get("content").cloned())
        .all(|content| content.is_null());

    assert!(
        all_null,
        "cassette {scenario} no longer records `message.content: null`, so it \
         no longer covers the null-content shape this control is about"
    );
}

fn assert_recorded_no_refusal(scenario: &str) {
    assert!(
        recorded_refusals(scenario).is_empty(),
        "control cassette {scenario} unexpectedly records a `message.refusal`"
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

fn assert_recorded_no_refusal_stream(scenario: &str) {
    assert!(
        !recorded_deltas(scenario).iter().any(|delta| delta
            .get("refusal")
            .and_then(Value::as_str)
            .is_some_and(|refusal| !refusal.is_empty())),
        "control cassette {scenario} unexpectedly records a `delta.refusal`"
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
