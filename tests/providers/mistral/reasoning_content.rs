//! Cassette-backed coverage for Mistral's reasoning (magistral-class) content.
//!
//! Mistral's reasoning models answer with `content` as a chunk array rather
//! than a string: a `thinking` chunk carrying the trace, then the answer's
//! `text` chunk. Rig read that array with a text-only join on both transports,
//! so the trace was dropped — and a turn whose `max_tokens` cap was spent
//! inside the trace came back with *nothing at all*. On the way back out the
//! trace was stripped outright, which Mistral's reasoning docs warn against:
//! replaying a tool-calling reasoning turn without its trace makes the model
//! re-issue the call it already made instead of answering from the result.
//!
//! ## Matrix
//!
//! | # | cell | transport | surface | shape |
//! |---|---|---|---|---|
//! | 1 | `blocking_keeps_the_reasoning_trace` | blocking | `CompletionModel` | trace + answer |
//! | 2 | `blocking_without_reasoning_effort_has_no_trace` | blocking | `CompletionModel` | **control**: plain string |
//! | 3 | `blocking_reasoning_effort_none_is_a_plain_string` | blocking | `CompletionModel` | **control**: `effort: none` |
//! | 4 | `blocking_truncated_thinking_still_yields_the_trace` | blocking | `CompletionModel` | trace only (cap spent thinking) |
//! | 5 | `blocking_reasoning_beside_a_tool_call` | blocking | `CompletionModel` | trace + tool call |
//! | 6 | `blocking_reasoning_on_the_magistral_alias` | blocking | `CompletionModel` | second model id |
//! | 7 | `blocking_agent_chat_keeps_the_trace_in_history` | blocking | `Agent::chat` | trace + answer |
//! | 8 | `blocking_reasoning_with_structured_output` | blocking | `CompletionModel` | trace + `json_schema` |
//! | 9 | `streaming_yields_the_reasoning_trace` | streaming | `CompletionModel` | trace + answer |
//! | 10 | `streaming_without_reasoning_effort_yields_no_trace` | streaming | `CompletionModel` | **control**: plain string |
//! | 11 | `streaming_reasoning_beside_a_tool_call` | streaming | `CompletionModel` | trace + tool call |
//! | 12 | `streaming_truncated_thinking_yields_the_trace` | streaming | `CompletionModel` | trace only |
//! | 13 | `streaming_reasoning_on_the_magistral_alias` | streaming | `CompletionModel` | second model id |
//! | 14 | `streaming_agent_stream_keeps_the_trace` | streaming | `Agent::stream_prompt` | trace + answer |
//! | 15 | `streaming_reasoning_with_structured_output` | streaming | `CompletionModel` | trace + `json_schema` |
//! | 16 | `streaming_terminal_aggregates_the_trace` | streaming | `CompletionModel` | terminal record |
//! | 17 | `roundtrip_replays_the_thinking_chunk` | blocking | `CompletionModel` | request bytes |
//! | 18 | `roundtrip_replays_the_trace_beside_a_tool_call` | blocking | `CompletionModel` | request bytes |
//! | 19 | `roundtrip_without_a_trace_sends_a_plain_string` | blocking | `CompletionModel` | **control**: request bytes |
//! | 20 | `roundtrip_lets_the_model_answer_from_the_tool_result` | blocking | `CompletionModel` | agent-level effect |
//! | 21 | `roundtrip_a_traceless_tool_history_sends_no_thinking_chunk` | blocking | `CompletionModel` | **control**: request bytes for a tool history |
//! | 22 | `roundtrip_replays_a_trace_captured_from_a_stream` | streaming → blocking | `CompletionModel` | cross-transport |
//! | 23 | `reasoning_history_on_a_model_without_the_capability_is_rejected` | blocking | `CompletionModel` | 400 |
//! | 24 | `an_unsupported_reasoning_effort_keeps_the_id_and_body` | blocking | `CompletionModel` | 400 |
//! | 25 | `bogus_key_reasoning_request_keeps_the_id_and_body` | blocking | `CompletionModel` | 401 |
//! | 26 | `parity_blocking_and_streaming_both_carry_a_trace` | both | `CompletionModel` | parity |
//!
//! Four more recorded cells live in `reasoning_roundtrip.rs` and
//! `reasoning_tool_roundtrip.rs`, which run the cross-provider reasoning
//! contract (`tests/common/reasoning.rs`) every other reasoning-capable
//! provider in the tree runs and Mistral had no entry in — including the
//! agent-loop cell whose history assertion requires a reasoning block.
//!
//! The wire shapes each cell relies on are unit-pinned next to the provider in
//! `crates/rig-core/src/providers/mistral/completion.rs` (`reasoning_tests`)
//! and `providers/openai/completion/streaming.rs`; those cover the schema-legal
//! shapes no live turn has been observed to send (a bare-string `thinking`
//! payload, several traces in one turn, a payload-less chunk).

use anyhow::Result;
use axum::http;
use futures::StreamExt;
use rig::completion::{CompletionModel as _, CompletionRequest};
use rig::message::{AssistantContent, Message, ToolChoice};
use rig::prelude::*;
use rig::providers::mistral;
use rig::streaming::StreamedAssistantContent;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use serde_json::json;

use super::support::{with_mistral_reasoning_cassette, with_mistral_reasoning_cassette_bogus_key};

/// Mistral's reasoning model. The live catalog lists `mistral-small-latest`
/// and `magistral-small-latest` as aliases of the same `mistral-small-2603`,
/// with `capabilities.reasoning: true` — so the reasoning cells cost no more
/// than every other Mistral cell.
const REASONING_MODEL: &str = mistral::MISTRAL_SMALL;
/// The same model under its reasoning-branded id, so one cell proves the
/// behaviour is not keyed on the identifier the other cells happen to use.
const MAGISTRAL_ALIAS: &str = "magistral-small-latest";
/// A model with `capabilities.reasoning: false`, used only to pin what Mistral
/// does with a reasoning input it cannot accept.
const NON_REASONING_MODEL: &str = mistral::MINISTRAL_3B;

/// A prompt whose answer is short and whose reasoning is not: the trace is the
/// bulk of the turn, so a cell that drops it is unmistakable.
const ARITHMETIC_PROMPT: &str = "What is 6*7? Answer with just the number.";

fn high_effort() -> serde_json::Value {
    json!({"reasoning_effort": "high"})
}

// ===========================================================================
// Blocking
// ===========================================================================

/// The headline cell: the trace Mistral returned reaches the caller as a
/// reasoning block, ahead of the answer. On `origin/main` the response carries
/// the answer alone.
#[tokio::test]
async fn blocking_keeps_the_reasoning_trace() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/blocking_keeps_the_reasoning_trace",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let response = model
                .completion_request(ARITHMETIC_PROMPT)
                .max_tokens(400)
                .temperature(0.0)
                .additional_params(high_effort())
                .send()
                .await?;

            assert_recorded_response_carries_a_thinking_chunk(
                "reasoning_content/blocking_keeps_the_reasoning_trace",
            );
            assert_trace_leads_the_answer(&response.choice);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// Control: the same model, same prompt, no `reasoning_effort`. Mistral answers
/// with a plain string and rig reports no reasoning — so cell 1 is about the
/// thinking chunk, not about array content generally.
#[tokio::test]
async fn blocking_without_reasoning_effort_has_no_trace() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/blocking_without_reasoning_effort_has_no_trace",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let response = model
                .completion_request(ARITHMETIC_PROMPT)
                .max_tokens(64)
                .temperature(0.0)
                .send()
                .await?;

            assert_recorded_response_has_no_thinking_chunk(
                "reasoning_content/blocking_without_reasoning_effort_has_no_trace",
            );
            assert_no_reasoning(&response.choice);
            anyhow::ensure!(
                response
                    .choice
                    .iter()
                    .any(|block| matches!(block, AssistantContent::Text(_))),
                "the control turn must still answer"
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// Control: `reasoning_effort: "none"` is the explicit off switch, and it
/// returns the plain-string form too.
#[tokio::test]
async fn blocking_reasoning_effort_none_is_a_plain_string() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/blocking_reasoning_effort_none_is_a_plain_string",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let response = model
                .completion_request(ARITHMETIC_PROMPT)
                .max_tokens(64)
                .temperature(0.0)
                .additional_params(json!({"reasoning_effort": "none"}))
                .send()
                .await?;

            assert_recorded_response_has_no_thinking_chunk(
                "reasoning_content/blocking_reasoning_effort_none_is_a_plain_string",
            );
            assert_no_reasoning(&response.choice);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The worst case: a cap spent entirely inside the trace. The response then
/// carries the thinking chunk and nothing else, so joining only the text parts
/// left the turn with no content at all — the `Length` finish reason was
/// everything the caller got back.
#[tokio::test]
async fn blocking_truncated_thinking_still_yields_the_trace() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/blocking_truncated_thinking_still_yields_the_trace",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let response = model
                .completion_request("What is 17*23? Think it through carefully.")
                .max_tokens(64)
                .temperature(0.0)
                .additional_params(high_effort())
                .send()
                .await?;

            assert_recorded_response_carries_a_thinking_chunk(
                "reasoning_content/blocking_truncated_thinking_still_yields_the_trace",
            );
            anyhow::ensure!(
                !response.choice.is_empty(),
                "a turn that spent its whole budget thinking still produced the trace, \
                 but the response carried no content at all"
            );
            anyhow::ensure!(
                response
                    .choice
                    .iter()
                    .any(|block| matches!(block, AssistantContent::Reasoning(_))),
                "the trace is the only thing this turn produced: {:?}",
                response.choice
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// A reasoning turn that ends in a tool call carries the trace beside the
/// call — the shape a reasoning agent sees on every turn of its loop.
#[tokio::test]
async fn blocking_reasoning_beside_a_tool_call() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/blocking_reasoning_beside_a_tool_call",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let response = model
                .completion_request("Add 2 and 3 using the tool.")
                .max_tokens(400)
                .temperature(0.0)
                .tool(add_tool_definition())
                .tool_choice(ToolChoice::Auto)
                .additional_params(high_effort())
                .send()
                .await?;

            assert_recorded_response_carries_a_thinking_chunk(
                "reasoning_content/blocking_reasoning_beside_a_tool_call",
            );
            anyhow::ensure!(
                response
                    .choice
                    .iter()
                    .any(|block| matches!(block, AssistantContent::Reasoning(_))),
                "the trace must survive beside the tool call: {:?}",
                response.choice
            );
            anyhow::ensure!(
                response
                    .choice
                    .iter()
                    .any(|block| matches!(block, AssistantContent::ToolCall(_))),
                "the tool call must still arrive: {:?}",
                response.choice
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The same model under its reasoning-branded id: the behaviour follows the
/// wire shape, not the model string.
#[tokio::test]
async fn blocking_reasoning_on_the_magistral_alias() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/blocking_reasoning_on_the_magistral_alias",
        |client| async move {
            let model = client.completion_model(MAGISTRAL_ALIAS);
            let response = model
                .completion_request(ARITHMETIC_PROMPT)
                .max_tokens(400)
                .temperature(0.0)
                .additional_params(high_effort())
                .send()
                .await?;

            assert_recorded_response_carries_a_thinking_chunk(
                "reasoning_content/blocking_reasoning_on_the_magistral_alias",
            );
            assert_trace_leads_the_answer(&response.choice);
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The agent surface, which is where a caller normally meets this: the trace
/// lands in the chat history the agent hands back, so the next turn can replay
/// it.
#[tokio::test]
async fn blocking_agent_chat_keeps_the_trace_in_history() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/blocking_agent_chat_keeps_the_trace_in_history",
        |client| async move {
            let agent = client
                .agent(REASONING_MODEL)
                .preamble("Answer with the number alone.")
                .max_tokens(400)
                .temperature(0.0)
                .additional_params(high_effort())
                .build();

            let mut history = Vec::<Message>::new();
            let answer = agent.chat(ARITHMETIC_PROMPT, &mut history).await?;

            assert_recorded_response_carries_a_thinking_chunk(
                "reasoning_content/blocking_agent_chat_keeps_the_trace_in_history",
            );
            anyhow::ensure!(!answer.trim().is_empty(), "the agent must answer");
            anyhow::ensure!(
                history.iter().any(|message| matches!(
                    message,
                    Message::Assistant { content, .. }
                        if content
                            .iter()
                            .any(|block| matches!(block, AssistantContent::Reasoning(_)))
                )),
                "the trace must reach the history the next turn replays: {history:#?}"
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// Reasoning beside a `json_schema` response format: the answer still parses
/// as the schema, and the trace does not leak into it.
#[tokio::test]
async fn blocking_reasoning_with_structured_output() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/blocking_reasoning_with_structured_output",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let request = model
                .completion_request("What is 6 times 7?")
                .preamble("Return only the requested structured object.".to_string())
                .max_tokens(400)
                .temperature(0.0)
                .additional_params(high_effort())
                .output_schema(schemars::schema_for!(Answer))
                .build();
            let response = model.completion(request).await?;

            assert_recorded_response_carries_a_thinking_chunk(
                "reasoning_content/blocking_reasoning_with_structured_output",
            );
            let text = assistant_text(&response.choice);
            let answer: Answer = serde_json::from_str(&text).map_err(|error| {
                anyhow::anyhow!("answer must parse as the schema: {error}: {text}")
            })?;
            anyhow::ensure!(answer.total == 42, "expected 42, got {}", answer.total);
            anyhow::ensure!(
                response
                    .choice
                    .iter()
                    .any(|block| matches!(block, AssistantContent::Reasoning(_))),
                "the trace must survive a structured turn: {:?}",
                response.choice
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

// ===========================================================================
// Streaming
// ===========================================================================

/// The streamed twin of cell 1. The trace arrives as reasoning deltas and the
/// answer as text; on `origin/main` the thinking deltas decode to neither and
/// the stream yields only the answer.
#[tokio::test]
async fn streaming_yields_the_reasoning_trace() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/streaming_yields_the_reasoning_trace",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let stream = model
                .completion_request(ARITHMETIC_PROMPT)
                .max_tokens(400)
                .temperature(0.0)
                .additional_params(high_effort())
                .stream()
                .await?;
            let streamed = collect(stream).await?;

            assert_recorded_response_carries_a_thinking_chunk(
                "reasoning_content/streaming_yields_the_reasoning_trace",
            );
            anyhow::ensure!(
                !streamed.reasoning.is_empty(),
                "the stream must deliver the trace it was sent"
            );
            anyhow::ensure!(
                !streamed.text.is_empty(),
                "the stream must still deliver the answer"
            );
            anyhow::ensure!(
                !streamed.text.contains(&streamed.reasoning),
                "the trace must not be spliced into the visible answer"
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// Control: a streamed turn with no `reasoning_effort` yields no reasoning at
/// all, so the cell above is about the thinking chunk specifically.
#[tokio::test]
async fn streaming_without_reasoning_effort_yields_no_trace() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/streaming_without_reasoning_effort_yields_no_trace",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let stream = model
                .completion_request(ARITHMETIC_PROMPT)
                .max_tokens(64)
                .temperature(0.0)
                .stream()
                .await?;
            let streamed = collect(stream).await?;

            assert_recorded_response_has_no_thinking_chunk(
                "reasoning_content/streaming_without_reasoning_effort_yields_no_trace",
            );
            anyhow::ensure!(
                streamed.reasoning.is_empty(),
                "no thinking chunk was sent, so no reasoning may be reported: {:?}",
                streamed.reasoning
            );
            anyhow::ensure!(!streamed.text.is_empty());
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// Streamed reasoning beside a streamed tool call: Mistral emits complete
/// single-chunk tool calls, so the trace and the call arrive in the same
/// stream without either displacing the other.
#[tokio::test]
async fn streaming_reasoning_beside_a_tool_call() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/streaming_reasoning_beside_a_tool_call",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let stream = model
                .completion_request("Add 2 and 3 using the tool.")
                .max_tokens(400)
                .temperature(0.0)
                .tool(add_tool_definition())
                .tool_choice(ToolChoice::Auto)
                .additional_params(high_effort())
                .stream()
                .await?;
            let streamed = collect(stream).await?;

            assert_recorded_response_carries_a_thinking_chunk(
                "reasoning_content/streaming_reasoning_beside_a_tool_call",
            );
            anyhow::ensure!(!streamed.reasoning.is_empty(), "the trace must survive");
            anyhow::ensure!(streamed.tool_calls > 0, "the tool call must still arrive");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The streamed twin of the truncated-thinking cell: the whole cap goes into
/// the trace, so the trace is the only thing the stream can deliver.
#[tokio::test]
async fn streaming_truncated_thinking_yields_the_trace() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/streaming_truncated_thinking_yields_the_trace",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let stream = model
                .completion_request("What is 17*23? Think it through carefully.")
                .max_tokens(64)
                .temperature(0.0)
                .additional_params(high_effort())
                .stream()
                .await?;
            let streamed = collect(stream).await?;

            assert_recorded_response_carries_a_thinking_chunk(
                "reasoning_content/streaming_truncated_thinking_yields_the_trace",
            );
            anyhow::ensure!(
                !streamed.reasoning.is_empty(),
                "the stream produced nothing at all for a turn that thought for its whole budget"
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The streamed twin of the alias cell.
#[tokio::test]
async fn streaming_reasoning_on_the_magistral_alias() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/streaming_reasoning_on_the_magistral_alias",
        |client| async move {
            let model = client.completion_model(MAGISTRAL_ALIAS);
            let stream = model
                .completion_request(ARITHMETIC_PROMPT)
                .max_tokens(400)
                .temperature(0.0)
                .additional_params(high_effort())
                .stream()
                .await?;
            let streamed = collect(stream).await?;

            assert_recorded_response_carries_a_thinking_chunk(
                "reasoning_content/streaming_reasoning_on_the_magistral_alias",
            );
            anyhow::ensure!(!streamed.reasoning.is_empty());
            anyhow::ensure!(!streamed.text.is_empty());
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The agent streaming surface: the trace arrives as reasoning items in the
/// multi-turn stream, not as text and not as nothing.
#[tokio::test]
async fn streaming_agent_stream_keeps_the_trace() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/streaming_agent_stream_keeps_the_trace",
        |client| async move {
            let agent = client
                .agent(REASONING_MODEL)
                .preamble("Answer with the number alone.")
                .max_tokens(400)
                .temperature(0.0)
                .additional_params(high_effort())
                .build();
            let stream = agent.stream_prompt(ARITHMETIC_PROMPT).await;
            let stats = crate::reasoning::collect_stream_stats(stream, "mistral").await;

            assert_recorded_response_carries_a_thinking_chunk(
                "reasoning_content/streaming_agent_stream_keeps_the_trace",
            );
            anyhow::ensure!(
                stats.errors.is_empty(),
                "the agent stream must not error: {:?}",
                stats.errors
            );
            anyhow::ensure!(
                stats.reasoning_delta_count > 0 || stats.reasoning_block_count > 0,
                "the trace must reach the agent stream: {:?}",
                stats.events
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// Reasoning beside a `json_schema` response format, streamed.
#[tokio::test]
async fn streaming_reasoning_with_structured_output() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/streaming_reasoning_with_structured_output",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let request = model
                .completion_request("What is 6 times 7?")
                .preamble("Return only the requested structured object.".to_string())
                .max_tokens(400)
                .temperature(0.0)
                .additional_params(high_effort())
                .output_schema(schemars::schema_for!(Answer))
                .build();
            let streamed = collect(model.stream(request).await?).await?;

            assert_recorded_response_carries_a_thinking_chunk(
                "reasoning_content/streaming_reasoning_with_structured_output",
            );
            let answer: Answer = serde_json::from_str(streamed.text.trim()).map_err(|error| {
                anyhow::anyhow!(
                    "streamed answer must parse as the schema: {error}: {}",
                    streamed.text
                )
            })?;
            anyhow::ensure!(answer.total == 42, "expected 42, got {}", answer.total);
            anyhow::ensure!(
                !streamed.reasoning.is_empty(),
                "the trace must survive a structured stream"
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The terminal record a stream aggregates must carry the trace too, or a
/// caller who reads only the final message loses what the deltas delivered.
#[tokio::test]
async fn streaming_terminal_aggregates_the_trace() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/streaming_terminal_aggregates_the_trace",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let mut stream = model
                .completion_request(ARITHMETIC_PROMPT)
                .max_tokens(400)
                .temperature(0.0)
                .additional_params(high_effort())
                .stream()
                .await?;
            while stream.next().await.transpose()?.is_some() {}

            assert_recorded_response_carries_a_thinking_chunk(
                "reasoning_content/streaming_terminal_aggregates_the_trace",
            );
            anyhow::ensure!(
                stream
                    .choice
                    .iter()
                    .any(|block| matches!(block, AssistantContent::Reasoning(_))),
                "the aggregated assistant message must carry the trace: {:?}",
                stream.choice
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

// ===========================================================================
// Round trip — what rig sends back
// ===========================================================================

/// Turn 1 produces a trace; turn 2 replays it. The cassette matches request
/// bodies, so the recorded second request *is* the assertion: on
/// `origin/main`, which strips the trace, the replay is a mock miss.
#[tokio::test]
async fn roundtrip_replays_the_thinking_chunk() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/roundtrip_replays_the_thinking_chunk",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let first = model
                .completion_request(ARITHMETIC_PROMPT)
                .max_tokens(400)
                .temperature(0.0)
                .additional_params(high_effort())
                .send()
                .await?;

            let history = vec![
                Message::user(ARITHMETIC_PROMPT),
                Message::Assistant {
                    id: None,
                    content: first.choice.clone(),
                },
            ];
            let second = model
                .completion_request("Now multiply that by 2. Answer with just the number.")
                .max_tokens(400)
                .temperature(0.0)
                .messages(history)
                .additional_params(high_effort())
                .send()
                .await?;

            assert_replayed_a_thinking_chunk(
                "reasoning_content/roundtrip_replays_the_thinking_chunk",
            );
            anyhow::ensure!(!second.choice.is_empty(), "the follow-up must answer");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The same replay for a turn whose trace rides beside a tool call — the shape
/// an agent loop replays on every turn after the first.
#[tokio::test]
async fn roundtrip_replays_the_trace_beside_a_tool_call() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/roundtrip_replays_the_trace_beside_a_tool_call",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let first = model
                .completion_request("Add 2 and 3 using the tool.")
                .max_tokens(400)
                .temperature(0.0)
                .tool(add_tool_definition())
                .tool_choice(ToolChoice::Auto)
                .additional_params(high_effort())
                .send()
                .await?;

            let second = model
                .completion(tool_result_followup(&model, &first.choice)?)
                .await?;

            assert_replayed_a_thinking_chunk(
                "reasoning_content/roundtrip_replays_the_trace_beside_a_tool_call",
            );
            anyhow::ensure!(!second.choice.is_empty(), "the follow-up must answer");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// Control: a history with no trace in it still flattens to Mistral's plain
/// string, so the replay above is the trace's doing and not a change to how
/// every assistant turn is sent.
#[tokio::test]
async fn roundtrip_without_a_trace_sends_a_plain_string() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/roundtrip_without_a_trace_sends_a_plain_string",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let history = vec![Message::user(ARITHMETIC_PROMPT), Message::assistant("42")];
            let response = model
                .completion_request("Now multiply that by 2. Answer with just the number.")
                .max_tokens(64)
                .temperature(0.0)
                .messages(history)
                .send()
                .await?;

            if let Some(requests) = recorded_request_bodies(
                "reasoning_content/roundtrip_without_a_trace_sends_a_plain_string",
            ) {
                let request = requests.first().expect("one recorded request");
                anyhow::ensure!(
                    !request.contains("\"thinking\""),
                    "a traceless history must not gain a thinking chunk: {request}"
                );
                anyhow::ensure!(
                    request.contains("\"content\":\"42\""),
                    "a traceless assistant turn still sends Mistral's plain string: {request}"
                );
            }
            anyhow::ensure!(!response.choice.is_empty());
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The replayed trace, end to end: the assistant turn goes back out with its
/// thinking chunk and the model answers from the tool result it already has,
/// rather than calling again. Pinned as the recorded turn's outcome — the
/// negative direction (what a *traceless* replay makes the model do) is not
/// reproducible and is deliberately not asserted anywhere; see the control
/// cell below.
#[tokio::test]
async fn roundtrip_lets_the_model_answer_from_the_tool_result() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/roundtrip_lets_the_model_answer_from_the_tool_result",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let first = model
                .completion_request("Add 2 and 3 using the tool.")
                .max_tokens(400)
                .temperature(0.0)
                .tool(add_tool_definition())
                .tool_choice(ToolChoice::Auto)
                .additional_params(high_effort())
                .send()
                .await?;

            let second = model
                .completion(tool_result_followup(&model, &first.choice)?)
                .await?;

            assert_replayed_a_thinking_chunk(
                "reasoning_content/roundtrip_lets_the_model_answer_from_the_tool_result",
            );
            anyhow::ensure!(
                !second
                    .choice
                    .iter()
                    .any(|block| matches!(block, AssistantContent::ToolCall(_))),
                "the recorded turn, with its trace replayed, answered from the tool result \
                 instead of calling again: {:?}",
                second.choice
            );
            anyhow::ensure!(
                assistant_text(&second.choice).contains('5'),
                "the answer must use the tool's result: {:?}",
                second.choice
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The control for the two replay cells above: a tool history assembled by
/// hand, with no reasoning block in it, still sends Mistral's plain assistant
/// turn and no thinking chunk. So the replay those cells assert is the trace's
/// doing and not a change to how every assistant turn is sent.
///
/// This cell deliberately does **not** assert what the traceless turn makes the
/// model *do*. Two hand-run live pairs had it re-issue the call it already made
/// instead of answering from the tool result — the incoherence Mistral's
/// reasoning docs warn about — but a third run through this very request shape
/// answered `5` correctly. Model behaviour that is not reproducible is not a
/// regression assertion; the reproducible direction (a replayed trace, and what
/// rig sends) is what the suite pins.
#[tokio::test]
async fn roundtrip_a_traceless_tool_history_sends_no_thinking_chunk() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/roundtrip_a_traceless_tool_history_sends_no_thinking_chunk",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let call = rig::message::ToolCall::from_wire(
                "traceless-call",
                rig::message::ToolFunction {
                    name: "add".to_string(),
                    arguments: json!({"a": 2, "b": 3}),
                },
            );
            let history = vec![
                Message::user("Add 2 and 3 using the tool."),
                Message::Assistant {
                    id: None,
                    content: vec![AssistantContent::ToolCall(call.clone())],
                },
                Message::tool_result(call.id.as_str(), "add", "5"),
            ];
            let response = model
                .completion_request("Answer the original question.")
                .max_tokens(400)
                .temperature(0.0)
                .messages(history)
                .tool(add_tool_definition())
                .tool_choice(ToolChoice::Auto)
                .additional_params(high_effort())
                .send()
                .await?;

            if let Some(requests) = recorded_request_bodies(
                "reasoning_content/roundtrip_a_traceless_tool_history_sends_no_thinking_chunk",
            ) {
                let request = requests.first().expect("one recorded request");
                anyhow::ensure!(
                    !request.contains("\"thinking\""),
                    "a history with no reasoning block must not gain a thinking chunk: {request}"
                );
                anyhow::ensure!(
                    request.contains("\"content\":\"\""),
                    "the tool-calling assistant turn still sends Mistral's plain string: {request}"
                );
            }
            anyhow::ensure!(!response.choice.is_empty(), "the turn must be accepted");
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// A trace captured from a *stream* replays the same way a blocking one does:
/// the two transports produce the same replayable block.
#[tokio::test]
async fn roundtrip_replays_a_trace_captured_from_a_stream() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/roundtrip_replays_a_trace_captured_from_a_stream",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let mut stream = model
                .completion_request(ARITHMETIC_PROMPT)
                .max_tokens(400)
                .temperature(0.0)
                .additional_params(high_effort())
                .stream()
                .await?;
            while stream.next().await.transpose()?.is_some() {}

            let history = vec![
                Message::user(ARITHMETIC_PROMPT),
                Message::Assistant {
                    id: None,
                    content: stream.choice.clone(),
                },
            ];
            let second = model
                .completion_request("Now multiply that by 2. Answer with just the number.")
                .max_tokens(400)
                .temperature(0.0)
                .messages(history)
                .additional_params(high_effort())
                .send()
                .await?;

            assert_replayed_a_thinking_chunk(
                "reasoning_content/roundtrip_replays_a_trace_captured_from_a_stream",
            );
            anyhow::ensure!(!second.choice.is_empty());
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

// ===========================================================================
// Errors
// ===========================================================================

/// Replaying a trace to a model without the capability is Mistral's own
/// explicit rejection, not a rig-invented one — and it is loud, where the
/// silent drop it replaces was not. Pinned so the trade is deliberate.
#[tokio::test]
async fn reasoning_history_on_a_model_without_the_capability_is_rejected() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/reasoning_history_on_a_model_without_the_capability_is_rejected",
        |client| async move {
            let model = client.completion_model(NON_REASONING_MODEL);
            let history = vec![
                Message::user(ARITHMETIC_PROMPT),
                Message::Assistant {
                    id: None,
                    content: vec![
                        AssistantContent::reasoning("6 multiplied by 7 equals 42."),
                        AssistantContent::text("42"),
                    ],
                },
            ];
            let error = model
                .completion_request("Now multiply that by 2.")
                .max_tokens(64)
                .temperature(0.0)
                .messages(history)
                .send()
                .await
                .map(|_| ())
                .expect_err("Mistral rejects reasoning input on a model without the capability");

            let body = error
                .provider_response_body()
                .expect("the rejection body must survive");
            anyhow::ensure!(
                body.contains("Reasoning input is not enabled"),
                "expected Mistral's own reasoning-capability rejection, got {body}"
            );
            anyhow::ensure!(
                error.provider_request_id().is_some(),
                "Mistral is a request-id-contract provider; the id must survive the failure"
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// An effort value the model does not offer is a 400 with a body naming the
/// values it does — and the correlation id rides it, as on every other Mistral
/// failure.
#[tokio::test]
async fn an_unsupported_reasoning_effort_keeps_the_id_and_body() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/an_unsupported_reasoning_effort_keeps_the_id_and_body",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let error = model
                .completion_request(ARITHMETIC_PROMPT)
                .max_tokens(64)
                .additional_params(json!({"reasoning_effort": "low"}))
                .send()
                .await
                .map(|_| ())
                .expect_err("`low` is not one of this model's efforts");

            let body = error
                .provider_response_body()
                .expect("the rejection body must survive");
            anyhow::ensure!(
                body.contains("reasoning_effort"),
                "expected Mistral's effort-value rejection, got {body}"
            );
            anyhow::ensure!(
                error.provider_request_id().is_some(),
                "the correlation id must survive a reasoning failure too"
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

/// The 401 half: a reasoning request with a bad key fails before the model
/// sees it, and still reports the id and the body.
#[tokio::test]
async fn bogus_key_reasoning_request_keeps_the_id_and_body() -> Result<()> {
    with_mistral_reasoning_cassette_bogus_key(
        "reasoning_content/bogus_key_reasoning_request_keeps_the_id_and_body",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let error = model
                .completion_request(ARITHMETIC_PROMPT)
                .max_tokens(64)
                .additional_params(high_effort())
                .send()
                .await
                .map(|_| ())
                .expect_err("an invalid key must fail");

            anyhow::ensure!(
                error.provider_response_status() == Some(http::StatusCode::UNAUTHORIZED),
                "expected a 401, got {:?}",
                error.provider_response_status()
            );
            anyhow::ensure!(error.provider_response_body().is_some());
            anyhow::ensure!(error.provider_request_id().is_some());
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

// ===========================================================================
// Parity
// ===========================================================================

/// The two transports answer the same request the same way: both carry a trace and
/// both carry the answer. Recorded as one cell so the comparison is between
/// two turns of the same fixture rather than across two recordings.
#[tokio::test]
async fn parity_blocking_and_streaming_both_carry_a_trace() -> Result<()> {
    with_mistral_reasoning_cassette(
        "reasoning_content/parity_blocking_and_streaming_both_carry_a_trace",
        |client| async move {
            let model = client.completion_model(REASONING_MODEL);
            let blocking = model
                .completion_request(ARITHMETIC_PROMPT)
                .max_tokens(400)
                .temperature(0.0)
                .additional_params(high_effort())
                .send()
                .await?;
            let streamed = collect(
                model
                    .completion_request(ARITHMETIC_PROMPT)
                    .max_tokens(400)
                    .temperature(0.0)
                    .additional_params(high_effort())
                    .stream()
                    .await?,
            )
            .await?;

            let blocking_reasoning: String = blocking
                .choice
                .iter()
                .filter_map(|block| match block {
                    AssistantContent::Reasoning(reasoning) => Some(reasoning.display_text()),
                    _ => None,
                })
                .collect();
            anyhow::ensure!(
                !blocking_reasoning.is_empty() && !streamed.reasoning.is_empty(),
                "both transports must report the trace they were sent"
            );
            anyhow::ensure!(
                !assistant_text(&blocking.choice).is_empty() && !streamed.text.is_empty(),
                "both transports must report the answer"
            );
            Ok::<_, anyhow::Error>(())
        },
    )
    .await
}

// ===========================================================================
// Helpers
// ===========================================================================

#[derive(Debug, Deserialize, Serialize, JsonSchema)]
struct Answer {
    total: i64,
}

fn add_tool_definition() -> rig::completion::ToolDefinition {
    rig::completion::ToolDefinition {
        name: "add".to_string(),
        description: "Add two integers".to_string(),
        parameters: json!({
            "type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"]
        }),
    }
}

/// Build the follow-up request for a turn that called `add`: the assistant
/// turn verbatim (trace included) plus the tool's result.
fn tool_result_followup(
    model: &mistral::CompletionModel,
    first: &[AssistantContent],
) -> Result<CompletionRequest> {
    let call = first
        .iter()
        .find_map(|block| match block {
            AssistantContent::ToolCall(call) => Some(call.clone()),
            _ => None,
        })
        .ok_or_else(|| anyhow::anyhow!("the first turn must have called the tool: {first:?}"))?;

    Ok(model
        .completion_request("Answer the original question.")
        .max_tokens(400)
        .temperature(0.0)
        .messages(vec![
            Message::user("Add 2 and 3 using the tool."),
            Message::Assistant {
                id: None,
                content: first.to_vec(),
            },
            Message::tool_result(call.id.as_str(), &call.function.name, "5"),
        ])
        .tool(add_tool_definition())
        .tool_choice(ToolChoice::Auto)
        .additional_params(high_effort())
        .build())
}

#[derive(Default)]
struct Streamed {
    text: String,
    reasoning: String,
    tool_calls: usize,
}

async fn collect(mut stream: rig::streaming::StreamingCompletionResponse) -> Result<Streamed> {
    let mut out = Streamed::default();
    while let Some(item) = stream.next().await {
        match item? {
            StreamedAssistantContent::Text(chunk) => out.text.push_str(&chunk.text),
            StreamedAssistantContent::ReasoningDelta { reasoning, .. } => {
                out.reasoning.push_str(&reasoning)
            }
            StreamedAssistantContent::Reasoning { reasoning, .. } => {
                // A completed block supersedes the deltas carrying the same
                // trace, so it replaces rather than appends.
                out.reasoning = reasoning.display_text();
            }
            StreamedAssistantContent::ToolCall { .. } => out.tool_calls += 1,
            _ => {}
        }
    }
    Ok(out)
}

fn assistant_text(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Text(text) => Some(text.text.as_str()),
            _ => None,
        })
        .collect()
}

fn assert_trace_leads_the_answer(choice: &[AssistantContent]) {
    assert!(
        matches!(choice.first(), Some(AssistantContent::Reasoning(_))),
        "the trace Mistral sent ahead of the answer must reach the caller, and lead: {choice:?}"
    );
    assert!(
        choice
            .iter()
            .any(|block| matches!(block, AssistantContent::Text(_))),
        "the answer must survive beside the trace: {choice:?}"
    );
}

fn assert_no_reasoning(choice: &[AssistantContent]) {
    assert!(
        !choice
            .iter()
            .any(|block| matches!(block, AssistantContent::Reasoning(_))),
        "no thinking chunk was sent, so no reasoning may be reported: {choice:?}"
    );
}

/// The raw recorded fixture, split into its `when:`/`then:` body strings.
///
/// `None` while recording: the fixture is written when the cassette is
/// finished, i.e. *after* the test body returns, so there is nothing to read
/// yet and reading the previous recording would assert against bytes this run
/// did not produce. Every replay — the key-free re-run each recording is
/// followed by, and CI — reads the real thing.
fn recorded_bodies(scenario: &str) -> Option<(Vec<String>, Vec<String>)> {
    if crate::cassettes::CassetteMode::current() == crate::cassettes::CassetteMode::Record {
        return None;
    }
    Some(read_recorded_bodies(scenario))
}

fn read_recorded_bodies(scenario: &str) -> (Vec<String>, Vec<String>) {
    let raw = std::fs::read_to_string(crate::cassettes::cassette_path("mistral", scenario))
        .expect("cassette should be readable");
    let lines: Vec<&str> = raw.lines().collect();
    let (mut requests, mut responses) = (Vec::new(), Vec::new());
    let mut in_response = false;
    let mut index = 0;

    while index < lines.len() {
        let line = lines[index];
        index += 1;
        match line {
            "when:" => in_response = false,
            "then:" => in_response = true,
            _ => {
                let Some(value) = line.strip_prefix("  body: ") else {
                    continue;
                };
                if value == "null" {
                    continue;
                }
                // An SSE body is a YAML *block scalar* (`|+`), whose payload is
                // the indented lines that follow — not the header. Reading the
                // header as the body made every streaming premise check either
                // panic or, worse, pass vacuously.
                let body = if value.starts_with('|') || value.starts_with('>') {
                    let mut block = Vec::new();
                    while index < lines.len() {
                        let candidate = lines[index];
                        if candidate.is_empty() {
                            block.push("");
                            index += 1;
                        } else if let Some(content) = candidate.strip_prefix("    ") {
                            block.push(content);
                            index += 1;
                        } else {
                            break;
                        }
                    }
                    block.join("\n")
                } else {
                    value.trim_matches('\'').replace("''", "'")
                };
                if in_response {
                    responses.push(body);
                } else {
                    requests.push(body);
                }
            }
        }
    }
    (requests, responses)
}

fn recorded_request_bodies(scenario: &str) -> Option<Vec<String>> {
    recorded_bodies(scenario).map(|(requests, _)| requests)
}

/// The premise checks are only worth anything if the bodies were actually
/// read. A YAML block scalar's header (`|+`) parsed as the body is the failure
/// that makes an absence assertion pass vacuously, so a body that short is
/// treated as a parse failure rather than as evidence.
fn assert_recorded_bodies_are_readable(scenario: &str, bodies: &[String]) {
    assert!(
        !bodies.is_empty(),
        "{scenario}: the recorded fixture carried no response body to read"
    );
    for body in bodies {
        assert!(
            body.len() > 8,
            "{scenario}: {body:?} is not a response body — the cassette's block scalar \
             was not folded into it"
        );
    }
}

/// A cell about the thinking chunk must be recorded against a turn that
/// actually carried one. Without this the provider could stop sending traces
/// and every cell here would keep passing while covering nothing.
fn assert_recorded_response_carries_a_thinking_chunk(scenario: &str) {
    let Some((_, responses)) = recorded_bodies(scenario) else {
        return;
    };
    assert_recorded_bodies_are_readable(scenario, &responses);
    assert!(
        responses
            .iter()
            .any(|body| body.contains("\"type\":\"thinking\"")),
        "{scenario}: the recorded turn must carry a thinking chunk, or this cell covers nothing"
    );
}

/// The control cells' own premise: the recorded turn carried no thinking chunk.
fn assert_recorded_response_has_no_thinking_chunk(scenario: &str) {
    let Some((_, responses)) = recorded_bodies(scenario) else {
        return;
    };
    assert_recorded_bodies_are_readable(scenario, &responses);
    assert!(
        responses
            .iter()
            .all(|body| !body.contains("\"type\":\"thinking\"")),
        "{scenario}: this control cell's premise is a turn with no thinking chunk"
    );
}

/// The replay cells' assertion, read off the recorded *request* the follow-up
/// turn sent.
fn assert_replayed_a_thinking_chunk(scenario: &str) {
    let Some(requests) = recorded_request_bodies(scenario) else {
        return;
    };
    assert!(
        requests.len() >= 2,
        "{scenario}: a replay cell records two requests, got {}",
        requests.len()
    );
    assert!(
        requests[1..]
            .iter()
            .any(|body| body.contains("\"type\":\"thinking\"")),
        "{scenario}: the follow-up request must replay the trace as Mistral's thinking chunk: {:?}",
        &requests[1..]
    );
}
