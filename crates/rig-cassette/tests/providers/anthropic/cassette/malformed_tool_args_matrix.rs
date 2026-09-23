//! Edge matrix for a streamed tool call whose arguments are not JSON
//! (rig#2447).
//!
//! **Bug.** Anthropic streams tool input as `input_json_delta` fragments and
//! closes the block with `content_block_stop`, so rig's accumulator treats a
//! close as "the wire promised a complete block" and, when the assembled
//! fragments are not JSON, raised a bare `ErrorReport`. The agent engine
//! turned any stream error into a fatal `StreamingError` before the
//! invalid-tool recovery seam saw it: one bad byte from the model ended the
//! run, the model never learned why, and the application got a string.
//!
//! **Fix.** The report carries a typed `MalformedToolInput` detail (name,
//! durable id, provider id, raw text, parser reason); the engine routes it
//! into the same `InvalidToolCallAction` recovery an unknown tool name gets.
//! Default stays fail-fast.
//!
//! **Fixtures.** Cells 1–2 are recorded live and are the controls. Cells 3–5
//! are hand-derived from cell 2: the recorded *response* stream's last
//! `input_json_delta` for the tool call has its `partial_json` replaced with
//! `"\u0001}"` — a control byte where a number belongs — so the assembled
//! input is not JSON. The corrupt turn's request section is byte-identical
//! to the control's. Cells 4–5 also need the follow-up interactions the
//! recovery produces: the engine rolls the turn back and sends a *feedback*
//! request (the malformed call with `{}` input — Anthropic requires a dict —
//! plus a `tool_result` carrying the hook's text). Those request bodies are
//! derived from the control's turn 2 by substituting that history, and are
//! answered with the control's own recorded responses (cell 4: the final
//! text; cell 5: the healthy tool call, then the tool-result follow-up with
//! the full five-message history). Each derived cell asserts its fixture
//! really carries the control byte, so a re-record cannot silently heal it.
//! Cells 3–5 were derived from an earlier recording of cell 2, so they still
//! carry that recording's scrubbed `REDACTED_<n>` ids rather than cell 2's
//! current ones.
//!
//! Anthropic is the provider for this matrix because its wire uses the
//! `Error` policy on block close; OpenAI-compatible wires use `Drop` /
//! `EmptyObject` and never reach this path. Bedrock (`contentBlockStop`,
//! same policy) needs AWS credentials to record and is left as follow-up.
//!
//! **How these cells fail on `origin/main`.** Cells 4–5 fail: the run ends
//! with the provider report before any hook runs. Cell 3 asserts that very
//! report and passes on both sides — it is the "default unchanged" control.
//! Cells 1–2 pass on both sides.
//!
//! | # | cell | transport | input on the wire | hook action | fixture |
//! |---|------|-----------|-------------------|-------------|---------|
//! | 1 | `blocking_healthy_control` | blocking | valid | — | recorded |
//! | 2 | `streaming_healthy_control` | streaming | valid | — | recorded |
//! | 3 | `streaming_malformed_fails_by_default` | streaming | corrupt | none (default) | derived from 2 |
//! | 4 | `streaming_malformed_skip_feeds_result_back` | streaming | corrupt | `Skip` | derived from 2 |
//! | 5 | `streaming_malformed_retry_reissues_request` | streaming | corrupt | `Retry` | derived from 2, + 2 follow-ups |
//!
//! Unit cells for the seam itself live beside the code:
//! `crates/rig-core/src/streaming/accumulator/tests.rs` (typed detail, raw
//! bytes) and `crates/rig-agent/src/agent/streaming/malformed_tool_args_tests.rs`
//! (one test per action, blocking-surface parity).

use futures::StreamExt;
use rig::agent::{
    AgentHook, HookContext, InvalidToolCallAction, InvalidToolCallContext, InvalidToolCallReason,
    MultiTurnStreamItem, StreamingError,
};
use rig::completion::PromptError;
use rig::driver::Bound;
use rig::prelude::*;
use rig::providers::anthropic;
use rig::providers::anthropic::wire::Anthropic;
use rig::streaming::StreamedUserContent;
use serde_json::Value;

use super::super::support::with_anthropic_cassette;
use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, collect_stream_final_response,
};

/// Every `partial_json` fragment of every recorded response stream in
/// `scenario`, in wire order.
fn recorded_partial_json(scenario: &str) -> Vec<String> {
    crate::cassettes::recorded_interaction_bodies("anthropic", scenario)
        .into_iter()
        .flat_map(|(_, response)| {
            response
                .lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter_map(|json| serde_json::from_str::<Value>(json).ok())
                .filter_map(|event| {
                    event
                        .get("delta")
                        .and_then(|delta| delta.get("partial_json"))
                        .and_then(Value::as_str)
                        .map(str::to_owned)
                })
                .collect::<Vec<_>>()
        })
        .collect()
}

fn assert_fixture_is_corrupt(scenario: &str) {
    let fragments = recorded_partial_json(scenario);
    assert!(
        !fragments.is_empty(),
        "{scenario}: fixture should stream tool input fragments"
    );
    let assembled = fragments.concat();
    assert!(
        serde_json::from_str::<Value>(&assembled).is_err(),
        "{scenario}: this derived fixture must assemble to invalid JSON, got {assembled:?}"
    );
    assert!(
        assembled.contains('\u{1}'),
        "{scenario}: the corruption is a control byte; a re-record healed it: {assembled:?}"
    );
}

fn assert_fixture_is_healthy(scenario: &str) {
    let fragments = recorded_partial_json(scenario);
    assert!(
        !fragments.is_empty(),
        "{scenario}: fixture should stream tool input fragments"
    );
    let assembled = fragments.concat();
    assert!(
        serde_json::from_str::<Value>(&assembled).is_ok(),
        "{scenario}: the recorded control must assemble to valid JSON, got {assembled:?}"
    );
}

#[derive(Clone)]
struct OnMalformed(InvalidToolCallAction);

impl AgentHook for OnMalformed {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        context: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        assert!(
            matches!(
                context.reason,
                InvalidToolCallReason::MalformedArguments { .. }
            ),
            "hook must be consulted for malformed arguments, got {:?}",
            context.reason
        );
        assert!(
            context
                .args
                .as_deref()
                .is_some_and(|raw| raw.contains('\u{1}')),
            "hook must see the raw wire text: {:?}",
            context.args
        );
        Some(self.0.clone())
    }
}

fn agent(client: Bound<Anthropic>) -> rig::agent::Agent {
    client
        .agent(anthropic::completion::CLAUDE_SONNET_4_6)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .default_max_turns(2)
        .build()
}

#[tokio::test]
async fn blocking_healthy_control() {
    with_anthropic_cassette(
        "malformed_tool_args_matrix/blocking_healthy_control",
        |client| async move {
            let response = agent(client)
                .prompt(STREAMING_TOOLS_PROMPT)
                .await
                .expect("blocking tool prompt should succeed");
            assert_mentions_expected_number(&response.output, -3);
        },
    )
    .await;
}

#[tokio::test]
async fn streaming_healthy_control() {
    with_anthropic_cassette(
        "malformed_tool_args_matrix/streaming_healthy_control",
        |client| async move {
            let mut stream = agent(client).prompt(STREAMING_TOOLS_PROMPT).stream();
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming tool prompt should succeed");
            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;
    assert_fixture_is_healthy("malformed_tool_args_matrix/streaming_healthy_control");
}

#[tokio::test]
async fn streaming_malformed_fails_by_default() {
    if crate::cassettes::skip_when_recording(
        "cell 3 is hand-derived from cell 2: its tool-input stream carries a control byte no provider emits",
    ) {
        return;
    }
    with_anthropic_cassette(
        "malformed_tool_args_matrix/streaming_malformed_fails_by_default",
        |client| async move {
            let mut stream = agent(client).prompt(STREAMING_TOOLS_PROMPT).stream();
            let mut error = None;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(MultiTurnStreamItem::ToolCall { .. }) => {
                        panic!("a malformed call must never be executed")
                    }
                    Ok(_) => {}
                    Err(err) => {
                        error = Some(err);
                        break;
                    }
                }
            }
            match error.expect("default policy is fail-fast") {
                StreamingError::Prompt(err) => match err {
                    PromptError::Report(report) => assert!(
                        report.message.contains("malformed JSON input"),
                        "{}",
                        report.message
                    ),
                    other => panic!("expected the provider report, got {other:?}"),
                },
                StreamingError::Report(report) => {
                    assert!(report.message.contains("malformed JSON input"));
                }
                other => panic!("expected the provider report, got {other:?}"),
            }
        },
    )
    .await;
    assert_fixture_is_corrupt("malformed_tool_args_matrix/streaming_malformed_fails_by_default");
}

#[tokio::test]
async fn streaming_malformed_skip_feeds_result_back() {
    if crate::cassettes::skip_when_recording(
        "cell 4 is hand-derived from cell 2, including the recovery follow-up interactions",
    ) {
        return;
    }
    with_anthropic_cassette(
        "malformed_tool_args_matrix/streaming_malformed_skip_feeds_result_back",
        |client| async move {
            let mut stream = agent(client)
                .prompt(STREAMING_TOOLS_PROMPT)
                .add_hook(OnMalformed(InvalidToolCallAction::skip(
                    "subtract: arguments were not valid JSON",
                )))
                .stream();
            let mut skipped = None;
            while let Some(item) = stream.next().await {
                match item {
                    Ok(MultiTurnStreamItem::StreamUserItem(StreamedUserContent::ToolResult {
                        tool_result,
                        ..
                    })) => skipped = Some(tool_result),
                    Ok(MultiTurnStreamItem::ToolCall { .. }) => {
                        panic!("a malformed call must never be executed")
                    }
                    // The follow-up turn is not recorded for this cell; the
                    // replay ends after the abandoned turn drains.
                    Ok(_) => {}
                    Err(_) => break,
                }
            }
            let skipped = skipped.expect("skip must emit a synthetic tool result");
            assert_eq!(skipped.name, "subtract");
            assert!(skipped.content.iter().any(|content| matches!(
                content,
                rig::message::ToolResultContent::Text(text)
                    if text.text.contains("not valid JSON") && !text.text.contains('\u{1}')
            )));
        },
    )
    .await;
    assert_fixture_is_corrupt(
        "malformed_tool_args_matrix/streaming_malformed_skip_feeds_result_back",
    );
}

#[tokio::test]
async fn streaming_malformed_retry_reissues_request() {
    if crate::cassettes::skip_when_recording(
        "cell 5 is hand-derived from cell 2, including the two retry follow-up interactions",
    ) {
        return;
    }
    with_anthropic_cassette(
        "malformed_tool_args_matrix/streaming_malformed_retry_reissues_request",
        |client| async move {
            // The retry consumes a model-call slot (documented budget
            // semantics), so this cell needs one more turn than the control.
            let mut stream = agent(client)
                .prompt(STREAMING_TOOLS_PROMPT)
                .max_turns(3)
                .max_invalid_tool_call_retries(1)
                .add_hook(OnMalformed(InvalidToolCallAction::retry(
                    "arguments were not valid JSON; call the tool again",
                )))
                .stream();
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("retry must recover through a second model request");
            assert_mentions_expected_number(&response, -3);
        },
    )
    .await;
    assert_fixture_is_corrupt(
        "malformed_tool_args_matrix/streaming_malformed_retry_reissues_request",
    );
    let interactions = crate::cassettes::recorded_interaction_bodies(
        "anthropic",
        "malformed_tool_args_matrix/streaming_malformed_retry_reissues_request",
    );
    assert!(
        interactions.len() >= 3,
        "retry needs the corrupt turn, the re-issued turn, and the tool-result follow-up; got {}",
        interactions.len()
    );
}
