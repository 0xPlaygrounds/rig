//! Edge matrix for a streamed tool call whose arguments are not JSON
//! (rig#2447).
//!
//! **Bug.** Anthropic streams tool input as `input_json_delta` fragments and
//! closes the block with `content_block_stop`. When the assembled fragments
//! were not JSON, the reply failed and the run ended: one bad byte from the
//! model ended the run, and the model never learned why.
//!
//! **Contract.** A malformed call never fails the reply. It is kept with
//! the arguments its text still states and the text itself in
//! `invalid_arguments`, and with no provider item, so it replays from its
//! canonical fields. The agent never runs the tool and never consults the
//! invalid-call hook: it answers the call with an `is_error` tool result
//! saying "The arguments for tool `X` are not a JSON object: ...", and the
//! run continues, as pi does.
//!
//! **Fixtures.** Cells 1 and 2 are recorded live and are the controls.
//! Cell 3 is hand-derived from cell 2: the recorded *response* stream's
//! last `input_json_delta` for the tool call has its `partial_json`
//! replaced with `"\u0001}"`, a control byte where a number belongs, so the
//! assembled input is not JSON. Its first request is byte-identical to the
//! control's. Its follow-up request is the control's turn 2 with the
//! history the agent sends substituted: the call rebuilt from its canonical
//! fields (the arguments the cut text states, `{"x": 2}`, and no `caller`,
//! which lived only in the provider item), and the `is_error` result naming
//! the raw text. It is answered with the control's own recorded final
//! response. The cell asserts its fixture really carries the control byte,
//! so a re-record cannot silently heal it. Re-recording cell 2 means
//! deriving cell 3 again from it the same way.
//!
//! Anthropic is the provider for this matrix because its wire states the
//! close of every tool-input block. Bedrock (`contentBlockStop`) needs AWS
//! credentials to record and is left as follow-up.
//!
//! | # | cell | transport | input on the wire | fixture |
//! |---|------|-----------|-------------------|---------|
//! | 1 | `blocking_healthy_control` | blocking | valid | recorded |
//! | 2 | `streaming_healthy_control` | streaming | valid | recorded |
//! | 3 | `streaming_malformed_call_is_answered_with_an_error` | streaming | corrupt | derived from 2 |
//!
//! Unit cells for the seam itself live beside the code:
//! `crates/rig-agent/src/agent/streaming/malformed_tool_args_tests.rs` (what
//! the model sees on the next request) and the history conformance row
//! `h06_malformed_arguments`.

use futures::StreamExt;
use rig::agent::{
    AgentHook, HookContext, InvalidToolCallAction, InvalidToolCallContext, MultiTurnStreamItem,
};
use rig::providers::anthropic;
use rig_test_support::cassette_models::AnthropicModels;
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

/// A hook that fails the cell if consulted: malformed arguments are not an
/// invalid call to resolve.
#[derive(Clone)]
struct NeverConsulted;

impl AgentHook for NeverConsulted {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        context: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        panic!("malformed arguments reached the invalid-call hook: {context:?}");
    }
}

fn agent(client: AnthropicModels) -> rig::agent::Agent {
    rig::AgentBuilder::new(client.completion(anthropic::completion::CLAUDE_SONNET_4_6))
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
            assert_mentions_expected_number(&response.output(), -3);
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
async fn streaming_malformed_call_is_answered_with_an_error() {
    if crate::cassettes::skip_when_recording(
        "cell 3 is hand-derived from cell 2: its tool-input stream carries a control byte no provider emits",
    ) {
        return;
    }
    with_anthropic_cassette(
        "malformed_tool_args_matrix/streaming_malformed_call_is_answered_with_an_error",
        |client| async move {
            let mut stream = agent(client)
                .prompt(STREAMING_TOOLS_PROMPT)
                .add_hook(NeverConsulted)
                .stream();
            let mut executed = false;
            let mut text = String::new();
            while let Some(item) = stream.next().await {
                match item.expect("a malformed call never ends the run") {
                    MultiTurnStreamItem::ToolExecutionCommitted { .. } => executed = true,
                    MultiTurnStreamItem::FinalResponse(response) => {
                        text = response.output().to_owned();
                    }
                    _ => {}
                }
            }
            assert!(!executed, "a malformed call must never be executed");
            assert_mentions_expected_number(&text, -3);
        },
    )
    .await;
    let scenario = "malformed_tool_args_matrix/streaming_malformed_call_is_answered_with_an_error";
    assert_fixture_is_corrupt(scenario);
    // The follow-up the model answers carries the error result, which the
    // cassette matched against this request.
    let interactions = crate::cassettes::recorded_interaction_bodies("anthropic", scenario);
    let follow_up: Value = interactions
        .get(1)
        .and_then(|(request, _)| serde_json::from_str(request).ok())
        .expect("the follow-up request is recorded");
    let result = &follow_up["messages"][2]["content"][0];
    assert_eq!(result["is_error"], Value::Bool(true));
    assert!(
        result["content"][0]["text"]
            .as_str()
            .is_some_and(|text| text
                .starts_with("The arguments for tool `subtract` are not a JSON object: ")),
        "{result}"
    );
}
