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
//! **Fixtures.** Every cell is recorded live. Cell 3 caps its first request
//! at [`CUT_AT`] output tokens, which cuts Sonnet 4.6's `subtract` call
//! inside its input, so the assembled input is not JSON and the turn stops
//! on `max_tokens`. Its follow-up carries the call rebuilt from the
//! arguments the cut text states, with no `caller` (which lived only in the
//! provider item), and the `is_error` result naming the raw text. The cell
//! asserts its fixture really holds a cut input, so a re-record that heals
//! it fails.
//!
//! Anthropic is the provider for this matrix because its wire states the
//! close of every tool-input block. Bedrock (`contentBlockStop`) needs AWS
//! credentials to record and is left as follow-up.
//!
//! | # | cell | transport | input on the wire | fixture |
//! |---|------|-----------|-------------------|---------|
//! | 1 | `blocking_healthy_control` | blocking | valid | recorded |
//! | 2 | `streaming_healthy_control` | streaming | valid | recorded |
//! | 3 | `streaming_malformed_call_is_answered_with_an_error` | streaming | cut by `max_tokens` | recorded |
//!
//! Unit cells for the seam itself live beside the code:
//! `crates/rig-agent/src/agent/streaming/malformed_tool_args_tests.rs` (what
//! the model sees on the next request) and the history conformance row
//! `h06_malformed_arguments`.

use futures::StreamExt;
use rig::agent::{
    AgentHook, CompletionCallAction, CompletionCallEvent, HookContext, InvalidToolCallAction,
    InvalidToolCallContext, MultiTurnStreamItem, RequestPatch,
};
use rig::providers::anthropic;
use rig_test_support::cassette_models::AnthropicModels;
use serde_json::Value;

use super::super::support::with_anthropic_cassette;
use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, collect_stream_final_response,
};

/// The tool input each recorded response stream in `scenario` assembles
/// from its `partial_json` fragments, for each response that streams any.
fn recorded_inputs(scenario: &str) -> Vec<String> {
    crate::cassettes::recorded_interaction_bodies("anthropic", scenario)
        .into_iter()
        .map(|(_, response)| {
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
                .collect::<String>()
        })
        .filter(|input| !input.is_empty())
        .collect()
}

fn assert_fixture_is_cut(scenario: &str) {
    let inputs = recorded_inputs(scenario);
    let first = inputs
        .first()
        .unwrap_or_else(|| panic!("{scenario}: fixture should stream tool input fragments"));
    assert!(
        serde_json::from_str::<Value>(first).is_err(),
        "{scenario}: the first call's input must be cut short of JSON; a re-record healed it: {first:?}"
    );
    let interactions = crate::cassettes::recorded_interaction_bodies("anthropic", scenario);
    assert!(
        interactions
            .first()
            .is_some_and(|(_, response)| response.contains(r#""stop_reason":"max_tokens""#)),
        "{scenario}: the first reply must stop on `max_tokens`"
    );
}

fn assert_fixture_is_healthy(scenario: &str) {
    let inputs = recorded_inputs(scenario);
    assert!(
        !inputs.is_empty(),
        "{scenario}: fixture should stream tool input fragments"
    );
    for input in inputs {
        assert!(
            serde_json::from_str::<Value>(&input).is_ok(),
            "{scenario}: the recorded control must assemble to valid JSON, got {input:?}"
        );
    }
}

/// The output cap that cuts Sonnet 4.6's first `subtract` call inside its
/// input.
const CUT_AT: u64 = 60;

/// Caps the first request at [`CUT_AT`] output tokens.
#[derive(Clone)]
struct CutFirstCall;

impl AgentHook for CutFirstCall {
    async fn on_completion_call(
        &self,
        _ctx: &HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        if event.turn == 1 {
            CompletionCallAction::patch(RequestPatch::new().max_tokens(CUT_AT))
        } else {
            CompletionCallAction::Continue
        }
    }
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

/// The calculator agent, allowed `calls` model calls.
fn agent(client: AnthropicModels, calls: usize) -> rig::agent::Agent {
    rig::AgentBuilder::new(client.completion(anthropic::completion::CLAUDE_SONNET_4_6))
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .default_max_turns(calls)
        .build()
}

#[tokio::test]
async fn blocking_healthy_control() {
    with_anthropic_cassette(
        "malformed_tool_args_matrix/blocking_healthy_control",
        |client| async move {
            let response = agent(client, 2)
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
            let mut stream = agent(client, 2).prompt(STREAMING_TOOLS_PROMPT).stream();
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
    with_anthropic_cassette(
        "malformed_tool_args_matrix/streaming_malformed_call_is_answered_with_an_error",
        |client| async move {
            // Room for the call the model makes again after the error result.
            let mut stream = agent(client, 4)
                .prompt(STREAMING_TOOLS_PROMPT)
                .add_hook(CutFirstCall)
                .add_hook(NeverConsulted)
                .stream();
            let mut executed = false;
            let mut text = String::new();
            while let Some(item) = stream.next().await {
                match item.expect("a malformed call never ends the run") {
                    MultiTurnStreamItem::ToolExecutionCommitted { tool_call } => {
                        executed |= tool_call.function.invalid_arguments.is_some();
                    }
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
    assert_fixture_is_cut(scenario);
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
