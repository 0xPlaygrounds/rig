//! A streamed Anthropic tool call the output limit cut inside its input.
//!
//! A cut call never fails the reply. It is kept with the arguments its text
//! still states, the text itself in `invalid_arguments`, and no provider
//! item, so it replays from its canonical fields. The agent never runs the
//! tool and does not offer the call to the invalid-call hook: it answers it
//! with an `is_error` tool result saying the reply reached the output limit,
//! and the run continues.
//!
//! The cut cell caps its first request at [`CUT_AT`] output tokens, which
//! cuts the `subtract` call inside its input, so the turn stops on
//! `max_tokens` with an input that is not JSON and no close of its block.
//! The cell asserts its fixture really holds a cut input, so a re-record
//! that heals it fails. A call that is malformed but not cut is answered
//! with the "not a JSON object" error, which the mock-wire tests of
//! rig-agent's `malformed_tool_args_tests` cover.

use std::sync::{Arc, Mutex};

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

/// Records every invalid call it is offered and leaves the decision to the
/// default.
#[derive(Clone, Default)]
struct Consulted(Arc<Mutex<Vec<InvalidToolCallContext>>>);

impl Consulted {
    fn seen(&self) -> Vec<InvalidToolCallContext> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .clone()
    }
}

impl AgentHook for Consulted {
    async fn on_invalid_tool_call(
        &self,
        _ctx: &HookContext,
        context: &InvalidToolCallContext,
    ) -> Option<InvalidToolCallAction> {
        self.0
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .push(context.clone());
        None
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
async fn streaming_cut_call_is_answered_as_cut_off() {
    with_anthropic_cassette(
        "malformed_tool_args_matrix/streaming_malformed_call_is_answered_with_an_error",
        |client| async move {
            let consulted = Consulted::default();
            // Room for the calls the model makes after the cut-off result.
            let mut stream = agent(client, 4)
                .prompt(STREAMING_TOOLS_PROMPT)
                .add_hook(CutFirstCall)
                .add_hook(consulted.clone())
                .stream();
            let mut executed = false;
            let mut text = String::new();
            while let Some(item) = stream.next().await {
                match item.expect("a cut call never ends the run") {
                    MultiTurnStreamItem::ToolExecutionCommitted { tool_call } => {
                        executed |= tool_call.function.invalid_arguments.is_some();
                    }
                    MultiTurnStreamItem::FinalResponse(response) => {
                        text = response.output().to_owned();
                    }
                    _ => {}
                }
            }
            assert!(!executed, "a cut call must never be executed");
            assert_mentions_expected_number(&text, -3);
            let seen = consulted.seen();
            assert!(
                seen.is_empty(),
                "a cut call is answered, not offered to the hook: {seen:?}"
            );
        },
    )
    .await;
    let scenario = "malformed_tool_args_matrix/streaming_malformed_call_is_answered_with_an_error";
    assert_fixture_is_cut(scenario);
}
