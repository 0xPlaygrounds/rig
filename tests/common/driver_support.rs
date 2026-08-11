//! Shared helpers for `AgentDriver` cassette suites.
//!
//! Every provider's driver suite drives the same protocol — prepare, send,
//! feed, dispatch, repeat — so the loop lives here and the per-provider tests
//! keep only what is provider-specific: the client, the model id, and the
//! assertions about the request the provider actually received.
//!
//! The assertions these helpers make are deliberately structural (a tool call
//! happened, the tool ran, the final answer is non-empty). Model wording varies
//! between recordings; request *shape* does not, and that is what the cassette
//! harness pins for us.

#![allow(dead_code)]

use rig::agent::{
    AgentDriver, DriveStep, InvalidToolCallAction, ModelTurnOutcome, PendingToolCall, TurnTools,
};
use rig::completion::{CompletionResponse, PromptError};
use rig::tool::ToolContext;

/// Preamble that reliably drives a tool call on every provider tested.
pub(crate) const FORCE_TOOLS_PREAMBLE: &str = "You are a calculator assistant. You MUST use the provided tools for every arithmetic operation instead of computing it yourself. After you have the tool results, reply with the final numeric answer in plain text.";

/// A prompt whose only sensible answer is one `add` call.
pub(crate) const ADD_PROMPT: &str = "What is 2 + 5?";

/// Advance the driver, expecting a request to send.
pub(crate) async fn expect_send(
    driver: &mut AgentDriver,
) -> (
    Box<rig::completion::CompletionRequestBuilder<rig::agent::ModelHandle>>,
    TurnTools,
    usize,
) {
    match driver.next_step().await.expect("next_step should succeed") {
        DriveStep::SendRequest {
            request,
            tools,
            turn,
        } => (request, tools, turn),
        other => panic!("expected SendRequest, got {other:?}"),
    }
}

/// Advance the driver, expecting pending tool calls.
pub(crate) async fn expect_execute_tools(
    driver: &mut AgentDriver,
) -> (Vec<PendingToolCall>, TurnTools) {
    match driver.next_step().await.expect("next_step should succeed") {
        DriveStep::ExecuteTools { calls, tools } => (calls, tools),
        other => panic!("expected ExecuteTools, got {other:?}"),
    }
}

/// Advance the driver, expecting the run to be finished.
pub(crate) async fn expect_done(driver: &mut AgentDriver) -> rig::agent::PromptResponse {
    match driver.next_step().await.expect("next_step should succeed") {
        DriveStep::Done(response) => response,
        other => panic!("expected Done, got {other:?}"),
    }
}

/// Dispatch every pending call through the turn that advertised it, and feed
/// the results back.
pub(crate) async fn dispatch_and_feed(
    driver: &mut AgentDriver,
    calls: &[PendingToolCall],
    tools: &TurnTools,
) {
    let mut context = ToolContext::new();
    let mut results = Vec::new();
    for call in calls {
        results.push(tools.execute_call(call, &mut context).await);
    }
    driver
        .tool_results(results)
        .expect("tool results should be accepted");
}

/// Feed a model response and assert the turn was accepted outright.
///
/// `ModelTurnOutcome` is `#[must_use]` because `NeedsResolution` must be
/// answered before the run may advance. These suites drive well-behaved turns,
/// so an unexpected resolution request is a test failure worth naming — not a
/// value to drop on the floor, which would resurface two steps later as an
/// unrelated protocol violation.
pub(crate) fn expect_turn_accepted(driver: &mut AgentDriver, response: &CompletionResponse) {
    match driver
        .model_response(response)
        .expect("the model turn should be accepted")
    {
        ModelTurnOutcome::Continue { .. } => {}
        other => panic!("expected the turn to be accepted outright, got {other:?}"),
    }
}

/// Drive a blocking run to completion, returning the final response.
///
/// The loop a caller writes by hand, in the shape the driver's own docs use —
/// including the arm most hand-written loops forget. A model that hallucinates
/// a tool name yields `NeedsResolution`, and the run cannot advance until it is
/// answered; this helper answers `Fail`, the correct default for a caller with
/// no recovery policy, so the run surfaces the invalid call rather than a
/// protocol violation about it.
pub(crate) async fn drive_to_completion(
    driver: &mut AgentDriver,
) -> Result<rig::agent::PromptResponse, PromptError> {
    loop {
        match driver.next_step().await? {
            DriveStep::SendRequest { request, .. } => {
                let response = request.send().await.map_err(PromptError::CompletionError)?;
                let mut outcome = driver.model_response(&response)?;
                while let ModelTurnOutcome::NeedsResolution(_) = outcome {
                    outcome = driver.resolve_invalid_tool_call(InvalidToolCallAction::Fail)?;
                }
            }
            DriveStep::ExecuteTools { calls, tools } => {
                let mut context = ToolContext::new();
                let mut results = Vec::new();
                for call in &calls {
                    results.push(tools.execute_call(call, &mut context).await);
                }
                driver.tool_results(results)?;
            }
            DriveStep::Done(response) => return Ok(response),
        }
    }
}

// Deliberately no "inspect the built request" helper. `build()` consumes the
// builder, so a test that inspects a request cannot also send it — and the
// recorded cassette body is the stronger assertion anyway: it is what the
// provider received, not what rig believed it was sending.
