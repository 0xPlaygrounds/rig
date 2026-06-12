//! Invalid tool-call recovery on [`AgentRun`] exercised against real Gemini
//! turns: the model's `add` call is recorded normally on the wire, while the
//! machine is fed restricted allowed-tool sets to trigger each recovery path
//! (fail, repair, skip, retry-budget exhaustion, bad repair).

use rig::agent::InvalidToolCallHookAction;
use rig::agent::run::{AgentRun, AgentRunStep, ModelTurnOutcome};
use rig::client::CompletionClient;
use rig::completion::PromptError;
use rig::message::ToolChoice;
use rig::providers::gemini;

use super::super::agent_run_support::{
    Add, FORCE_TOOLS_PREAMBLE, Subtract, Sum, assistant_tool_call_names, call_model,
    execute_pending_calls, history_has_assistant_tool_call, tool_names,
    user_content_tool_result_texts,
};
use super::super::support::with_gemini_cassette;
use crate::support::{assert_mentions_expected_number, assert_nonempty_response};

const SKIP_REASON: &str = "The add tool is disabled for this request.";

/// Drive a fresh single-tool run to its first `NeedsResolution`, returning
/// the run mid-resolution.
async fn run_until_invalid_add_call(
    agent: &super::super::agent_run_support::GeminiAgent,
    allowed: &std::collections::BTreeSet<String>,
    retries: usize,
) -> AgentRun {
    let executable = tool_names(&["add"]);
    let mut run = AgentRun::new("What is 21 + 21? Use the add tool.")
        .max_turns(2)
        .max_invalid_tool_call_retries(retries);
    let AgentRunStep::CallModel {
        prompt, history, ..
    } = run.next_step().expect("run should advance")
    else {
        panic!("a fresh run starts with a model call");
    };
    let outcome = run
        .model_response(call_model(agent, prompt, history, &executable, allowed).await)
        .expect("model turn should be ingested");
    let ModelTurnOutcome::NeedsResolution(context) = outcome else {
        panic!("the add call must be rejected for this turn: {outcome:?}");
    };
    assert_eq!(context.tool_name, "add");
    run
}

#[tokio::test]
async fn fail_resolution_returns_unknown_tool_call() {
    with_gemini_cassette(
        "agent_run_recovery/fail_resolution_returns_unknown_tool_call",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool_choice(ToolChoice::Required)
                .build();

            let mut run = run_until_invalid_add_call(&agent, &tool_names(&[]), 0).await;
            let error = run
                .resolve_invalid_tool_call(InvalidToolCallHookAction::fail())
                .expect_err("fail resolution must error the run");

            let PromptError::UnknownToolCall {
                tool_name,
                available_tools,
                allowed_tools,
                chat_history,
            } = error
            else {
                panic!("expected UnknownToolCall, got {error:?}");
            };
            assert_eq!(tool_name, "add");
            assert_eq!(available_tools, vec!["add".to_string()]);
            assert!(allowed_tools.is_empty());
            assert!(
                history_has_assistant_tool_call(&chat_history, "add"),
                "the diagnostic history must include the rejected assistant turn: {chat_history:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn repair_renames_tool_call_and_executes_it() {
    with_gemini_cassette(
        "agent_run_recovery/repair_renames_tool_call_and_executes_it",
        |client| async move {
            // `sum` is registered alongside `add` so the post-repair wire
            // history references a tool Gemini saw advertised.
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool(Sum)
                .build();
            let machine_names = tool_names(&["sum"]);

            let mut run =
                AgentRun::new("Use the add tool to compute 2 + 3, then state the result.")
                    .max_turns(3);
            let mut repaired_calls = 0_usize;

            let response = loop {
                match run.next_step().expect("run should advance") {
                    AgentRunStep::CallModel {
                        prompt, history, ..
                    } => {
                        let repaired_before = repaired_calls;
                        let mut outcome = run
                            .model_response(
                                call_model(&agent, prompt, history, &machine_names, &machine_names)
                                    .await,
                            )
                            .expect("model turn should be ingested");
                        while let ModelTurnOutcome::NeedsResolution(context) = outcome {
                            assert_eq!(context.tool_name, "add");
                            outcome = run
                                .resolve_invalid_tool_call(InvalidToolCallHookAction::repair("sum"))
                                .expect("repair to an allowed tool should be accepted");
                            repaired_calls += 1;
                            assert!(repaired_calls < 6, "repair loop did not converge");
                        }
                        let ModelTurnOutcome::Continue {
                            response_hook_suppressed,
                        } = outcome
                        else {
                            panic!("repaired turns continue: {outcome:?}");
                        };
                        assert_eq!(
                            response_hook_suppressed,
                            repaired_calls > repaired_before,
                            "exactly the recovered turns suppress the response hook"
                        );
                    }
                    AgentRunStep::CallTools { calls } => {
                        for call in &calls {
                            assert_eq!(
                                call.tool_call.function.name, "sum",
                                "the repaired name must reach the driver"
                            );
                            assert!(call.preresolved_result.is_none());
                        }
                        run.tool_results(execute_pending_calls(&calls))
                            .expect("tool results should be accepted");
                    }
                    AgentRunStep::Done(response) => break response,
                }
            };

            assert!(repaired_calls >= 1, "at least one call should be repaired");
            assert_mentions_expected_number(&response.output, 5);

            // The repaired name is what history records; the original name
            // never reaches the conversation.
            let messages = response.messages.clone().expect("run reports its messages");
            let recorded: Vec<String> = messages
                .iter()
                .flat_map(assistant_tool_call_names)
                .collect();
            assert!(recorded.iter().any(|name| name == "sum"), "{recorded:?}");
            assert!(
                !recorded.iter().any(|name| name == "add"),
                "the unrepaired name must not be recorded: {recorded:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn skip_suppresses_every_call_in_the_turn() {
    with_gemini_cassette(
        "agent_run_recovery/skip_suppresses_every_call_in_the_turn",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool(Subtract)
                .build();
            let executable = tool_names(&["add", "subtract"]);
            // `add` is disallowed for the first turn.
            let restricted = tool_names(&["subtract"]);

            let mut run = AgentRun::new(
                "Compute 3 + 5 and 10 - 4. You MUST call the add tool and the subtract tool together in your first response, as two parallel function calls, then report both results.",
            )
            .max_turns(3);
            let mut skipped_turn_calls: Option<Vec<String>> = None;

            let response = loop {
                match run.next_step().expect("run should advance") {
                    AgentRunStep::CallModel {
                        prompt, history, ..
                    } => {
                        let (allowed, expect_invalid) = if skipped_turn_calls.is_none() {
                            (&restricted, true)
                        } else {
                            (&executable, false)
                        };
                        let mut outcome = run
                            .model_response(
                                call_model(&agent, prompt, history, &executable, allowed).await,
                            )
                            .expect("model turn should be ingested");
                        if let ModelTurnOutcome::NeedsResolution(context) = outcome {
                            assert!(expect_invalid, "only the first turn restricts tools");
                            assert_eq!(context.tool_name, "add");
                            outcome = run
                                .resolve_invalid_tool_call(InvalidToolCallHookAction::skip(
                                    SKIP_REASON,
                                ))
                                .expect("skip should be accepted");
                        }
                        assert!(matches!(outcome, ModelTurnOutcome::Continue { .. }));
                    }
                    AgentRunStep::CallTools { calls } => {
                        if skipped_turn_calls.is_none() {
                            // Recovery skipped `add`, so no call in this turn
                            // may execute: each one is preresolved.
                            let mut names = Vec::new();
                            for call in &calls {
                                names.push(call.tool_call.function.name.clone());
                                let preresolved = call
                                    .preresolved_result
                                    .clone()
                                    .expect("every call in a skipped turn is preresolved");
                                let texts = user_content_tool_result_texts(&preresolved);
                                if call.tool_call.function.name == "add" {
                                    assert!(
                                        texts.iter().any(|text| text.contains(SKIP_REASON)),
                                        "the skipped call carries the hook reason: {texts:?}"
                                    );
                                } else {
                                    assert!(
                                        texts.iter().any(|text| text
                                            .contains("another tool call in the same assistant turn was invalid")),
                                        "peers carry the not-executed marker: {texts:?}"
                                    );
                                }
                            }
                            skipped_turn_calls = Some(names);
                        }
                        run.tool_results(execute_pending_calls(&calls))
                            .expect("tool results should be accepted");
                    }
                    AgentRunStep::Done(response) => break response,
                }
            };

            let first_turn = skipped_turn_calls.expect("the model should call tools");
            assert!(
                first_turn.iter().any(|name| name == "add"),
                "the skipped add call still reaches the driver: {first_turn:?}"
            );
            assert_nonempty_response(&response.output);
        },
    )
    .await;
}

#[tokio::test]
async fn retry_with_exhausted_budget_fails_with_unknown_tool_call() {
    with_gemini_cassette(
        "agent_run_recovery/retry_with_exhausted_budget_fails_with_unknown_tool_call",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool_choice(ToolChoice::Required)
                .build();

            // Zero retry budget: the first retry resolution must fail.
            let mut run = run_until_invalid_add_call(&agent, &tool_names(&[]), 0).await;
            let error = run
                .resolve_invalid_tool_call(InvalidToolCallHookAction::retry(
                    "Try a different tool.",
                ))
                .expect_err("retry without budget must error the run");

            let PromptError::UnknownToolCall { tool_name, .. } = error else {
                panic!("expected UnknownToolCall, got {error:?}");
            };
            assert_eq!(tool_name, "add");
        },
    )
    .await;
}

#[tokio::test]
async fn repair_to_disallowed_name_fails_with_unknown_tool_call() {
    with_gemini_cassette(
        "agent_run_recovery/repair_to_disallowed_name_fails_with_unknown_tool_call",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool_choice(ToolChoice::Required)
                .build();

            let mut run = run_until_invalid_add_call(&agent, &tool_names(&["subtract"]), 0).await;
            let error = run
                .resolve_invalid_tool_call(InvalidToolCallHookAction::repair("multiply"))
                .expect_err("repairing to a disallowed name must error the run");

            let PromptError::UnknownToolCall {
                tool_name,
                allowed_tools,
                ..
            } = error
            else {
                panic!("expected UnknownToolCall, got {error:?}");
            };
            assert_eq!(
                tool_name, "multiply",
                "the error names the rejected repair target"
            );
            assert_eq!(allowed_tools, vec!["subtract".to_string()]);
        },
    )
    .await;
}
