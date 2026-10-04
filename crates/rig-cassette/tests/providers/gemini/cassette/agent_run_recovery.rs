//! Invalid tool-call recovery on [`AgentRun`] exercised against real Gemini
//! turns: the model's `add` call is recorded normally on the wire, while the
//! machine is fed restricted allowed-tool sets to trigger each recovery path
//! (fail, repair, skip, retry-budget exhaustion, bad repair).

use rig::agent::InvalidToolCallAction;
use rig::agent::run::{AgentRun, AgentRunStep, ModelTurnOutcome};
use rig::completion::PromptError;
use rig::message::ToolChoice;
use rig::providers::gemini;
use rig_agent::test_utils::validate_unknown_tool_failure;

use super::super::agent_run_support::{
    FORCE_TOOLS_PREAMBLE, GeminiAgent, assistant_tool_call_names, call_model,
    execute_pending_calls, tool_names,
};
use super::super::support::with_gemini_cassette;
use crate::support::assert_mentions_expected_number;

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
async fn repair_renames_tool_call_and_executes_it() {
    with_gemini_cassette(
        "agent_run_recovery/repair_renames_tool_call_and_executes_it",
        |client| async move {
            // `sum` is registered alongside `add` so the post-repair wire
            // history references a tool Gemini saw advertised.
            let agent = GeminiAgent::new(
                client.completion(gemini::completion::GEMINI_2_5_FLASH),
                FORCE_TOOLS_PREAMBLE,
                &["add", "sum"],
                None,
            );
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
                                .resolve_invalid_tool_call(InvalidToolCallAction::repair("sum"))
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
            assert_mentions_expected_number(&response.output(), 5);

            // The repaired name is what history records; the original name
            // never reaches the conversation.
            let messages = response.messages;
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
async fn repair_to_disallowed_name_fails_with_unknown_tool_call() {
    with_gemini_cassette(
        "agent_run_recovery/repair_to_disallowed_name_fails_with_unknown_tool_call",
        |client| async move {
            let agent = GeminiAgent::new(
                client.completion(gemini::completion::GEMINI_2_5_FLASH),
                FORCE_TOOLS_PREAMBLE,
                &["add"],
                Some(ToolChoice::Required),
            );

            let mut run = run_until_invalid_add_call(&agent, &tool_names(&["subtract"]), 0).await;
            let error = run
                .resolve_invalid_tool_call(InvalidToolCallAction::repair("multiply"))
                .expect_err("repairing to a disallowed name must error the run");

            validate_unknown_tool_failure(&error, "multiply", &["subtract"])
                .expect("portable rejected-repair diagnostics should hold");

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
