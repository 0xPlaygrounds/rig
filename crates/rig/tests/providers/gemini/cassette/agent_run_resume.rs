//! The "serialized state alone" guarantee: an [`AgentRun`] suspended at any
//! point can be serialized, dropped, deserialized in a fresh context, and
//! driven to completion against real Gemini turns.

use rig::agent::InvalidToolCallHookAction;
use rig::agent::run::{AgentRun, AgentRunStep, ModelTurnOutcome};
use rig::client::CompletionClient;
use rig::providers::gemini;

use super::super::agent_run_support::{
    Add, FORCE_TOOLS_PREAMBLE, call_model, execute_pending_calls, history_has_assistant_tool_call,
    is_tool_result_user_message, tool_names, tool_result_texts,
};
use super::super::support::with_gemini_cassette;
use crate::support::{assert_mentions_expected_number, assert_nonempty_response};

/// Serialize the run, drop it, and bring it back from the JSON alone.
fn roundtrip(run: AgentRun) -> AgentRun {
    let suspended = serde_json::to_string(&run).expect("run state should serialize");
    drop(run);
    serde_json::from_str(&suspended).expect("run state should deserialize")
}

#[tokio::test]
async fn resume_from_serialized_state_mid_tool_execution() {
    with_gemini_cassette(
        "agent_run_resume/resume_from_serialized_state_mid_tool_execution",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .build();
            let names = tool_names(&["add"]);

            let mut run = AgentRun::new("What is 21 + 21? Use the add tool.").max_turns(2);
            let pending_calls = loop {
                match run.next_step().expect("run should advance") {
                    AgentRunStep::CallModel {
                        prompt, history, ..
                    } => {
                        let outcome = run
                            .model_response(
                                call_model(&agent, prompt, history, &names, &names).await,
                            )
                            .expect("model turn should be accepted");
                        assert!(matches!(outcome, ModelTurnOutcome::Continue { .. }));
                    }
                    AgentRunStep::CallTools { calls } => break calls,
                    AgentRunStep::Done(response) => {
                        panic!("the model should call the add tool before answering: {response:?}")
                    }
                }
            };

            // Suspend while tool calls are pending; resume from the JSON alone.
            let mut resumed = roundtrip(run);
            assert!(!resumed.is_done());
            assert_eq!(resumed.turn(), 1);
            assert_eq!(resumed.completion_calls().len(), 1);

            // The resumed run re-emits the pending calls from its own state,
            // idempotently.
            for attempt in 0..2 {
                let AgentRunStep::CallTools { calls } =
                    resumed.next_step().expect("resumed run should advance")
                else {
                    panic!("resumed run must re-emit the pending tool calls");
                };
                assert_eq!(
                    calls.len(),
                    pending_calls.len(),
                    "attempt {attempt}: resumed pending calls must match the suspended ones"
                );
                for (resumed_call, original) in calls.iter().zip(&pending_calls) {
                    assert_eq!(resumed_call.tool_call.id, original.tool_call.id);
                    assert_eq!(
                        resumed_call.tool_call.function.name,
                        original.tool_call.function.name
                    );
                    assert_eq!(
                        resumed_call.tool_call.function.arguments,
                        original.tool_call.function.arguments
                    );
                }
            }

            let AgentRunStep::CallTools { calls } =
                resumed.next_step().expect("resumed run should advance")
            else {
                panic!("resumed run must still be executing tools");
            };
            resumed
                .tool_results(execute_pending_calls(&calls))
                .expect("tool results should be accepted");

            let response = loop {
                match resumed.next_step().expect("resumed run should advance") {
                    AgentRunStep::CallModel {
                        prompt, history, ..
                    } => {
                        assert!(
                            history_has_assistant_tool_call(&history, "add"),
                            "resumed history must thread the pre-suspension tool call: {history:?}"
                        );
                        let outcome = resumed
                            .model_response(
                                call_model(&agent, prompt, history, &names, &names).await,
                            )
                            .expect("model turn should be accepted");
                        assert!(matches!(outcome, ModelTurnOutcome::Continue { .. }));
                    }
                    AgentRunStep::CallTools { calls } => {
                        resumed
                            .tool_results(execute_pending_calls(&calls))
                            .expect("tool results should be accepted");
                    }
                    AgentRunStep::Done(response) => break response,
                }
            };

            assert_mentions_expected_number(&response.output, 42);
            assert_eq!(resumed.completion_calls().len(), resumed.turn());
        },
    )
    .await;
}

#[tokio::test]
async fn resume_while_invalid_tool_call_awaits_resolution() {
    with_gemini_cassette(
        "agent_run_resume/resume_while_invalid_tool_call_awaits_resolution",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .build();
            let executable = tool_names(&["add"]);
            // Machine-side restriction: the model's `add` call is disallowed
            // for this turn even though it was advertised on the wire.
            let restricted = tool_names(&["subtract"]);

            let mut run = AgentRun::new("What is 21 + 21? Use the add tool.").max_turns(2);
            let AgentRunStep::CallModel {
                prompt, history, ..
            } = run.next_step().expect("run should advance")
            else {
                panic!("a fresh run starts with a model call");
            };
            let outcome = run
                .model_response(call_model(&agent, prompt, history, &executable, &restricted).await)
                .expect("model turn should be ingested");
            let ModelTurnOutcome::NeedsResolution(context) = outcome else {
                panic!("the add call must be rejected for this turn: {outcome:?}");
            };
            assert_eq!(context.tool_name, "add");
            assert!(!context.is_streaming);

            // Suspend mid-resolution; the resumed run re-derives the pending
            // invalid call from its own state.
            let mut resumed = roundtrip(run);
            let rederived = resumed
                .pending_invalid_tool_call()
                .expect("resumed run re-derives the pending invalid tool call");
            assert_eq!(rederived.tool_name, "add");
            assert_eq!(rederived.available_tools, vec!["add".to_string()]);
            assert_eq!(rederived.allowed_tools, vec!["subtract".to_string()]);

            let outcome = resumed
                .resolve_invalid_tool_call(InvalidToolCallHookAction::skip(
                    "The add tool is disabled for this request.",
                ))
                .expect("skip resolution should be accepted");
            assert!(
                matches!(
                    outcome,
                    ModelTurnOutcome::Continue {
                        response_hook_suppressed: true
                    }
                ),
                "recovered turns suppress the response hook"
            );

            // The skipped call comes back preresolved; the driver must not
            // execute it.
            let AgentRunStep::CallTools { calls } =
                resumed.next_step().expect("resumed run should advance")
            else {
                panic!("the skipped call must still be answered via CallTools");
            };
            assert_eq!(calls.len(), 1);
            let preresolved = calls
                .first()
                .and_then(|call| call.preresolved_result.clone())
                .expect("skipped calls carry a preresolved result");
            resumed
                .tool_results(vec![preresolved])
                .expect("preresolved results should be accepted");

            let response = loop {
                match resumed.next_step().expect("resumed run should advance") {
                    AgentRunStep::CallModel {
                        prompt, history, ..
                    } => {
                        let outcome = resumed
                            .model_response(
                                call_model(&agent, prompt, history, &executable, &executable).await,
                            )
                            .expect("model turn should be accepted");
                        assert!(matches!(outcome, ModelTurnOutcome::Continue { .. }));
                    }
                    AgentRunStep::CallTools { calls } => {
                        resumed
                            .tool_results(execute_pending_calls(&calls))
                            .expect("tool results should be accepted");
                    }
                    AgentRunStep::Done(response) => break response,
                }
            };
            assert_nonempty_response(&response.output);
        },
    )
    .await;
}

#[tokio::test]
async fn resume_after_invalid_tool_call_retry_rollback() {
    with_gemini_cassette(
        "agent_run_resume/resume_after_invalid_tool_call_retry_rollback",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .build();
            let executable = tool_names(&["add"]);
            let nothing_allowed = tool_names(&[]);
            const FEEDBACK: &str = "Tools are temporarily unavailable. Answer the question directly in plain text without calling any tools.";

            let mut run = AgentRun::new("What is 21 + 21?")
                .max_turns(3)
                .max_invalid_tool_call_retries(1);
            let AgentRunStep::CallModel {
                prompt, history, ..
            } = run.next_step().expect("run should advance")
            else {
                panic!("a fresh run starts with a model call");
            };
            let outcome = run
                .model_response(
                    call_model(&agent, prompt, history, &executable, &nothing_allowed).await,
                )
                .expect("model turn should be ingested");
            let ModelTurnOutcome::NeedsResolution(_) = outcome else {
                panic!("the tool call must be rejected for this turn: {outcome:?}");
            };

            let outcome = run
                .resolve_invalid_tool_call(InvalidToolCallHookAction::retry(FEEDBACK))
                .expect("retry should be accepted within budget");
            assert!(matches!(outcome, ModelTurnOutcome::TurnRetried));

            // Suspend right after the rollback; resume and take the retry turn.
            let mut resumed = roundtrip(run);
            let AgentRunStep::CallModel {
                prompt,
                history,
                turn,
            } = resumed.next_step().expect("resumed run should advance")
            else {
                panic!("a rolled-back run must retry with a model call");
            };
            assert_eq!(turn, 2, "the retry consumes multi-turn depth");
            assert!(
                history_has_assistant_tool_call(&history, "add"),
                "the rolled-back assistant turn stays in history: {history:?}"
            );
            assert!(
                is_tool_result_user_message(&prompt),
                "the retry prompt is the corrective tool-results message: {prompt:?}"
            );
            assert!(
                tool_result_texts(&prompt).iter().any(|text| text.contains(FEEDBACK)),
                "the retry prompt must carry the hook feedback: {prompt:?}"
            );

            // Take the retry turn with the full allowed set, then drive the
            // run to completion normally.
            let outcome = resumed
                .model_response(call_model(&agent, prompt, history, &executable, &executable).await)
                .expect("retry model turn should be accepted");
            assert!(matches!(outcome, ModelTurnOutcome::Continue { .. }));

            let response = loop {
                match resumed.next_step().expect("resumed run should advance") {
                    AgentRunStep::CallModel {
                        prompt, history, ..
                    } => {
                        let outcome = resumed
                            .model_response(
                                call_model(&agent, prompt, history, &executable, &executable).await,
                            )
                            .expect("model turn should be accepted");
                        assert!(matches!(outcome, ModelTurnOutcome::Continue { .. }));
                    }
                    AgentRunStep::CallTools { calls } => {
                        resumed
                            .tool_results(execute_pending_calls(&calls))
                            .expect("tool results should be accepted");
                    }
                    AgentRunStep::Done(response) => break response,
                }
            };

            assert_nonempty_response(&response.output);
            assert!(resumed.completion_calls().len() >= 2);
        },
    )
    .await;
}
