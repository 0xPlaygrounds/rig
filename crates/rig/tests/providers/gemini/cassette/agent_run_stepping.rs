//! Hand-driving the sans-IO [`AgentRun`] state machine against real Gemini
//! turns: stepping protocol, multi-turn tool threading, parallel tool calls,
//! and `max_turns` exhaustion.

use rig::agent::run::{AgentRun, AgentRunStep, ModelTurnOutcome};
use rig::client::CompletionClient;
use rig::completion::PromptError;
use rig::message::{Message, ToolChoice, UserContent};
use rig::providers::gemini;

use super::super::agent_run_support::{
    Add, FORCE_TOOLS_PREAMBLE, Subtract, call_model, execute_pending_calls,
    history_has_assistant_tool_call, is_tool_result_user_message, sum_completion_call_usage,
    tool_names,
};
use super::super::support::with_gemini_cassette;
use crate::support::{
    BASIC_PREAMBLE, BASIC_PROMPT, assert_mentions_expected_number, assert_nonempty_response,
};

#[tokio::test]
async fn hand_driven_single_turn_completes() {
    with_gemini_cassette(
        "agent_run_stepping/hand_driven_single_turn_completes",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(BASIC_PREAMBLE)
                .build();
            let names = tool_names(&[]);

            let mut run = AgentRun::new(BASIC_PROMPT);
            let response = loop {
                match run.next_step().expect("run should advance") {
                    AgentRunStep::CallModel {
                        prompt,
                        history,
                        turn,
                    } => {
                        assert_eq!(turn, 1, "a tool-free run makes exactly one model call");
                        assert!(
                            history.is_empty(),
                            "first turn of a history-free run starts empty: {history:?}"
                        );
                        let outcome = run
                            .model_response(
                                call_model(&agent, prompt, history, &names, &names).await,
                            )
                            .expect("model turn should be accepted");
                        assert!(
                            matches!(
                                outcome,
                                ModelTurnOutcome::Continue {
                                    response_hook_suppressed: false
                                }
                            ),
                            "unrecovered turns must not suppress the response hook"
                        );
                    }
                    AgentRunStep::CallTools { calls } => {
                        panic!("tool-free run must not request tool execution: {calls:?}")
                    }
                    AgentRunStep::Done(response) => break response,
                }
            };

            assert_nonempty_response(&response.output);
            assert!(run.is_done());
            assert_eq!(
                run.response()
                    .expect("done run exposes its response")
                    .output,
                response.output
            );
            assert_eq!(run.turn(), 1);
            assert_eq!(response.completion_calls.len(), 1);
            assert_eq!(run.usage(), response.usage);
            assert_eq!(
                sum_completion_call_usage(&response.completion_calls),
                response.usage,
                "aggregate usage must equal the sum of per-call usage"
            );
            assert!(
                response.usage.total_tokens > 0,
                "cassette-recorded usage should be non-zero"
            );

            let messages = response.messages.clone().expect("run reports its messages");
            assert_eq!(messages.as_slice(), run.messages());
            assert_eq!(
                messages.len(),
                2,
                "single turn accumulates [user prompt, assistant reply]: {messages:?}"
            );
            assert!(matches!(messages.first(), Some(Message::User { .. })));
            assert!(matches!(messages.last(), Some(Message::Assistant { .. })));
        },
    )
    .await;
}

#[tokio::test]
async fn hand_driven_multi_turn_tool_run_completes() {
    with_gemini_cassette(
        "agent_run_stepping/hand_driven_multi_turn_tool_run_completes",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool(Subtract)
                .build();
            let names = tool_names(&["add", "subtract"]);

            let mut run = AgentRun::new(
                "Use the tools to compute (7 + 4) - 2: first compute 7 + 4 with the add tool, then subtract 2 from that result with the subtract tool, then state the final result.",
            )
            .max_turns(5);
            let mut executed_tools: Vec<String> = Vec::new();

            let response = loop {
                match run.next_step().expect("run should advance") {
                    AgentRunStep::CallModel {
                        prompt,
                        history,
                        turn,
                    } => {
                        if turn > 1 {
                            assert!(
                                is_tool_result_user_message(&prompt),
                                "follow-up turns are prompted by the pending tool results: {prompt:?}"
                            );
                            assert!(
                                history.iter().any(|m| matches!(m, Message::Assistant { .. })),
                                "follow-up history threads the prior assistant turn: {history:?}"
                            );
                        }
                        let outcome = run
                            .model_response(call_model(&agent, prompt, history, &names, &names).await)
                            .expect("model turn should be accepted");
                        assert!(matches!(outcome, ModelTurnOutcome::Continue { .. }));
                    }
                    AgentRunStep::CallTools { calls } => {
                        for call in &calls {
                            assert!(
                                call.preresolved_result.is_none(),
                                "no recovery happened, so no call should be preresolved"
                            );
                            assert!(
                                call.internal_call_id.is_none(),
                                "non-streamed turns carry no internal call ids"
                            );
                            executed_tools.push(call.tool_call.function.name.clone());
                        }
                        run.tool_results(execute_pending_calls(&calls))
                            .expect("tool results should be accepted");
                    }
                    AgentRunStep::Done(response) => break response,
                }
            };

            assert!(
                executed_tools.iter().any(|name| name == "add"),
                "the add tool should run: {executed_tools:?}"
            );
            assert!(
                executed_tools.iter().any(|name| name == "subtract"),
                "the subtract tool should run: {executed_tools:?}"
            );
            assert_mentions_expected_number(&response.output, 9);
            assert!(run.turn() >= 2, "tool use forces at least two model calls");
            assert_eq!(
                response.completion_calls.len(),
                run.turn(),
                "every model call records exactly one completion call"
            );
            assert_eq!(
                sum_completion_call_usage(&response.completion_calls),
                response.usage
            );

            let messages = response.messages.clone().expect("run reports its messages");
            assert!(history_has_assistant_tool_call(&messages, "add"));
            assert!(history_has_assistant_tool_call(&messages, "subtract"));
            assert!(
                messages.iter().any(is_tool_result_user_message),
                "tool results must be threaded into history: {messages:?}"
            );
        },
    )
    .await;
}

#[tokio::test]
async fn hand_driven_parallel_tool_calls_arrive_in_one_step() {
    with_gemini_cassette(
        "agent_run_stepping/hand_driven_parallel_tool_calls_arrive_in_one_step",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool(Subtract)
                .build();
            let names = tool_names(&["add", "subtract"]);

            let mut run = AgentRun::new(
                "Compute 3 + 5 and 10 - 4. You MUST call the add tool and the subtract tool together in your first response, as two parallel function calls, then report both results.",
            )
            .max_turns(3);
            let mut first_tool_step: Option<Vec<String>> = None;

            let response = loop {
                match run.next_step().expect("run should advance") {
                    AgentRunStep::CallModel {
                        prompt, history, ..
                    } => {
                        let outcome = run
                            .model_response(call_model(&agent, prompt, history, &names, &names).await)
                            .expect("model turn should be accepted");
                        assert!(matches!(outcome, ModelTurnOutcome::Continue { .. }));
                    }
                    AgentRunStep::CallTools { calls } => {
                        if first_tool_step.is_none() {
                            first_tool_step = Some(
                                calls
                                    .iter()
                                    .map(|call| call.tool_call.function.name.clone())
                                    .collect(),
                            );
                        }
                        // Deliver results in reverse emission order: the
                        // machine accepts them in any order.
                        let mut results = execute_pending_calls(&calls);
                        results.reverse();
                        run.tool_results(results)
                            .expect("tool results in any order should be accepted");
                    }
                    AgentRunStep::Done(response) => break response,
                }
            };

            let first_step = first_tool_step.expect("the model should call tools");
            assert_eq!(
                first_step.len(),
                2,
                "both calls should arrive in a single CallTools step: {first_step:?}"
            );
            assert!(first_step.iter().any(|name| name == "add"));
            assert!(first_step.iter().any(|name| name == "subtract"));
            assert_mentions_expected_number(&response.output, 8);
            assert_mentions_expected_number(&response.output, 6);
        },
    )
    .await;
}

#[tokio::test]
async fn max_turns_error_carries_pending_tool_results_message() {
    with_gemini_cassette(
        "agent_run_stepping/max_turns_error_carries_pending_tool_results_message",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool_choice(ToolChoice::Required)
                .build();
            let names = tool_names(&["add"]);

            // max_turns 0: the run errors once tool results demand a second
            // model call.
            let mut run = AgentRun::new("What is 21 + 21? Use the add tool.");
            let error = loop {
                match run.next_step() {
                    Ok(AgentRunStep::CallModel {
                        prompt, history, ..
                    }) => {
                        let outcome = run
                            .model_response(call_model(&agent, prompt, history, &names, &names).await)
                            .expect("model turn should be accepted");
                        assert!(matches!(outcome, ModelTurnOutcome::Continue { .. }));
                    }
                    Ok(AgentRunStep::CallTools { calls }) => {
                        run.tool_results(execute_pending_calls(&calls))
                            .expect("tool results should be accepted");
                    }
                    Ok(AgentRunStep::Done(response)) => {
                        panic!("run should exhaust max_turns before completing: {response:?}")
                    }
                    Err(error) => break error,
                }
            };

            let PromptError::MaxTurnsError {
                max_turns,
                chat_history,
                prompt,
            } = error
            else {
                panic!("expected MaxTurnsError, got {error:?}");
            };
            assert_eq!(max_turns, 0);
            // Pins the divergence resolved by #1899: the error carries the
            // actual pending message (the tool-results user message), not a
            // reconstruction of its text.
            assert!(
                is_tool_result_user_message(&prompt),
                "MaxTurnsError must carry the pending tool-results message: {prompt:?}"
            );
            assert!(
                matches!(
                    &*prompt,
                    Message::User { content }
                        if content.iter().all(|item| matches!(item, UserContent::ToolResult(_)))
                ),
                "the pending message should consist of tool results only: {prompt:?}"
            );
            assert!(
                history_has_assistant_tool_call(&chat_history, "add"),
                "the error history must include the recorded assistant tool-call turn: {chat_history:?}"
            );
        },
    )
    .await;
}
