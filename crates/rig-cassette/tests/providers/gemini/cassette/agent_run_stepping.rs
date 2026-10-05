//! Hand-driving the sans-IO [`AgentRun`] state machine against real Gemini
//! turns: stepping protocol, multi-turn tool threading, parallel tool calls,
//! and `max_turns` exhaustion.

use rig::agent::run::{AgentRun, AgentRunStep, ModelTurnOutcome};
use rig::message::Message;
use rig::providers::gemini;

use super::super::agent_run_support::{
    GeminiAgent, call_model, sum_completion_call_usage, tool_names,
};
use super::super::support::with_gemini_cassette;
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
async fn hand_driven_single_turn_completes() {
    with_gemini_cassette(
        "agent_run_stepping/hand_driven_single_turn_completes",
        |client| async move {
            let agent = GeminiAgent::new(
                client.completion(gemini::completion::GEMINI_2_5_FLASH),
                BASIC_PREAMBLE,
                &[],
                None,
            );
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

            assert_nonempty_response(&response.output());
            assert!(run.is_done());
            assert_eq!(
                run.response()
                    .expect("done run exposes its response")
                    .output(),
                response.output()
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
                response.usage.total_tokens.is_some_and(|n| n > 0),
                "cassette-recorded usage should be non-zero"
            );

            let messages = response.messages;
            assert_eq!(messages.as_slice(), run.messages());
            assert_eq!(
                messages.len(),
                2,
                "single turn accumulates [user prompt, assistant reply]: {messages:?}"
            );
            assert!(matches!(messages.first(), Some(Message::User { .. })));
            assert!(matches!(messages.last(), Some(Message::Assistant(_))));
        },
    )
    .await;
}
