//! Durable runner checkpoint coverage against Gemini cassette replay.

use rig::agent::AgentRunnerOutcome;
use rig::client::CompletionClient;
use rig::completion::PromptError;
use rig::message::ToolChoice;
use rig::providers::gemini;

use super::super::agent_run_support::{Add, FORCE_TOOLS_PREAMBLE, Subtract};

#[tokio::test]
async fn runner_checkpoint_round_trips_before_tool_execution() {
    super::super::support::with_gemini_cassette(
        "agent_run_stepping/hand_driven_multi_turn_tool_run_completes",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool(Subtract)
                .build();
            let prompt = "Use the tools to compute (7 + 4) - 2: first compute 7 + 4 with the add tool, then subtract 2 from that result with the subtract tool, then state the final result.";

            let checkpoint = match agent
                .runner(prompt)
                .max_turns(3)
                .run_until_interruption()
                .await
                .expect("the first model turn should succeed")
            {
                AgentRunnerOutcome::Interrupted(run) => run,
                AgentRunnerOutcome::Completed(_) => panic!("the tool turn should interrupt"),
                _ => panic!("unexpected runner outcome"),
            };
            assert_eq!(
                checkpoint.pending_tool_calls().map(|calls| calls.len()),
                Some(1)
            );

            let checkpoint = serde_json::from_slice(
                &serde_json::to_vec(&checkpoint).expect("checkpoint should serialize"),
            )
            .expect("checkpoint should deserialize");
            let checkpoint = match agent
                .runner(prompt)
                .max_turns(3)
                .resume(checkpoint)
                .run_until_interruption()
                .await
                .expect("the resumed runner should reach the second tool")
            {
                AgentRunnerOutcome::Interrupted(run) => run,
                AgentRunnerOutcome::Completed(_) => panic!("subtract should require approval"),
                _ => panic!("unexpected runner outcome"),
            };

            let outcome = agent
                .runner(prompt)
                .max_turns(3)
                .resume(checkpoint)
                .run_until_interruption()
                .await
                .expect("the second resumed runner should finish");
            let AgentRunnerOutcome::Completed(response) = outcome else {
                panic!("the resumed run should complete");
            };
            assert!(response.output.contains('9'));
        },
    )
    .await;
}

#[tokio::test]
async fn runner_checkpoint_preserves_parallel_tool_batch() {
    super::super::support::with_gemini_cassette(
        "agent_run_stepping/hand_driven_parallel_tool_calls_arrive_in_one_step",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool(Subtract)
                .build();
            let prompt = "Compute 3 + 5 and 10 - 4. You MUST call the add tool and the subtract tool together in your first response, as two parallel function calls, then report both results.";

            let checkpoint = match agent
                .runner(prompt)
                .max_turns(3)
                .run_until_interruption()
                .await
                .expect("parallel tool turn should interrupt")
            {
                AgentRunnerOutcome::Interrupted(run) => run,
                AgentRunnerOutcome::Completed(_) => panic!("expected pending parallel tools"),
                _ => panic!("unexpected runner outcome"),
            };
            let calls = checkpoint
                .pending_tool_calls()
                .expect("checkpoint should contain pending calls");
            assert_eq!(calls.len(), 2);
            assert!(calls.iter().any(|call| call.tool_call.function.name == "add"));
            assert!(calls
                .iter()
                .any(|call| call.tool_call.function.name == "subtract"));

            let outcome = agent
                .runner(prompt)
                .max_turns(3)
                .resume(checkpoint)
                .run_until_interruption()
                .await
                .expect("parallel tools should execute and complete");
            let AgentRunnerOutcome::Completed(response) = outcome else {
                panic!("parallel run should complete after one approved batch");
            };
            assert!(response.output.contains('8'));
            assert!(response.output.contains('6'));
        },
    )
    .await;
}

#[tokio::test]
async fn runner_max_turns_error_keeps_pending_tool_results() {
    super::super::support::with_gemini_cassette(
        "agent_run_stepping/max_turns_error_carries_pending_tool_results_message",
        |client| async move {
            let agent = client
                .agent(gemini::completion::GEMINI_2_5_FLASH)
                .preamble(FORCE_TOOLS_PREAMBLE)
                .tool(Add)
                .tool_choice(ToolChoice::Required)
                .build();

            let error = agent
                .runner("What is 21 + 21? Use the add tool.")
                .max_turns(2)
                .run()
                .await
                .expect_err("required tool calls should exhaust the model-call budget");
            let PromptError::MaxTurnsError {
                max_turns,
                prompt,
                chat_history,
            } = error
            else {
                panic!("expected MaxTurnsError");
            };
            assert_eq!(max_turns, 2);
            assert!(matches!(*prompt, rig::message::Message::User { .. }));
            assert!(chat_history.iter().any(|message| {
                matches!(message, rig::message::Message::Assistant { content, .. }
                    if content.iter().any(|item| matches!(item, rig::message::AssistantContent::ToolCall(_))))
            }));
        },
    )
    .await;
}
