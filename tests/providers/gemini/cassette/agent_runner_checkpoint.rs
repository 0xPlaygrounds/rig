//! Durable runner checkpoint coverage against Gemini cassette replay.

use rig::agent::AgentRunnerOutcome;
use rig::client::CompletionClient;
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
