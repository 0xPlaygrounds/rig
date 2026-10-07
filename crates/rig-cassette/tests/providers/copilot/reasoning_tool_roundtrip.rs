//! Copilot reasoning-enabled tool roundtrip tests.

use std::sync::Arc;
use std::sync::atomic::AtomicUsize;

use rig::completion::Message;

use crate::copilot::{live_client, live_responses_model, with_copilot_cassette};
use crate::reasoning::{self, WeatherTool};

#[tokio::test]
#[ignore = "requires Copilot credentials or existing OAuth cache"]
async fn streaming() {
    let call_count = Arc::new(AtomicUsize::new(0));
    let agent = rig::AgentBuilder::new(live_client().await.completion(live_responses_model()))
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .options(
            rig::completion::GenerationOptions::default().reasoning(rig::completion::Effort::High),
        )
        .build();

    let stream = agent
        .prompt(reasoning::TOOL_USER_PROMPT)
        .history(Vec::<Message>::new())
        .max_turns(3)
        .stream();

    let stats = reasoning::collect_stream_stats(stream, "copilot").await;
    reasoning::assert_universal(&stats, &call_count, "copilot");

    if stats.reasoning_block_count > 0 {
        assert!(
            stats.reasoning_has_encrypted || stats.reasoning_content_types.contains(&"Summary"),
            "[copilot] Expected encrypted or summary reasoning content. Got: {:?}",
            stats.reasoning_content_types
        );
    }
}

#[tokio::test]
async fn nonstreaming() {
    with_copilot_cassette(
        "reasoning_tool_roundtrip/nonstreaming",
        |client| async move {
            let call_count = Arc::new(AtomicUsize::new(0));
            let agent = rig::AgentBuilder::new(client.completion(live_responses_model()))
                .preamble(reasoning::TOOL_SYSTEM_PROMPT)
                .max_tokens(4096)
                .tool(WeatherTool::new(call_count.clone()))
                .options(
                    rig::completion::GenerationOptions::default()
                        .reasoning(rig::completion::Effort::High),
                )
                .default_max_turns(2)
                .build();

            let result = agent
                .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
                .await
                .expect("[copilot] Non-streaming chat failed");

            reasoning::assert_nonstreaming_universal(&result.output(), &call_count, "copilot");
        },
    )
    .await;
}
