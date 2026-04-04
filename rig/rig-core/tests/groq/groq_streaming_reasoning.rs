//! Migrated from `examples/groq_streaming_reasoning.rs`.

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::groq::{self, DEEPSEEK_R1_DISTILL_LLAMA_70B};
use rig::streaming::StreamingPrompt;

use crate::support::{assert_nonempty_response, collect_stream_final_response};

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn parsed_reasoning_stream() {
    let client = groq::Client::from_env();
    let agent = client
        .agent(DEEPSEEK_R1_DISTILL_LLAMA_70B)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .additional_params(serde_json::json!({ "reasoning_format": "parsed" }))
        .build();

    let mut stream = agent.stream_prompt("Entertain me!").await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
