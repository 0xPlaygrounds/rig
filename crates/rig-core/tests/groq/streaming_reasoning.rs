//! Migrated from `examples/groq_streaming_reasoning.rs`.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::providers::groq;
use rig_core::streaming::StreamingPrompt;

use crate::support::{assert_nonempty_response, collect_stream_final_response};

use super::STREAMING_REASONING_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn parsed_reasoning_stream() {
    let client = groq::Client::from_env().expect("client should build");
    let agent = client
        .agent(STREAMING_REASONING_MODEL)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .additional_params(serde_json::json!({ "reasoning_format": "parsed" }))
        .build();

    let mut stream = agent.stream_prompt("Entertain me!").await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
