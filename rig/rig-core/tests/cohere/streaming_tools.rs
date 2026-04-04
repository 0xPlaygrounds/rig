//! Cohere streaming tools smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::cohere;
use rig::streaming::StreamingPrompt;

use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires COHERE_API_KEY"]
async fn streaming_tools_smoke() {
    let client = cohere::Client::from_env();
    let agent = client
        .agent(cohere::COMMAND_R)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_TOOLS_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}
