//! Cohere streaming smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::cohere;
use rig::streaming::StreamingPrompt;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires COHERE_API_KEY"]
async fn streaming_smoke() {
    let client = cohere::Client::from_env();
    let agent = client
        .agent(cohere::COMMAND)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
