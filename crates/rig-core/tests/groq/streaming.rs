//! Groq streaming smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::providers::groq;
use rig_core::streaming::StreamingPrompt;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

use super::STREAMING_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn streaming_smoke() {
    let client = groq::Client::from_env().expect("client should build");
    let agent = client
        .agent(STREAMING_MODEL)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
