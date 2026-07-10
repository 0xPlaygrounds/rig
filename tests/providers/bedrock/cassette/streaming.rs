//! AWS Bedrock streaming replay smoke tests.

use rig::bedrock;
use rig::client::CompletionClient;
use rig::streaming::StreamingPrompt;

use super::super::support::with_bedrock_cassette;
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT,
    Subtract, assert_mentions_expected_number, assert_nonempty_response,
    collect_stream_final_response,
};

#[tokio::test]
async fn streaming_smoke() {
    with_bedrock_cassette("streaming/streaming_smoke", |client| async move {
        let agent = client
            .agent(bedrock::completion::AMAZON_NOVA_LITE)
            .preamble(STREAMING_PREAMBLE)
            .build();

        let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}

#[tokio::test]
async fn streaming_tools_smoke() {
    with_bedrock_cassette("streaming/streaming_tools_smoke", |client| async move {
        let agent = client
            .agent(bedrock::completion::AMAZON_NOVA_LITE)
            .preamble(STREAMING_TOOLS_PREAMBLE)
            .max_tokens(1024)
            .tool(Subtract)
            .default_max_turns(2)
            .build();

        let mut stream = agent.stream_prompt(STREAMING_TOOLS_PROMPT).await;
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming tool prompt should succeed");

        assert_mentions_expected_number(&response, -3);
    })
    .await;
}
