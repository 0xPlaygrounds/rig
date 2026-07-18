//! ChatGPT non-interactive OAuth cassette coverage.

use rig::prelude::AgentClientExt;
use rig::providers::chatgpt;
use rig::streaming::StreamingPrompt;

use super::super::support::with_chatgpt_noninteractive_oauth_cassette;
use crate::support::{
    BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
async fn cached_oauth_allows_noninteractive_streaming_completion() {
    with_chatgpt_noninteractive_oauth_cassette(
        "noninteractive_oauth/cached_oauth_allows_noninteractive_streaming_completion",
        |client| async move {
            client
                .authorize()
                .await
                .expect("cached OAuth auth should not require device flow");

            let agent = client
                .agent(chatgpt::GPT_5_4)
                .preamble(BASIC_PREAMBLE)
                .build();
            let mut stream = agent.stream_prompt(BASIC_PROMPT).await;
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("non-interactive OAuth streaming completion should succeed");

            assert_nonempty_response(&response);
        },
    )
    .await;
}
