//! ChatGPT non-interactive OAuth cassette coverage.

use rig::prelude::*;
use rig::providers::chatgpt;

use super::super::support::with_chatgpt_noninteractive_oauth_cassette;
use crate::support::{
    BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
async fn cached_oauth_allows_noninteractive_streaming_completion() {
    with_chatgpt_noninteractive_oauth_cassette(
        "noninteractive_oauth/cached_oauth_allows_noninteractive_streaming_completion",
        |client| async move {
            // The harness already resolved the cached credential without a
            // device flow — the yielded provider holds it — so the only thing
            // left to prove is that a turn goes out on it.
            let agent = client
                .agent(chatgpt::GPT_5_4)
                .preamble(BASIC_PREAMBLE)
                .build();
            let mut stream = agent.prompt(BASIC_PROMPT).stream();
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("non-interactive OAuth streaming completion should succeed");

            assert_nonempty_response(&response);
        },
    )
    .await;
}
