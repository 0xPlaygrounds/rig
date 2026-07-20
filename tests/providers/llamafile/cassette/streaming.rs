//! Llamafile streaming completion smoke test.
//!
//! Replays by default; set `RIG_PROVIDER_TEST_MODE=record` to record against a
//! local OpenAI-compatible llama.cpp-family server (see `cassette_support`).

use rig::client::CompletionClient;
use rig::streaming::StreamingPrompt;

use super::super::cassette_support::{CASSETTE_CHAT_MODEL, with_llamafile_cassette};
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
async fn streaming_smoke() {
    with_llamafile_cassette("streaming/streaming_smoke", |client| async move {
        let agent = client
            .agent(CASSETTE_CHAT_MODEL)
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
