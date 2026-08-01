//! Groq streaming smoke test.

use rig::prelude::*;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

use super::STREAMING_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn streaming_smoke() {
    let agent = rig::providers::groq::Client::from_env()
        .expect("client should build")
        .agent(STREAMING_MODEL)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.runner(STREAMING_PROMPT).stream_run();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
