//! Cohere streaming smoke test.

use rig::prelude::*;
use rig::providers::cohere::{self, wire::Cohere};

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires COHERE_API_KEY"]
async fn streaming_smoke() {
    let cohere = Cohere::from_env()
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
    let agent = cohere
        .agent(cohere::COMMAND_A_03_2025)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.prompt(STREAMING_PROMPT).stream();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
