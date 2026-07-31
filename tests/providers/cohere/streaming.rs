//! Cohere streaming smoke test.

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};
use rig::prelude::*;
use rig::providers::cohere;

#[tokio::test]
#[ignore = "requires COHERE_API_KEY"]
async fn streaming_smoke() {
    let cfg = cohere::functions::Config::from_env(cohere::COMMAND).expect("config should build");
    let agent = AgentBuilder::new(ProviderConfig::Cohere(cfg))
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.runner(STREAMING_PROMPT).stream_run();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
