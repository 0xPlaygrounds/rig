//! Together streaming smoke test.

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};
use rig::prelude::*;
use rig::providers::together;

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn streaming_smoke() {
    let cfg = together::functions::Config::from_env(together::LLAMA_3_8B_CHAT_HF)
        .expect("config should build");
    let agent = AgentBuilder::new(ProviderConfig::Together(cfg))
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.runner(STREAMING_PROMPT).stream_run();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
