//! Cohere streaming tools smoke test.

use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, collect_stream_final_response,
};
use rig::prelude::*;
use rig::providers::cohere;

#[tokio::test]
#[ignore = "requires COHERE_API_KEY"]
async fn streaming_tools_smoke() {
    let cfg = cohere::functions::Config::from_env(cohere::COMMAND_R).expect("config should build");
    let agent = AgentBuilder::new(ProviderConfig::Cohere(cfg))
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = Box::pin(agent.runner(STREAMING_TOOLS_PROMPT).stream_run());
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}
