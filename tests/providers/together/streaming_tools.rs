//! Together streaming tools smoke test.

use rig::providers::openai::wire::{OpenAI, TOGETHER};
use rig::providers::together;
use rig_test_support::endpoint::Endpoint;

use crate::support::{
    Adder, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, Subtract,
    assert_mentions_expected_number, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn streaming_tools_smoke() {
    let provider = Endpoint::new(
        OpenAI::from_env_with(&TOGETHER).expect("config should build from env"),
        rig::rig_reqwest::shared(),
    );
    let agent = provider
        .agent(together::LLAMA_2_70B_CHAT_TOGETHER)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = agent.prompt(STREAMING_TOOLS_PROMPT).stream();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}
