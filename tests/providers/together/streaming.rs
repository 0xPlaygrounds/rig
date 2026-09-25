//! Together streaming smoke test.

use rig::providers::openai::wire::{OpenAI, TOGETHER};
use rig::providers::together;
use rig_test_support::endpoint::Endpoint;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn streaming_smoke() {
    let provider = Endpoint::new(
        OpenAI::from_env_with(&TOGETHER).expect("config should build from env"),
        rig::rig_reqwest::shared(),
    );
    let agent = provider
        .agent(together::LLAMA_3_8B_CHAT_HF)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.prompt(STREAMING_PROMPT).stream();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
