//! Together streaming smoke test.

use rig::prelude::*;
use rig::providers::openai::wire::{OpenAI, TOGETHER};
use rig::providers::together;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

#[tokio::test]
#[ignore = "requires TOGETHER_API_KEY"]
async fn streaming_smoke() {
    let provider = OpenAI::from_env_with(&TOGETHER)
        .expect("config should build from env")
        .bound()
        .expect("transport should build");
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
