//! Migrated from `examples/groq_streaming_reasoning.rs`.

use rig::prelude::*;
use rig::providers::openai::wire::{GROQ, OpenAI};

use crate::support::{assert_nonempty_response, collect_stream_final_response};

use super::STREAMING_REASONING_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn parsed_reasoning_stream() {
    let groq = OpenAI::from_env_with(&GROQ)
        .expect("GROQ_API_KEY should be set")
        .bound()
        .expect("transport should build");
    let agent = groq
        .agent(STREAMING_REASONING_MODEL)
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .additional_params(serde_json::json!({ "reasoning_format": "parsed" }))
        .build();

    let mut stream = agent.prompt("Entertain me!").stream();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
