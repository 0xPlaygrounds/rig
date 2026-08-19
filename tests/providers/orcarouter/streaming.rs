//! OrcaRouter streaming smoke tests.

use rig::prelude::*;
use rig::providers::orcarouter;
use rig::streaming::{StreamingChat, StreamingPrompt};

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT,
    collect_stream_final_response, collect_stream_observation,
};

#[tokio::test]
#[ignore = "requires ORCAROUTER_API_KEY"]
async fn streaming_smoke() {
    let client = orcarouter::Client::from_env().expect("client should build");
    let agent = client
        .agent(orcarouter::ORCAROUTER_AUTO)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert!(!response.trim().is_empty(), "streamed response was empty");
}

#[tokio::test]
#[ignore = "requires ORCAROUTER_API_KEY"]
async fn streaming_tools_smoke() {
    let client = orcarouter::Client::from_env().expect("client should build");
    let agent = client
        .agent(orcarouter::ORCAROUTER_AUTO)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .default_max_turns(2)
        .tool(crate::support::Adder)
        .tool(crate::support::Subtract)
        .build();

    let mut stream = agent
        .stream_chat(
            STREAMING_TOOLS_PROMPT,
            Vec::<rig::completion::Message>::new(),
        )
        .await;

    let observation = collect_stream_observation(&mut stream).await;
    assert!(
        observation.errors.is_empty(),
        "stream should not emit errors: {:?}",
        observation.errors
    );
    assert!(
        observation.got_final_response,
        "stream should emit a final response"
    );
}
