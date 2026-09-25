//! ChatGPT streaming smoke tests.

use crate::chatgpt::{LIVE_MODEL, live_client};
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};
use rig::wire::Wire as _;

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn streaming_smoke() {
    let agent = rig::AgentBuilder::new(
        live_client()
            .await
            .completion(LIVE_MODEL)
            .on(rig::transport()),
    )
    .preamble(STREAMING_PREAMBLE)
    .build();

    let mut stream = agent.prompt(STREAMING_PROMPT).stream();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("ChatGPT stream should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn example_streaming_prompt() {
    let agent = rig::AgentBuilder::new(
        live_client()
            .await
            .completion(LIVE_MODEL)
            .on(rig::transport()),
    )
    .preamble("Be precise and concise.")
    .temperature(0.5)
    .build();

    let mut stream = agent
        .prompt("When and where and what type is the next solar eclipse?")
        .stream();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
