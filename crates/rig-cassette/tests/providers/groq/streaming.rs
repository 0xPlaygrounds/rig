//! Groq streaming smoke test.

use rig::providers::openai::wire::{GROQ, OpenAI};
use rig_test_support::endpoint::Endpoint;

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

use super::STREAMING_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn streaming_smoke() {
    let groq = Endpoint::new(
        OpenAI::from_env_with(&GROQ).expect("GROQ_API_KEY should be set"),
        rig::rig_reqwest::shared(),
    );
    let agent = groq
        .agent(STREAMING_MODEL)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.prompt(STREAMING_PROMPT).stream();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
