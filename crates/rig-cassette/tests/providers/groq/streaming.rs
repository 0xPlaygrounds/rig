//! Groq streaming smoke test.

use rig::providers::openai::wire::{GROQ, OpenAI};

use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};

use super::STREAMING_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn streaming_smoke() {
    let groq = OpenAI::from_env_with(&GROQ).expect("GROQ_API_KEY should be set");
    let agent = rig::AgentBuilder::new(rig::model(groq.completion(STREAMING_MODEL)))
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.prompt(STREAMING_PROMPT).stream();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
