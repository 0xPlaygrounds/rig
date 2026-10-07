//! Migrated from `examples/groq_streaming_reasoning.rs`.

use rig::providers::groq::extension::{GroqExt, GroqOptions, ReasoningFormat};
use rig::providers::openai::wire::GROQ;
use rig_test_support::cassette_models::OpenAiModels;

use crate::support::{assert_nonempty_response, collect_stream_final_response};

use super::STREAMING_REASONING_MODEL;

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn parsed_reasoning_stream() {
    let groq = OpenAiModels::from_env_for(&GROQ).expect("GROQ_API_KEY should be set");
    let agent = rig::AgentBuilder::new(groq.completion(STREAMING_REASONING_MODEL))
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .provider_options(
            rig::completion::ProviderOptions::new()
                .with::<GroqExt>(&GroqOptions::new().reasoning_format(ReasoningFormat::Parsed))
                .expect("Groq options serialize"),
        )
        .build();

    let mut stream = agent.prompt("Entertain me!").stream();
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}
