//! OpenAI streaming coverage, including the migrated example path.

use rig::providers::openai;

use super::super::support::with_openai_cassette;
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
    collect_stream_final_response_and_provider_final,
};

#[tokio::test]
async fn streaming_smoke() {
    with_openai_cassette("streaming/streaming_smoke", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble(STREAMING_PREAMBLE)
            .build();

        let mut stream = agent.runner(STREAMING_PROMPT).stream_run();
        let (response, provider_final) =
            collect_stream_final_response_and_provider_final(&mut stream)
                .await
                .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
        assert!(provider_final.usage.total_tokens > 0);
    })
    .await;
}

#[tokio::test]
async fn example_streaming_prompt() {
    with_openai_cassette("streaming/example_streaming_prompt", |client| async move {
        let agent = client
            .agent(openai::GPT_4O)
            .preamble("Be precise and concise.")
            .temperature(0.5)
            .build();

        let mut stream = agent
            .runner("When and where and what type is the next solar eclipse?")
            .stream_run();
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
    })
    .await;
}
