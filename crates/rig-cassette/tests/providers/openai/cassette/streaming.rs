//! OpenAI streaming smoke coverage. The recording is also the source the
//! stream-fault cells cut their frames from (`stream_faults.rs`).

use rig::providers::openai;

use super::super::support::with_openai_cassette;
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response,
    collect_stream_final_response_and_provider_final,
};

#[tokio::test]
async fn streaming_smoke() {
    with_openai_cassette("streaming/streaming_smoke", |client| async move {
        let agent = rig::AgentBuilder::new(client.openai.completion(openai::GPT_4O))
            .preamble(STREAMING_PREAMBLE)
            .build();

        let mut stream = agent.prompt(STREAMING_PROMPT).stream();
        let (response, provider_final) =
            collect_stream_final_response_and_provider_final(&mut stream)
                .await
                .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
        assert!(provider_final.usage.total_tokens.is_some_and(|n| n > 0));
    })
    .await;
}
