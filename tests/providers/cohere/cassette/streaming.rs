//! Cassette-backed Cohere streaming completion coverage.

use rig::completion::GetTokenUsage;
use rig::prelude::*;
use rig::streaming::StreamingPrompt;

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response,
    collect_stream_final_response_and_provider_final,
};

#[tokio::test]
async fn streaming_smoke() {
    with_cohere_cassette("streaming/streaming_smoke", |client| async move {
        // Capped so the recorded SSE body stays reviewable; uncapped, the model can
        // run to its 8k output limit and the fixture balloons past 800 KB.
        let agent = client
            .agent(CASSETTE_MODEL)
            .preamble(STREAMING_PREAMBLE)
            .max_tokens(64)
            .build();

        let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
        let (response, provider_final) =
            collect_stream_final_response_and_provider_final(&mut stream)
                .await
                .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);

        let tokens = provider_final
            .usage
            .as_ref()
            .and_then(|usage| usage.tokens.as_ref())
            .expect("`message-end` should carry `usage.tokens`");
        let usage = provider_final.token_usage();

        assert_eq!(
            usage.input_tokens,
            tokens.input_tokens.expect("input token count") as u64
        );
        assert_eq!(
            usage.output_tokens,
            tokens.output_tokens.expect("output token count") as u64
        );
        assert_eq!(usage.total_tokens, usage.input_tokens + usage.output_tokens);
        assert!(usage.total_tokens > 0, "streamed usage should be non-zero");

        let cached = provider_final
            .usage
            .as_ref()
            .and_then(|usage| usage.cached_tokens)
            .expect("`message-end` should carry `cached_tokens`");
        assert_eq!(usage.cached_input_tokens, cached as u64);
    })
    .await;
}
