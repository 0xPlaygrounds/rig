//! Cassette-backed Cohere streaming completion coverage.

use super::super::{CASSETTE_MODEL, support::with_cohere_cassette};
use crate::support::{
    Observed, STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response,
    collect_stream_final_response_and_provider_final,
};

#[tokio::test]
async fn streaming_smoke() {
    let usage = Observed::default();
    let parked = usage.clone();
    with_cohere_cassette("streaming/streaming_smoke", |client| async move {
        // Capped so the recorded SSE body stays reviewable; uncapped, the model can
        // run to its 8k output limit and the fixture balloons past 800 KB.
        let agent = rig::AgentBuilder::new(client.completion(CASSETTE_MODEL))
            .preamble(STREAMING_PREAMBLE)
            .max_tokens(64)
            .build();

        let mut stream = agent.prompt(STREAMING_PROMPT).stream();
        let (response, provider_final) =
            collect_stream_final_response_and_provider_final(&mut stream)
                .await
                .expect("streaming prompt should succeed");

        assert_nonempty_response(&response);
        parked.put(provider_final.usage);
    })
    .await;

    // The usage is the stream's usage chunk, counted as Cohere sent it.
    let usage = usage.take();
    let frame =
        crate::raw_capture::chat::recorded_sole_usage_frame("cohere", "streaming/streaming_smoke");
    let count = |pointer: &str| frame.pointer(pointer).and_then(serde_json::Value::as_u64);
    assert_eq!(usage.input_tokens, count("/usage/prompt_tokens"));
    assert_eq!(usage.output_tokens, count("/usage/completion_tokens"));
    assert_eq!(usage.total_tokens, count("/usage/total_tokens"));
    assert_eq!(
        usage.cached_input_tokens,
        count("/usage/prompt_tokens_details/cached_tokens")
    );
}
