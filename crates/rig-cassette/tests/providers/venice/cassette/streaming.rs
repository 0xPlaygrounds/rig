//! Cassette-backed Venice streaming coverage.

use super::super::{DEFAULT_MODEL, support::with_venice_cassette};
use crate::support::{
    STREAMING_PREAMBLE, STREAMING_PROMPT, assert_nonempty_response, collect_stream_final_response,
};
use rig::wire::Wire as _;

#[tokio::test]
async fn streaming_smoke() {
    with_venice_cassette("streaming/streaming_smoke", |client| async move {
        let agent = rig::AgentBuilder::new(client.completion(DEFAULT_MODEL).on(rig::transport()))
            .preamble(STREAMING_PREAMBLE)
            .build();
        let mut stream = agent.prompt(STREAMING_PROMPT).stream();
        let response = collect_stream_final_response(&mut stream)
            .await
            .expect("streaming prompt should succeed");
        assert_nonempty_response(&response);
    })
    .await;
}
