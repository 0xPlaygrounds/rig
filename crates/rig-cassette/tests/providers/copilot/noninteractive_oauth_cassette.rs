//! Copilot non-interactive OAuth cassette coverage.

use rig::prelude::*;

use crate::copilot::{LIVE_MODEL, with_copilot_noninteractive_oauth_cassette};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
async fn cached_oauth_allows_noninteractive_completion() {
    with_copilot_noninteractive_oauth_cassette(
        "noninteractive_oauth/cached_oauth_allows_noninteractive_completion",
        |client| async move {
            // Reaching the closure at all is the assertion the deleted
            // `Client::authorize()` made: the harness resolved the cached
            // API key through `copilot::auth` with `allow_device_flow =
            // false`, so a credential that needed a device flow would have
            // failed there rather than prompting.

            let response = client
                .agent(LIVE_MODEL)
                .preamble(BASIC_PREAMBLE)
                .build()
                .prompt(BASIC_PROMPT)
                .await
                .expect("non-interactive OAuth completion should succeed");

            assert_nonempty_response(&response.output);
        },
    )
    .await;
}
