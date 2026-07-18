//! Copilot non-interactive OAuth cassette coverage.

use rig::client::AgentClientExt;
use rig::completion::Prompt;

use crate::copilot::{LIVE_MODEL, with_copilot_noninteractive_oauth_cassette};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
async fn cached_oauth_allows_noninteractive_completion() {
    with_copilot_noninteractive_oauth_cassette(
        "noninteractive_oauth/cached_oauth_allows_noninteractive_completion",
        |client| async move {
            client
                .authorize()
                .await
                .expect("cached OAuth auth should not require device flow");

            let response = client
                .agent(LIVE_MODEL)
                .preamble(BASIC_PREAMBLE)
                .build()
                .prompt(BASIC_PROMPT)
                .await
                .expect("non-interactive OAuth completion should succeed");

            assert_nonempty_response(&response);
        },
    )
    .await;
}
