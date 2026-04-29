//! ChatGPT agent completion smoke test.

use rig_core::client::CompletionClient;
use rig_core::completion::Prompt;

use crate::chatgpt::{LIVE_MODEL, live_client};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn completion_smoke() {
    let agent = live_client()
        .agent(LIVE_MODEL)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
