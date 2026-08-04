//! ChatGPT agent completion smoke test.

use crate::chatgpt::{LIVE_MODEL, live_agent};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn completion_smoke() {
    let agent = live_agent(LIVE_MODEL)
        .await
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response);
}
