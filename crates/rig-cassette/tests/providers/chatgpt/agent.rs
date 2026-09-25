//! ChatGPT agent completion smoke test.

use crate::chatgpt::{LIVE_MODEL, live_client};
use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};
use rig::wire::Wire as _;

#[tokio::test]
#[ignore = "requires ChatGPT credentials or existing OAuth cache"]
async fn completion_smoke() {
    let agent = rig::AgentBuilder::new(
        live_client()
            .await
            .completion(LIVE_MODEL)
            .on(rig::transport()),
    )
    .preamble(BASIC_PREAMBLE)
    .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed");

    assert_nonempty_response(&response.output);
}
