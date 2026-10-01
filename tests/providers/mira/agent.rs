//! Mira agent completion smoke test.

use rig::providers::mira;
use rig::providers::openai;

use crate::support::{BASIC_PREAMBLE, BASIC_PROMPT, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires MIRA_API_KEY"]
async fn completion_smoke() {
    let provider = mira::from_env().expect("config should build from env");
    let agent = rig::AgentBuilder::new(provider.completion(openai::GPT_4O))
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("completion should succeed")
        .output();

    assert_nonempty_response(&response);
}
