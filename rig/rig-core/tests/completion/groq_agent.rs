//! Groq agent completion smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::providers::groq;

use crate::support::{PREAMBLE, PROMPT, assert_nontrivial_response};

#[tokio::test]
#[ignore = "requires GROQ_API_KEY"]
async fn completion_smoke() {
    let client = groq::Client::from_env();
    let agent = client
        .agent(groq::DEEPSEEK_R1_DISTILL_LLAMA_70B)
        .preamble(PREAMBLE)
        .build();

    let response = agent
        .prompt(PROMPT)
        .await
        .expect("completion should succeed");

    assert_nontrivial_response(&response);
}
