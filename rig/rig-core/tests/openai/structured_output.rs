//! OpenAI structured output smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::TypedPrompt;
use rig::providers::openai;

use crate::support::{STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_nonempty_response};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn structured_output_smoke() {
    let client = openai::Client::from_env();
    let agent = client.agent(openai::GPT_4O).build();

    let response: SmokeStructuredOutput = agent
        .prompt_typed(STRUCTURED_OUTPUT_PROMPT)
        .await
        .expect("structured output prompt should succeed");

    assert_nonempty_response(&response.title);
    assert_nonempty_response(&response.category);
    assert_nonempty_response(&response.summary);
}
