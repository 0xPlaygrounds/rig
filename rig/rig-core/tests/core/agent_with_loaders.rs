//! Migrated from `examples/agent_with_loaders.rs`.

use rig::agent::AgentBuilder;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::loaders::FileLoader;
use rig::providers::openai;

use crate::support::{LOADERS_GLOB, LOADERS_PROMPT, assert_loader_answer_is_relevant};

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn file_loader_context_prompt() {
    let client = openai::Client::from_env();
    let model = client.completion_model(openai::GPT_4O);
    let files = FileLoader::with_glob(LOADERS_GLOB)
        .expect("glob should parse")
        .read_with_path()
        .ignore_errors()
        .into_iter();

    let agent = files
        .fold(AgentBuilder::new(model), |builder, (path, content)| {
            builder.context(format!("Rust test module {path:?}:\n{content}").as_str())
        })
        .build();

    let response = agent
        .prompt(LOADERS_PROMPT)
        .await
        .expect("loader prompt should succeed");

    assert_loader_answer_is_relevant(&response);
}
