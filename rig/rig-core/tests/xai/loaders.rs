//! xAI loaders smoke test.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::loaders::FileLoader;
use rig::providers::xai;

use crate::support::{LOADERS_GLOB, LOADERS_PROMPT, assert_loader_answer_is_relevant};

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn loaders_smoke() {
    let client = xai::Client::from_env();
    let examples = FileLoader::with_glob(LOADERS_GLOB)
        .expect("examples glob should parse")
        .read_with_path()
        .ignore_errors()
        .into_iter();

    let agent = examples
        .fold(
            client.agent(xai::completion::GROK_3_MINI),
            |builder, (path, content)| {
                builder.context(format!("Rust Example {path:?}:\n{content}").as_str())
            },
        )
        .build();

    let response = agent
        .prompt(LOADERS_PROMPT)
        .await
        .expect("loader prompt should succeed");

    assert_loader_answer_is_relevant(&response);
}
