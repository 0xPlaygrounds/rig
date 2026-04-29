//! xAI loaders smoke test.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::loaders::FileLoader;
use rig_core::providers::xai;

use crate::support::{LOADERS_GLOB, LOADERS_PROMPT, assert_loader_answer_is_relevant};

#[tokio::test]
#[ignore = "requires XAI_API_KEY"]
async fn loaders_smoke() {
    let client = xai::Client::from_env().expect("client should build");
    let examples = FileLoader::with_glob(LOADERS_GLOB)
        .expect("examples glob should parse")
        .read_with_path()
        .ignore_errors()
        .into_iter();

    let agent = examples
        .fold(client.agent(xai::GROK_4), |builder, (path, content)| {
            builder.context(format!("Rust Example {path:?}:\n{content}").as_str())
        })
        .preamble(
            "Use only the provided Rust Example contexts. \
             Exactly one of these files is the correct answer: agent_with_loaders.rs, streaming.rs, tools.rs. \
             Reply with exactly one of those file names and nothing else.",
        )
        .build();

    let response = agent
        .prompt(
            format!(
                "{LOADERS_PROMPT} Choose only from these exact file names: agent_with_loaders.rs, streaming.rs, tools.rs. Reply with just the exact file name."
            ),
        )
        .await
        .expect("loader prompt should succeed");

    assert_loader_answer_is_relevant(&response);
}
