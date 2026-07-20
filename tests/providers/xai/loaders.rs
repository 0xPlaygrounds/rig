//! xAI loaders smoke test.

use rig::client::CompletionClient;
use rig::completion::Prompt;
use rig::loaders::FileLoader;
use rig::providers::xai;

use super::support::with_xai_cassette;
use crate::support::{LOADERS_GLOB, LOADERS_PROMPT, assert_loader_answer_is_relevant};

#[tokio::test]
async fn loaders_smoke() {
    with_xai_cassette("loaders/loaders_smoke", |client| async move {
        let examples = FileLoader::with_glob(LOADERS_GLOB)
            .expect("examples glob should parse")
            .read_with_path()
            .ignore_errors()
            .into_iter();

        let agent = examples
            .fold(client.agent(xai::GROK_4), |builder, (path, content)| {
                let file_name = path
                    .file_name()
                    .and_then(|name| name.to_str())
                    .expect("loader fixture path should have a UTF-8 file name");

                builder.context(format!("Rust Example {file_name}:\n{content}").as_str())
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
    })
    .await;
}
