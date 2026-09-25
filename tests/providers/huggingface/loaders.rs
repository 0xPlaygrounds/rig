//! Hugging Face loaders smoke test.

use rig::loaders::FileLoader;
use rig::providers::openai::wire::{HUGGINGFACE, OpenAI};
use rig_test_support::endpoint::Endpoint;

use crate::support::{LOADERS_GLOB, LOADERS_PROMPT, assert_loader_answer_is_relevant};

#[tokio::test]
#[ignore = "requires HUGGINGFACE_API_KEY"]
async fn loaders_smoke() {
    let provider = Endpoint::new(
        OpenAI::from_env_with(&HUGGINGFACE).expect("config should build from env"),
        rig::rig_reqwest::shared(),
    );
    let examples = FileLoader::with_glob(LOADERS_GLOB)
        .expect("examples glob should parse")
        .read_with_path()
        .ignore_errors()
        .into_iter();

    let agent = examples
        .fold(
            provider.agent("deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"),
            |builder, (path, content)| {
                builder.context(format!("Rust Example {path:?}:\n{content}").as_str())
            },
        )
        .build();

    let response = agent
        .prompt(LOADERS_PROMPT)
        .await
        .expect("loader prompt should succeed");

    assert_loader_answer_is_relevant(&response.output);
}
