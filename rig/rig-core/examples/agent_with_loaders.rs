//! Demonstrates loading real example files into agent context.
//! Requires `OPENAI_API_KEY`.
//! Run it to see the model identify the example that uses `FileLoader::with_glob`.

use anyhow::Result;
use rig::agent::AgentBuilder;
use rig::client::{CompletionClient, ProviderClient};
use rig::completion::Prompt;
use rig::loaders::FileLoader;
use rig::providers::openai;

const LOADERS_GLOB: &str = "rig-core/examples/*.rs";
const LOADERS_PROMPT: &str = "Which example builds an agent from files loaded via FileLoader::with_glob(\"rig-core/examples/*.rs\")? Answer with just the file name.";

fn load_example_contexts() -> Result<impl Iterator<Item = (std::path::PathBuf, String)>> {
    Ok(FileLoader::with_glob(LOADERS_GLOB)?
        .read_with_path()
        .ignore_errors()
        .into_iter())
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = openai::Client::from_env()?;
    let model = client.completion_model(openai::GPT_4O);
    let files = load_example_contexts()?;

    let agent = files
        .fold(AgentBuilder::new(model), |builder, (path, content)| {
            let context = format!("Rust example {path:?}:\n{content}");
            builder.context(&context)
        })
        .build();

    let response = agent.prompt(LOADERS_PROMPT).await?;
    println!("{response}");

    Ok(())
}
