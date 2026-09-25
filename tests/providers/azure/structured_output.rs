use rig::providers::openai::{
    GPT_5_MINI,
    wire::{AZURE, OpenAI},
};
use rig_test_support::endpoint::Endpoint;
use schemars::JsonSchema;
use serde::Deserialize;

#[tokio::test]
#[ignore = "requires AZURE_API_KEY or AZURE_TOKEN, plus AZURE_API_VERSION and AZURE_ENDPOINT"]
async fn test_azure_structured_output() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    #[derive(Debug, Deserialize, JsonSchema)]
    struct Person {
        name: String,
        age: u32,
    }

    let azure = Endpoint::new(OpenAI::from_env_with(&AZURE)?, rig::rig_reqwest::shared());
    let agent = azure
        .agent(GPT_5_MINI)
        .preamble("You are a helpful assistant that extracts personal details.")
        .max_tokens(100)
        .output_schema::<Person>()
        .build();

    let result: Person = agent
        .prompt_typed("Hello! My name is John Doe and I'm 54 years old.")
        .await?
        .output;

    anyhow::ensure!(
        result.name == "John Doe",
        "expected name John Doe, got {}",
        result.name
    );
    anyhow::ensure!(result.age == 54, "expected age 54, got {}", result.age);

    tracing::info!("Extracted person: {:?}", result);
    Ok(())
}
