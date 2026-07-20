use rig::{
    client::{CompletionClient, ProviderClient},
    completion::TypedPrompt,
    providers::{azure::Client, openai::GPT_5_MINI},
};
use schemars::JsonSchema;
use serde::Deserialize;

#[tokio::test]
#[ignore = "requires AZURE_OPENAI_API_KEY and related Azure env vars"]
async fn test_azure_structured_output() -> anyhow::Result<()> {
    let _ = tracing_subscriber::fmt::try_init();

    #[derive(Debug, Deserialize, JsonSchema)]
    struct Person {
        name: String,
        age: u32,
    }

    let client = Client::from_env()?;
    let agent = client
        .agent(GPT_5_MINI)
        .preamble("You are a helpful assistant that extracts personal details.")
        .max_tokens(100)
        .output_schema::<Person>()
        .build();

    let result: Person = agent
        .prompt_typed("Hello! My name is John Doe and I'm 54 years old.")
        .await?;

    anyhow::ensure!(
        result.name == "John Doe",
        "expected name John Doe, got {}",
        result.name
    );
    anyhow::ensure!(result.age == 54, "expected age 54, got {}", result.age);

    tracing::info!("Extracted person: {:?}", result);
    Ok(())
}
