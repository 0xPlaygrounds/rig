use anyhow::Result;
use rig::client::ProviderClient;
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Person {
    #[schemars(required)]
    first_name: Option<String>,
    #[schemars(required)]
    last_name: Option<String>,
    #[schemars(required)]
    job: Option<String>,
}

#[tokio::main]
async fn main() -> Result<()> {
    let client = openai::Client::from_env();
    let extractor = client.extractor::<Person>(openai::GPT_4).build();

    let person = extractor
        .extract("Hello my name is John Doe! I am a software engineer.")
        .await?;
    println!("{}", serde_json::to_string_pretty(&person)?);

    let response = extractor
        .extract_with_usage("Jane Smith is a data scientist.")
        .await?;
    println!("{}", serde_json::to_string_pretty(&response.data)?);
    println!("total tokens: {}", response.usage.total_tokens);

    Ok(())
}
