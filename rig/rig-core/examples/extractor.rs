use rig::prelude::*;
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A record representing a person
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Person {
    /// The person's first name, if provided (null otherwise)
    #[schemars(required)]
    pub first_name: Option<String>,
    /// The person's last name, if provided (null otherwise)
    #[schemars(required)]
    pub last_name: Option<String>,
    /// The person's job, if provided (null otherwise)
    #[schemars(required)]
    pub job: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_client = openai::Client::from_env();

    // Create extractor
    let data_extractor = openai_client.extractor::<Person>(openai::GPT_4).build();

    // Example 1: Extract without usage tracking (original API)
    println!("=== Example 1: Extract without usage ===");
    let person = data_extractor
        .extract("Hello my name is John Doe! I am a software engineer.")
        .await?;
    println!(
        "Extracted data: {}",
        serde_json::to_string_pretty(&person).unwrap()
    );

    // Example 2: Extract with usage tracking (new API)
    println!("\n=== Example 2: Extract with usage tracking ===");
    let response = data_extractor
        .extract_with_usage("Jane Smith is a data scientist.")
        .await?;
    println!(
        "Extracted data: {}",
        serde_json::to_string_pretty(&response.data).unwrap()
    );
    println!("Token usage:");
    println!("  Input tokens: {}", response.usage.input_tokens);
    println!("  Output tokens: {}", response.usage.output_tokens);
    println!("  Total tokens: {}", response.usage.total_tokens);

    Ok(())
}
