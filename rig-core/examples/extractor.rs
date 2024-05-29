use std::env;

// use llm::client::{Client, OpenAIClient};
use llm::providers::{cohere::Client as CohereClient, openai::Client as OpenAIClient};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
/// A record representing a person
struct Person {
    /// The person's first name, if provided (null otherwise)
    pub first_name: Option<String>,
    /// The person's last name, if provided (null otherwise)
    pub last_name: Option<String>,
    /// The person's job, if provided (null otherwise)
    pub job: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = OpenAIClient::new(&openai_api_key);

    // Create extractor
    let data_extractor = openai_client.extractor::<Person>("gpt-4").build();

    let person = data_extractor
        .extract("Hello my name is John Doe! I am a software engineer.")
        .await?;

    println!("GPT-4: {}", serde_json::to_string_pretty(&person).unwrap());

    // Create Cohere client
    let cohere_api_key = env::var("COHERE_API_KEY").expect("COHERE_API_KEY not set");
    let cohere_client = CohereClient::new(&cohere_api_key);

    // Create extractor
    let data_extractor = cohere_client.extractor::<Person>("command-r").build();

    let person = data_extractor
        .extract("Hello my name is John Doe! I am a software engineer.")
        .await?;

    println!("Coral: {}", serde_json::to_string_pretty(&person).unwrap());

    Ok(())
}
