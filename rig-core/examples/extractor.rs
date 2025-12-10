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
    let person = data_extractor
        .extract("Hello my name is John Doe! I am a software engineer.")
        .await?;

    println!("GPT-4: {}", serde_json::to_string_pretty(&person).unwrap());

    Ok(())
}
