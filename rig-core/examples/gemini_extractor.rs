use rig::providers::gemini;
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
    tracing_subscriber::fmt()
        .with_max_level(tracing::Level::DEBUG)
        .with_target(false)
        .init();

    // Create Gemini client
    let client = gemini::Client::from_env();

    // Create extractor
    let data_extractor = client
        .extractor::<Person>(gemini::completion::GEMINI_2_0_FLASH)
        .build();

    let person = data_extractor
        .extract("Hello my name is John Doe! I am a software engineer.")
        .await?;

    println!("GEMINI: {}", serde_json::to_string_pretty(&person).unwrap());

    Ok(())
}
