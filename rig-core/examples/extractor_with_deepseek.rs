use rig::providers::deepseek;
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
    // Create DeepSeek client
    let deepseek_client = deepseek::Client::from_env();

    // Create extractor
    let data_extractor = deepseek_client
        .extractor::<Person>(deepseek::DEEPSEEK_CHAT)
        .build();

    let person = data_extractor
        .extract("Hello my name is John Doe! I am a software engineer.")
        .await?;

    println!(
        "DeepSeek: {}",
        serde_json::to_string_pretty(&person).unwrap()
    );

    Ok(())
}
