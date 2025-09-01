use rig::prelude::*;
use rig::providers::deepseek;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A record representing a person
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
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
    tracing_subscriber::fmt().init();

    let deepseek_api_key = std::env::var("DEEPSEEK_API_KEY")
        .expect("DEEPSEEK_API_KEY should exist as an environment variable");
    // Create DeepSeek client
    let deepseek_client = deepseek::Client::builder(&deepseek_api_key)
        .base_url("https://api.deepseek.com/beta")
        .build()?;

    // Create extractor
    let data_extractor = deepseek_client
        .extractor::<Person>(deepseek::DEEPSEEK_CHAT)
        .retries(2)
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
