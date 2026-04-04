//! Gemini-specific extractor example showing how to pass `additional_params`.
//! The generic extractor flow lives in `extractor.rs`.

use rig::client::{CompletionClient, ProviderClient};
use rig::providers::gemini;
use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig,
};
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
    let gen_cfg = GenerationConfig::default();
    let cfg = AdditionalParameters::default().with_config(gen_cfg);

    // Create Gemini client
    let client = gemini::Client::from_env();

    // Create extractor
    let data_extractor = client
        .extractor::<Person>(gemini::completion::GEMINI_2_0_FLASH)
        .additional_params(serde_json::to_value(cfg).unwrap())
        .build();

    let person = data_extractor
        .extract("Hello my name is John Doe! I am a software engineer.")
        .await?;

    println!("GEMINI: {}", serde_json::to_string_pretty(&person)?);

    Ok(())
}
