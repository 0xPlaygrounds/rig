//! Demonstrates typed extraction and extraction with usage metadata.
//! Requires `OPENAI_API_KEY`.
//! Run it to compare a plain structured extraction with a usage-aware one.
//!
use anyhow::Result;
use rig::prelude::*;
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

const FIRST_INPUT: &str = "Hello my name is John Doe! I am a software engineer.";
const SECOND_INPUT: &str = "Jane Smith is a data scientist.";

#[tokio::main]
async fn main() -> Result<()> {
    let client = openai::Client::from_env()?;
    let agent = client.agent(openai::GPT_4).build();

    let person: Person = agent.extractor(FIRST_INPUT).run().await?;
    println!("{}", serde_json::to_string_pretty(&person)?);

    let outcome = agent
        .extractor(SECOND_INPUT)
        .run_with_usage::<Person>()
        .await?;
    println!("{}", serde_json::to_string_pretty(&outcome.value)?);
    println!("total tokens: {}", outcome.usage.total_tokens);

    Ok(())
}
