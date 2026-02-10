use hyper_util::client::legacy::Client;
use rig::client::Nothing;
use rig::prelude::*;
use rig::providers::ollama::OllamaExt;
use rig::{completion::Prompt, providers::ollama};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// A character from a fictional story
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Character {
    /// The character's full name
    name: String,
    /// The character's age in years
    age: u32,
    /// A brief biography of the character
    bio: String,
    /// The character's personality traits
    traits: Vec<String>,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = ollama::Client::from_env();

    let agent = client
        .agent("qwen3:4b")
        .preamble("You are a creative fiction writer. Create detailed characters.")
        .output_schema::<Character>()
        .build();

    let response = agent
        .prompt("Create a protagonist for a sci-fi novel set on Mars.")
        .await?;

    let character: Character = serde_json::from_str(&response)?;

    println!("{}", serde_json::to_string_pretty(&character)?);

    Ok(())
}
