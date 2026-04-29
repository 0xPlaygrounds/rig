//! Migrated from `examples/ollama_structured_output.rs`.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::Prompt;
use rig_core::providers::ollama;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::support::assert_nonempty_response;

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Character {
    name: String,
    age: u32,
    bio: String,
    traits: Vec<String>,
}

#[tokio::test]
#[ignore = "requires a local Ollama server"]
async fn structured_output_prompt() {
    let client = ollama::Client::from_env().expect("client should build");
    let agent = client
        .agent("qwen3:4b")
        .preamble("You are a creative fiction writer. Create detailed characters.")
        .output_schema::<Character>()
        .build();

    let response = agent
        .prompt("Create a protagonist for a sci-fi novel set on Mars.")
        .await
        .expect("prompt should succeed");
    let character: Character =
        serde_json::from_str(&response).expect("schema response should deserialize");

    assert_nonempty_response(&character.name);
    assert_nonempty_response(&character.bio);
    assert!(character.age > 0, "character age should be positive");
    assert!(!character.traits.is_empty(), "traits should not be empty");
    assert!(
        character
            .traits
            .iter()
            .all(|trait_name| !trait_name.trim().is_empty()),
        "traits should not contain empty entries"
    );
}
