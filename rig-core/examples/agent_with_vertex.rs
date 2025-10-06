use rig::prelude::*;

use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig,
};
use rig::{completion::Prompt, providers};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let client = providers::vertex::Client::from_env();

    let gen_cfg = GenerationConfig {
        top_k: Some(1),
        top_p: Some(0.95),
        candidate_count: Some(1),
        ..Default::default()
    };
    let cfg = AdditionalParameters::default().with_config(gen_cfg);

    // Create agent with a single context prompt
    let comedian_agent = client
        .agent("gemini-2.5-flash")
        .preamble("You are a comedian here to entertain the user using humour and jokes.")
        .additional_params(serde_json::to_value(cfg).unwrap())
        .build();

    // Prompt the agent and print the response
    let response = comedian_agent.prompt("Entertain me!").await?;

    println!("{response}");

    Ok(())
}
