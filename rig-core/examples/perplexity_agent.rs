use std::env;

use rig::{
    completion::Prompt,
    providers::{self, perplexity::SONAR},
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client
    let client = providers::perplexity::Client::new(
        &env::var("PERPLEXITY_API_KEY").expect("PERPLEXITY_API_KEY not set"),
    );

    // Create agent with a single context prompt
    let agent = client
        .agent(SONAR)
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .additional_params(json!({
            "return_related_questions": true,
            "return_images": true
        }))
        .build();

    // Prompt the agent and print the response
    let response = agent
        .prompt("When and where and what type is the next solar eclipse?")
        .await?;
    println!("{}", response);

    Ok(())
}
