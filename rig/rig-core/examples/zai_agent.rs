use rig::prelude::*;
use rig::{
    completion::Prompt,
    providers::zai::{self},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create Zai client
    let client = zai::Client::from_env();

    // Create agent with a single context prompt
    let agent = client
        .agent(zai::completion::GLM_4_7)
        .preamble("Be precise and concise.")
        .temperature(0.5)
        .build();

    // Prompt the agent and print the response
    let response = agent
        .prompt("When and where and what type is the next solar eclipse?")
        .await?;

    println!("{response}");

    Ok(())
}
