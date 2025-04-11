use std::env;

use rig::{completion::Prompt, providers::inception::{ClientBuilder, MERCURY_CODER_SMALL}};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create Inception Labs client
    let client =
        ClientBuilder::new(&env::var("INCEPTION_API_KEY").expect("INCEPTION_API_KEY not set"))
            .build();

    // Create agent with a single context prompt
    let agent = client
        .agent(MERCURY_CODER_SMALL)
        .preamble("You are a helpful AI assistant.")
        .temperature(0.0)
        .build();

    // Prompt the agent and print the response
    let response = agent.prompt("Hello, how are you?").await?;
    println!("{}", response);

    Ok(())
}
