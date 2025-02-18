use rig::completion::Prompt;
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the Mira client with your API key
    let client = rig::providers::mira::Client::new(
        "mira-api-key",
    )?;

    // Create an agent with the Mira model
    let agent = client
        .agent("claude-3.5-sonnet")
        .preamble("You are a helpful AI assistant.")
        .temperature(0.7)
        .build();

    // Send a message to the agent
    let response = agent.prompt("What are the 7 wonders of the world?").await?;

    // Print the response
    println!("Assistant: {}", response);

    // List available models
    println!("\nAvailable models:");
    let models = client.list_models().await?;
    for model in models {
        println!("- {}", model);
    }

    // Get user credits
    let credits = client.get_user_credits().await?;
    println!("\nUser credits: {:?}", credits);

    Ok(())
}
