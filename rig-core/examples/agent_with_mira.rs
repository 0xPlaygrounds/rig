use rig::providers::mira::{self, AiRequest, ChatMessage};
use std::error::Error;

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
    // Initialize the Mira client with your API key
    let client = mira::Client::new("mira-api-key")?;

    // Create a chat request
    let request = AiRequest {
        model: "claude-3.5-sonnet".to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: "What are the three laws of robotics?".to_string(),
        }],
        temperature: Some(0.7),
        max_tokens: Some(500),
        stream: None,
    };

    // Generate a response
    let response = client.generate(request).await?;

    // Print the response
    if let Some(choice) = response.choices.first() {
        println!("Assistant: {}", choice.message.content);
    }

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
