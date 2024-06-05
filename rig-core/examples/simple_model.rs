use std::env;

use rig::{
    completion::Prompt,
    providers::{cohere, openai},
};

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Create OpenAI client and model
    let openai_api_key = env::var("OPENAI_API_KEY").expect("OPENAI_API_KEY not set");
    let openai_client = openai::Client::new(&openai_api_key);

    let gpt4 = openai_client.model("gpt-4").temperature(0.0).build();

    // Create Cohere client and model
    let cohere_api_key = env::var("COHERE_API_KEY").expect("COHERE_API_KEY not set");
    let cohere_client = cohere::Client::new(&cohere_api_key);

    let command_r = cohere_client.model("command-r").temperature(0.0).build();

    // Prompt the models and print their response
    println!("Question: Who are you?");
    println!("GPT-4: {:?}", gpt4.chat("Who are you?", vec![]).await?);
    println!("Coral: {:?}", command_r.chat("Who are you?", vec![]).await?);

    Ok(())
}
