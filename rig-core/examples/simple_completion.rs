use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and model
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.agent("gpt-4").build();

    // Prompt the model and build a request
    let request = gpt4.completion("Who are you?").await;

    // Send the request into a response
    let response = request.send().await.expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
