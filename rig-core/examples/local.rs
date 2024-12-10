use rig::{completion::Prompt, providers::local};

#[tokio::main]
async fn main() {
    let ollama_client = local::Client::new();

    let llama3 = ollama_client.agent("llama3.1:8b-instruct-q8_0").build();

    // Prompt the model and print its response
    let response = llama3
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt ollama");

    println!("Ollama: {response}");
}
