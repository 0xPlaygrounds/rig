# Rig
A library for developing LLM-powered Rust applications.

## Installation
```bash
cargo add rig-core
```

## Example
```rust
use rig::{completion::Prompt, providers::openai};

#[tokio::main]
async fn main() {
    // Create OpenAI client and model
    let openai_client = openai::Client::from_env();

    let gpt4 = openai_client.model("gpt-4").build();

    // Prompt the model and print its response
    let response = gpt4
        .prompt("Who are you?")
        .await
        .expect("Failed to prompt GPT-4");

    println!("GPT-4: {response}");
}
```

Note: This requires the `OPENAI_API_KEY` environment variable to be set.
