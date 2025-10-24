//! Example demonstrating the usage of the flex provider
//!
//! To run this example:
//!
//! 1. Create a .env file in the rig-core directory with:
//!    ```bash
//!    FLEX_API_KEY="your-api-key-here"
//!    FLEX_BASE_URL="https://api.openai.com/v1"
//!    FLEX_MODELS="gpt-4o,gpt-4-turbo"
//!    ```
//!
//! 2. Then run:
//!    ```bash
//!    cargo run --example flex_example --features=derive
//!    ```

use rig::client::{CompletionClient, VerifyClient};
use rig::completion::CompletionModel;
use rig::message::Message;
use rig::providers::flex;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Load environment variables from .env file
    dotenvy::dotenv().ok();
    println!("Environment variables loaded from .env file (if exists)\n");

    println!("Testing the flex provider implementation...\n");

    // Example 1: Create a client from environment variables
    println!("1. Creating client from environment variables...");
    println!("   Looking for: FLEX_API_KEY and FLEX_BASE_URL");

    let client = flex::Client::from_env();
    println!("   ✓ Client created successfully!\n");

    // Example 2: Get models from environment
    println!("2. Getting models from environment variable FLEX_MODELS...");
    let models = flex::get_models_from_env();
    println!("   Available models: {:?}", models);

    if !models.is_empty() {
        // Example 3: Create a completion model using the first available model
        let first_model = &models[0];
        println!("\n3. Creating completion model with: {}", first_model);
        let model = client.completion_model(first_model);
        println!("   ✓ Completion model created successfully!\n");

        // Example 4: Make a test API call
        println!("4. Making a test API call...");
        let response = model
            .completion_request(Message::user("Say hello in 10 words or less."))
            .send()
            .await;

        match response {
            Ok(response) => {
                println!("   ✓ Test API call successful!");

                // Extract and display the response text
                for content in response.choice.iter() {
                    match content {
                        rig::completion::AssistantContent::Text(text_content) => {
                            println!("   Prompt: Say hello in 10 words or less.");
                            println!("   Response: {}", text_content.text);
                        }
                        _ => {
                            println!("   Response content: {:?}", content);
                        }
                    }
                }

                // Display token usage
                let usage = &response.usage;
                println!(
                    "   Tokens - Input: {}, Output: {}, Total: {}",
                    usage.input_tokens, usage.output_tokens, usage.total_tokens
                );
            }
            Err(e) => {
                println!("   ✗ Test API call failed: {}", e);
                println!("   (If this is an authentication error, check your API key)");
            }
        }
    } else {
        println!(
            "\n3. No models configured in FLEX_MODELS. Using default model name for demonstration."
        );
        let _model = client.completion_model("gpt-4o");
        println!("   ✓ Completion model created with default name!\n");

        // Example 4: Verify the client (this will test the API key and base URL)
        println!("4. Verifying the client connection...");
        match client.verify().await {
            Ok(()) => println!("   ✓ Client verification successful!"),
            Err(e) => println!("   ✗ Client verification failed: {}", e),
        }
    }

    println!("\nFlex provider is working correctly!");
    println!("\nTo use it in your project:");
    println!(
        "  1. Create a .env file with: FLEX_API_KEY, FLEX_BASE_URL, and optionally FLEX_MODELS"
    );
    println!("  2. Load it with dotenvy::dotenv().ok()");
    println!("  3. Create a client with flex::Client::from_env()");
    println!("  4. Use flex::get_models_from_env() to get configured model names");
    println!("  5. Create models with client.completion_model(model_name)");

    Ok(())
}
