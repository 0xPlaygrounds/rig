//! This example demonstrates how to use OpenRouter's provider selection and prioritization feature.
//!
//! Provider selection allows you to:
//! - Select specific providers (allow/ignore)
//! - Prioritize providers by order
//! - Sort providers by throughput, price, latency, or quality
//! - Require specific capabilities (zero data retention, specific quantization levels)
//!
//! For more information, see: https://openrouter.ai/docs/guides/routing/provider-selection

use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::openrouter::{
    self, ProviderPreferences, ProviderRequire, ProviderSort, Quantization,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    // Initialize the OpenRouter client
    let client = openrouter::Client::from_env();

    // Example 1: Zero Data Retention with Throughput Sorting
    // This configuration only uses providers with zero data retention policies
    // and sorts them by throughput (fastest providers first)
    println!("=== Example 1: Zero Data Retention + Throughput Sorting ===\n");

    let preferences = ProviderPreferences::new()
        .zero_data_retention()
        .fastest();

    let agent = client
        .agent(openrouter::GEMINI_FLASH_2_0)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("Say hello in one sentence.").await?;
    println!("Response: {}\n", response);

    // Example 2: Specific Provider Order with Quantization Requirements
    // This prioritizes Anthropic and OpenAI providers, requiring INT8 quantization
    println!("=== Example 2: Provider Order + Quantization Requirements ===\n");

    let preferences = ProviderPreferences::new()
        .order(["Anthropic", "OpenAI", "Google"])
        .require(
            ProviderRequire::new()
                .quantization([Quantization::Int8, Quantization::Fp16]),
        )
        .sort(ProviderSort::Price);

    let agent = client
        .agent(openrouter::CLAUDE_3_7_SONNET)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("What's 2+2? Answer briefly.").await?;
    println!("Response: {}\n", response);

    // Example 3: Allow-list only specific providers
    // Only use Anthropic and OpenAI providers
    println!("=== Example 3: Provider Allow-list ===\n");

    let preferences = ProviderPreferences::new()
        .allow(["Anthropic", "OpenAI"])
        .sort(ProviderSort::Quality);

    let agent = client
        .agent(openrouter::CLAUDE_3_7_SONNET)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("Name one planet.").await?;
    println!("Response: {}\n", response);

    // Example 4: Using ignore-list to exclude providers
    // Exclude certain providers from selection
    println!("=== Example 4: Provider Ignore-list ===\n");

    let preferences = ProviderPreferences::new()
        .ignore(["SomeProviderToAvoid"])
        .lowest_latency();

    let agent = client
        .agent(openrouter::GEMINI_FLASH_2_0)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("Say 'hello' in French.").await?;
    println!("Response: {}\n", response);

    // Example 5: Combining with other additional_params
    // You can combine provider preferences with other OpenRouter-specific parameters
    println!("=== Example 5: Combined with Other Parameters ===\n");

    let combined_params = json!({
        "provider": ProviderPreferences::new()
            .order(["Google"])
            .sort(ProviderSort::Throughput),
        "transforms": ["middle-out"]  // Other OpenRouter-specific parameter
    });

    let agent = client
        .agent(openrouter::GEMINI_FLASH_2_0)
        .preamble("You are a helpful assistant.")
        .additional_params(combined_params)
        .build();

    let response = agent.prompt("Count to 3.").await?;
    println!("Response: {}\n", response);

    println!("All examples completed successfully!");

    Ok(())
}
