//! This example demonstrates how to use OpenRouter's provider selection and prioritization feature.
//!
//! Provider selection allows you to:
//! - Select specific providers (only/ignore)
//! - Prioritize providers by order
//! - Sort providers by throughput, price, or latency
//! - Require specific capabilities (zero data retention, specific quantization levels)
//! - Set throughput and latency thresholds
//! - Set maximum price ceilings
//!
//! For more information, see: https://openrouter.ai/docs/guides/routing/provider-selection

use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::openrouter::{
    self, DataCollection, MaxPrice, PercentileThresholds, ProviderPreferences,
    ProviderSortStrategy, Quantization, ThroughputThreshold,
};
use serde_json::json;

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = openrouter::Client::from_env();

    println!("=== Example 1: Provider Order with Fallbacks ===\n");

    let preferences = ProviderPreferences::new()
        .order(["anthropic", "openai"])
        .allow_fallbacks(true);

    let agent = client
        .agent(openrouter::GEMINI_FLASH_2_0)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("Say hello in one sentence.").await?;
    println!("Response: {}\n", response);

    println!("=== Example 2: Fixed Allowlist (No Fallbacks) ===\n");

    let preferences = ProviderPreferences::new()
        .only(["azure", "together"])
        .allow_fallbacks(false);

    let agent = client
        .agent(openrouter::GEMINI_FLASH_2_0)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    match agent.prompt("What's 2+2?").await {
        Ok(response) => println!("Response: {}\n", response),
        Err(e) => println!("Expected error (no matching provider): {}\n", e),
    }

    println!("=== Example 3: Provider Blocklist ===\n");

    let preferences = ProviderPreferences::new().ignore(["deepinfra"]);

    let agent = client
        .agent(openrouter::GEMINI_FLASH_2_0)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("Name one planet.").await?;
    println!("Response: {}\n", response);

    println!("=== Example 4: Sort by Latency ===\n");

    let preferences = ProviderPreferences::new().sort(ProviderSortStrategy::Latency);

    let agent = client
        .agent(openrouter::GEMINI_FLASH_2_0)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("Say 'hello' in French.").await?;
    println!("Response: {}\n", response);

    println!("=== Example 5: Price Sort with Throughput Threshold ===\n");

    let preferences = ProviderPreferences::new()
        .sort(ProviderSortStrategy::Price)
        .preferred_min_throughput(ThroughputThreshold::Percentile(
            PercentileThresholds::new().p90(50.0),
        ));

    let agent = client
        .agent(openrouter::GEMINI_FLASH_2_0)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("Count to 3.").await?;
    println!("Response: {}\n", response);

    println!("=== Example 6: Require Parameter Support ===\n");

    let preferences = ProviderPreferences::new().require_parameters(true);

    let agent = client
        .agent(openrouter::GEMINI_FLASH_2_0)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("What color is the sky?").await?;
    println!("Response: {}\n", response);

    println!("=== Example 7: Data Policy and ZDR ===\n");

    let preferences = ProviderPreferences::new()
        .data_collection(DataCollection::Deny)
        .zdr(true);

    let agent = client
        .agent(openrouter::GEMINI_FLASH_2_0)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("Name a fruit.").await?;
    println!("Response: {}\n", response);

    println!("=== Example 8: Quantization Filter ===\n");

    let preferences = ProviderPreferences::new()
        .quantizations([Quantization::Int8, Quantization::Fp16]);

    let agent = client
        .agent(openrouter::GEMINI_FLASH_2_0)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("Name an animal.").await?;
    println!("Response: {}\n", response);

    println!("=== Example 9: Maximum Price Ceiling ===\n");

    let preferences = ProviderPreferences::new()
        .max_price(MaxPrice::new().prompt(0.001).completion(0.002));

    let agent = client
        .agent(openrouter::GEMINI_FLASH_2_0)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("Name a country.").await?;
    println!("Response: {}\n", response);

    println!("=== Example 10: Combined Configuration ===\n");

    let combined_params = json!({
        "provider": ProviderPreferences::new()
            .order(["google"])
            .sort(ProviderSortStrategy::Throughput)
            .zdr(true),
        "transforms": ["middle-out"]
    });

    let agent = client
        .agent(openrouter::GEMINI_FLASH_2_0)
        .preamble("You are a helpful assistant.")
        .additional_params(combined_params)
        .build();

    let response = agent.prompt("Say goodbye.").await?;
    println!("Response: {}\n", response);

    println!("All examples completed successfully!");

    Ok(())
}
