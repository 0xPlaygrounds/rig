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
//!
//! This example uses deepseek/deepseek-v3.2 with providers:
//! DeepInfra, DeepSeek, Chutes, AtlasCloud, NovitaAI, Parasail, SiliconFlow, Google Vertex

use rig::completion::Prompt;
use rig::prelude::*;
use rig::providers::openrouter::{
    self, DataCollection, MaxPrice, PercentileThresholds, ProviderPreferences,
    ProviderSortStrategy, Quantization, ThroughputThreshold,
};
use serde_json::json;

const DEEPSEEK_V3_2: &str = "deepseek/deepseek-v3.2";

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = openrouter::Client::from_env();

    println!("=== Example 1: Provider Order with Fallbacks ===\n");

    let preferences = ProviderPreferences::new()
        .order(["DeepInfra", "DeepSeek", "Chutes"])
        .allow_fallbacks(true);

    let agent = client
        .agent(DEEPSEEK_V3_2)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("Say hello in one sentence.").await?;
    println!("Response: {}\n", response);

    println!("=== Example 2: Fixed Allowlist (No Fallbacks) ===\n");

    let preferences = ProviderPreferences::new()
        .only(["DeepInfra", "AtlasCloud"])
        .allow_fallbacks(false);

    let agent = client
        .agent(DEEPSEEK_V3_2)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    match agent.prompt("What's 2+2?").await {
        Ok(response) => println!("Response: {}\n", response),
        Err(e) => println!("Error: {}\n", e),
    }

    println!("=== Example 3: Provider Blocklist ===\n");

    let preferences = ProviderPreferences::new().ignore(["Google Vertex"]);

    let agent = client
        .agent(DEEPSEEK_V3_2)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("Name one planet.").await?;
    println!("Response: {}\n", response);

    println!("=== Example 4: Sort by Latency ===\n");

    let preferences = ProviderPreferences::new().sort(ProviderSortStrategy::Latency);

    let agent = client
        .agent(DEEPSEEK_V3_2)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("Say 'hello' in French.").await?;
    println!("Response: {}\n", response);

    println!("=== Example 5: Price Sort with Throughput Threshold ===\n");

    let preferences = ProviderPreferences::new()
        .sort(ProviderSortStrategy::Price)
        .preferred_min_throughput(ThroughputThreshold::Percentile(
            PercentileThresholds::new().p90(15.0),
        ));

    let agent = client
        .agent(DEEPSEEK_V3_2)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("Count to 3.").await?;
    println!("Response: {}\n", response);

    println!("=== Example 6: Require Parameter Support ===\n");

    let preferences = ProviderPreferences::new().require_parameters(true);

    let agent = client
        .agent(DEEPSEEK_V3_2)
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
        .agent(DEEPSEEK_V3_2)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    match agent.prompt("Name a fruit.").await {
        Ok(response) => println!("Response: {}\n", response),
        Err(e) => println!("Error (ZDR providers may not be available): {}\n", e),
    }

    println!("=== Example 8: Quantization Filter (fp8) ===\n");

    let preferences = ProviderPreferences::new().quantizations([Quantization::Fp8]);

    let agent = client
        .agent(DEEPSEEK_V3_2)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("Name an animal.").await?;
    println!("Response: {}\n", response);

    println!("=== Example 9: Maximum Price Ceiling ===\n");

    let preferences =
        ProviderPreferences::new().max_price(MaxPrice::new().prompt(0.30).completion(0.50));

    let agent = client
        .agent(DEEPSEEK_V3_2)
        .preamble("You are a helpful assistant.")
        .additional_params(preferences.to_json())
        .build();

    let response = agent.prompt("Name a country.").await?;
    println!("Response: {}\n", response);

    println!("=== Example 10: Combined Configuration ===\n");

    let combined_params = json!({
        "provider": ProviderPreferences::new()
            .order(["DeepSeek", "DeepInfra", "Parasail"])
            .sort(ProviderSortStrategy::Throughput),
        "transforms": ["middle-out"]
    });

    let agent = client
        .agent(DEEPSEEK_V3_2)
        .preamble("You are a helpful assistant.")
        .additional_params(combined_params)
        .build();

    let response = agent.prompt("Say goodbye.").await?;
    println!("Response: {}\n", response);

    println!("All examples completed successfully!");

    Ok(())
}
