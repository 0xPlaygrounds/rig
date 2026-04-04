//! Migrated from `examples/openai_structured_output.rs`.

use rig::client::{CompletionClient, ProviderClient};
use rig::completion::{Prompt, TypedPrompt};
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::support::assert_nonempty_response;

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Conditions {
    temperature_f: f64,
    humidity_pct: u8,
    description: String,
}

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct WeatherForecast {
    city: String,
    current: Conditions,
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn prompt_typed_and_output_schema() {
    let client = openai::Client::from_env();
    let agent = client
        .agent(openai::GPT_4O)
        .preamble("You are a helpful weather assistant. Respond with realistic weather data.")
        .build();

    let forecast: WeatherForecast = agent
        .prompt_typed("What's the weather forecast for New York City today?")
        .await
        .expect("prompt_typed should succeed");
    assert_nonempty_response(&forecast.city);
    assert_nonempty_response(&forecast.current.description);

    let extended = agent
        .prompt_typed::<WeatherForecast>("What's the weather forecast for Los Angeles?")
        .extended_details()
        .await
        .expect("extended prompt_typed should succeed");
    assert_nonempty_response(&extended.output.city);
    assert!(extended.usage.total_tokens > 0, "usage should be populated");

    let agent_with_schema = client
        .agent(openai::GPT_4O)
        .preamble("You are a helpful weather assistant. Respond with realistic weather data.")
        .output_schema::<WeatherForecast>()
        .build();
    let response = agent_with_schema
        .prompt("What's the weather forecast for Chicago?")
        .await
        .expect("output schema prompt should succeed");
    let parsed: WeatherForecast =
        serde_json::from_str(&response).expect("schema response should deserialize");
    assert_nonempty_response(&parsed.city);
}
