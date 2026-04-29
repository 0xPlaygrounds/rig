//! OpenAI structured output coverage, including the migrated example path.

use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::{Prompt, TypedPrompt};
use rig_core::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::support::{
    STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_contains_any_case_insensitive,
    assert_nonempty_response, assert_smoke_structured_output,
};

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

fn assert_weather_forecast(forecast: &WeatherForecast, expected_city: &[&str]) {
    assert_nonempty_response(&forecast.city);
    assert_contains_any_case_insensitive(&forecast.city, expected_city);
    assert_nonempty_response(&forecast.current.description);
    assert!(
        forecast.current.temperature_f.is_finite(),
        "temperature should be finite"
    );
    assert!(
        (-100.0..=150.0).contains(&forecast.current.temperature_f),
        "temperature should be in a plausible Fahrenheit range, got {}",
        forecast.current.temperature_f
    );
    assert!(
        forecast.current.humidity_pct <= 100,
        "humidity should be within 0..=100, got {}",
        forecast.current.humidity_pct
    );
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn structured_output_smoke() {
    let client = openai::Client::from_env().expect("client should build");
    let agent = client.agent(openai::GPT_4O).build();

    let response: SmokeStructuredOutput = agent
        .prompt_typed(STRUCTURED_OUTPUT_PROMPT)
        .await
        .expect("structured output prompt should succeed");

    assert_smoke_structured_output(&response);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn prompt_typed_and_output_schema() {
    let client = openai::Client::from_env().expect("client should build");
    let agent = client
        .agent(openai::GPT_4O)
        .preamble("You are a helpful weather assistant. Respond with realistic weather data.")
        .build();

    let forecast: WeatherForecast = agent
        .prompt_typed("What's the weather forecast for New York City today?")
        .await
        .expect("prompt_typed should succeed");
    assert_weather_forecast(&forecast, &["new york", "nyc"]);

    let extended = agent
        .prompt_typed::<WeatherForecast>("What's the weather forecast for Los Angeles?")
        .extended_details()
        .await
        .expect("extended prompt_typed should succeed");
    assert_weather_forecast(&extended.output, &["los angeles", "la"]);
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
    assert_weather_forecast(&parsed, &["chicago"]);
}
