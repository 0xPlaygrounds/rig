//! Copilot structured output coverage, including the migrated example path.

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::copilot::{LIVE_MODEL, with_copilot_cassette};
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
async fn structured_output_smoke() {
    with_copilot_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let agent = client.agent(LIVE_MODEL).build();

            let response: SmokeStructuredOutput = agent
                .prompt_typed(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("structured output prompt should succeed");

            assert_smoke_structured_output(&response);
        },
    )
    .await;
}

#[tokio::test]
async fn prompt_typed_and_output_schema() {
    with_copilot_cassette(
        "structured_output/prompt_typed_and_output_schema",
        |client| async move {
            let agent = client
                .agent(LIVE_MODEL)
                .preamble(
                    "You are a helpful weather assistant. Respond with realistic weather data.",
                )
                .build();

            let forecast: WeatherForecast = agent
                .prompt_typed("What's the weather forecast for New York City today?")
                .await
                .expect("prompt_typed should succeed");
            assert_weather_forecast(&forecast, &["new york", "nyc"]);

            // `prompt_typed(..).extended_details()` is gone; `extract_native` is
            // the typed-plus-usage successor (same native structured-output
            // request as the agent above).
            let extended = rig::extract::extract_native::<WeatherForecast>(
                rig::agent::AgentConfig::new().with_preamble(
                    "You are a helpful weather assistant. Respond with realistic weather data.",
                ),
                client.provider_config(LIVE_MODEL),
                std::sync::Arc::new(rig::provider::Runtime::new()),
                "What's the weather forecast for Los Angeles?",
                0,
            )
            .await
            .expect("extended structured-output extraction should succeed");
            assert_weather_forecast(&extended.value, &["los angeles", "la"]);
            assert!(extended.usage.total_tokens > 0, "usage should be populated");

            let agent_with_schema = client
                .agent(LIVE_MODEL)
                .preamble(
                    "You are a helpful weather assistant. Respond with realistic weather data.",
                )
                .output_schema::<WeatherForecast>()
                .build();
            let response = agent_with_schema
                .prompt("What's the weather forecast for Chicago?")
                .await
                .expect("output schema prompt should succeed");
            let parsed: WeatherForecast =
                serde_json::from_str(&response).expect("schema response should deserialize");
            assert_weather_forecast(&parsed, &["chicago"]);
        },
    )
    .await;
}
use rig::prelude::*;
