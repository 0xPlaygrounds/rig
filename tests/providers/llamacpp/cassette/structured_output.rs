//! llama.cpp structured output coverage, including the migrated example path.

use rig::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use crate::support::{
    STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_contains_any_case_insensitive,
    assert_nonempty_response, assert_smoke_structured_output,
};

use super::super::cassette_support::*;

const WEATHER_PREAMBLE: &str = "You are a helpful weather assistant. Return ONLY JSON matching exactly this schema: \
     {\"city\": string, \"current\": {\"temperature_f\": number, \"humidity_pct\": integer, \"description\": string}}. \
     Every field is required. Use Fahrenheit for temperature_f. Do not omit keys. Do not wrap the JSON in markdown.";

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
    with_llamacpp_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let agent = client.agent(CASSETTE_MODEL).build();

            let response: SmokeStructuredOutput = agent
                .prompt_typed(STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("structured output prompt should succeed")
                .output;

            assert_smoke_structured_output(&response);
        },
    )
    .await;
}

#[tokio::test]
async fn prompt_typed_structured_output() {
    with_llamacpp_cassette("structured_output/prompt_typed_structured_output", |client| async move {
        let model = CASSETTE_MODEL;
        let agent = client
            .agent(model)
            .preamble(WEATHER_PREAMBLE)
            .temperature(0.0)
            .build();

        let forecast: WeatherForecast = agent
            .prompt_typed(
                "Return JSON weather data for New York City today with fields city, current.temperature_f, current.humidity_pct, and current.description.",
            )
            .await
            .expect("prompt_typed should succeed").output;
        assert_weather_forecast(&forecast, &["new york", "nyc"]);
    })
    .await;
}

#[tokio::test]
async fn prompt_typed_extended_details_structured_output() {
    with_llamacpp_cassette("structured_output/prompt_typed_extended_details_structured_output", |client| async move {
        let model = CASSETTE_MODEL;
        let agent = client
            .agent(model)
            .preamble(WEATHER_PREAMBLE)
            .temperature(0.0)
            .build();

        let extended = agent
            .prompt_typed::<WeatherForecast>(
                "Return JSON weather data for Los Angeles with fields city, current.temperature_f, current.humidity_pct, and current.description.",
            )
            .await
            .expect("extended prompt_typed should succeed");
        assert_weather_forecast(&extended.output, &["los angeles", "la"]);
        assert!(extended.usage.total_tokens > 0, "usage should be populated");
    })
    .await;
}

#[tokio::test]
async fn output_schema_structured_output() {
    with_llamacpp_cassette("structured_output/output_schema_structured_output", |client| async move {
        let model = CASSETTE_MODEL;
        let agent_with_schema = client
            .agent(model)
            .preamble(WEATHER_PREAMBLE)
            .temperature(0.0)
            .output_schema::<WeatherForecast>()
            .build();
        let response = agent_with_schema
            .prompt(
                "Return JSON weather data for Chicago with fields city, current.temperature_f, current.humidity_pct, and current.description.",
            )
            .await
            .expect("output schema prompt should succeed");
        let parsed: WeatherForecast =
            serde_json::from_str(&response.output).expect("schema response should deserialize");
        assert_weather_forecast(&parsed, &["chicago"]);
    })
    .await;
}
