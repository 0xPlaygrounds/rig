//! OpenAI structured output coverage, including the migrated example path.

use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt, TypedPrompt};
use rig::prelude::AgentClientExt;
use rig::providers::openai;
use rig_agent::test_utils::decode_structured_output;
use rig_bevy::{LocalRuntime, TenantId};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::super::support::with_openai_cassette;
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
    with_openai_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let agent = client.agent(openai::GPT_4O).build();

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
async fn bevy_local_native_structured_output() {
    with_openai_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let model = client.completion_model(openai::GPT_4O);
            let request = model
                .completion_request(STRUCTURED_OUTPUT_PROMPT)
                .output_schema(schemars::schema_for!(SmokeStructuredOutput))
                .build();
            let mut runtime = LocalRuntime::new(model, TenantId::new());
            let result = runtime
                .run_structured::<SmokeStructuredOutput>(
                    request,
                    rig_bevy::OutputMode::Native,
                    true,
                    false,
                )
                .await
                .expect("Bevy structured run");
            assert_smoke_structured_output(&result.output);
        },
    )
    .await;
}

#[tokio::test]
async fn prompt_typed_and_output_schema() {
    with_openai_cassette(
        "structured_output/prompt_typed_and_output_schema",
        |client| async move {
            let agent = client
                .agent(openai::GPT_4O)
                .preamble(
                    "You are a helpful weather assistant. Respond with realistic weather data.",
                )
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
                decode_structured_output("openai_output_schema_weather", &response)
                    .expect("schema response should deserialize");
            assert_weather_forecast(&parsed, &["chicago"]);
        },
    )
    .await;
}
