//! OpenAI structured output coverage, including the migrated example path.

use rig::agent::OutputMode;
use rig::client::CompletionClient;
use rig::completion::{Prompt, TypedPrompt};
use rig::providers::openai;
use rig::test_utils::RecordingHttpClient;
use rig_agent::test_utils::decode_structured_output;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

use super::super::support::with_openai_cassette;
use crate::support::{
    STRUCTURED_OUTPUT_PROMPT, SmokeStructuredOutput, assert_contains_any_case_insensitive,
    assert_nonempty_response, assert_smoke_structured_output, smoke_structured_output_value,
};

fn output_tool_response(name: &str) -> String {
    serde_json::json!({
        "id": "resp_runtime_acceptance",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-4o",
        "output": [{
            "type": "function_call",
            "id": "fc_runtime_acceptance",
            "arguments": smoke_structured_output_value().to_string(),
            "call_id": "call_runtime_acceptance",
            "name": name,
            "status": "completed"
        }],
        "tools": []
    })
    .to_string()
}

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
            let agent = client
                .agent(openai::GPT_4O)
                .output_schema::<SmokeStructuredOutput>()
                .output_mode(OutputMode::Native)
                .build();

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
async fn classic_tool_mode_maps_through_openai_responses() {
    let http = RecordingHttpClient::new(output_tool_response("final_result"));
    let client = openai::Client::builder()
        .api_key("test-key")
        .http_client(http.clone())
        .build()
        .expect("OpenAI test client should build");
    let agent = client
        .agent(openai::GPT_4O)
        .output_schema::<SmokeStructuredOutput>()
        .output_mode(OutputMode::Tool)
        .build();

    let response = agent
        .prompt(STRUCTURED_OUTPUT_PROMPT)
        .await
        .expect("classic OpenAI Tool-mode run should succeed");
    let structured: SmokeStructuredOutput =
        decode_structured_output("openai_classic_tool_mode", &response)
            .expect("output-tool arguments should deserialize");
    assert_smoke_structured_output(&structured);

    let requests = http.requests();
    assert_eq!(requests.len(), 1);
    let body: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("request should be JSON");
    assert_eq!(body["tools"][0]["name"], "final_result");
    assert_ne!(
        body.pointer("/text/format/type").and_then(|v| v.as_str()),
        Some("json_schema")
    );
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
