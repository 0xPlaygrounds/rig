use rig::prelude::*;
use rig::{completion::Prompt, providers::openai};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// Current weather conditions at a point in time
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Conditions {
    /// The temperature in degrees Fahrenheit
    temperature_f: f64,
    /// The humidity as a percentage (0-100)
    humidity_pct: u8,
    /// A short description (e.g. "sunny", "partly cloudy", "rain")
    description: String,
    /// Wind information
    wind: Wind,
}

/// Wind speed and direction
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Wind {
    /// Wind speed in miles per hour
    speed_mph: f64,
    /// Cardinal or intercardinal direction (e.g. "NW", "SSE")
    direction: String,
}

/// A multi-day weather forecast for a given city
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct WeatherForecast {
    /// The name of the city
    city: String,
    /// Current conditions right now
    current: Conditions,
    /// Forecast for the next three days
    daily_forecast: Vec<DayForecast>,
}

/// Forecast for a single day
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct DayForecast {
    /// The day of the week (e.g. "Monday")
    day: String,
    /// Expected high temperature in degrees Fahrenheit
    high_f: f64,
    /// Expected low temperature in degrees Fahrenheit
    low_f: f64,
    /// Expected conditions for the day
    conditions: Conditions,
}

#[tokio::main]
async fn main() -> Result<(), anyhow::Error> {
    let client = openai::Client::from_env();

    // Build an agent with a structured output schema.
    // The provider will constrain the model's response to valid JSON matching the schema.
    let agent = client
        .agent("gpt-5.2")
        .preamble("You are a helpful weather assistant. Respond with realistic weather data.")
        .output_schema::<WeatherForecast>()
        .build();

    let response = agent
        .prompt("What's the weather forecast for New York City today?")
        .await?;

    // The response is a JSON string conforming to the WeatherForecast schema.
    let forecast: WeatherForecast = serde_json::from_str(&response)?;

    println!("{}", serde_json::to_string_pretty(&forecast)?);

    Ok(())
}
