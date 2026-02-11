//! This example demonstrates structured output with automatic type inference.
//!
//! Rig provides two ways to get structured output from an LLM:
//!
//! 1. **`prompt_typed<T>()`** (recommended) - Ergonomic API that automatically handles
//!    schema generation and deserialization:
//!    ```ignore
//!    let forecast: WeatherForecast = agent.prompt_typed("...").await?;
//!    ```
//!
//! 2. **`output_schema::<T>()` on builder** - Set schema at build time, manually deserialize:
//!    ```ignore
//!    let agent = client.agent("gpt-4o").output_schema::<WeatherForecast>().build();
//!    let response = agent.prompt("...").await?;
//!    let forecast: WeatherForecast = serde_json::from_str(&response)?;
//!    ```

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

    let agent = client
        .agent("gpt-4o")
        .preamble("You are a helpful weather assistant. Respond with realistic weather data.")
        .build();

    // Type can be inferred from the variable binding
    let forecast: WeatherForecast = agent
        .prompt_typed("What's the weather forecast for New York City today?")
        .await?;

    println!("=== Method 1: prompt_typed<T>() ===");
    println!("{}", serde_json::to_string_pretty(&forecast)?);

    // Or use turbofish syntax for explicit type specification
    let forecast = agent
        .prompt_typed::<WeatherForecast>("What's the weather forecast for Los Angeles?")
        .await?;

    println!("\n=== With turbofish syntax ===");
    println!("{}", serde_json::to_string_pretty(&forecast)?);

    // This approach sets the schema at agent build time. The response is a
    // JSON string that you must manually deserialize.
    // This method is more suited towards agent being used as tools
    // where you might still want to send the output to a parent agent,
    // as the LLM can still use the raw data in that case (a raw JSON string is still a string)
    let agent_with_schema = client
        .agent("gpt-4o")
        .preamble("You are a helpful weather assistant. Respond with realistic weather data.")
        .output_schema::<WeatherForecast>()
        .build();

    let response = agent_with_schema
        .prompt("What's the weather forecast for Chicago?")
        .await?;

    // Manual deserialization required
    let forecast: WeatherForecast = serde_json::from_str(&response)?;

    println!("\n=== Method 2: output_schema on builder ===");
    println!("{}", serde_json::to_string_pretty(&forecast)?);

    Ok(())
}
