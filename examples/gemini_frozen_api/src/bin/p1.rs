use rig::AgentBuilder;
use rig::providers::gemini::{self, Gemini};
use rig::tool::PortableTool;
use serde::Deserialize;
use serde_json::{Value, json};

#[derive(Deserialize)]
struct CityArgs {
    city: String,
}

#[derive(Deserialize)]
struct CelsiusArgs {
    celsius: f64,
}

struct Weather;

impl PortableTool for Weather {
    const NAME: &'static str = "get_weather";
    type Args = CityArgs;
    type Output = Value;
    type Error = std::convert::Infallible;

    fn description(&self) -> String {
        "Current weather for a city, in Celsius.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        })
    }

    async fn call(&self, args: CityArgs) -> Result<Value, Self::Error> {
        Ok(json!({ "city": args.city, "celsius": 21.5, "sky": "clear" }))
    }
}

struct LocalTime;

impl PortableTool for LocalTime {
    const NAME: &'static str = "local_time";
    type Args = CityArgs;
    type Output = Value;
    type Error = std::convert::Infallible;

    fn description(&self) -> String {
        "The current local time in a city, 24h clock.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "city": { "type": "string" } },
            "required": ["city"]
        })
    }

    async fn call(&self, args: CityArgs) -> Result<Value, Self::Error> {
        Ok(json!({ "city": args.city, "time": "14:05" }))
    }
}

struct ToFahrenheit;

impl PortableTool for ToFahrenheit {
    const NAME: &'static str = "to_fahrenheit";
    type Args = CelsiusArgs;
    type Output = f64;
    type Error = std::convert::Infallible;

    fn description(&self) -> String {
        "Convert Celsius to Fahrenheit.".to_owned()
    }

    fn parameters(&self) -> Value {
        json!({
            "type": "object",
            "properties": { "celsius": { "type": "number" } },
            "required": ["celsius"]
        })
    }

    async fn call(&self, args: CelsiusArgs) -> Result<f64, Self::Error> {
        Ok(args.celsius * 9.0 / 5.0 + 32.0)
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let gemini = Gemini::from_env()?;

    let agent = AgentBuilder::new(gemini.completion(gemini::GEMINI_3_8_FLASH))
        .preamble("You are a travel assistant. Use the tools; never guess weather or time.")
        .tool(Weather)
        .tool(LocalTime)
        .tool(ToFahrenheit)
        .build();

    let response = agent
        .prompt("Is it a sensible hour to call my friend in Lisbon, and how warm is it there in Fahrenheit?")
        .max_turns(6)
        .await?;

    println!("{}", response.output);
    Ok(())
}
