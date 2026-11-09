//! The weather tool and typed reply the `prompt_typed()` tool-call suites share.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::tool::{Tool, ToolContext};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// No doc comment: schemars would send it as the schema description.
#[derive(Debug, Deserialize, JsonSchema, Serialize)]
pub struct WeatherResponse {
    pub city: String,
    pub weather: String,
}

#[derive(Debug, Deserialize)]
pub struct WeatherArgs {
    pub city: String,
}

/// The `weather` tool. It answers that a city's weather is all fire and
/// brimstone and counts its calls across clones.
#[derive(Clone, Default)]
pub struct WeatherTool {
    call_count: Arc<AtomicUsize>,
}

impl WeatherTool {
    /// The number of times any clone of this tool ran.
    pub fn calls(&self) -> usize {
        self.call_count.load(Ordering::SeqCst)
    }
}

impl Tool for WeatherTool {
    const NAME: &'static str = "weather";
    type Error = std::io::Error;
    type Args = WeatherArgs;
    type Output = String;

    fn description(&self) -> String {
        "Get the current weather for a city.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "city": { "type": "string" }
            },
            "required": ["city"]
        })
    }

    async fn call(
        &self,
        _context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.call_count.fetch_add(1, Ordering::SeqCst);
        Ok(format!(
            "The weather in {} is all fire and brimstone",
            args.city
        ))
    }
}
