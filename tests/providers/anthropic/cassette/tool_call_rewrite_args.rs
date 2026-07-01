//! Verifies that a `Flow::RewriteArgs` hook rewrites a tool call's arguments
//! before the tool executes, end-to-end through a real Anthropic round-trip.
//!
//! The `get_weather` tool advertises only a `location` parameter, so the model
//! never sends a `units` field. A default hook injects `units: "celsius"` on the
//! `ToolCall` event via `Flow::rewrite_args`, and the tool records what it was
//! actually called with. A recorded `units` value therefore proves the rewritten
//! arguments reached execution — and the blocking and streaming tests assert the
//! same behavior, since both drivers share the same tool-execution seam.

use std::sync::{Arc, Mutex};

use rig::agent::{AgentHook, Flow, StepEvent};
use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt, ToolDefinition};
use rig::providers::anthropic;
use rig::streaming::StreamingPrompt;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::json;

use super::super::support::with_anthropic_cassette;
use crate::support::collect_stream_final_response;

const WEATHER_PROMPT: &str =
    "What is the weather in Tokyo right now? Use the get_weather tool to find out.";

/// What `get_weather` was actually invoked with, recorded per call.
#[derive(Clone, Debug, PartialEq)]
struct ObservedCall {
    location: String,
    units: Option<String>,
}

#[derive(Deserialize)]
struct WeatherArgs {
    location: String,
    // Absent from the tool's advertised schema, so the model never sends it: a
    // value here can only have been injected by the rewrite hook.
    #[serde(default)]
    units: Option<String>,
}

#[derive(Debug, thiserror::Error)]
#[error("weather error")]
struct WeatherError;

/// A tool that records the arguments it observed so the test can assert the
/// rewritten arguments — not the model's original ones — reached execution.
#[derive(Clone, Default)]
struct GetWeather {
    calls: Arc<Mutex<Vec<ObservedCall>>>,
}

impl GetWeather {
    fn observations(&self) -> Vec<ObservedCall> {
        self.calls.lock().expect("calls lock").clone()
    }
}

impl Tool for GetWeather {
    const NAME: &'static str = "get_weather";
    type Error = WeatherError;
    type Args = WeatherArgs;
    type Output = String;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Get the current weather for a location.".to_string(),
            parameters: json!({
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "City to get the weather for, e.g. 'Tokyo'"
                    }
                },
                "required": ["location"]
            }),
        }
    }

    async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
        self.calls.lock().expect("calls lock").push(ObservedCall {
            location: args.location.clone(),
            units: args.units.clone(),
        });
        let unit_label = match args.units.as_deref() {
            Some("celsius") => "18 degrees Celsius",
            Some("fahrenheit") => "64 degrees Fahrenheit",
            _ => "18 degrees",
        };
        Ok(format!(
            "It is {unit_label} and sunny in {}.",
            args.location
        ))
    }
}

/// A guardrail hook that injects `units: "celsius"` into every `get_weather`
/// call before it runs — the parameter-normalization use case `RewriteArgs`
/// exists for. It reads the model's emitted arguments, adds the field, and
/// returns the rewritten object via [`Flow::rewrite_args`].
struct PinUnitsToCelsius;

impl<M: CompletionModel> AgentHook<M> for PinUnitsToCelsius {
    async fn on_event(&self, event: StepEvent<'_, M>) -> Flow {
        if let StepEvent::ToolCall {
            tool_name, args, ..
        } = event
            && tool_name == GetWeather::NAME
        {
            let mut value: serde_json::Value =
                serde_json::from_str(args).unwrap_or_else(|_| json!({}));
            if let Some(object) = value.as_object_mut() {
                object.insert("units".to_string(), json!("celsius"));
            }
            return Flow::rewrite_args(value);
        }
        Flow::cont()
    }
}

fn assert_units_were_injected(observations: &[ObservedCall]) {
    assert!(
        !observations.is_empty(),
        "the get_weather tool should have been called"
    );
    for call in observations {
        assert_eq!(
            call.units.as_deref(),
            Some("celsius"),
            "the hook must inject units=celsius into every call (the model cannot, \
             since units is not in the tool's schema); saw {call:?}"
        );
    }
}

#[tokio::test]
async fn tool_call_args_rewritten_by_hook_blocking() {
    let weather = GetWeather::default();
    let probe = weather.clone();

    with_anthropic_cassette(
        "tool_call_rewrite_args/tool_call_args_rewritten_by_hook_blocking",
        move |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble("You are a weather assistant. Always use the get_weather tool to answer.")
                .tool(weather)
                .add_hook(PinUnitsToCelsius)
                .build();

            let response = agent
                .prompt(WEATHER_PROMPT)
                .max_turns(5)
                .await
                .expect("weather prompt should succeed");

            assert!(!response.is_empty(), "agent should produce a final answer");
        },
    )
    .await;

    assert_units_were_injected(&probe.observations());
}

#[tokio::test]
async fn tool_call_args_rewritten_by_hook_streaming() {
    let weather = GetWeather::default();
    let probe = weather.clone();

    with_anthropic_cassette(
        "tool_call_rewrite_args/tool_call_args_rewritten_by_hook_streaming",
        move |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble("You are a weather assistant. Always use the get_weather tool to answer.")
                .tool(weather)
                .add_hook(PinUnitsToCelsius)
                .build();

            let mut stream = agent.stream_prompt(WEATHER_PROMPT).multi_turn(5).await;
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming weather prompt should succeed");

            assert!(!response.is_empty(), "stream should produce a final answer");
        },
    )
    .await;

    assert_units_were_injected(&probe.observations());
}
