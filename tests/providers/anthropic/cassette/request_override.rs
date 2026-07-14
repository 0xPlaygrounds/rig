//! Verifies that a `CompletionCallAction::Patch` hook steers the model request per
//! turn, end-to-end through a real Anthropic round-trip.
//!
//! The agent registers two tools (`get_weather` and `get_time`). On the first
//! turn a hook returns `CompletionCallAction::patch(...)` that narrows the advertised
//! tools to `["get_weather"]` and forces `tool_choice = Required`. The recorded
//! request to Anthropic therefore advertises only `get_weather` (not `get_time`)
//! and carries a non-auto tool choice — proving the per-turn override reached the
//! wire. The blocking and streaming tests assert the same, since both drivers
//! resolve the override through the shared request builder.

use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};

use rig::agent::{AgentHook, CompletionCallAction, CompletionCallEvent, RequestPatch};
use rig::client::CompletionClient;
use rig::completion::{CompletionModel, Prompt};
use rig::message::ToolChoice;
use rig::providers::anthropic;
use rig::streaming::StreamingPrompt;
use rig::tool::Tool;
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::with_anthropic_cassette;
use crate::support::collect_stream_final_response;

const PROMPT: &str = "I'm planning a trip to Paris. Use a tool to help me prepare.";
const PREAMBLE: &str =
    "You are a travel assistant. Use the available tools to gather information before answering.";

#[derive(Deserialize)]
struct WeatherArgs {
    location: String,
}

#[derive(Deserialize)]
struct TimeArgs {
    #[allow(dead_code)]
    timezone: String,
}

#[derive(Debug, thiserror::Error)]
#[error("tool error")]
struct ToolErr;

/// `get_weather` — the only tool the override leaves advertised on turn 1.
#[derive(Clone, Default)]
struct GetWeather {
    calls: Arc<AtomicUsize>,
}

impl Tool for GetWeather {
    const NAME: &'static str = "get_weather";
    type Error = ToolErr;
    type Args = WeatherArgs;
    type Output = String;

    fn description(&self) -> String {
        "Get the current weather for a location.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": { "location": { "type": "string", "description": "City name" } },
            "required": ["location"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.calls.fetch_add(1, Ordering::SeqCst);
        Ok(format!(
            "It is 18 degrees Celsius and clear in {}.",
            args.location
        ))
    }
}

/// `get_time` — registered on the agent but filtered out of turn 1 by the
/// override's `active_tools` allow-list, so the model is never told about it.
#[derive(Clone, Default)]
struct GetTime;

impl Tool for GetTime {
    const NAME: &'static str = "get_time";
    type Error = ToolErr;
    type Args = TimeArgs;
    type Output = String;

    fn description(&self) -> String {
        "Get the current time in a timezone.".to_string()
    }

    fn parameters(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": { "timezone": { "type": "string", "description": "IANA timezone" } },
            "required": ["timezone"]
        })
    }

    async fn call(
        &self,
        _context: &mut rig::tool::ToolContext,
        _args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        Ok("12:00".to_string())
    }
}

/// A hook that, on the first turn only, narrows the advertised tools to
/// `get_weather` and forces a tool call. Later turns are left untouched (the
/// override is per-turn and non-sticky), so the model can answer with text.
struct ForceWeatherOnlyOnFirstTurn;

impl<M: CompletionModel> AgentHook<M> for ForceWeatherOnlyOnFirstTurn {
    async fn on_completion_call(
        &self,
        _ctx: &rig::agent::HookContext,
        event: CompletionCallEvent<'_>,
    ) -> CompletionCallAction {
        if event.turn == 1 {
            CompletionCallAction::patch(
                RequestPatch::new()
                    .active_tools([GetWeather::NAME])
                    .tool_choice(ToolChoice::Required),
            )
        } else {
            CompletionCallAction::continue_run()
        }
    }
}

/// Read the first recorded Anthropic request and assert the override hit the
/// wire: only `get_weather` is advertised (not `get_time`) and a non-auto tool
/// choice is set.
fn assert_first_request_was_overridden(scenario: &str) {
    let cassette_path = crate::cassettes::cassette_path("anthropic", scenario);
    let contents = std::fs::read_to_string(&cassette_path).unwrap_or_else(|e| {
        panic!(
            "cassette {} should be readable: {e}",
            cassette_path.display()
        )
    });

    let first_request_body = serde_yaml::Deserializer::from_str(&contents)
        .filter_map(|doc| Interaction::deserialize(doc).ok())
        .find_map(|i| i.when.body)
        .and_then(|body| serde_json::from_str::<Value>(&body).ok())
        .expect("cassette should contain a first request with a JSON body");

    let tool_names: Vec<&str> = first_request_body
        .get("tools")
        .and_then(Value::as_array)
        .map(|tools| {
            tools
                .iter()
                .filter_map(|t| t.get("name").and_then(Value::as_str))
                .collect()
        })
        .unwrap_or_default();

    assert!(
        tool_names.contains(&GetWeather::NAME),
        "the overridden request must advertise get_weather, saw {tool_names:?}"
    );
    assert!(
        !tool_names.contains(&GetTime::NAME),
        "active_tools must filter get_time off the first request, saw {tool_names:?}"
    );
    assert!(
        first_request_body
            .get("tool_choice")
            .is_some_and(|tc| { tc.get("type").and_then(Value::as_str) != Some("auto") }),
        "tool_choice override must set a non-auto choice, saw {:?}",
        first_request_body.get("tool_choice")
    );
}

#[derive(Deserialize)]
struct Interaction {
    when: RecordedRequest,
}

#[derive(Deserialize)]
struct RecordedRequest {
    body: Option<String>,
}

#[tokio::test]
async fn request_overridden_by_hook_blocking() {
    let weather = GetWeather::default();
    let probe = weather.clone();

    with_anthropic_cassette(
        "request_override/request_overridden_by_hook_blocking",
        move |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble(PREAMBLE)
                .tool(weather)
                .tool(GetTime)
                .add_hook(ForceWeatherOnlyOnFirstTurn)
                .build();

            let response = agent
                .prompt(PROMPT)
                .max_turns(5)
                .await
                .expect("blocking prompt should succeed");

            assert!(!response.is_empty(), "agent should produce a final answer");
        },
    )
    .await;

    assert!(
        probe.calls.load(Ordering::SeqCst) >= 1,
        "the forced tool_choice should make the model call get_weather"
    );
    assert_first_request_was_overridden("request_override/request_overridden_by_hook_blocking");
}

#[tokio::test]
async fn request_overridden_by_hook_streaming() {
    let weather = GetWeather::default();
    let probe = weather.clone();

    with_anthropic_cassette(
        "request_override/request_overridden_by_hook_streaming",
        move |client| async move {
            let agent = client
                .agent(anthropic::completion::CLAUDE_SONNET_4_6)
                .preamble(PREAMBLE)
                .tool(weather)
                .tool(GetTime)
                .add_hook(ForceWeatherOnlyOnFirstTurn)
                .build();

            let mut stream = agent.stream_prompt(PROMPT).max_turns(5).await;
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming prompt should succeed");

            assert!(!response.is_empty(), "stream should produce a final answer");
        },
    )
    .await;

    assert!(
        probe.calls.load(Ordering::SeqCst) >= 1,
        "the forced tool_choice should make the model call get_weather"
    );
    assert_first_request_was_overridden("request_override/request_overridden_by_hook_streaming");
}
