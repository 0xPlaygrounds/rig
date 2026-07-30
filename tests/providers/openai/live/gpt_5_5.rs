//! Dedicated GPT-5.5 live smoke tests.

use std::sync::Arc;

use base64::{Engine, prelude::BASE64_STANDARD};
use rig::AgentBuilder;
use rig::agent::AgentConfig;
use rig::completion::Message;
use rig::completion::message::Image;
use rig::extract::{ExtractOptions, extract_with_options};
use rig::message::{DocumentSourceKind, ImageDetail, ImageMediaType};
use rig::provider::{ProviderConfig, Runtime};
use rig::providers::openai;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[cfg(feature = "websocket")]
use rig::completion::CompletionRequest;
#[cfg(feature = "websocket")]
use rig::providers::openai::responses_api::websocket::ResponsesWebSocketEvent;

use crate::reasoning::{self, ReasoningRoundtripAgent, WeatherTool};
use crate::support::{
    Adder, BASIC_PREAMBLE, BASIC_PROMPT, EXTRACTOR_TEXT, IMAGE_FIXTURE_PATH, STREAMING_PREAMBLE,
    STREAMING_PROMPT, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, SmokePerson,
    SmokeStructuredOutput, Subtract, TOOLS_PREAMBLE, TOOLS_PROMPT,
    assert_contains_any_case_insensitive, assert_mentions_expected_number,
    assert_nonempty_response, assert_smoke_structured_output, collect_stream_final_response,
};

#[derive(Debug, Deserialize, JsonSchema, Serialize)]
struct Gpt55Event {
    title: String,
    category: String,
    summary: String,
}

/// Responses-API provider config for `model`, built from the environment.
fn responses_provider(model: &str) -> ProviderConfig {
    ProviderConfig::OpenAiResponses(
        openai::responses_api::functions::Config::from_env(model).expect("config should build"),
    )
}

/// Chat-Completions provider config for `model`, built from the environment.
fn completions_provider(model: &str) -> ProviderConfig {
    ProviderConfig::OpenAi(openai::functions::Config::from_env(model).expect("config should build"))
}

fn responses_agent(model: &str) -> AgentBuilder {
    AgentBuilder::new(responses_provider(model))
}

fn completions_agent(model: &str) -> AgentBuilder {
    AgentBuilder::new(completions_provider(model))
}

fn gpt_5_5_reasoning_params() -> serde_json::Value {
    serde_json::json!({
        "reasoning": { "effort": "xhigh" }
    })
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_prompt_smoke() {
    let agent = responses_agent(openai::GPT_5_5)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_streaming_prompt_smoke() {
    let agent = responses_agent(openai::GPT_5_5)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = Box::pin(agent.runner(STREAMING_PROMPT).stream_run());
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_tools_smoke() {
    let agent = responses_agent(openai::GPT_5_5)
        .preamble(TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent
        .prompt(TOOLS_PROMPT)
        .await
        .expect("tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_streaming_tools_smoke() {
    let agent = responses_agent(openai::GPT_5_5)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = Box::pin(
        agent
            .runner(STREAMING_TOOLS_PROMPT)
            .max_turns(3)
            .stream_run(),
    );
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_structured_output_smoke() {
    let agent = responses_agent(openai::GPT_5_5).build();

    let response: Gpt55Event = agent
        .prompt_typed("Return a concise event object for a local Rust meetup in Seattle.")
        .await
        .expect("typed prompt should succeed");

    assert_nonempty_response(&response.title);
    assert_nonempty_response(&response.category);
    assert_nonempty_response(&response.summary);

    let agent = responses_agent(openai::GPT_5_5)
        .output_schema::<SmokeStructuredOutput>()
        .build();
    let response = agent
        .prompt("Return a concise event object for a local Rust meetup in Seattle.")
        .await
        .expect("output schema prompt should succeed");
    let structured: SmokeStructuredOutput =
        serde_json::from_str(&response).expect("structured output should deserialize");
    assert_smoke_structured_output(&structured);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_extractor_smoke() {
    let response = extract_with_options::<SmokePerson>(
        AgentConfig::new(),
        responses_provider(openai::GPT_5_5),
        Arc::new(Runtime::new()),
        EXTRACTOR_TEXT,
        ExtractOptions::classic_extractor(),
    )
    .await
    .expect("extractor request should succeed");

    assert_nonempty_response(
        response
            .value
            .first_name
            .as_deref()
            .expect("first name should be present"),
    );
    assert_nonempty_response(
        response
            .value
            .last_name
            .as_deref()
            .expect("last name should be present"),
    );
    assert!(response.usage.total_tokens > 0, "usage should be populated");
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_image_input_smoke() {
    let agent = responses_agent(openai::GPT_5_5)
        .preamble("You are an image describer.")
        .build();
    let image_bytes = std::fs::read(IMAGE_FIXTURE_PATH).expect("fixture image should be readable");
    let image = Image {
        data: DocumentSourceKind::base64(&BASE64_STANDARD.encode(image_bytes)),
        media_type: Some(ImageMediaType::JPEG),
        detail: Some(ImageDetail::Auto),
        ..Default::default()
    };

    let response = agent
        .prompt(image)
        .await
        .expect("image prompt should succeed");

    assert_nonempty_response(&response);
    assert_contains_any_case_insensitive(&response, &["ant", "insect"]);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_reasoning_nonstreaming_smoke() {
    reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
        responses_provider(openai::GPT_5_5),
        Some(gpt_5_5_reasoning_params()),
    ))
    .await;
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_reasoning_streaming_smoke() {
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        responses_provider(openai::GPT_5_5),
        Some(gpt_5_5_reasoning_params()),
    ))
    .await;
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_reasoning_tool_roundtrip_smoke() {
    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let agent = responses_agent(openai::GPT_5_5)
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(gpt_5_5_reasoning_params())
        .build();

    let result = agent
        .chat(reasoning::TOOL_USER_PROMPT, &mut Vec::<Message>::new())
        .await
        .expect("reasoning tool chat should succeed");

    reasoning::assert_nonstreaming_universal(&result, &call_count, "openai");
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_reasoning_streaming_tool_roundtrip_smoke() {
    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let agent = responses_agent(openai::GPT_5_5)
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(gpt_5_5_reasoning_params())
        .build();

    let stream = Box::pin(
        agent
            .runner(reasoning::TOOL_USER_PROMPT)
            .history(Vec::<Message>::new())
            .max_turns(3)
            .stream_run(),
    );

    let stats = reasoning::collect_stream_stats(stream, "openai").await;
    reasoning::assert_universal(&stats, &call_count, "openai");
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn chat_completions_prompt_smoke() {
    let agent = completions_agent(openai::GPT_5_5)
        .preamble(BASIC_PREAMBLE)
        .build();

    let response = agent
        .prompt(BASIC_PROMPT)
        .await
        .expect("chat completions prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn chat_completions_streaming_prompt_smoke() {
    let agent = completions_agent(openai::GPT_5_5)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = Box::pin(agent.runner(STREAMING_PROMPT).stream_run());
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("chat completions streaming prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn chat_completions_tools_smoke() {
    let agent = completions_agent(openai::GPT_5_5)
        .preamble(TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let response = agent
        .prompt(TOOLS_PROMPT)
        .await
        .expect("chat completions tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn chat_completions_streaming_tools_smoke() {
    let agent = completions_agent(openai::GPT_5_5)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = Box::pin(agent.runner(STREAMING_TOOLS_PROMPT).stream_run());
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("chat completions streaming tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn chat_completions_structured_output_smoke() {
    let agent = completions_agent(openai::GPT_5_5)
        .output_schema::<SmokeStructuredOutput>()
        .build();

    let response = agent
        .prompt("Return a concise event object for a local Rust meetup in Seattle.")
        .await
        .expect("chat completions output schema prompt should succeed");
    let structured: SmokeStructuredOutput =
        serde_json::from_str(&response).expect("structured output should deserialize");

    assert_smoke_structured_output(&structured);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn chat_completions_extractor_smoke() {
    let response = extract_with_options::<SmokePerson>(
        AgentConfig::new(),
        completions_provider(openai::GPT_5_5),
        Arc::new(Runtime::new()),
        EXTRACTOR_TEXT,
        ExtractOptions::classic_extractor(),
    )
    .await
    .expect("chat completions extractor request should succeed");

    assert_nonempty_response(
        response
            .value
            .first_name
            .as_deref()
            .expect("first name should be present"),
    );
    assert_nonempty_response(
        response
            .value
            .last_name
            .as_deref()
            .expect("last name should be present"),
    );
    assert!(response.usage.total_tokens > 0, "usage should be populated");
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn chat_completions_image_input_smoke() {
    let agent = completions_agent(openai::GPT_5_5)
        .preamble("You are an image describer.")
        .build();
    let image_bytes = std::fs::read(IMAGE_FIXTURE_PATH).expect("fixture image should be readable");
    let image = Image {
        data: DocumentSourceKind::base64(&BASE64_STANDARD.encode(image_bytes)),
        media_type: Some(ImageMediaType::JPEG),
        detail: Some(ImageDetail::Auto),
        ..Default::default()
    };

    let response = agent
        .prompt(image)
        .await
        .expect("chat completions image prompt should succeed");

    assert_nonempty_response(&response);
    assert_contains_any_case_insensitive(&response, &["ant", "insect"]);
}

#[cfg(feature = "websocket")]
#[tokio::test]
#[ignore = "requires OPENAI_API_KEY and --features websocket"]
async fn responses_websocket_smoke() -> anyhow::Result<()> {
    let cfg = openai::responses_api::functions::Config::from_env(openai::GPT_5_5)
        .expect("config should build");
    let mut session = openai::responses_api::websocket::connect(cfg).await?;

    let request =
        CompletionRequest::from_prompt("Explain one benefit of websocket mode in one sentence.");
    session.send(request).await?;

    let mut streamed_text = String::new();
    loop {
        match session.next_event().await? {
            ResponsesWebSocketEvent::Item(item) => {
                if let rig::providers::openai::responses_api::streaming::ItemChunkKind::OutputTextDelta(delta) =
                    item.data
                {
                    streamed_text.push_str(&delta.delta);
                }
            }
            ResponsesWebSocketEvent::Response(chunk) => {
                if matches!(
                    chunk.kind,
                    rig::providers::openai::responses_api::streaming::ResponseChunkKind::ResponseCompleted
                        | rig::providers::openai::responses_api::streaming::ResponseChunkKind::ResponseFailed
                        | rig::providers::openai::responses_api::streaming::ResponseChunkKind::ResponseIncomplete
                ) {
                    break;
                }
            }
            ResponsesWebSocketEvent::Done(_) => {}
            ResponsesWebSocketEvent::Error(error) => return Err(anyhow::anyhow!(error.to_string())),
        }
    }

    assert_nonempty_response(&streamed_text);
    session.close().await?;
    Ok(())
}
