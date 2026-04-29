//! Dedicated GPT-5.5 live smoke tests.

use base64::{Engine, prelude::BASE64_STANDARD};
use rig_core::client::{CompletionClient, ProviderClient};
use rig_core::completion::message::Image;
use rig_core::completion::{Chat, Message};
use rig_core::completion::{Prompt, TypedPrompt};
use rig_core::message::{DocumentSourceKind, ImageDetail, ImageMediaType};
use rig_core::providers::openai;
use rig_core::streaming::{StreamingChat, StreamingPrompt};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

#[cfg(feature = "websocket")]
use rig_core::completion::CompletionModel;
#[cfg(feature = "websocket")]
use rig_core::providers::openai::responses_api::websocket::ResponsesWebSocketEvent;

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

fn gpt_5_5_reasoning_params() -> serde_json::Value {
    serde_json::json!({
        "reasoning": { "effort": "xhigh" }
    })
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_prompt_smoke() {
    let client = openai::Client::from_env().expect("client should build");
    let agent = client
        .agent(openai::GPT_5_5)
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
    let client = openai::Client::from_env().expect("client should build");
    let agent = client
        .agent(openai::GPT_5_5)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_tools_smoke() {
    let client = openai::Client::from_env().expect("client should build");
    let agent = client
        .agent(openai::GPT_5_5)
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
    let client = openai::Client::from_env().expect("client should build");
    let agent = client
        .agent(openai::GPT_5_5)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = agent
        .stream_prompt(STREAMING_TOOLS_PROMPT)
        .multi_turn(3)
        .await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("streaming tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_structured_output_smoke() {
    let client = openai::Client::from_env().expect("client should build");
    let agent = client.agent(openai::GPT_5_5).build();

    let response: Gpt55Event = agent
        .prompt_typed("Return a concise event object for a local Rust meetup in Seattle.")
        .await
        .expect("typed prompt should succeed");

    assert_nonempty_response(&response.title);
    assert_nonempty_response(&response.category);
    assert_nonempty_response(&response.summary);

    let agent = client
        .agent(openai::GPT_5_5)
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
    let client = openai::Client::from_env().expect("client should build");
    let extractor = client.extractor::<SmokePerson>(openai::GPT_5_5).build();

    let response = extractor
        .extract_with_usage(EXTRACTOR_TEXT)
        .await
        .expect("extractor request should succeed");

    assert_nonempty_response(
        response
            .data
            .first_name
            .as_deref()
            .expect("first name should be present"),
    );
    assert_nonempty_response(
        response
            .data
            .last_name
            .as_deref()
            .expect("last name should be present"),
    );
    assert!(response.usage.total_tokens > 0, "usage should be populated");
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_image_input_smoke() {
    let client = openai::Client::from_env().expect("client should build");
    let agent = client
        .agent(openai::GPT_5_5)
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
    let client = openai::Client::from_env().expect("client should build");
    reasoning::run_reasoning_roundtrip_nonstreaming(ReasoningRoundtripAgent::new(
        client.completion_model(openai::GPT_5_5),
        Some(gpt_5_5_reasoning_params()),
    ))
    .await;
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_reasoning_streaming_smoke() {
    let client = openai::Client::from_env().expect("client should build");
    reasoning::run_reasoning_roundtrip_streaming(ReasoningRoundtripAgent::new(
        client.completion_model(openai::GPT_5_5),
        Some(gpt_5_5_reasoning_params()),
    ))
    .await;
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_reasoning_tool_roundtrip_smoke() {
    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let client = openai::Client::from_env().expect("client should build");
    let agent = client
        .agent(openai::GPT_5_5)
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(gpt_5_5_reasoning_params())
        .build();

    let result = agent
        .chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .await
        .expect("reasoning tool chat should succeed");

    reasoning::assert_nonstreaming_universal(&result, &call_count, "openai");
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn responses_reasoning_streaming_tool_roundtrip_smoke() {
    let call_count = std::sync::Arc::new(std::sync::atomic::AtomicUsize::new(0));
    let client = openai::Client::from_env().expect("client should build");
    let agent = client
        .agent(openai::GPT_5_5)
        .preamble(reasoning::TOOL_SYSTEM_PROMPT)
        .max_tokens(4096)
        .tool(WeatherTool::new(call_count.clone()))
        .additional_params(gpt_5_5_reasoning_params())
        .build();

    let stream = agent
        .stream_chat(reasoning::TOOL_USER_PROMPT, Vec::<Message>::new())
        .multi_turn(3)
        .await;

    let stats = reasoning::collect_stream_stats(stream, "openai").await;
    reasoning::assert_universal(&stats, &call_count, "openai");
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn chat_completions_prompt_smoke() {
    let client = openai::Client::from_env()
        .expect("client should build")
        .completions_api();
    let agent = client
        .agent(openai::GPT_5_5)
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
    let client = openai::Client::from_env()
        .expect("client should build")
        .completions_api();
    let agent = client
        .agent(openai::GPT_5_5)
        .preamble(STREAMING_PREAMBLE)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("chat completions streaming prompt should succeed");

    assert_nonempty_response(&response);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn chat_completions_tools_smoke() {
    let client = openai::Client::from_env()
        .expect("client should build")
        .completions_api();
    let agent = client
        .agent(openai::GPT_5_5)
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
    let client = openai::Client::from_env()
        .expect("client should build")
        .completions_api();
    let agent = client
        .agent(openai::GPT_5_5)
        .preamble(STREAMING_TOOLS_PREAMBLE)
        .tool(Adder)
        .tool(Subtract)
        .build();

    let mut stream = agent.stream_prompt(STREAMING_TOOLS_PROMPT).await;
    let response = collect_stream_final_response(&mut stream)
        .await
        .expect("chat completions streaming tool prompt should succeed");

    assert_mentions_expected_number(&response, -3);
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn chat_completions_structured_output_smoke() {
    let client = openai::Client::from_env()
        .expect("client should build")
        .completions_api();
    let agent = client
        .agent(openai::GPT_5_5)
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
    let client = openai::Client::from_env()
        .expect("client should build")
        .completions_api();
    let extractor = client.extractor::<SmokePerson>(openai::GPT_5_5).build();

    let response = extractor
        .extract_with_usage(EXTRACTOR_TEXT)
        .await
        .expect("chat completions extractor request should succeed");

    assert_nonempty_response(
        response
            .data
            .first_name
            .as_deref()
            .expect("first name should be present"),
    );
    assert_nonempty_response(
        response
            .data
            .last_name
            .as_deref()
            .expect("last name should be present"),
    );
    assert!(response.usage.total_tokens > 0, "usage should be populated");
}

#[tokio::test]
#[ignore = "requires OPENAI_API_KEY"]
async fn chat_completions_image_input_smoke() {
    let client = openai::Client::from_env()
        .expect("client should build")
        .completions_api();
    let agent = client
        .agent(openai::GPT_5_5)
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
    let client = openai::Client::from_env().expect("client should build");
    let model = client.completion_model(openai::GPT_5_5);
    let mut session = client.responses_websocket(openai::GPT_5_5).await?;

    let request = model
        .completion_request("Explain one benefit of websocket mode in one sentence.")
        .build();
    session.send(request).await?;

    let mut streamed_text = String::new();
    loop {
        match session.next_event().await? {
            ResponsesWebSocketEvent::Item(item) => {
                if let rig_core::providers::openai::responses_api::streaming::ItemChunkKind::OutputTextDelta(delta) =
                    item.data
                {
                    streamed_text.push_str(&delta.delta);
                }
            }
            ResponsesWebSocketEvent::Response(chunk) => {
                if matches!(
                    chunk.kind,
                    rig_core::providers::openai::responses_api::streaming::ResponseChunkKind::ResponseCompleted
                        | rig_core::providers::openai::responses_api::streaming::ResponseChunkKind::ResponseFailed
                        | rig_core::providers::openai::responses_api::streaming::ResponseChunkKind::ResponseIncomplete
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
