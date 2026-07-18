//! Anthropic Messages acceptance through the experimental ECS runtime.

use rig::{
    bevy::{
        BevyModelExt, HostedRuntime, LocalRuntime, OutputMode, StreamingRunEvent,
        StructuredOutputPolicy,
    },
    client::CompletionClient,
    completion::CompletionModel,
    providers::anthropic,
    test_utils::RecordingHttpClient,
};

use super::super::support::with_anthropic_cassette;
use crate::support::{
    BASIC_PREAMBLE, BASIC_PROMPT, PortableAdder, PortableSubtract, STREAMING_PREAMBLE,
    STREAMING_PROMPT, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, STRUCTURED_OUTPUT_PROMPT,
    SmokeStructuredOutput, assert_mentions_expected_number, assert_smoke_structured_output,
    bevy_synthetic_output_tool_name, smoke_structured_output_value,
};

fn text_response(text: &str) -> String {
    serde_json::json!({
        "content": [{"type": "text", "text": text}],
        "id": "msg_bevy_runtime_acceptance",
        "model": anthropic::completion::CLAUDE_SONNET_4_6,
        "role": "assistant",
        "type": "message",
        "stop_reason": "end_turn",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_read_input_tokens": null,
            "cache_creation_input_tokens": null
        }
    })
    .to_string()
}

fn output_tool_response(name: &str) -> String {
    serde_json::json!({
        "content": [{
            "type": "tool_use",
            "id": "toolu_bevy_runtime_acceptance",
            "name": name,
            "input": smoke_structured_output_value()
        }],
        "id": "msg_bevy_runtime_acceptance",
        "model": anthropic::completion::CLAUDE_SONNET_4_6,
        "role": "assistant",
        "type": "message",
        "stop_reason": "tool_use",
        "stop_sequence": null,
        "usage": {
            "input_tokens": 1,
            "output_tokens": 1,
            "cache_read_input_tokens": null,
            "cache_creation_input_tokens": null
        }
    })
    .to_string()
}

fn has_typed_blocking_final<M>(model: &M, result: &rig::bevy::LocalRunResult) -> bool
where
    M: CompletionModel,
    M::Response: std::any::Any + Send + Sync,
{
    let _ = model;
    result.raw_final::<M::Response>().is_some()
}

async fn run_typed_stream<M>(
    model: &M,
    runtime: &mut LocalRuntime,
    agent: rig::bevy::AgentId,
    prompt: &str,
) -> rig::bevy::StreamingRunResult<M::StreamingResponse>
where
    M: CompletionModel + Send + Sync + 'static,
    M::Response: std::any::Any + Send + Sync,
    M::StreamingResponse: std::any::Any + Send + Sync,
{
    let result = runtime
        .run_streaming::<M::StreamingResponse>(agent, prompt)
        .await
        .expect("ECS streaming provider run should succeed");
    let provisional = result.events.iter().position(|event| {
        matches!(
            event,
            StreamingRunEvent::Runtime(runtime)
                if matches!(runtime.as_ref(), rig::bevy::RunEvent::Provisional { .. })
        )
    });
    let provider_final = result
        .events
        .iter()
        .position(|event| matches!(event, StreamingRunEvent::ProviderFinal { .. }));
    assert!(matches!((provisional, provider_final), (Some(delta), Some(final_)) if delta < final_));
    let _ = model;
    result
}

#[tokio::test]
async fn blocking_exposes_concrete_provider_final() {
    with_anthropic_cassette("agent/completion_smoke", |client| async move {
        let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
        let mut runtime = LocalRuntime::new().expect("runtime should build");
        let agent = runtime
            .spawn_agent(
                model
                    .clone()
                    .into_bevy_agent_builder()
                    .preamble(BASIC_PREAMBLE)
                    .build(),
            )
            .expect("agent should spawn");

        let result = runtime
            .run_blocking(agent, BASIC_PROMPT)
            .await
            .expect("ECS blocking provider run should succeed");

        assert!(result.text.as_ref().is_some_and(|text| !text.is_empty()));
        assert!(has_typed_blocking_final(&model, &result));
    })
    .await;
}

#[tokio::test]
async fn tool_mode_maps_through_anthropic_messages() {
    let output_tool = bevy_synthetic_output_tool_name::<SmokeStructuredOutput>();
    let http = RecordingHttpClient::new(output_tool_response(&output_tool));
    let client = anthropic::Client::builder()
        .api_key("test-key")
        .http_client(http.clone())
        .build()
        .expect("Anthropic test client should build");
    let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
    let mut runtime = LocalRuntime::new().expect("runtime should build");
    let agent = runtime
        .spawn_agent(
            model
                .clone()
                .into_bevy_agent_builder()
                .structured_output::<SmokeStructuredOutput>(StructuredOutputPolicy {
                    mode: OutputMode::Tool,
                    max_retries: 0,
                    best_effort: false,
                })
                .build(),
        )
        .expect("agent should spawn");

    let result = runtime
        .run_blocking(agent, STRUCTURED_OUTPUT_PROMPT)
        .await
        .expect("Bevy Anthropic Tool-mode run should succeed");
    let structured: SmokeStructuredOutput = serde_json::from_value(
        result
            .structured_output
            .clone()
            .expect("validated structured output"),
    )
    .expect("structured output should deserialize");
    assert_smoke_structured_output(&structured);
    assert!(has_typed_blocking_final(&model, &result));

    let requests = http.requests();
    assert_eq!(requests.len(), 1);
    let body: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("request should be JSON");
    assert_eq!(body["tools"][0]["name"], output_tool);
    assert!(body.get("output_config").is_none());
}

#[tokio::test]
async fn prompted_mode_maps_through_anthropic_messages() {
    let output = smoke_structured_output_value();
    let http = RecordingHttpClient::new(text_response(&output.to_string()));
    let client = anthropic::Client::builder()
        .api_key("test-key")
        .http_client(http.clone())
        .build()
        .expect("Anthropic test client should build");
    let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
    let mut runtime = LocalRuntime::new().expect("runtime should build");
    let agent = runtime
        .spawn_agent(
            model
                .clone()
                .into_bevy_agent_builder()
                .structured_output::<SmokeStructuredOutput>(StructuredOutputPolicy {
                    mode: OutputMode::Prompted,
                    max_retries: 0,
                    best_effort: false,
                })
                .build(),
        )
        .expect("agent should spawn");

    let result = runtime
        .run_blocking(agent, STRUCTURED_OUTPUT_PROMPT)
        .await
        .expect("Bevy Anthropic Prompted-mode run should succeed");
    let structured: SmokeStructuredOutput = serde_json::from_value(
        result
            .structured_output
            .clone()
            .expect("validated structured output"),
    )
    .expect("structured output should deserialize");
    assert_smoke_structured_output(&structured);
    assert!(has_typed_blocking_final(&model, &result));

    let requests = http.requests();
    assert_eq!(requests.len(), 1);
    let body: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("request should be JSON");
    assert!(body.get("tools").is_none());
    assert!(body.get("output_config").is_none());
    assert!(
        body["system"][0]["text"]
            .as_str()
            .is_some_and(|text| text.contains("Respond with only JSON"))
    );
}

#[tokio::test]
async fn hosted_surface_exposes_only_redacted_provider_diagnostics() {
    with_anthropic_cassette("agent/completion_smoke", |client| async move {
        let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
        let mut local = LocalRuntime::new().expect("runtime should build");
        let agent = local
            .spawn_agent(
                model
                    .into_bevy_agent_builder()
                    .preamble(BASIC_PREAMBLE)
                    .build(),
            )
            .expect("agent should spawn");
        let hosted = HostedRuntime::new(local);
        let handle = hosted
            .start_run(agent, BASIC_PROMPT)
            .await
            .expect("hosted run should start");
        let result = hosted
            .drive_to_terminal(handle)
            .await
            .expect("hosted provider run should finish");
        let diagnostic = hosted
            .provider_diagnostic(handle)
            .await
            .expect("diagnostic lookup should succeed")
            .expect("provider diagnostic should exist");

        assert!(diagnostic.available);
        let final_text = result.text.as_deref().expect("hosted final text");
        assert!(!format!("{diagnostic:?}").contains(final_text));
    })
    .await;
}

#[tokio::test]
async fn streaming_exposes_deltas_and_concrete_provider_final() {
    with_anthropic_cassette("streaming/streaming_smoke", |client| async move {
        let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
        let mut runtime = LocalRuntime::new().expect("runtime should build");
        let agent = runtime
            .spawn_agent(
                model
                    .clone()
                    .into_bevy_agent_builder()
                    .preamble(STREAMING_PREAMBLE)
                    .build(),
            )
            .expect("agent should spawn");

        let result = run_typed_stream(&model, &mut runtime, agent, STREAMING_PROMPT).await;

        assert!(
            result
                .result
                .text
                .as_ref()
                .is_some_and(|text| !text.is_empty())
        );
    })
    .await;
}

#[tokio::test]
async fn streaming_tools_roundtrip_through_owned_effects() {
    with_anthropic_cassette(
        "streaming_tools/streaming_tools_smoke",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let mut runtime = LocalRuntime::new().expect("runtime should build");
            let agent = runtime
                .spawn_agent(
                    model
                        .clone()
                        .into_bevy_agent_builder()
                        .preamble(STREAMING_TOOLS_PREAMBLE)
                        .max_model_calls(2)
                        .build(),
                )
                .expect("agent should spawn");
            runtime
                .install_tool(agent, PortableAdder)
                .expect("portable adder should install");
            runtime
                .install_tool(agent, PortableSubtract)
                .expect("portable subtract should install");

            let result =
                run_typed_stream(&model, &mut runtime, agent, STREAMING_TOOLS_PROMPT).await;

            assert_mentions_expected_number(result.result.text.as_deref().expect("final text"), -3);
        },
    )
    .await;
}

#[tokio::test]
async fn structured_output_validates_in_the_ecs_runtime() {
    with_anthropic_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let model = client.completion_model(anthropic::completion::CLAUDE_SONNET_4_6);
            let mut runtime = LocalRuntime::new().expect("runtime should build");
            let agent = runtime
                .spawn_agent(
                    model
                        .clone()
                        .into_bevy_agent_builder()
                        .structured_output::<SmokeStructuredOutput>(
                            StructuredOutputPolicy::default(),
                        )
                        .build(),
                )
                .expect("agent should spawn");

            let result = runtime
                .run_blocking(agent, STRUCTURED_OUTPUT_PROMPT)
                .await
                .expect("ECS structured provider run should succeed");
            let structured: SmokeStructuredOutput = serde_json::from_value(
                result
                    .structured_output
                    .clone()
                    .expect("validated structured output"),
            )
            .expect("structured output should deserialize");

            assert_smoke_structured_output(&structured);
            assert!(has_typed_blocking_final(&model, &result));
        },
    )
    .await;
}
