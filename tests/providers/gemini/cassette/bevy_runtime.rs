//! Gemini Generate Content acceptance through the experimental ECS runtime.

use rig::providers::gemini::completion::gemini_api_types::{
    AdditionalParameters, GenerationConfig, ThinkingConfig, ThinkingLevel,
};
use rig::{
    bevy::{
        BevyModelExt, HostedRuntime, LocalRuntime, OutputMode, StreamingRunEvent,
        StructuredOutputPolicy,
    },
    client::CompletionClient,
    completion::CompletionModel,
    providers::gemini,
    test_utils::{MockHttpResponse, SequencedHttpClient},
};

use super::super::support::with_gemini_cassette;
use crate::support::{
    BASIC_PREAMBLE, BASIC_PROMPT, PortableAdder, PortableSubtract, STREAMING_PREAMBLE,
    STREAMING_PROMPT, STREAMING_TOOLS_PREAMBLE, STREAMING_TOOLS_PROMPT, STRUCTURED_OUTPUT_PROMPT,
    SmokeStructuredOutput, assert_mentions_expected_number, assert_smoke_structured_output,
    smoke_structured_output_value,
};

fn text_response(text: &str, response_id: &str) -> String {
    serde_json::json!({
        "candidates": [{
            "content": {
                "parts": [{"text": text}],
                "role": "model"
            },
            "finishReason": "STOP",
            "index": 0
        }],
        "modelVersion": gemini::completion::GEMINI_2_5_FLASH,
        "responseId": response_id,
        "usageMetadata": {
            "promptTokenCount": 1,
            "candidatesTokenCount": 1,
            "totalTokenCount": 2
        }
    })
    .to_string()
}

fn streaming_tool_params() -> serde_json::Value {
    serde_json::to_value(AdditionalParameters::default().with_config(GenerationConfig::default()))
        .expect("Gemini additional params should serialize")
}

fn streaming_params() -> serde_json::Value {
    let config = GenerationConfig {
        thinking_config: Some(ThinkingConfig {
            thinking_budget: None,
            thinking_level: Some(ThinkingLevel::Medium),
            include_thoughts: Some(true),
        }),
        ..GenerationConfig::default()
    };
    serde_json::to_value(AdditionalParameters::default().with_config(config))
        .expect("Gemini streaming params should serialize")
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
    with_gemini_cassette("agent/completion_smoke", |client| async move {
        let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
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
async fn hosted_surface_exposes_only_redacted_provider_diagnostics() {
    with_gemini_cassette("agent/completion_smoke", |client| async move {
        let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
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
    with_gemini_cassette("streaming/streaming_smoke", |client| async move {
        let model = client.completion_model(gemini::completion::GEMINI_3_FLASH_PREVIEW);
        let mut runtime = LocalRuntime::new().expect("runtime should build");
        let agent = runtime
            .spawn_agent(
                model
                    .clone()
                    .into_bevy_agent_builder()
                    .preamble(STREAMING_PREAMBLE)
                    .additional_params(streaming_params())
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
    with_gemini_cassette(
        "streaming_tools/streaming_tools_smoke",
        |client| async move {
            let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
            let mut runtime = LocalRuntime::new().expect("runtime should build");
            let agent = runtime
                .spawn_agent(
                    model
                        .clone()
                        .into_bevy_agent_builder()
                        .preamble(STREAMING_TOOLS_PREAMBLE)
                        .additional_params(streaming_tool_params())
                        .max_model_calls(3)
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
async fn native_structured_output_validates_in_the_ecs_runtime() {
    with_gemini_cassette(
        "structured_output/structured_output_smoke",
        |client| async move {
            let model = client.completion_model("gemini-3-flash-preview");
            let mut runtime = LocalRuntime::new().expect("runtime should build");
            let agent = runtime
                .spawn_agent(
                    model
                        .clone()
                        .into_bevy_agent_builder()
                        .structured_output::<SmokeStructuredOutput>(StructuredOutputPolicy {
                            mode: OutputMode::Native,
                            ..StructuredOutputPolicy::default()
                        })
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

#[tokio::test]
async fn invalid_native_output_recovers_through_gemini_generate_content() {
    let valid = smoke_structured_output_value();
    let http = SequencedHttpClient::new([
        MockHttpResponse::success(text_response("not valid JSON", "gemini-bevy-invalid")),
        MockHttpResponse::success(text_response(&valid.to_string(), "gemini-bevy-valid")),
    ]);
    let client = gemini::Client::builder()
        .api_key("test-key")
        .http_client(http.clone())
        .build()
        .expect("Gemini test client should build");
    let model = client.completion_model(gemini::completion::GEMINI_2_5_FLASH);
    let mut runtime = LocalRuntime::new().expect("runtime should build");
    let agent = runtime
        .spawn_agent(
            model
                .clone()
                .into_bevy_agent_builder()
                .max_model_calls(2)
                .structured_output::<SmokeStructuredOutput>(StructuredOutputPolicy {
                    mode: OutputMode::Native,
                    max_retries: 1,
                    best_effort: false,
                })
                .build(),
        )
        .expect("agent should spawn");

    let result = runtime
        .run_blocking(agent, STRUCTURED_OUTPUT_PROMPT)
        .await
        .expect("Bevy Gemini structured-output recovery should succeed");
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
    assert_eq!(requests.len(), 2);
    assert_eq!(http.remaining_responses(), 0);
    for request in &requests {
        let body: serde_json::Value =
            serde_json::from_slice(&request.body).expect("request should be JSON");
        assert!(
            body.pointer("/generationConfig/responseJsonSchema")
                .is_some()
        );
    }
    let second: serde_json::Value =
        serde_json::from_slice(&requests[1].body).expect("request should be JSON");
    assert!(
        second
            .to_string()
            .contains("previous response did not satisfy the required JSON schema")
    );
}
