use anyhow::Result;
use rig::OneOrMany;
use rig::agent::{AgentHook, Flow, RequestOverride, StepEvent};
use rig::client::CompletionClient;
use rig::completion::{
    CompletionModel, CompletionRequest, Message, Prompt, ProviderToolDefinition,
};
use rig::providers::openai;
use rig::streaming::StreamingPrompt;
use serde::Deserialize;
use serde_json::{Value, json};

use super::super::support::with_openai_completions_cassette_result;
use crate::support::{assert_nonempty_response, collect_stream_final_response};

const HOSTED_TOOL_NAME: &str = "hosted_lookup";

#[derive(Clone)]
struct ProviderToolOverrideHook;

impl<M> AgentHook<M> for ProviderToolOverrideHook
where
    M: CompletionModel,
{
    async fn on_event(&self, event: StepEvent<'_, M>) -> Flow {
        match event {
            StepEvent::CompletionCall { .. } => {
                Flow::override_request(RequestOverride::new().additional_params(json!({
                    "tools": [hosted_custom_tool()]
                })))
            }
            _ => Flow::cont(),
        }
    }
}

fn hosted_custom_tool() -> ProviderToolDefinition {
    ProviderToolDefinition::new("custom").with_config(
        "custom",
        json!({
            "name": HOSTED_TOOL_NAME,
            "description": "A provider-hosted lookup tool. Do not use it unless explicitly asked."
        }),
    )
}

#[tokio::test]
async fn streaming_agent_forwards_provider_tools() -> Result<()> {
    with_openai_completions_cassette_result(
        "provider_tools/streaming_agent_forwards_provider_tools",
        |client| async move {
            let agent = client
                .agent(openai::GPT_5_NANO)
                .preamble("Reply concisely. Do not use tools unless explicitly asked.")
                .provider_tool(hosted_custom_tool())
                .build();

            let mut stream = agent.stream_prompt("Reply exactly: ok").await;
            let response = collect_stream_final_response(&mut stream)
                .await
                .expect("streaming provider-tool prompt should succeed");
            assert_nonempty_response(&response);

            Ok::<(), anyhow::Error>(())
        },
    )
    .await?;

    let body = recorded_request_body("provider_tools/streaming_agent_forwards_provider_tools");
    assert_eq!(body["stream"], true, "expected streaming request: {body:#}");
    assert_hosted_custom_tool_present(&body);

    Ok(())
}

#[tokio::test]
async fn request_override_additional_params_can_inject_provider_tools() -> Result<()> {
    with_openai_completions_cassette_result(
        "provider_tools/request_override_additional_params_can_inject_provider_tools",
        |client| async move {
            let agent = client
                .agent(openai::GPT_5_NANO)
                .preamble("Reply concisely. Do not use tools unless explicitly asked.")
                .build();

            let response = agent
                .prompt("Reply exactly: ok")
                .add_hook(ProviderToolOverrideHook)
                .await
                .expect("hook-injected provider-tool prompt should succeed");
            assert_nonempty_response(&response);

            Ok::<(), anyhow::Error>(())
        },
    )
    .await?;

    let body = recorded_request_body(
        "provider_tools/request_override_additional_params_can_inject_provider_tools",
    );
    assert_hosted_custom_tool_present(&body);

    Ok(())
}

#[tokio::test]
async fn mixed_rig_and_provider_tools_with_schema_defers_response_format() -> Result<()> {
    with_openai_completions_cassette_result(
        "provider_tools/mixed_rig_and_provider_tools_with_schema_defers_response_format",
        |client| async move {
            let model = client.completion_model(openai::GPT_5_NANO);
            let request = CompletionRequest {
                model: None,
                preamble: Some(
                    "Reply concisely. Do not call tools unless the user explicitly asks."
                        .to_string(),
                ),
                chat_history: OneOrMany::one(Message::user(
                    "Say a friendly one-sentence greeting without using tools.",
                )),
                documents: vec![],
                tools: vec![rig::completion::ToolDefinition {
                    name: "weather".to_string(),
                    description: "Get the current weather for a city.".to_string(),
                    parameters: json!({
                        "type": "object",
                        "properties": {
                            "city": { "type": "string" }
                        },
                        "required": ["city"]
                    }),
                }],
                temperature: None,
                max_tokens: None,
                tool_choice: None,
                additional_params: Some(json!({
                    "tools": [hosted_custom_tool()]
                })),
                output_schema: Some(
                    serde_json::from_value(json!({
                        "title": "GreetingAnswer",
                        "type": "object",
                        "properties": {
                            "answer": { "type": "string" }
                        },
                        "required": ["answer"]
                    }))
                    .expect("schema should deserialize"),
                ),
            };

            model
                .completion(request)
                .await
                .expect("mixed provider/Rig tool completion request should succeed");

            Ok::<(), anyhow::Error>(())
        },
    )
    .await?;

    let body = recorded_request_body(
        "provider_tools/mixed_rig_and_provider_tools_with_schema_defers_response_format",
    );
    let tools = body["tools"]
        .as_array()
        .unwrap_or_else(|| panic!("expected tools array in request: {body:#}"));
    assert_eq!(tools.len(), 2, "expected Rig and provider tools: {body:#}");
    assert!(
        tools
            .iter()
            .any(|tool| tool["type"] == "function" && tool["function"]["name"] == "weather"),
        "expected Rig function tool in request: {body:#}"
    );
    assert_hosted_custom_tool_present(&body);
    assert!(
        body.get("response_format").is_none(),
        "initial mixed executable-tool turn should defer response_format: {body:#}"
    );

    Ok(())
}

#[test]
fn invalid_provider_tools_payload_fails_before_http() {
    let request = CompletionRequest {
        model: None,
        preamble: None,
        chat_history: OneOrMany::one(Message::user("Hello")),
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: Some(json!({
            "tools": { "type": "custom" }
        })),
        output_schema: None,
    };

    let result = rig::providers::openai::completion::CompletionRequest::try_from((
        openai::GPT_5_NANO.to_string(),
        request,
    ));

    assert!(
        matches!(
            result,
            Err(rig::completion::CompletionError::RequestError(_))
        ),
        "invalid provider tools payload should be rejected before sending"
    );
}

#[derive(Deserialize)]
struct RecordedInteraction {
    when: RecordedRequest,
}

#[derive(Deserialize)]
struct RecordedRequest {
    body: Option<String>,
}

fn recorded_request_body(scenario: &str) -> Value {
    let cassette_path = crate::cassettes::cassette_path("openai", scenario);
    let contents = std::fs::read_to_string(&cassette_path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable after recording/replay: {error}",
            cassette_path.display()
        )
    });

    serde_yaml::Deserializer::from_str(&contents)
        .find_map(|document| {
            let interaction = RecordedInteraction::deserialize(document)
                .expect("cassette interaction should deserialize");
            interaction
                .when
                .body
                .and_then(|body| serde_json::from_str::<Value>(&body).ok())
        })
        .unwrap_or_else(|| panic!("expected cassette {scenario} to contain a JSON request body"))
}

fn assert_hosted_custom_tool_present(body: &Value) {
    let tools = body["tools"]
        .as_array()
        .unwrap_or_else(|| panic!("expected tools array in request: {body:#}"));
    assert!(
        tools
            .iter()
            .any(|tool| tool["type"] == "custom" && tool["custom"]["name"] == HOSTED_TOOL_NAME),
        "expected provider-hosted custom tool in request: {body:#}"
    );
}
