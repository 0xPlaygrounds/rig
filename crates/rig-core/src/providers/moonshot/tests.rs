use super::{ANTHROPIC_BASE_URLS, Moonshot};
use crate::completion::CompletionRequest;
use crate::message::{AssistantContent, Message, Reasoning, ToolCall, ToolChoice, ToolFunction};
use crate::providers::openai::completion::{
    CompletionRequest as OpenAICompletionRequest, OpenAICompatibleProvider, OpenAIRequestParams,
};

fn prepared_body(request: CompletionRequest, model: &str) -> serde_json::Value {
    let mut request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
        model: model.to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: Moonshot::SUPPORTS_RESPONSE_FORMAT,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request should convert");
    Moonshot
        .prepare_request(&mut request)
        .expect("prepare_request should succeed");
    serde_json::to_value(request).expect("request should serialize")
}

#[test]
fn test_client_initialization() {
    let _client = crate::providers::moonshot::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed");
    let _client_from_builder = crate::providers::moonshot::Client::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");
    let _anthropic_client = crate::providers::moonshot::AnthropicClient::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("AnthropicClient::new() failed");
    let _anthropic_client_from_builder = crate::providers::moonshot::AnthropicClient::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("AnthropicClient::builder() failed");
}

#[test]
fn moonshot_preserves_reasoning_content_in_assistant_history() {
    let assistant = Message::Assistant {
        id: None,
        content: vec![
            AssistantContent::Reasoning(Reasoning::new("tool planning")),
            AssistantContent::ToolCall(ToolCall::from_wire(
                "call_1",
                ToolFunction {
                    name: "lookup".to_string(),
                    arguments: serde_json::json!({}),
                },
            )),
        ],
    };

    let request = CompletionRequest {
        model: Some("kimi-k2-thinking".to_string()),
        chat_history: vec![assistant],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let body = prepared_body(request, "kimi-k2-thinking");
    assert_eq!(
        body["messages"][0]["reasoning_content"],
        serde_json::json!("tool planning")
    );
}

#[test]
fn moonshot_joins_multiple_reasoning_blocks_with_newline() {
    // A replayed assistant turn carrying two distinct reasoning blocks must
    // keep them newline-separated on the wire, not glued together.
    let assistant = Message::Assistant {
        id: None,
        content: vec![
            AssistantContent::Reasoning(Reasoning::new("first thought")),
            AssistantContent::Reasoning(Reasoning::new("second thought")),
            AssistantContent::Text("done".into()),
        ],
    };

    let request = CompletionRequest {
        model: Some("kimi-k2-thinking".to_string()),
        chat_history: vec![assistant],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let body = prepared_body(request, "kimi-k2-thinking");
    assert_eq!(
        body["messages"][0]["reasoning_content"],
        serde_json::json!("first thought\nsecond thought")
    );
}

#[test]
fn moonshot_specific_tool_choice_is_rejected() {
    let request = CompletionRequest {
        model: Some("kimi-k2.5".to_string()),
        chat_history: vec![Message::user("Use a tool.")],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: Some(ToolChoice::Specific {
            function_names: vec!["lookup".to_string()],
        }),
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let mut request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
        model: "kimi-k2.5".to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: Moonshot::SUPPORTS_RESPONSE_FORMAT,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request should convert");

    let error = Moonshot
        .prepare_request(&mut request)
        .expect_err("specific tool choice should be rejected");
    assert!(error.to_string().contains("specific tool"));
}

#[test]
fn moonshot_required_tool_choice_is_coerced() {
    let request = CompletionRequest {
        model: Some("kimi-k2.5".to_string()),
        chat_history: vec![Message::user("Use a tool.")],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: Some(ToolChoice::Required),
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let body = prepared_body(request, "kimi-k2.5");
    assert_eq!(body["tool_choice"], "auto");
    assert_eq!(
        body["messages"]
            .as_array()
            .and_then(|messages| messages.last())
            .and_then(|message| message.get("content"))
            .and_then(|content| content.as_str()),
        Some("Please select a tool to handle the current issue.")
    );
}

#[test]
fn normalize_openai_style_base_to_anthropic_base() {
    assert_eq!(
        ANTHROPIC_BASE_URLS
            .normalize("https://api.moonshot.ai/v1")
            .as_deref(),
        Some("https://api.moonshot.ai/anthropic")
    );
    assert_eq!(
        ANTHROPIC_BASE_URLS
            .normalize("https://api.moonshot.cn/v1")
            .as_deref(),
        Some("https://api.moonshot.cn/anthropic")
    );
    assert_eq!(
        ANTHROPIC_BASE_URLS
            .normalize("https://proxy.example.com/v1")
            .as_deref(),
        Some("https://proxy.example.com/anthropic")
    );
}

#[test]
fn normalize_preserves_existing_anthropic_base() {
    assert_eq!(
        ANTHROPIC_BASE_URLS
            .normalize("https://proxy.example.com/anthropic")
            .as_deref(),
        Some("https://proxy.example.com/anthropic")
    );
}

#[test]
fn anthropic_primary_override_wins() {
    let override_url = ANTHROPIC_BASE_URLS.resolve(
        Some("https://primary.example.com/anthropic"),
        Some("https://api.moonshot.cn/v1"),
    );

    assert_eq!(
        override_url.as_deref(),
        Some("https://primary.example.com/anthropic")
    );
}
