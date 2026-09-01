use super::*;
use crate::providers::openai::completion::{
    CompletionRequest as OpenAICompletionRequest, OpenAICompatibleProvider, OpenAIRequestParams,
};
use crate::test_utils::MockCompletionModel;

#[test]
fn test_client_initialization() {
    let _client = crate::providers::perplexity::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed");
    let _client_from_builder = crate::providers::perplexity::Client::builder()
        .api_key("dummy-key")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");
}

#[test]
fn perplexity_finalize_flattens_text_only_content_arrays() {
    let mut body = serde_json::json!({
        "model": SONAR,
        "messages": [
            {"role": "system", "content": [{"type": "text", "text": "Be brief."}]},
            {"role": "user", "content": [
                {"type": "text", "text": "First."},
                {"type": "text", "text": "Second."}
            ]},
            {"role": "user", "content": [
                {"type": "text", "text": "Look:"},
                {"type": "image_url", "image_url": {"url": "https://example.com/i.png"}}
            ]}
        ]
    });

    PerplexityExt
        .finalize_request_body(&mut body)
        .expect("finalize should succeed");

    assert_eq!(body["messages"][0]["content"], "Be brief.");
    assert_eq!(body["messages"][1]["content"], "First.\nSecond.");
    // Mixed content stays an array for the API's multimodal handling.
    assert!(body["messages"][2]["content"].is_array());
}

#[test]
fn perplexity_drops_tool_choice_instead_of_erroring() {
    // Multi-name Specific errors on tool-supporting providers; with
    // SUPPORTS_TOOLS = false it must be dropped before that validation.
    let mut request = crate::completion::CompletionRequest {
        model: None,
        chat_history: vec!["Hello!".into()],
        documents: vec![],
        max_tokens: None,
        temperature: None,
        tools: vec![],
        tool_choice: Some(crate::message::ToolChoice::Specific {
            function_names: vec!["a".to_string(), "b".to_string()],
        }),
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    };
    request.tools = vec![crate::completion::ToolDefinition {
        name: "lookup".to_string(),
        description: String::new(),
        parameters: serde_json::json!({}),
    }];

    let converted = OpenAICompletionRequest::try_from(OpenAIRequestParams {
        model: SONAR.to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: PerplexityExt::SUPPORTS_RESPONSE_FORMAT,
        supports_image_tool_results: false,
        supports_tools: PerplexityExt::SUPPORTS_TOOLS,
    })
    .expect("unsupported tools should be dropped, not an error");

    let json = serde_json::to_value(converted).expect("request should serialize");
    assert!(
        json.get("tools")
            .is_none_or(|tools| tools.as_array().is_none_or(|tools| tools.is_empty()))
    );
    assert!(json.get("tool_choice").is_none());
}

#[test]
fn perplexity_finalize_strips_tool_history_and_preserves_alternation() {
    let mut body = serde_json::json!({
        "model": SONAR,
        "messages": [
            {"role": "user", "content": "Look it up."},
            {"role": "assistant", "tool_calls": [
                {"id": "call_1", "type": "function", "function": {"name": "lookup", "arguments": "{}"}}
            ]},
            {"role": "tool", "tool_call_id": "call_1", "content": "result"},
            {"role": "assistant", "content": "It is crimson.", "reasoning_content": "hmm"},
            {"role": "user", "content": "Thanks!"}
        ]
    });

    PerplexityExt
        .finalize_request_body(&mut body)
        .expect("finalize should succeed");

    let messages = body["messages"].as_array().expect("messages array");
    let roles = messages
        .iter()
        .map(|m| m["role"].as_str().unwrap_or_default())
        .collect::<Vec<_>>();
    assert_eq!(roles, ["user", "assistant", "user"]);
    assert_eq!(messages[1]["content"], "It is crimson.");
    assert!(messages[1].get("reasoning_content").is_none());
    assert!(messages[1].get("tool_calls").is_none());
}

#[test]
fn perplexity_prepare_request_drops_tools() {
    let request = crate::completion::CompletionRequestBuilder::new(
        MockCompletionModel::default(),
        "What's new today?",
    )
    .tool(crate::completion::ToolDefinition {
        name: "lookup".to_string(),
        description: "Lookup".to_string(),
        parameters: serde_json::json!({"type":"object","properties":{},"required":[]}),
    })
    .tool_choice(crate::message::ToolChoice::Required)
    .build();

    let mut request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
        model: SONAR.to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: PerplexityExt::SUPPORTS_RESPONSE_FORMAT,
        supports_image_tool_results: false,
        supports_tools: false,
    })
    .expect("request should convert");
    PerplexityExt
        .prepare_request(&mut request)
        .expect("prepare_request should succeed");

    let body = serde_json::to_value(request).expect("request should serialize");
    assert!(body.get("tools").is_none());
    assert!(body.get("tool_choice").is_none());
}
