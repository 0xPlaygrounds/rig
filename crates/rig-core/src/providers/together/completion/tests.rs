use crate::client::CompletionClient;
use crate::completion::{CompletionError, CompletionModel};
use crate::message;
use crate::providers::openai::completion::{
    CompletionRequest as OpenAICompletionRequest, OpenAIRequestParams,
};
use crate::test_utils::RecordingHttpClient;

use super::super::client::Client;

#[tokio::test]
async fn completion_preserves_raw_provider_error_json_on_api_error_envelope() {
    let body = r#"{"error":"model unavailable","code":"model_overloaded"}"#;
    let http_client = RecordingHttpClient::with_error_response(http::StatusCode::ACCEPTED, body);
    let client = Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model("meta-llama/Meta-Llama-3-70B-Instruct-Turbo");
    let request = model.completion_request("hello").build();

    let error = model
        .completion(request)
        .await
        .expect_err("completion should fail with provider error envelope");

    match &error {
        CompletionError::ProviderResponse(stored) => {
            assert_eq!(stored.body, body);
            assert_eq!(stored.status, Some(http::StatusCode::ACCEPTED));
            assert_eq!(error.provider_response_body(), Some(body));
            assert_eq!(
                error.provider_response_status(),
                Some(http::StatusCode::ACCEPTED)
            );
            let json = error
                .provider_response_json()
                .expect("raw body should be valid JSON")
                .expect("parsed JSON should be present");
            assert_eq!(json["code"], "model_overloaded");
            assert_eq!(json["error"], "model unavailable");
        }
        other => panic!("expected ProviderResponse, got {other:?}"),
    }
}

#[test]
fn together_request_conversion_errors_when_all_messages_are_filtered() {
    let request = crate::completion::CompletionRequest {
        chat_history: vec![message::Message::Assistant {
            id: None,
            content: vec![message::AssistantContent::reasoning("hidden")],
        }],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        model: None,
        output_schema: None,
        record_telemetry_content: false,
    };

    let result = OpenAICompletionRequest::try_from(OpenAIRequestParams {
        model: "meta-llama/test-model".to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: false,
        supports_image_tool_results: false,
        supports_tools: true,
    });
    assert!(matches!(result, Err(CompletionError::RequestError(_))));
}
