use crate::providers::openai::completion::{
    CompletionRequest as OpenAICompletionRequest, OpenAICompatibleProvider, OpenAIRequestParams,
};
use crate::{completion::CompletionRequestBuilder, test_utils::MockCompletionModel};

/// An OpenAI-style nested error body (`{"error": {"message": ...}}`) on a
/// 2xx status must classify as the error envelope — not fail both untagged
/// arms and surface as a serde error that loses the provider body.
#[test]
fn nested_error_object_parses_as_the_error_envelope() {
    #[derive(serde::Deserialize)]
    struct Success {
        #[allow(dead_code)]
        choices: Vec<serde_json::Value>,
    }

    let nested = r#"{"error":{"message":"model not found","type":"invalid_request_error"}}"#;
    match serde_json::from_str::<super::ApiResponse<Success>>(nested)
        .expect("nested error envelope should deserialize")
    {
        super::ApiResponse::Err(err) => assert!(err.message.contains("model not found")),
        super::ApiResponse::Ok(_) => panic!("error body must classify as the error envelope"),
    }

    let plain = r#"{"error":"over capacity"}"#;
    match serde_json::from_str::<super::ApiResponse<Success>>(plain)
        .expect("string error envelope should deserialize")
    {
        super::ApiResponse::Err(err) => assert_eq!(err.message, "over capacity"),
        super::ApiResponse::Ok(_) => panic!("error body must classify as the error envelope"),
    }
}

#[test]
fn groq_request_maps_output_schema_max_tokens_and_specific_tool_choice() {
    let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Return JSON")
        .max_tokens(64)
        .tool(crate::completion::ToolDefinition {
            name: "choose_beta".to_string(),
            description: "Choose beta".to_string(),
            parameters: serde_json::json!({"type":"object","properties":{},"required":[]}),
        })
        .tool_choice(crate::message::ToolChoice::Specific {
            function_names: vec!["choose_beta".to_string()],
        })
        .output_schema(schemars::schema_for!(serde_json::Value))
        .build();

    let request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
        model: "llama-3.3-70b-versatile".to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: true,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("Groq request should convert");
    let json = serde_json::to_value(request).expect("request should serialize");

    assert_eq!(json["max_tokens"], 64);
    assert_eq!(
        json["tool_choice"],
        serde_json::json!({"type":"function","function":{"name":"choose_beta"}})
    );
    // The shared path defers `response_format` while tools are present and
    // no tool result exists yet (see `should_apply_response_format`).
    assert_eq!(json["response_format"], serde_json::Value::Null);

    let no_tools_request =
        CompletionRequestBuilder::new(MockCompletionModel::default(), "Return JSON")
            .output_schema(schemars::schema_for!(serde_json::Value))
            .build();
    let no_tools_request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
        model: "llama-3.3-70b-versatile".to_string(),
        request: no_tools_request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: true,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request should convert");
    let json = serde_json::to_value(no_tools_request).expect("request should serialize");
    assert_eq!(json["response_format"]["type"], "json_schema");
    assert_eq!(json["response_format"]["json_schema"]["strict"], true);
}

#[tokio::test]
async fn transcription_routes_model_in_multipart_body() {
    use crate::client::transcription::TranscriptionClient;
    use crate::test_utils::RecordingHttpClient;
    use crate::transcription::TranscriptionModel as _;

    let http_client = RecordingHttpClient::new(r#"{"text":"transcribed"}"#);
    let client = super::Client::builder()
        .api_key("test-key")
        .http_client(http_client.clone())
        .build()
        .expect("build client");
    let model = client.transcription_model(super::WHISPER_LARGE_V3);

    let response = model
        .transcription_request()
        .data(vec![1, 2, 3])
        .filename(Some("audio.mp3".to_owned()))
        .send()
        .await
        .expect("transcription should succeed");

    assert_eq!(response.text, "transcribed");
    let request = http_client
        .requests()
        .into_iter()
        .next()
        .expect("request should be captured");
    assert_eq!(
        request.uri,
        "https://api.groq.com/openai/v1/audio/transcriptions"
    );
    let body = String::from_utf8_lossy(&request.body);
    assert!(
        body.contains("name=\"model\"\r\n\r\nwhisper-large-v3\r\n"),
        "{body}"
    );
    assert!(
        body.contains("name=\"file\"; filename=\"audio.mp3\""),
        "{body}"
    );
}

#[test]
fn groq_prepare_request_merges_native_tools_into_compound_custom() {
    let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "search")
        .tool(crate::completion::ToolDefinition {
            name: "local_tool".to_string(),
            description: "A local function tool".to_string(),
            parameters: serde_json::json!({"type":"object","properties":{},"required":[]}),
        })
        .additional_params(serde_json::json!({
            "tools": [{"type": "browser_search"}, {"type": "browser_search"}],
        }))
        .build();

    let mut request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
        model: "llama-3.3-70b-versatile".to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: true,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request should convert");

    super::GroqExt
        .prepare_request(&mut request)
        .expect("prepare_request should succeed");

    let json = serde_json::to_value(request).expect("request should serialize");
    assert_eq!(
        json["compound_custom"]["enabled_tools"],
        serde_json::json!([{"type": "browser_search"}])
    );
    // The rig-level function tool array must survive the native-tool merge.
    assert_eq!(json["tools"][0]["function"]["name"], "local_tool");
}

#[test]
fn groq_reasoning_params_flatten_into_request_body() {
    let additional_params = serde_json::to_value(super::GroqAdditionalParameters {
        reasoning_format: Some(super::ReasoningFormat::Parsed),
        include_reasoning: Some(true),
        extra: None,
    })
    .expect("params should serialize");
    let request = CompletionRequestBuilder::new(MockCompletionModel::default(), "Think about it")
        .additional_params(additional_params)
        .build();

    let request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
        model: "llama-3.3-70b-versatile".to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: true,
        supports_image_tool_results: false,
        supports_tools: true,
    })
    .expect("request should convert");
    let json = serde_json::to_value(request).expect("request should serialize");

    assert_eq!(json["reasoning_format"], "parsed");
    assert_eq!(json["include_reasoning"], true);
}

#[test]
fn test_client_initialization() {
    let _client = crate::providers::groq::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed");
    let builder: crate::providers::groq::ClientBuilder =
        crate::providers::groq::Client::builder().api_key("dummy-key");
    let _client_from_builder = builder
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");
}

#[tokio::test]
async fn completion_preserves_raw_provider_error_json_on_api_error_envelope() {
    use crate::client::CompletionClient;
    use crate::completion::{CompletionError, CompletionModel};
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"message":"model overloaded","type":"server_error","code":"503"}"#;
    let http_client = RecordingHttpClient::with_error_response(http::StatusCode::ACCEPTED, body);
    let client = super::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model("llama-3.3-70b-versatile");
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
            let json = error
                .provider_response_json()
                .expect("raw body should be valid JSON")
                .expect("parsed JSON should be present");
            assert_eq!(json["code"], "503");
        }
        other => panic!("expected ProviderResponse, got {other:?}"),
    }
}

#[tokio::test]
async fn completion_http_non_success_preserves_status_and_body() {
    use crate::client::CompletionClient;
    use crate::completion::{CompletionError, CompletionModel};
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"service unavailable","code":"503"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = super::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model("llama-3.3-70b-versatile");
    let request = model.completion_request("hello").build();

    let error = model
        .completion(request)
        .await
        .expect_err("completion should fail with non-success status");

    // rig#2314: a provider with a request-id contract preserves its
    // non-success responses as ProviderResponse, so the transport id has
    // a home on the error; this mock sent no header, so the id is None.
    assert!(matches!(error, CompletionError::ProviderResponse(_)));
    assert_eq!(error.provider_request_id(), None);
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn transcription_http_non_success_preserves_status_and_body() {
    use crate::client::transcription::TranscriptionClient;
    use crate::test_utils::RecordingHttpClient;
    use crate::transcription::{TranscriptionError, TranscriptionModel as _};

    let body = r#"{"error":{"message":"bad audio","code":"400"}}"#;
    let http_client = RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
    let client = super::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.transcription_model("whisper-large-v3");

    let Err(error) = model
        .transcription_request()
        .data(vec![0u8; 16])
        .send()
        .await
    else {
        panic!("transcription should fail with non-success status")
    };

    assert!(matches!(error, TranscriptionError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::BAD_REQUEST)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}
