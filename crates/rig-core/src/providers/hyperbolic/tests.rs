#[test]
fn hyperbolic_prepare_request_drops_tools_and_tool_choice() {
    use crate::providers::openai::completion::{
        CompletionRequest as OpenAICompletionRequest, OpenAICompatibleProvider, OpenAIRequestParams,
    };

    let request = crate::completion::CompletionRequestBuilder::new(
        crate::test_utils::MockCompletionModel::default(),
        "hello",
    )
    .tool(crate::completion::ToolDefinition {
        name: "lookup".to_string(),
        description: "Lookup".to_string(),
        parameters: serde_json::json!({"type":"object","properties":{},"required":[]}),
    })
    .tool_choice(crate::message::ToolChoice::Required)
    .output_schema(schemars::schema_for!(serde_json::Value))
    .build();

    let mut request = OpenAICompletionRequest::try_from(OpenAIRequestParams {
        model: "meta-llama/Meta-Llama-3.1-8B-Instruct".to_string(),
        request,
        strict_tools: false,
        tool_result_array_content: false,
        supports_response_format: super::HyperbolicExt::SUPPORTS_RESPONSE_FORMAT,
        supports_image_tool_results: false,
        supports_tools: false,
    })
    .expect("request should convert");
    super::HyperbolicExt
        .prepare_request(&mut request)
        .expect("prepare_request should succeed");

    let body = serde_json::to_value(request).expect("request should serialize");
    assert!(body.get("tools").is_none());
    assert!(body.get("tool_choice").is_none());
    assert!(body.get("response_format").is_none());
}

#[test]
fn test_client_initialization() {
    let _client = crate::providers::hyperbolic::Client::new_with(
        "dummy-key",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed");
    let builder: crate::providers::hyperbolic::ClientBuilder =
        crate::providers::hyperbolic::Client::builder().api_key("dummy-key");
    let _client_from_builder = builder
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");
}

#[tokio::test]
async fn completion_non_success_preserves_status_and_body() {
    use crate::client::CompletionClient;
    use crate::completion::{CompletionError, CompletionModel};
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"boom"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = super::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model(super::LLAMA_3_1_8B);
    let request = model.completion_request("hello").build();

    let error = model
        .completion(request)
        .await
        .expect_err("completion should fail with non-success status");

    assert!(matches!(error, CompletionError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn completion_2xx_error_envelope_preserves_status_and_body() {
    use crate::client::CompletionClient;
    use crate::completion::{CompletionError, CompletionModel};
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"message":"boom"}"#;
    let http_client = RecordingHttpClient::new(body); // 200 OK
    let client = super::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.completion_model(super::LLAMA_3_1_8B);
    let request = model.completion_request("hello").build();

    let error = model
        .completion(request)
        .await
        .expect_err("completion should fail with provider error envelope");

    match &error {
        CompletionError::ProviderResponse(stored) => {
            assert_eq!(stored.body, body);
            assert_eq!(stored.status, Some(http::StatusCode::OK));
        }
        other => panic!("expected ProviderResponse, got {other:?}"),
    }
}

#[cfg(feature = "image")]
#[tokio::test]
async fn image_generation_non_success_preserves_status_and_body() {
    use crate::client::image_generation::ImageGenerationClient;
    use crate::image_generation::{
        ImageGenerationError, ImageGenerationModel as _, ImageGenerationRequest,
    };
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"boom"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = super::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.image_generation_model(super::SDXL1_0_BASE);

    let request = ImageGenerationRequest {
        prompt: "draw a cat".to_string(),
        width: 256,
        height: 256,
        additional_params: None,
    };

    let error = model
        .image_generation(request)
        .await
        .expect_err("image generation should fail with non-success status");

    assert!(matches!(error, ImageGenerationError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[cfg(feature = "image")]
#[tokio::test]
async fn image_generation_2xx_error_envelope_preserves_status_and_body() {
    use crate::client::image_generation::ImageGenerationClient;
    use crate::image_generation::{
        ImageGenerationError, ImageGenerationModel as _, ImageGenerationRequest,
    };
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"message":"boom"}"#;
    let http_client = RecordingHttpClient::new(body); // 200 OK
    let client = super::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.image_generation_model(super::SDXL1_0_BASE);

    let request = ImageGenerationRequest {
        prompt: "draw a cat".to_string(),
        width: 256,
        height: 256,
        additional_params: None,
    };

    let error = model
        .image_generation(request)
        .await
        .expect_err("image generation should fail with provider error envelope");

    match &error {
        ImageGenerationError::ProviderResponse(stored) => {
            assert_eq!(stored.body, body);
            assert_eq!(stored.status, Some(http::StatusCode::OK));
        }
        other => panic!("expected ProviderResponse, got {other:?}"),
    }
}

#[cfg(feature = "audio")]
#[tokio::test]
async fn audio_generation_non_success_preserves_status_and_body() {
    use crate::audio_generation::{
        AudioGenerationError, AudioGenerationModel as _, AudioGenerationRequest,
    };
    use crate::client::audio_generation::AudioGenerationClient;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"boom"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
    let client = super::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.audio_generation_model("EN");

    let request = AudioGenerationRequest {
        text: "hello".to_string(),
        voice: "default".to_string(),
        speed: 1.0,
        additional_params: None,
    };

    let error = model
        .audio_generation(request)
        .await
        .expect_err("audio generation should fail with non-success status");

    assert!(matches!(error, AudioGenerationError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::SERVICE_UNAVAILABLE)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[cfg(feature = "audio")]
#[tokio::test]
async fn audio_generation_2xx_error_envelope_preserves_status_and_body() {
    use crate::audio_generation::{
        AudioGenerationError, AudioGenerationModel as _, AudioGenerationRequest,
    };
    use crate::client::audio_generation::AudioGenerationClient;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"message":"boom"}"#;
    let http_client = RecordingHttpClient::new(body); // 200 OK
    let client = super::Client::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.audio_generation_model("EN");

    let request = AudioGenerationRequest {
        text: "hello".to_string(),
        voice: "default".to_string(),
        speed: 1.0,
        additional_params: None,
    };

    let error = model
        .audio_generation(request)
        .await
        .expect_err("audio generation should fail with provider error envelope");

    match &error {
        AudioGenerationError::ProviderResponse(stored) => {
            assert_eq!(stored.body, body);
            assert_eq!(stored.status, Some(http::StatusCode::OK));
        }
        other => panic!("expected ProviderResponse, got {other:?}"),
    }
}
