use super::*;
use crate::client::embeddings::EmbeddingsClient;
use crate::completion::{CompletionError, CompletionRequest};
use crate::embeddings::EmbeddingError;

#[cfg(any(feature = "image", feature = "audio"))]
fn test_client(
    http_client: crate::test_utils::RecordingHttpClient,
) -> Client<crate::test_utils::RecordingHttpClient> {
    Client::builder()
        .api_key("test-key")
        .azure_endpoint("https://example.openai.azure.com".to_string())
        .http_client(http_client)
        .build()
        .expect("build client")
}

#[cfg(feature = "image")]
#[tokio::test]
async fn image_generation_client_routes_to_the_deployment() {
    use crate::client::image_generation::ImageGenerationClient;
    use crate::image_generation::{ImageGenerationModel as _, ImageGenerationRequest};
    use crate::test_utils::RecordingHttpClient;

    let http_client = RecordingHttpClient::new(r#"{"created":0,"data":[{"b64_json":"aW1hZ2U="}]}"#);
    let client = test_client(http_client.clone());
    let model = client.image_generation_model("image-deployment");

    let response = model
        .image_generation(ImageGenerationRequest {
            prompt: "draw a cat".to_owned(),
            width: 256,
            height: 256,
            additional_params: None,
        })
        .await
        .expect("image generation should succeed");

    assert_eq!(response.image, b"image");
    let requests = http_client.requests();
    assert_eq!(
        requests[0].uri,
        "https://example.openai.azure.com/openai/deployments/image-deployment/images/generations?api-version=2024-10-21"
    );
    let body: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
    assert!(body.get("model").is_none());
    assert_eq!(body["response_format"], "b64_json");
}

#[cfg(feature = "image")]
#[tokio::test]
async fn image_generation_non_success_response_preserves_status_and_body() {
    use crate::client::image_generation::ImageGenerationClient;
    use crate::image_generation::{
        ImageGenerationError, ImageGenerationModel as ImageGenerationModelTrait,
        ImageGenerationRequest,
    };
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"invalid image request"}}"#;
    let http_client = RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
    let model = test_client(http_client).image_generation_model("dall-e-3");

    let error = model
        .image_generation(ImageGenerationRequest {
            prompt: "draw a cat".to_string(),
            width: 256,
            height: 256,
            additional_params: None,
        })
        .await
        .expect_err("image generation should fail with non-success status");

    assert!(matches!(error, ImageGenerationError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::BAD_REQUEST)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[cfg(feature = "audio")]
#[test]
fn audio_api_version_can_be_overridden() {
    let client = Client::builder()
        .api_key("test-key")
        .azure_endpoint("https://example.openai.azure.com".to_owned())
        .audio_api_version("2026-01-01-preview")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("build client");
    let request = client
        .post_audio_generation("tts-deployment")
        .expect("build audio request")
        .body(Vec::<u8>::new())
        .expect("finish audio request");

    assert_eq!(
        request.uri(),
        "https://example.openai.azure.com/openai/deployments/tts-deployment/audio/speech?api-version=2026-01-01-preview"
    );
}

#[cfg(feature = "audio")]
#[tokio::test]
async fn audio_generation_routes_to_the_deployment() {
    use crate::audio_generation::{AudioGenerationModel as _, AudioGenerationRequest};
    use crate::client::audio_generation::AudioGenerationClient;
    use crate::test_utils::RecordingHttpClient;

    let http_client = RecordingHttpClient::new("audio");
    let client = test_client(http_client.clone());
    let model = client.audio_generation_model("tts-deployment");

    let response = model
        .audio_generation(AudioGenerationRequest {
            text: "hello".to_owned(),
            voice: "alloy".to_owned(),
            speed: 1.0,
            additional_params: None,
        })
        .await
        .expect("audio generation should succeed");

    assert_eq!(response.audio, b"audio");
    let requests = http_client.requests();
    assert_eq!(
        requests[0].uri,
        "https://example.openai.azure.com/openai/deployments/tts-deployment/audio/speech?api-version=2025-04-01-preview"
    );
    let body: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
    assert!(body.get("model").is_none());
    assert_eq!(body["input"], "hello");
    assert_eq!(body["voice"], "alloy");
}

#[cfg(feature = "audio")]
#[tokio::test]
async fn audio_generation_non_success_response_preserves_status_and_body() {
    use crate::audio_generation::{
        AudioGenerationError, AudioGenerationModel as _, AudioGenerationRequest,
    };
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"invalid voice"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::UNPROCESSABLE_ENTITY, body);
    let model = AudioGenerationModel::new(test_client(http_client), "tts-1");

    let Err(error) = model
        .audio_generation(AudioGenerationRequest {
            text: "hello".to_string(),
            voice: "alloy".to_string(),
            speed: 1.0,
            additional_params: None,
        })
        .await
    else {
        panic!("audio generation should fail with non-success status")
    };

    assert!(matches!(error, AudioGenerationError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::UNPROCESSABLE_ENTITY)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn transcription_http_non_success_preserves_status_and_body() {
    use crate::test_utils::RecordingHttpClient;
    use crate::transcription::{TranscriptionError, TranscriptionModel as _};

    let body = r#"{"error":{"message":"bad audio","type":"invalid_request_error"}}"#;
    let http_client = RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
    let client = Client::builder()
        .api_key("test-key")
        .azure_endpoint("https://example.openai.azure.com".to_string())
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = TranscriptionModel::new(client, "whisper");

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

#[tokio::test]
async fn transcription_routes_deployment_in_url_not_multipart_body() {
    use crate::test_utils::RecordingHttpClient;
    use crate::transcription::TranscriptionModel as _;

    let http_client = RecordingHttpClient::new(r#"{"text":"transcribed"}"#);
    let client = Client::builder()
        .api_key("test-key")
        .azure_endpoint("https://example.openai.azure.com".to_owned())
        .http_client(http_client.clone())
        .build()
        .expect("build client");
    let model = TranscriptionModel::new(client, "whisper-deployment");

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
        "https://example.openai.azure.com/openai/deployments/whisper-deployment/audio/translations?api-version=2024-10-21"
    );
    let body = String::from_utf8_lossy(&request.body);
    assert!(!body.contains("name=\"model\""), "{body}");
    assert!(
        body.contains("name=\"file\"; filename=\"audio.mp3\""),
        "{body}"
    );
}

#[tokio::test]
async fn embedding_http_non_success_preserves_status_and_body() {
    use crate::embeddings::EmbeddingModel as _;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"bad embedding","type":"invalid_request_error"}}"#;
    let http_client = RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
    let client = Client::builder()
        .api_key("test-key")
        .azure_endpoint("https://example.openai.azure.com".to_string())
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.embedding_model(TEXT_EMBEDDING_3_SMALL);

    let Err(error) = model.embed_texts(vec!["Hello, world!".to_string()]).await else {
        panic!("embedding should fail with non-success status")
    };

    assert!(matches!(error, EmbeddingError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::BAD_REQUEST)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn embedding_preserves_deployment_url_and_body_and_reports_usage() {
    use crate::embeddings::EmbeddingModel as _;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{
            "object": "list",
            "model": "text-embedding-3-small",
            "usage": { "prompt_tokens": 4, "total_tokens": 4 },
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2] }]
        }"#;
    let http_client = RecordingHttpClient::new(body);
    let client = Client::builder()
        .api_key("test-key")
        .azure_endpoint("https://example.openai.azure.com".to_string())
        .http_client(http_client.clone())
        .build()
        .expect("build client");
    let model = client.embedding_model(TEXT_EMBEDDING_3_SMALL);

    let response = model
        .embed_texts_response(vec!["Hello, world!".to_string()])
        .await
        .expect("embedding should succeed");

    // Usage is now surfaced instead of the zero-usage default.
    assert_eq!(response.usage.input_tokens, 4);
    assert_eq!(response.usage.total_tokens, 4);
    assert_eq!(response.embeddings.len(), 1);

    // The deployment stays in the URL and the body carries no `model`
    // field, matching the hand-rolled request this replaced.
    let requests = http_client.requests();
    assert_eq!(
        requests[0].uri,
        format!(
            "https://example.openai.azure.com/openai/deployments/{TEXT_EMBEDDING_3_SMALL}/embeddings?api-version=2024-10-21"
        )
    );
    let request_body: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
    assert_eq!(request_body.get("model"), None);
    assert_eq!(request_body["dimensions"], serde_json::json!(1_536));
    assert_eq!(request_body["input"], serde_json::json!(["Hello, world!"]));
}

#[tokio::test]
async fn completion_pins_deployment_url_under_model_override() {
    use crate::completion::CompletionModel as _;
    use crate::test_utils::RecordingHttpClient;

    // The error response keeps the test independent of response parsing;
    // only the captured request matters here.
    let http_client = RecordingHttpClient::with_error_response(
        http::StatusCode::BAD_REQUEST,
        r#"{"error":{"message":"x"}}"#,
    );
    let client = Client::builder()
        .api_key("test-key")
        .azure_endpoint("https://example.openai.azure.com".to_string())
        .http_client(http_client.clone())
        .build()
        .expect("build client");
    let model = super::CompletionModel::new(client, GPT_4O_MINI);

    let _ = model
        .completion(CompletionRequest {
            model: Some("other-deployment".to_string()),
            chat_history: vec!["Hello!".into()],
            documents: vec![],
            max_tokens: None,
            temperature: None,
            tools: vec![],
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        })
        .await;

    let requests = http_client.requests();
    let request = requests.first().expect("request should be captured");
    // The deployment URL stays pinned to the configured model; the
    // override only changes the body.
    assert!(
        request
            .uri
            .contains("/openai/deployments/gpt-4o-mini/chat/completions"),
        "unexpected uri: {}",
        request.uri
    );
    let body: serde_json::Value =
        serde_json::from_slice(&request.body).expect("captured body should be JSON");
    assert_eq!(body["model"], "other-deployment");
}

#[tokio::test]
async fn completion_http_non_success_preserves_status_and_body() {
    use crate::completion::CompletionModel as _;
    use crate::test_utils::RecordingHttpClient;

    let body = r#"{"error":{"message":"bad completion","type":"invalid_request_error"}}"#;
    let http_client = RecordingHttpClient::with_error_response(http::StatusCode::BAD_REQUEST, body);
    let client = Client::builder()
        .api_key("test-key")
        .azure_endpoint("https://example.openai.azure.com".to_string())
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = super::CompletionModel::new(client, GPT_4O_MINI);

    let Err(error) = model
        .completion(CompletionRequest {
            model: None,
            chat_history: vec![
                crate::message::Message::system("You are a helpful assistant."),
                "Hello!".into(),
            ],
            documents: vec![],
            max_tokens: Some(100),
            temperature: Some(0.0),
            tools: vec![],
            tool_choice: None,
            additional_params: None,
            output_schema: None,
            record_telemetry_content: false,
        })
        .await
    else {
        panic!("completion should fail with non-success status")
    };

    assert!(matches!(error, CompletionError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::BAD_REQUEST)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}

#[tokio::test]
async fn test_client_initialization() {
    let _client = crate::providers::azure::Client::builder()
        .api_key("test")
        .azure_endpoint("test".to_string()) // add your endpoint here!
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");
}
