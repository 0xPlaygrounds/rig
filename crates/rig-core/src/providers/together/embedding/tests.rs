use super::BGE_BASE_EN_V1_5;
use crate::client::EmbeddingsClient;
use crate::embeddings::{EmbeddingError, EmbeddingModel as _};
use crate::providers::{openai::embedding::EncodingFormat, together};
use crate::test_utils::RecordingHttpClient;

const RESPONSE_BODY: &str = r#"{
        "object": "list",
        "model": "BAAI/bge-base-en-v1.5",
        "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3] }]
    }"#;

fn client(http_client: RecordingHttpClient) -> together::Client<RecordingHttpClient> {
    together::Client::builder()
        .api_key("dummy-key")
        .http_client(http_client)
        .build()
        .expect("client should build")
}

#[tokio::test]
async fn together_embeddings_send_dimensions_to_v1_path() {
    let http_client = RecordingHttpClient::new(RESPONSE_BODY);
    let model = client(http_client.clone()).embedding_model_with_ndims(BGE_BASE_EN_V1_5, 3);

    let response = model
        .embed_texts_response(["hello".to_string()])
        .await
        .expect("embedding request should succeed");

    assert_eq!(response.usage.total_tokens, 0);
    let requests = http_client.requests();
    assert_eq!(requests.len(), 1);
    assert!(requests[0].uri.ends_with("/v1/embeddings"));
    let body: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
    assert_eq!(body["dimensions"], serde_json::json!(3));
    assert_eq!(body["model"], BGE_BASE_EN_V1_5);
}

#[tokio::test]
async fn together_embeddings_omit_dimensions_when_unset() {
    let http_client = RecordingHttpClient::new(RESPONSE_BODY);
    let model = client(http_client.clone()).embedding_model(BGE_BASE_EN_V1_5);

    model
        .embed_texts(["hello".to_string()])
        .await
        .expect("embedding request should succeed");

    let body: serde_json::Value = serde_json::from_slice(&http_client.requests()[0].body)
        .expect("request body should be JSON");
    assert!(body.get("dimensions").is_none());
}

#[tokio::test]
async fn together_rejects_unsupported_openai_parameters_before_sending() {
    let cases = [
        (RecordingHttpClient::new(RESPONSE_BODY), "encoding_format"),
        (RecordingHttpClient::new(RESPONSE_BODY), "user"),
    ];

    for (http_client, parameter) in cases {
        let model = match parameter {
            "encoding_format" => client(http_client.clone())
                .embedding_model(BGE_BASE_EN_V1_5)
                .encoding_format(EncodingFormat::Float),
            "user" => client(http_client.clone())
                .embedding_model(BGE_BASE_EN_V1_5)
                .user("user-123"),
            _ => unreachable!("test case must use a supported parameter name"),
        };
        let error = model
            .embed_texts(["hello".to_string()])
            .await
            .expect_err("unsupported parameter should fail");
        assert!(matches!(
            error,
            EmbeddingError::UnsupportedParameter {
                provider: "together",
                parameter: actual
            } if actual == parameter
        ));
        assert!(http_client.requests().is_empty());
    }
}

#[tokio::test]
async fn together_rejects_base64_consistently_before_sending() {
    let http_client = RecordingHttpClient::new(RESPONSE_BODY);
    let model = client(http_client.clone())
        .embedding_model(BGE_BASE_EN_V1_5)
        .encoding_format(EncodingFormat::Base64);

    let error = model
        .embed_texts(["hello".to_string()])
        .await
        .expect_err("numeric response parser should reject base64");

    assert!(matches!(
        error,
        EmbeddingError::UnsupportedResponseEncoding {
            provider: "together",
            encoding_format: "base64"
        }
    ));
    assert!(http_client.requests().is_empty());
}

#[tokio::test]
async fn together_error_envelope_preserves_raw_response() {
    let body = r#"{"error":{"message":"invalid model"},"code":"invalid_request"}"#;
    let http_client = RecordingHttpClient::with_error_response(http::StatusCode::ACCEPTED, body);
    let model = client(http_client).embedding_model(BGE_BASE_EN_V1_5);

    let error = model
        .embed_texts(["hello".to_string()])
        .await
        .expect_err("provider error envelope should fail");

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::ACCEPTED)
    );
}

#[tokio::test]
async fn together_non_success_preserves_raw_response() {
    let body = r#"{"error":{"message":"invalid api key"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::UNAUTHORIZED, body);
    let model = client(http_client).embedding_model(BGE_BASE_EN_V1_5);

    let error = model
        .embed_texts(["hello".to_string()])
        .await
        .expect_err("non-success response should fail");

    assert_eq!(error.provider_response_body(), Some(body));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::UNAUTHORIZED)
    );
}
