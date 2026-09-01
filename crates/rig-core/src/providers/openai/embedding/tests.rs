use super::*;
use crate::client::EmbeddingsClient;
use crate::embeddings::EmbeddingModel as _;
use crate::http_client::{LazyBody, MultipartForm, Request, Response, StreamingResponse};
use crate::providers::openai::CompletionsClient;
use crate::test_utils::RecordingHttpClient;
use bytes::Bytes;
use std::future::{self, Future};

#[derive(Clone)]
struct CustomHttpClient;

impl HttpClientExt for CustomHttpClient {
    fn send<T, U>(
        &self,
        _req: Request<T>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        T: Into<Bytes> + WasmCompatSend,
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        future::ready(Err(http_client::Error::StreamEnded))
    }

    fn send_multipart<U>(
        &self,
        _req: Request<MultipartForm>,
    ) -> impl Future<Output = http_client::Result<Response<LazyBody<U>>>> + WasmCompatSend + 'static
    where
        U: From<Bytes> + WasmCompatSend + 'static,
    {
        future::ready(Err(http_client::Error::StreamEnded))
    }

    fn send_streaming<T>(
        &self,
        _req: Request<T>,
    ) -> impl Future<Output = http_client::Result<StreamingResponse>> + WasmCompatSend
    where
        T: Into<Bytes> + WasmCompatSend,
    {
        future::ready(Err(http_client::Error::StreamEnded))
    }
}

const RESPONSE_BODY: &str = r#"{
        "object": "list",
        "model": "text-embedding-3-small",
        "usage": { "prompt_tokens": 4, "total_tokens": 4 },
        "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2] }]
    }"#;

#[test]
fn embedding_model_accepts_backend_without_default_or_debug() {
    let client = CompletionsClient::builder()
        .api_key("test-key")
        .http_client(CustomHttpClient)
        .build()
        .expect("build client");

    let model = client.embedding_model(TEXT_EMBEDDING_3_SMALL);

    assert_eq!(model.ndims(), 1_536);
}

#[tokio::test]
async fn openai_embeddings_preserve_path_parameters_and_usage() {
    let http_client = RecordingHttpClient::new(RESPONSE_BODY);
    let client = CompletionsClient::builder()
        .api_key("test-key")
        .http_client(http_client.clone())
        .build()
        .expect("build client");
    let model = client
        .embedding_model(TEXT_EMBEDDING_3_SMALL)
        .encoding_format(EncodingFormat::Float)
        .user("user-123");

    let response = model
        .embed_texts_response(["hello".to_string()])
        .await
        .expect("embedding should succeed");

    assert_eq!(response.usage.input_tokens, 4);
    assert_eq!(response.usage.total_tokens, 4);
    let requests = http_client.requests();
    assert_eq!(requests[0].uri, "https://api.openai.com/v1/embeddings");
    let body: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
    assert_eq!(body["dimensions"], serde_json::json!(1_536));
    assert_eq!(body["encoding_format"], serde_json::json!("float"));
    assert_eq!(body["user"], serde_json::json!("user-123"));
}

/// The normalized response carries the provider name, the wire's model,
/// the `x-request-id` transport id, and the raw payload — which
/// round-trips back to the provider type.
#[tokio::test]
async fn openai_embeddings_normalize_identity_and_raw() {
    let mut headers = http::HeaderMap::new();
    headers.insert("x-request-id", "req_embed_0001".parse().expect("header"));
    // `with_error_response_headers` with 200 is the one unary double that
    // carries response headers.
    let http_client = RecordingHttpClient::with_error_response_headers(
        http::StatusCode::OK,
        RESPONSE_BODY,
        headers,
    );
    let client = CompletionsClient::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.embedding_model(TEXT_EMBEDDING_3_SMALL);

    let response = model
        .embed_texts_response(["hello".to_string()])
        .await
        .expect("embedding should succeed");

    assert_eq!(response.provider, "openai");
    assert_eq!(response.model.as_deref(), Some("text-embedding-3-small"));
    assert_eq!(
        response.provider_request_id.as_deref(),
        Some("req_embed_0001")
    );
    assert_eq!(
        response.identity().provider_request_id.as_deref(),
        Some("req_embed_0001")
    );
    assert_eq!(response.embeddings.len(), 1);
    assert_eq!(response.embeddings[0].document, "hello");

    let raw: CompatibleEmbeddingResponse =
        serde_json::from_value(response.raw.clone()).expect("raw round-trips");
    assert_eq!(raw.data.len(), 1);
    assert_eq!(raw.usage.map(|usage| usage.total_tokens), Some(4));

    let typed = model
        .raw_embed_texts(["hello".to_string()])
        .await
        .expect("raw route should succeed");
    assert_eq!(typed.model, "text-embedding-3-small");
}

/// A width of zero is rig's "unknown", not a declaration.
///
/// `default_ndims` returning `None` lands here through
/// `unwrap_or_default()`, and `GenericEmbeddingModel::new(client, model, 0)`
/// is how a caller says they do not know the width. Treating either as a
/// claim would make the width check turn every such handle into a hard
/// error on its first request.
#[tokio::test]
async fn a_zero_width_is_unknown_rather_than_a_claim() {
    let http_client = RecordingHttpClient::new(RESPONSE_BODY);
    let client = CompletionsClient::builder()
        .api_key("test-key")
        .http_client(http_client.clone())
        .build()
        .expect("build client");

    let embeddings = client
        .embedding_model_with_ndims(TEXT_EMBEDDING_3_SMALL, 0)
        .embed_texts(["hello".to_string()])
        .await
        .expect("a zero width must not be checked against the response");

    assert_eq!(
        embeddings[0].vec.len(),
        2,
        "the response's own width is what came back"
    );
}

/// `ada-002` accepts no `dimensions`, so a width asked for is a width the
/// caller will not get — on the wire *and* in the answer.
///
/// Both halves are asserted. The request never carries the field, which is
/// the older claim; and because ada cannot resize, the response comes back
/// at its own width and the declared one is refused rather than reported.
/// Silently keeping `ndims() == 512` beside 1,536-wide vectors is what
/// sizes a vector-store index that cannot hold them.
#[tokio::test]
async fn openai_ada_dimensions_remain_absent_from_the_wire() {
    let http_client = RecordingHttpClient::new(RESPONSE_BODY);
    let client = CompletionsClient::builder()
        .api_key("test-key")
        .http_client(http_client.clone())
        .build()
        .expect("build client");

    let error = client
        .embedding_model_with_ndims(TEXT_EMBEDDING_ADA_002, 512)
        .embed_texts(["hello".to_string()])
        .await
        .expect_err("ada cannot return the width the caller declared");

    let requests = http_client.requests();
    let body: serde_json::Value =
        serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
    assert!(body.get("dimensions").is_none());

    assert!(
        matches!(
            error,
            EmbeddingError::MismatchedDimensions {
                provider: "openai",
                requested: 512,
                ..
            }
        ),
        "{error}"
    );
}

#[tokio::test]
async fn openai_rejects_base64_before_sending() {
    let http_client = RecordingHttpClient::new(RESPONSE_BODY);
    let client = CompletionsClient::builder()
        .api_key("test-key")
        .http_client(http_client.clone())
        .build()
        .expect("build client");
    let model = client
        .embedding_model(TEXT_EMBEDDING_3_SMALL)
        .encoding_format(EncodingFormat::Base64);

    let error = model
        .embed_texts(["hello".to_string()])
        .await
        .expect_err("numeric response parser should reject base64");

    assert!(matches!(
        error,
        EmbeddingError::UnsupportedResponseEncoding {
            provider: "openai",
            encoding_format: "base64"
        }
    ));
    assert!(http_client.requests().is_empty());
}

#[test]
fn public_openai_embedding_response_requires_usage() {
    let body = r#"{
            "object": "list",
            "model": "text-embedding-3-small",
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1] }]
        }"#;

    assert!(serde_json::from_str::<EmbeddingResponse>(body).is_err());
}

#[tokio::test]
async fn embedding_preserves_raw_provider_error_json_on_api_error_envelope() {
    let body = r#"{"message":"embedding quota exceeded","type":"insufficient_quota"}"#;
    let http_client = RecordingHttpClient::with_error_response(http::StatusCode::ACCEPTED, body);
    let client = CompletionsClient::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.embedding_model("text-embedding-3-small");

    let error = model
        .embed_texts(["hello".to_string()])
        .await
        .expect_err("embedding should fail with provider error envelope");

    match &error {
        EmbeddingError::ProviderResponse(stored) => {
            assert_eq!(stored.body, body);
            assert_eq!(stored.status, Some(http::StatusCode::ACCEPTED));
            assert_eq!(error.provider_response_body(), Some(body));
            let json = error
                .provider_response_json()
                .expect("raw body should be valid JSON")
                .expect("parsed JSON should be present");
            assert_eq!(json["type"], "insufficient_quota");
        }
        other => panic!("expected ProviderResponse, got {other:?}"),
    }
}

#[tokio::test]
async fn embedding_http_non_success_preserves_status_and_body() {
    let body = r#"{"error":{"message":"invalid api key","type":"invalid_request_error"}}"#;
    let http_client =
        RecordingHttpClient::with_error_response(http::StatusCode::UNAUTHORIZED, body);
    let client = CompletionsClient::builder()
        .api_key("test-key")
        .http_client(http_client)
        .build()
        .expect("build client");
    let model = client.embedding_model("text-embedding-3-small");

    let error = model
        .embed_texts(["hello".to_string()])
        .await
        .expect_err("embedding should fail with non-success status");

    assert!(matches!(error, EmbeddingError::HttpError(_)));
    assert_eq!(
        error.provider_response_status(),
        Some(http::StatusCode::UNAUTHORIZED)
    );
    assert_eq!(error.provider_response_body(), Some(body));
}
