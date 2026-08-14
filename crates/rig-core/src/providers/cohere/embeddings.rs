use super::{client::ApiResponse, client::Client};
use crate::{
    embeddings::{self, EmbeddingError},
    http_client::HttpClientExt,
    wasm_compat::*,
};
use base64::{
    Engine as _,
    engine::general_purpose::{STANDARD, URL_SAFE_NO_PAD},
};
use serde::Deserialize;
use serde_json::json;
use sha2::{Digest, Sha256};

const MAX_IMAGE_BYTES: usize = 5_000_000;

#[derive(Deserialize)]
pub struct EmbeddingResponse {
    #[serde(default)]
    pub response_type: Option<String>,
    pub id: String,
    pub embeddings: Vec<Vec<serde_json::Number>>,
    pub texts: Vec<String>,
    #[serde(default)]
    pub meta: Option<Meta>,
}

#[derive(Deserialize)]
pub struct Meta {
    pub api_version: ApiVersion,
    pub billed_units: BilledUnits,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Deserialize)]
pub struct ApiVersion {
    pub version: String,
    #[serde(default)]
    pub is_deprecated: Option<bool>,
    #[serde(default)]
    pub is_experimental: Option<bool>,
}

#[derive(Deserialize, Debug)]
pub struct BilledUnits {
    #[serde(default)]
    pub input_tokens: u32,
    #[serde(default)]
    pub output_tokens: u32,
    #[serde(default)]
    pub search_units: u32,
    #[serde(default)]
    pub classifications: u32,
    #[serde(default)]
    pub images: u32,
}

impl std::fmt::Display for BilledUnits {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "Input tokens: {}\nOutput tokens: {}\nSearch units: {}\nClassifications: {}",
            self.input_tokens, self.output_tokens, self.search_units, self.classifications
        )?;
        if self.images > 0 {
            write!(f, "\nImages: {}", self.images)?;
        }
        Ok(())
    }
}

#[derive(Deserialize)]
struct ImageEmbeddingResponse {
    embeddings: FloatEmbeddings,
    #[serde(default)]
    meta: Option<Meta>,
}

#[derive(Deserialize)]
struct FloatEmbeddings {
    #[serde(rename = "float")]
    values: Vec<Vec<serde_json::Number>>,
}

#[derive(Debug, thiserror::Error)]
enum ImageInputError {
    #[error("Cohere image embeddings support PNG, JPEG, WebP, or GIF file bytes")]
    UnsupportedFormat,
    #[error("Cohere image embeddings accept at most 5 MB per image; received {actual_bytes} bytes")]
    TooLarge { actual_bytes: usize },
}

fn image_media_type(bytes: &[u8]) -> Option<&'static str> {
    if bytes.starts_with(b"\x89PNG\r\n\x1a\n") {
        Some("image/png")
    } else if bytes.starts_with(b"\xff\xd8\xff") {
        Some("image/jpeg")
    } else if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
        Some("image/gif")
    } else if bytes.starts_with(b"RIFF") && bytes.get(8..12) == Some(b"WEBP".as_slice()) {
        Some("image/webp")
    } else {
        None
    }
}

fn validate_image(bytes: &[u8]) -> Result<&'static str, EmbeddingError> {
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(EmbeddingError::DocumentError(Box::new(
            ImageInputError::TooLarge {
                actual_bytes: bytes.len(),
            },
        )));
    }

    image_media_type(bytes)
        .ok_or_else(|| EmbeddingError::DocumentError(Box::new(ImageInputError::UnsupportedFormat)))
}

fn image_data_url(bytes: &[u8], media_type: &str) -> String {
    format!("data:{media_type};base64,{}", STANDARD.encode(bytes))
}

fn image_document(bytes: &[u8], media_type: &str) -> String {
    let digest = Sha256::digest(bytes);
    format!("{media_type};sha256={}", URL_SAFE_NO_PAD.encode(digest))
}

#[derive(Clone)]
pub struct EmbeddingModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
    pub input_type: String,
    ndims: usize,
}

/// Cohere `embed-english-v3.0` image embedding model.
///
/// Cohere Embed v3 accepts one image per request, so batch calls are sent as
/// ordered individual requests.
#[derive(Clone)]
pub struct ImageEmbeddingModel<T = reqwest::Client> {
    client: Client<T>,
}

impl<T> embeddings::EmbeddingModel for EmbeddingModel<T>
where
    T: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    const MAX_DOCUMENTS: usize = 96;
    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>, dims: Option<usize>) -> Self {
        let model = model.into();
        let dims = dims
            .or(super::model_dimensions_from_identifier(&model))
            .unwrap_or_default();

        Self::new(client.clone(), model, "search_document", dims)
    }

    fn ndims(&self) -> usize {
        self.ndims
    }

    async fn embed_texts(
        &self,
        documents: impl IntoIterator<Item = String>,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let documents = documents.into_iter().collect::<Vec<_>>();

        let body = json!({
            "model": self.model.to_string(),
            "texts": documents,
            "input_type": self.input_type
        });

        let body = serde_json::to_vec(&body)?;

        let req = self
            .client
            .post("/v1/embed")?
            .body(body)
            .map_err(|e| EmbeddingError::HttpError(e.into()))?;

        let response = self
            .client
            .send::<_, Vec<u8>>(req)
            .await
            .map_err(EmbeddingError::HttpError)?;

        let status = response.status();
        let raw_body = response.into_body().await?;

        if status.is_success() {
            let body: ApiResponse<EmbeddingResponse> = serde_json::from_slice(raw_body.as_slice())?;

            match body {
                ApiResponse::Ok(response) => {
                    match response.meta {
                        Some(meta) => tracing::info!(target: "rig",
                            "Cohere embeddings billed units: {}",
                            meta.billed_units,
                        ),
                        None => tracing::info!(target: "rig",
                            "Cohere embeddings billed units: n/a",
                        ),
                    };

                    if response.embeddings.len() != documents.len() {
                        return Err(EmbeddingError::DocumentError(
                            format!(
                                "Expected {} embeddings, got {}",
                                documents.len(),
                                response.embeddings.len()
                            )
                            .into(),
                        ));
                    }

                    Ok(response
                        .embeddings
                        .into_iter()
                        .zip(documents.into_iter())
                        .map(|(embedding, document)| embeddings::Embedding {
                            document,
                            vec: embedding.into_iter().filter_map(|n| n.as_f64()).collect(),
                        })
                        .collect())
                }
                ApiResponse::Err(error) => {
                    tracing::warn!(
                        message = %error.message,
                        "Cohere returned an error response"
                    );
                    Err(EmbeddingError::from_http_response(
                        status,
                        String::from_utf8_lossy(&raw_body),
                    ))
                }
            }
        } else {
            Err(EmbeddingError::from_http_response(
                status,
                String::from_utf8_lossy(&raw_body),
            ))
        }
    }
}

impl<T> embeddings::ImageEmbeddingModel for ImageEmbeddingModel<T>
where
    T: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    const MAX_DOCUMENTS: usize = 1;

    fn ndims(&self) -> usize {
        1_024
    }

    async fn embed_images(
        &self,
        images: impl IntoIterator<Item = Vec<u8>> + WasmCompatSend,
    ) -> Result<Vec<embeddings::Embedding>, EmbeddingError> {
        let images = images
            .into_iter()
            .map(|bytes| {
                let media_type = validate_image(&bytes)?;
                let document = image_document(&bytes, media_type);
                Ok((bytes, media_type, document))
            })
            .collect::<Result<Vec<_>, EmbeddingError>>()?;
        let mut embeddings = Vec::with_capacity(images.len());

        for (image, media_type, document) in images {
            let data_url = image_data_url(&image, media_type);
            embeddings.push(self.embed_image_data_url(data_url, document).await?);
        }

        Ok(embeddings)
    }
}

impl<T> EmbeddingModel<T> {
    pub fn new(
        client: Client<T>,
        model: impl Into<String>,
        input_type: &str,
        ndims: usize,
    ) -> Self {
        Self {
            client,
            model: model.into(),
            input_type: input_type.to_string(),
            ndims,
        }
    }

    pub fn with_model(client: Client<T>, model: &str, input_type: &str, ndims: usize) -> Self {
        Self {
            client,
            model: model.into(),
            input_type: input_type.into(),
            ndims,
        }
    }
}

impl<T> ImageEmbeddingModel<T> {
    pub(crate) fn new(client: Client<T>) -> Self {
        Self { client }
    }
}

impl<T> ImageEmbeddingModel<T>
where
    T: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    async fn embed_image_data_url(
        &self,
        data_url: String,
        document: String,
    ) -> Result<embeddings::Embedding, EmbeddingError> {
        let body = json!({
            "model": super::EMBED_ENGLISH_V3,
            "images": [&data_url],
            "input_type": "image",
            "embedding_types": ["float"],
        });
        let body = serde_json::to_vec(&body)?;

        let request = self
            .client
            .post("/v1/embed")?
            .body(body)
            .map_err(|error| EmbeddingError::HttpError(error.into()))?;
        let response = self
            .client
            .send::<_, Vec<u8>>(request)
            .await
            .map_err(EmbeddingError::HttpError)?;
        let status = response.status();
        let raw_body = response.into_body().await?;

        if !status.is_success() {
            return Err(EmbeddingError::from_http_response(
                status,
                String::from_utf8_lossy(&raw_body),
            ));
        }

        let body: ApiResponse<ImageEmbeddingResponse> =
            serde_json::from_slice(raw_body.as_slice())?;
        let response = match body {
            ApiResponse::Ok(response) => response,
            ApiResponse::Err(error) => {
                tracing::warn!(
                    message = %error.message,
                    "Cohere returned an error response"
                );
                return Err(EmbeddingError::from_http_response(
                    status,
                    String::from_utf8_lossy(&raw_body),
                ));
            }
        };

        match response.meta {
            Some(meta) => tracing::info!(target: "rig",
                "Cohere embeddings billed units: {}",
                meta.billed_units,
            ),
            None => tracing::info!(target: "rig", "Cohere embeddings billed units: n/a"),
        }

        if response.embeddings.values.len() != 1 {
            return Err(EmbeddingError::DocumentError(
                format!(
                    "Expected 1 image embedding, got {}",
                    response.embeddings.values.len()
                )
                .into(),
            ));
        }

        let vector = response
            .embeddings
            .values
            .into_iter()
            .next()
            .ok_or_else(|| {
                EmbeddingError::ResponseError(
                    "Cohere returned an empty image embedding response".to_string(),
                )
            })?;

        Ok(embeddings::Embedding {
            document,
            vec: vector
                .into_iter()
                .filter_map(|number| number.as_f64())
                .collect(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[tokio::test]
    async fn embeddings_non_success_preserves_status_and_body() {
        use crate::embeddings::EmbeddingModel as _;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"boom"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = crate::providers::cohere::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.embedding_model(
            crate::providers::cohere::EMBED_ENGLISH_V3,
            "search_document",
        );

        let error = model
            .embed_texts(["hello".to_string()])
            .await
            .expect_err("should fail with non-success status");

        assert!(matches!(error, EmbeddingError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn embeddings_2xx_error_envelope_preserves_status_and_body() {
        use crate::embeddings::EmbeddingModel as _;
        use crate::test_utils::RecordingHttpClient;

        // Deserializes to `ApiResponse::Err(ApiErrorResponse { message })` on a 200 OK.
        let body = r#"{"message":"boom"}"#;
        let http_client = RecordingHttpClient::new(body);
        let client = crate::providers::cohere::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");
        let model = client.embedding_model(
            crate::providers::cohere::EMBED_ENGLISH_V3,
            "search_document",
        );

        let error = model
            .embed_texts(["hello".to_string()])
            .await
            .expect_err("should fail with provider error envelope");

        match &error {
            EmbeddingError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::OK));
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }

    #[test]
    fn image_data_urls_detect_every_cohere_image_format() {
        let cases: &[(&[u8], &str)] = &[
            (b"\x89PNG\r\n\x1a\n", "image/png"),
            (b"\xff\xd8\xff", "image/jpeg"),
            (b"GIF89a", "image/gif"),
            (b"RIFF\0\0\0\0WEBP", "image/webp"),
        ];

        for &(bytes, expected_media_type) in cases {
            let result = validate_image(bytes);
            assert!(
                matches!(result, Ok(media_type) if media_type == expected_media_type),
                "expected {expected_media_type}"
            );
            assert!(
                image_data_url(bytes, expected_media_type)
                    .starts_with(&format!("data:{expected_media_type};base64,"))
            );
        }
    }

    #[test]
    fn image_documents_are_stable_without_retaining_image_bytes() {
        let first = image_document(b"\x89PNG\r\n\x1a\nfirst", "image/png");
        let second = image_document(b"\x89PNG\r\n\x1a\nother", "image/png");

        assert_eq!(
            first,
            image_document(b"\x89PNG\r\n\x1a\nfirst", "image/png")
        );
        assert_ne!(first, second);
        assert!(first.starts_with("image/png;sha256="));
        assert!(!first.contains("first"));
    }

    #[test]
    fn image_data_url_rejects_unsupported_and_oversized_inputs() {
        assert!(matches!(
            validate_image(b"not an image"),
            Err(EmbeddingError::DocumentError(_))
        ));
        assert!(matches!(
            validate_image(&vec![0; MAX_IMAGE_BYTES + 1]),
            Err(EmbeddingError::DocumentError(_))
        ));
    }

    #[tokio::test]
    async fn image_batches_are_fully_validated_before_any_request() {
        use crate::embeddings::ImageEmbeddingModel as _;
        use crate::test_utils::RecordingHttpClient;

        let http_client = RecordingHttpClient::default();
        let client = crate::providers::cohere::Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("build client");

        let error = client
            .image_embedding_model()
            .embed_images([b"\x89PNG\r\n\x1a\n".to_vec(), b"not an image".to_vec()])
            .await
            .expect_err("invalid batch should fail before transport");

        assert!(matches!(error, EmbeddingError::DocumentError(_)));
        assert!(http_client.requests().is_empty());
    }

    #[tokio::test]
    async fn image_embeddings_non_success_preserves_status_and_body() {
        use crate::embeddings::ImageEmbeddingModel as _;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"error":{"message":"boom"}}"#;
        let http_client =
            RecordingHttpClient::with_error_response(http::StatusCode::SERVICE_UNAVAILABLE, body);
        let client = crate::providers::cohere::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");

        let error = client
            .image_embedding_model()
            .embed_image(b"\x89PNG\r\n\x1a\n")
            .await
            .expect_err("should fail with non-success status");

        assert!(matches!(error, EmbeddingError::HttpError(_)));
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::SERVICE_UNAVAILABLE)
        );
        assert_eq!(error.provider_response_body(), Some(body));
    }

    #[tokio::test]
    async fn image_embeddings_2xx_error_envelope_preserves_status_and_body() {
        use crate::embeddings::ImageEmbeddingModel as _;
        use crate::test_utils::RecordingHttpClient;

        let body = r#"{"message":"boom"}"#;
        let http_client = RecordingHttpClient::new(body);
        let client = crate::providers::cohere::Client::builder()
            .api_key("test-key")
            .http_client(http_client)
            .build()
            .expect("build client");

        let error = client
            .image_embedding_model()
            .embed_image(b"\x89PNG\r\n\x1a\n")
            .await
            .expect_err("should fail with provider error envelope");

        match &error {
            EmbeddingError::ProviderResponse(stored) => {
                assert_eq!(stored.body, body);
                assert_eq!(stored.status, Some(http::StatusCode::OK));
            }
            other => panic!("expected ProviderResponse, got {other:?}"),
        }
    }
}
