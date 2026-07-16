use super::client::MistralExt;
use crate::{
    embeddings::EmbeddingError,
    providers::openai::embedding::{
        EmbeddingDimensions, GenericEmbeddingModel, OpenAIEmbeddingsCompatible,
    },
};

pub const MISTRAL_EMBED: &str = "mistral-embed";
/// Codestral embedding model with configurable output dimensions.
pub const CODESTRAL_EMBED: &str = "codestral-embed";

pub const MAX_DOCUMENTS: usize = 1024;

impl OpenAIEmbeddingsCompatible for MistralExt {
    const PROVIDER_NAME: &'static str = "mistral";
    const SUPPORTS_USER: bool = false;

    fn embeddings_path(&self) -> String {
        "/v1/embeddings".to_string()
    }

    fn embedding_dimensions(
        &self,
        model: &str,
        dimensions: Option<usize>,
    ) -> Result<Option<EmbeddingDimensions>, EmbeddingError> {
        let Some(dimensions) = dimensions else {
            return Ok(None);
        };

        if !matches!(model, "codestral-embed" | "codestral-embed-2505") {
            return Err(EmbeddingError::UnsupportedParameter {
                provider: Self::PROVIDER_NAME,
                parameter: "dimensions",
            });
        }

        if dimensions > 3_072 {
            return Err(EmbeddingError::InvalidParameterValue {
                provider: Self::PROVIDER_NAME,
                parameter: "dimensions",
                requirement: "to be at most 3072 for Codestral Embed",
            });
        }

        Ok(Some(EmbeddingDimensions::OutputDimension(dimensions)))
    }
}

pub type EmbeddingModel<H = reqwest::Client> = GenericEmbeddingModel<MistralExt, H>;

#[cfg(test)]
mod tests {
    use super::{CODESTRAL_EMBED, MISTRAL_EMBED};
    use crate::client::EmbeddingsClient;
    use crate::embeddings::{EmbeddingError, EmbeddingModel as _};
    use crate::providers::{mistral, openai::embedding::EncodingFormat};
    use crate::test_utils::RecordingHttpClient;

    const RESPONSE_BODY: &str = r#"{
        "id": "emb-1",
        "object": "list",
        "model": "mistral-embed",
        "usage": { "prompt_tokens": 5, "total_tokens": 5 },
        "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3] }]
    }"#;

    fn client(http_client: RecordingHttpClient) -> mistral::Client<RecordingHttpClient> {
        mistral::Client::builder()
            .api_key("dummy-key")
            .http_client(http_client)
            .build()
            .expect("client should build")
    }

    #[tokio::test]
    async fn codestral_embeddings_map_dimensions_and_mistral_usage() {
        let http_client = RecordingHttpClient::new(RESPONSE_BODY);
        let model = client(http_client.clone())
            .embedding_model_with_ndims(CODESTRAL_EMBED, 512)
            .encoding_format(EncodingFormat::Float);

        let response = model
            .embed_texts_with_usage(["hello".to_string()])
            .await
            .expect("embedding request should succeed");

        assert_eq!(response.embeddings[0].vec, vec![0.1, 0.2, 0.3]);
        assert_eq!(response.usage.input_tokens, 5);
        assert_eq!(response.usage.total_tokens, 5);

        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].uri.ends_with("/v1/embeddings"));
        let body: serde_json::Value =
            serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
        assert_eq!(body["output_dimension"], serde_json::json!(512));
        assert_eq!(body["encoding_format"], serde_json::json!("float"));
        assert!(body.get("dimensions").is_none());
        assert!(body.get("user").is_none());
    }

    #[tokio::test]
    async fn mistral_embed_rejects_dimensions_before_sending() {
        let http_client = RecordingHttpClient::new(RESPONSE_BODY);
        let model = client(http_client.clone()).embedding_model_with_ndims(MISTRAL_EMBED, 512);

        let error = model
            .embed_texts(["hello".to_string()])
            .await
            .expect_err("fixed-size model should reject dimensions");

        assert!(matches!(
            error,
            EmbeddingError::UnsupportedParameter {
                provider: "mistral",
                parameter: "dimensions"
            }
        ));
        assert!(http_client.requests().is_empty());
    }

    #[tokio::test]
    async fn codestral_embed_rejects_dimensions_above_maximum_before_sending() {
        let http_client = RecordingHttpClient::new(RESPONSE_BODY);
        let model = client(http_client.clone()).embedding_model_with_ndims(CODESTRAL_EMBED, 3_073);

        let error = model
            .embed_texts(["hello".to_string()])
            .await
            .expect_err("out-of-range dimensions should fail");

        assert!(matches!(
            error,
            EmbeddingError::InvalidParameterValue {
                provider: "mistral",
                parameter: "dimensions",
                ..
            }
        ));
        assert!(http_client.requests().is_empty());
    }

    #[tokio::test]
    async fn mistral_rejects_base64_before_sending() {
        let http_client = RecordingHttpClient::new(RESPONSE_BODY);
        let model = client(http_client.clone())
            .embedding_model(MISTRAL_EMBED)
            .encoding_format(EncodingFormat::Base64);

        let error = model
            .embed_texts(["hello".to_string()])
            .await
            .expect_err("unsupported response encoding should fail");

        assert!(matches!(
            error,
            EmbeddingError::UnsupportedResponseEncoding {
                provider: "mistral",
                encoding_format: "base64"
            }
        ));
        assert!(http_client.requests().is_empty());
    }

    #[tokio::test]
    async fn mistral_rejects_unsupported_user_before_sending() {
        let http_client = RecordingHttpClient::new(RESPONSE_BODY);
        let model = client(http_client.clone())
            .embedding_model(MISTRAL_EMBED)
            .user("user-123");

        let error = model
            .embed_texts(["hello".to_string()])
            .await
            .expect_err("unsupported user should fail");

        assert!(matches!(
            error,
            EmbeddingError::UnsupportedParameter {
                provider: "mistral",
                parameter: "user"
            }
        ));
        assert!(http_client.requests().is_empty());
    }
}
