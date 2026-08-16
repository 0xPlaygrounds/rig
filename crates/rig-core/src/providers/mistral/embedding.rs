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

/// Most inputs Mistral accepts in one `/v1/embeddings` request. Verified
/// against the live API: 256 succeeds, 257 is rejected with
/// `"Too many inputs in request, split into more batches."`.
pub const MAX_DOCUMENTS: usize = 256;

/// Output dimensions of `mistral-embed`, which has no other width.
const MISTRAL_EMBED_NDIMS: usize = 1024;

/// Output dimensions `codestral-embed` returns when the request names none.
///
/// Configurable up to 3072 through `output_dimension`, but a caller who asks
/// for nothing still gets vectors of this width — verified live, a
/// dimension-less request returns 1536 floats. Reporting the *configurable*
/// half as "unknown" left `ndims()` at 0, which is not a width a vector store
/// can size itself from.
const CODESTRAL_EMBED_NDIMS: usize = 1536;

impl OpenAIEmbeddingsCompatible for MistralExt {
    const PROVIDER_NAME: &'static str = "mistral";
    const SUPPORTS_USER: bool = false;
    const MAX_DOCUMENTS: usize = MAX_DOCUMENTS;

    fn default_ndims(model: &str) -> Option<usize> {
        // Mistral's models are absent from OpenAI's table, so without this
        // every Mistral embedding model reported `ndims() == 0`. Codestral's
        // width is configurable, but a model whose default is documented and
        // observable still has a default: leaving it unreported declared 0
        // dimensions for vectors 1536 wide.
        match model {
            MISTRAL_EMBED | "mistral-embed-2312" => Some(MISTRAL_EMBED_NDIMS),
            CODESTRAL_EMBED | "codestral-embed-2505" => Some(CODESTRAL_EMBED_NDIMS),
            _ => None,
        }
    }

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

        // A model naming its own default width is not a request at all — it is
        // the shared path echoing back the dimension `default_ndims` reported,
        // and sending it would change the bytes of every request that names no
        // width. Send nothing and let the model emit its native width. This
        // applies to Codestral too, whose width *is* configurable: declaring a
        // default must not silently start populating the request.
        if Self::default_ndims(model) == Some(dimensions) {
            return Ok(None);
        }

        if !matches!(model, "codestral-embed" | "codestral-embed-2505") {
            // Any other value is a real request for a parameter Mistral does
            // not accept for a fixed-width model.
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

#[cfg(test)]
mod batch_tests {
    use super::*;

    /// The chunk size `EmbeddingsBuilder` uses. Recording the 256-input
    /// success it guards would commit ~5 MB of returned vectors to a fixture;
    /// the cap it must stay under is pinned live in
    /// `tests/providers/mistral/capability_edges.rs`.
    #[test]
    fn builder_chunks_at_mistrals_cap_not_openais() {
        assert_eq!(MAX_DOCUMENTS, 256);
        assert_eq!(
            <super::super::EmbeddingModel as crate::embeddings::EmbeddingModel>::MAX_DOCUMENTS,
            256,
            "the generic model must take the provider's cap; the shared default is OpenAI's 1024, \
             which Mistral rejects"
        );
    }

    /// `mistral-embed` is fixed-width, and its width is what `ndims()` must
    /// report — a model declaring 0 cannot size a vector store.
    #[test]
    fn mistral_embed_declares_its_width_without_requesting_it() {
        assert_eq!(MistralExt::default_ndims(MISTRAL_EMBED), Some(1024));
        assert_eq!(MistralExt::default_ndims("mistral-embed-2312"), Some(1024));

        // The declared width must not become a `dimensions` request field:
        // Mistral rejects that parameter for every model but Codestral.
        assert!(matches!(
            MistralExt.embedding_dimensions(MISTRAL_EMBED, Some(1024)),
            Ok(None)
        ));
        // Any other value is still a genuine request for the parameter.
        assert!(
            MistralExt
                .embedding_dimensions(MISTRAL_EMBED, Some(512))
                .is_err()
        );
    }

    /// Codestral's width is *configurable*, not unknown: a request naming no
    /// dimension still returns 1536-wide vectors, so that is the width
    /// `ndims()` must declare. Reporting `None` here left it at 0.
    #[test]
    fn codestral_embed_declares_its_default_width() {
        assert_eq!(MistralExt::default_ndims(CODESTRAL_EMBED), Some(1536));
        assert_eq!(
            MistralExt::default_ndims("codestral-embed-2505"),
            Some(1536)
        );
        assert_eq!(MistralExt::default_ndims("mistral-ocr-latest"), None);
    }

    /// Codestral is the one Mistral model that takes a width on the request,
    /// under Mistral's own `output_dimension` spelling — including the
    /// default the declaration above now echoes back.
    #[test]
    fn codestral_embed_sends_its_width_as_output_dimension() {
        // Declaring the default must not start populating the request: a model
        // asked for its own native width sends no width at all, so a plain
        // `embedding_model("codestral-embed")` puts the same bytes on the wire
        // it always did.
        assert!(matches!(
            MistralExt.embedding_dimensions(CODESTRAL_EMBED, Some(1536)),
            Ok(None)
        ));
        assert!(matches!(
            MistralExt.embedding_dimensions(CODESTRAL_EMBED, Some(512)),
            Ok(Some(EmbeddingDimensions::OutputDimension(512)))
        ));
        assert!(
            MistralExt
                .embedding_dimensions(CODESTRAL_EMBED, Some(3_073))
                .is_err(),
            "3072 is Codestral's ceiling"
        );
    }
}
