// ================================================================
//! Doubleword Embeddings Integration
//! From [Doubleword Inference API](https://docs.doubleword.ai/inference-api/models)
// ================================================================

use core::ops::RangeInclusive;

use crate::{
    embeddings::EmbeddingError,
    providers::openai::embedding::{
        EmbeddingDimensions, GenericEmbeddingModel, OpenAIEmbeddingsCompatible,
    },
};

use super::client::DoublewordExt;

// ================================================================
// Doubleword Embedding API
// ================================================================
pub const QWEN3_EMBEDDING_8B: &str = "Qwen/Qwen3-Embedding-8B";

/// Output widths Doubleword documents for [`QWEN3_EMBEDDING_8B`] on its model
/// page (<https://docs.doubleword.ai/inference-api/models/qwen-qwen3-embedding-8b>):
/// "Output Dimensions: 32-4096 Configurable". The top of the range is also the
/// width the model returns when the request names none, which is why one
/// constant can serve as both — a second model whose maximum and default
/// differ would need them apart.
const QWEN3_EMBEDDING_8B_DIMENSIONS: RangeInclusive<usize> = 32..=4_096;

/// The documented output-dimension range of a Doubleword embedding model, or
/// `None` for a model this build does not know.
///
/// One table backs both halves of the dimension contract — the width
/// [`OpenAIEmbeddingsCompatible::default_ndims`] reports and the values
/// [`OpenAIEmbeddingsCompatible::embedding_dimensions`] will put on the wire —
/// so the two cannot drift apart into a model that reports a width it would
/// refuse to request. The rejection *message* is a `&'static str` the error
/// type cannot format from a range, so it repeats the bounds by hand; adding a
/// second model here means revisiting it.
fn documented_dimensions(model: &str) -> Option<RangeInclusive<usize>> {
    (model == QWEN3_EMBEDDING_8B).then_some(QWEN3_EMBEDDING_8B_DIMENSIONS)
}

impl OpenAIEmbeddingsCompatible for DoublewordExt {
    const PROVIDER_NAME: &'static str = "doubleword";

    // Doubleword responses are not guaranteed to carry usage; usage is
    // reported when present and zero otherwise.
    const REQUIRES_USAGE: bool = false;
    const SUPPORTS_ENCODING_FORMAT: bool = false;
    const SUPPORTS_USER: bool = false;

    fn default_ndims(model: &str) -> Option<usize> {
        // Doubleword's models are absent from OpenAI's table, so without this
        // the provider's only embedding model reported `ndims() == 0` — and a
        // vector store sized from `ndims()` built a zero-width index.
        documented_dimensions(model).map(|dimensions| *dimensions.end())
    }

    fn embedding_dimensions(
        &self,
        model: &str,
        dimensions: Option<usize>,
    ) -> Result<Option<EmbeddingDimensions>, EmbeddingError> {
        let Some(dimensions) = dimensions else {
            return Ok(None);
        };

        if dimensions == 0 {
            return Err(EmbeddingError::InvalidParameterValue {
                provider: Self::PROVIDER_NAME,
                parameter: "dimensions",
                requirement: "to be greater than zero",
            });
        }

        let Some(documented) = documented_dimensions(model) else {
            // An embedding model this build does not know: the caller's width
            // is the only width there is, so send it and let the API rule.
            return Ok(Some(EmbeddingDimensions::Dimensions(dimensions)));
        };

        if dimensions == *documented.end() {
            // A model naming its own native width is not a request for
            // truncation — it is the shared path echoing back what
            // `default_ndims` reported. Send nothing and let the model emit
            // that width, which is the same vector either way.
            return Ok(None);
        }

        if !documented.contains(&dimensions) {
            // Worth catching here rather than on the wire, in both
            // directions. Above the ceiling Doubleword silently clamps to the
            // native width, which would leave `ndims()` describing vectors the
            // API never returned — the very mismatch this hook exists to
            // prevent. Below the floor it is not dependable: the identical
            // request answers `422 Unprocessable request` or `200` with a
            // sub-floor vector at random (six of fifteen live probes at 1, 2,
            // 8, 16 and 31 were rejected; every probe at 32 and above
            // succeeded), so a width rig cannot promise is better refused than
            // half-honoured.
            return Err(EmbeddingError::InvalidParameterValue {
                provider: Self::PROVIDER_NAME,
                parameter: "dimensions",
                requirement: "to be between 32 and 4096",
            });
        }

        Ok(Some(EmbeddingDimensions::Dimensions(dimensions)))
    }
}

/// Doubleword embedding model, driven by the shared OpenAI-compatible
/// embeddings path.
pub type EmbeddingModel<T = reqwest::Client> = GenericEmbeddingModel<DoublewordExt, T>;

#[cfg(test)]
mod tests {
    use super::QWEN3_EMBEDDING_8B;
    use crate::client::EmbeddingsClient;
    use crate::embeddings::{EmbeddingError, EmbeddingModel as _};
    use crate::providers::{doubleword, openai::embedding::EncodingFormat};
    use crate::test_utils::RecordingHttpClient;

    /// A width Doubleword never returns, so a test that reads it back proves
    /// the body it asserts on is the body that was sent.
    const RESPONSE_BODY: &str = r#"{
        "object": "list",
        "model": "Qwen/Qwen3-Embedding-8B",
        "usage": { "prompt_tokens": 2, "total_tokens": 2 },
        "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2, 0.3] }]
    }"#;

    fn client(http_client: RecordingHttpClient) -> doubleword::Client<RecordingHttpClient> {
        doubleword::Client::builder()
            .api_key("dummy-key")
            .http_client(http_client)
            .build()
            .expect("client should build")
    }

    /// The `dimensions` value a request carried, or `None` when it carried the
    /// field not at all — the distinction the whole hook turns on.
    async fn sent_dimensions(model: &str, ndims: Option<usize>) -> Option<serde_json::Value> {
        let http_client = RecordingHttpClient::new(RESPONSE_BODY);
        let embedding_model = match ndims {
            Some(ndims) => client(http_client.clone()).embedding_model_with_ndims(model, ndims),
            None => client(http_client.clone()).embedding_model(model),
        };

        embedding_model
            .embed_texts(["probe".to_string()])
            .await
            .expect("embedding request should succeed");

        let requests = http_client.requests();
        assert_eq!(requests.len(), 1);
        assert!(requests[0].uri.ends_with("/v1/embeddings"));
        let body: serde_json::Value =
            serde_json::from_slice(&requests[0].body).expect("request body should be JSON");
        body.get("dimensions").cloned()
    }

    async fn rejected_dimensions(model: &str, ndims: usize) -> EmbeddingError {
        let http_client = RecordingHttpClient::new(RESPONSE_BODY);
        let error = client(http_client.clone())
            .embedding_model_with_ndims(model, ndims)
            .embed_texts(["probe".to_string()])
            .await
            .expect_err("out-of-range dimensions should fail");

        assert!(
            http_client.requests().is_empty(),
            "a rejected width must not reach the wire"
        );
        error
    }

    #[test]
    fn default_ndims_reports_the_native_width_doubleword_returns() {
        let model = doubleword::Client::new("dummy-key")
            .expect("client should build")
            .embedding_model(QWEN3_EMBEDDING_8B);

        assert_eq!(model.ndims(), 4_096);
    }

    #[test]
    fn an_unknown_embedding_model_still_reports_no_width() {
        let model = doubleword::Client::new("dummy-key")
            .expect("client should build")
            .embedding_model("Qwen/Qwen4-Embedding-Unreleased");

        assert_eq!(model.ndims(), 0);
    }

    #[tokio::test]
    async fn the_native_width_is_not_echoed_back_onto_the_wire() {
        // `default_ndims` hands the shared path 4096, which then offers it
        // back as a "requested" width. Sending it would be a no-op the
        // recorded requests of every existing caller would no longer match.
        assert_eq!(sent_dimensions(QWEN3_EMBEDDING_8B, None).await, None);
        assert_eq!(sent_dimensions(QWEN3_EMBEDDING_8B, Some(4_096)).await, None);
    }

    #[tokio::test]
    async fn a_requested_width_reaches_the_wire() {
        for ndims in [32_usize, 64, 512, 1_024, 4_095] {
            assert_eq!(
                sent_dimensions(QWEN3_EMBEDDING_8B, Some(ndims)).await,
                Some(serde_json::json!(ndims)),
                "dimensions={ndims} should be sent verbatim"
            );
        }
    }

    #[tokio::test]
    async fn a_zero_width_is_rejected_before_sending() {
        assert!(matches!(
            rejected_dimensions(QWEN3_EMBEDDING_8B, 0).await,
            EmbeddingError::InvalidParameterValue {
                provider: "doubleword",
                parameter: "dimensions",
                requirement: "to be greater than zero"
            }
        ));
    }

    #[tokio::test]
    async fn an_openai_named_model_cannot_bypass_zero_width_validation() {
        assert!(matches!(
            rejected_dimensions(crate::providers::openai::TEXT_EMBEDDING_ADA_002, 0).await,
            EmbeddingError::InvalidParameterValue {
                provider: "doubleword",
                parameter: "dimensions",
                requirement: "to be greater than zero"
            }
        ));
    }

    #[tokio::test]
    async fn widths_outside_the_documented_range_are_rejected_before_sending() {
        for ndims in [1_usize, 31, 4_097, 8_192] {
            assert!(
                matches!(
                    rejected_dimensions(QWEN3_EMBEDDING_8B, ndims).await,
                    EmbeddingError::InvalidParameterValue {
                        provider: "doubleword",
                        parameter: "dimensions",
                        ..
                    }
                ),
                "dimensions={ndims} should be rejected"
            );
        }
    }

    #[tokio::test]
    async fn an_unknown_embedding_model_passes_the_requested_width_through() {
        // No table entry means no documented range to police, so the caller's
        // width is the only width there is and the API gets to rule on it.
        assert_eq!(
            sent_dimensions("Qwen/Qwen4-Embedding-Unreleased", Some(8_192)).await,
            Some(serde_json::json!(8_192))
        );
    }

    #[tokio::test]
    async fn an_unknown_embedding_model_still_rejects_zero_width() {
        assert!(matches!(
            rejected_dimensions("Qwen/Qwen4-Embedding-Unreleased", 0).await,
            EmbeddingError::InvalidParameterValue {
                provider: "doubleword",
                parameter: "dimensions",
                requirement: "to be greater than zero"
            }
        ));
    }

    #[tokio::test]
    async fn unsupported_request_options_still_fail_before_sending() {
        let http_client = RecordingHttpClient::new(RESPONSE_BODY);
        let error = client(http_client.clone())
            .embedding_model(QWEN3_EMBEDDING_8B)
            .user("user-123")
            .embed_texts(["probe".to_string()])
            .await
            .expect_err("unsupported user should fail");
        assert!(matches!(
            error,
            EmbeddingError::UnsupportedParameter {
                provider: "doubleword",
                parameter: "user"
            }
        ));
        assert!(http_client.requests().is_empty());

        let http_client = RecordingHttpClient::new(RESPONSE_BODY);
        let error = client(http_client.clone())
            .embedding_model(QWEN3_EMBEDDING_8B)
            .encoding_format(EncodingFormat::Float)
            .embed_texts(["probe".to_string()])
            .await
            .expect_err("unsupported encoding format should fail");
        assert!(matches!(
            error,
            EmbeddingError::UnsupportedParameter {
                provider: "doubleword",
                parameter: "encoding_format"
            }
        ));
        assert!(http_client.requests().is_empty());
    }
}
