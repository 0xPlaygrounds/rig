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

/// Output dimensions of `mistral-embed`. `codestral-embed` is configurable and
/// is left to the caller's `dimensions`.
const MISTRAL_EMBED_NDIMS: usize = 1024;

impl OpenAIEmbeddingsCompatible for MistralExt {
    const PROVIDER_NAME: &'static str = "mistral";
    // Mistral reports its transport id on every response, embeddings
    // included; inheriting the trait's `None` default silently dropped it
    // (latent since the id capture landed for completions in #2313's wake).
    // Pinned by `embedding_matrix/bug_mistral_request_id_dropped`.
    const REQUEST_ID_HEADER: Option<&'static str> = Some("mistral-correlation-id");
    const SUPPORTS_USER: bool = false;
    const MAX_DOCUMENTS: usize = MAX_DOCUMENTS;

    fn default_ndims(model: &str) -> Option<usize> {
        // Mistral's models are absent from OpenAI's table, so without this
        // every Mistral embedding model reported `ndims() == 0`.
        matches!(model, MISTRAL_EMBED | "mistral-embed-2312").then_some(MISTRAL_EMBED_NDIMS)
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

        if !matches!(model, "codestral-embed" | "codestral-embed-2505") {
            // A fixed-width model naming its own width is not a request for
            // the unsupported parameter — it is the shared path echoing back
            // the dimension `default_ndims` reported. Send nothing and let the
            // model emit its native width. Any *other* value is still a real
            // request for a parameter Mistral does not accept here.
            if Self::default_ndims(model) == Some(dimensions) {
                return Ok(None);
            }

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

pub type EmbeddingModel<H> = GenericEmbeddingModel<MistralExt, H>;

#[cfg(test)]
mod tests;

#[cfg(test)]
mod batch_tests;
