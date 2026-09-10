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

use super::client::Doubleword;

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

impl OpenAIEmbeddingsCompatible for Doubleword {
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
pub type EmbeddingModel<T = crate::http_client::BoxedHttpClient> =
    GenericEmbeddingModel<Doubleword, T>;

#[cfg(test)]
mod tests;
