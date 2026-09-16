use crate::embeddings::EmbeddingError;
use base64::{Engine as _, engine::general_purpose::STANDARD};
use serde::{Deserialize, Serialize};

const MAX_IMAGE_BYTES: usize = 5_000_000;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    #[serde(default)]
    pub response_type: Option<String>,
    pub id: String,
    pub embeddings: Vec<Vec<serde_json::Number>>,
    pub texts: Vec<String>,
    #[serde(default)]
    pub meta: Option<Meta>,
}

/// The error envelope Cohere can answer a `/v1/embed` **200** with instead
/// of embeddings: `{"message":"…"}`.
///
/// Decoding it is the whole point — it proves the body is the envelope and
/// nothing else — but the error the consumer sees is built from the raw
/// body, so the provider's payload rides out verbatim.
#[derive(Debug, Deserialize)]
pub(super) struct ErrorEnvelope {
    #[allow(dead_code)]
    message: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Meta {
    pub api_version: ApiVersion,
    pub billed_units: BilledUnits,
    #[serde(default)]
    pub warnings: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ApiVersion {
    pub version: String,
    #[serde(default)]
    pub is_deprecated: Option<bool>,
    #[serde(default)]
    pub is_experimental: Option<bool>,
}

/// Cohere's `meta.billed_units`. Token counters are absent when the request
/// was not billed in tokens (image embeds bill `images`), so they are
/// `Option`; the non-token counters default to zero.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct BilledUnits {
    #[serde(skip_serializing_if = "Option::is_none")]
    pub input_tokens: Option<u32>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub output_tokens: Option<u32>,
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
            self.input_tokens.unwrap_or(0),
            self.output_tokens.unwrap_or(0),
            self.search_units,
            self.classifications
        )?;
        if self.images > 0 {
            write!(f, "\nImages: {}", self.images)?;
        }
        Ok(())
    }
}

/// One Cohere `/v1/embed` answer for a single image, one per input image:
/// Cohere Embed v3 accepts a single image per call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ImageEmbeddingResponse {
    #[serde(default)]
    pub id: Option<String>,
    pub embeddings: FloatEmbeddings,
    #[serde(default)]
    pub meta: Option<Meta>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FloatEmbeddings {
    #[serde(rename = "float")]
    pub values: Vec<Vec<serde_json::Number>>,
}

impl BilledUnits {
    /// Maps the billed token counters straight through; `total_tokens` is
    /// the sum of whichever counters Cohere sent (an embed bills input only,
    /// so it reports `input_tokens` and `total_tokens`, no `output_tokens`).
    pub(super) fn to_usage(&self) -> crate::completion::Usage {
        let input_tokens = self.input_tokens.map(u64::from);
        let output_tokens = self.output_tokens.map(u64::from);
        let total_tokens = match (input_tokens, output_tokens) {
            (None, None) => None,
            (input, output) => Some(input.unwrap_or(0) + output.unwrap_or(0)),
        };
        crate::completion::Usage {
            input_tokens,
            output_tokens,
            total_tokens,
            ..Default::default()
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub(super) enum ImageInputError {
    #[error("Cohere image embeddings support PNG, JPEG, WebP, or GIF file bytes")]
    UnsupportedFormat,
    #[error("Cohere image embeddings accept at most 5 MB per image; received {actual_bytes} bytes")]
    TooLarge { actual_bytes: usize },
}

/// The media type Cohere will accept these bytes as, or the reason it will
/// not. Sniffing is [`crate::embeddings::image_media_type`]'s; the
/// acceptance policy (the formats and the 5 MB ceiling) is Cohere's.
pub(super) fn validate_image(bytes: &[u8]) -> Result<&'static str, EmbeddingError> {
    if bytes.len() > MAX_IMAGE_BYTES {
        return Err(EmbeddingError::DocumentError(Box::new(
            ImageInputError::TooLarge {
                actual_bytes: bytes.len(),
            },
        )));
    }

    crate::embeddings::image_media_type(bytes)
        .ok_or_else(|| EmbeddingError::DocumentError(Box::new(ImageInputError::UnsupportedFormat)))
}

pub(super) fn image_data_url(bytes: &[u8], media_type: &str) -> String {
    format!("data:{media_type};base64,{}", STANDARD.encode(bytes))
}

#[cfg(test)]
mod tests;
