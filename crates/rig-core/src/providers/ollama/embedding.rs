//! Ollama's embedding wire, `POST /api/embed`, and its embedding models.
//!
//! ```
//! use rig_core::providers::ollama::{ALL_MINILM, OllamaConfig};
//! let model = OllamaConfig::new().client().embedding(ALL_MINILM, None);
//! assert_eq!(model.wire.ndims, 384);
//! ```

use crate::error::{EncodeError, ProviderError};
use crate::operation::Embedding;
use crate::wire::{
    Body, Capabilities, Decoder, Descriptor, Encoded, Flow, Framing, Mode, Out, Wire, WireEvent,
    WireFrame,
};
use serde::{Deserialize, Serialize};

use super::{OllamaConfig, PROVIDER_NAME};

/// The `all-minilm` embedding model.
pub const ALL_MINILM: &str = "all-minilm";
/// The `nomic-embed-text` embedding model.
pub const NOMIC_EMBED_TEXT: &str = "nomic-embed-text";
/// The `mxbai-embed-large` embedding model.
pub const MXBAI_EMBED_LARGE: &str = "mxbai-embed-large";
/// The `bge-m3` multilingual embedding model.
pub const BGE_M3: &str = "bge-m3";
/// The `embeddinggemma` embedding model.
pub const EMBEDDINGGEMMA: &str = "embeddinggemma";
/// The `qwen3-embedding` embedding model family; dimensions vary by size, so pass them explicitly.
pub const QWEN3_EMBEDDING: &str = "qwen3-embedding";

fn model_dimensions_from_identifier(identifier: &str) -> Option<usize> {
    match identifier {
        ALL_MINILM => Some(384),
        NOMIC_EMBED_TEXT => Some(768),
        MXBAI_EMBED_LARGE => Some(1024),
        BGE_M3 => Some(1024),
        EMBEDDINGGEMMA => Some(768),
        _ => None,
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingResponse {
    pub model: String,
    pub embeddings: Vec<Vec<f64>>,
    #[serde(default)]
    pub total_duration: Option<u64>,
    #[serde(default)]
    pub load_duration: Option<u64>,
    #[serde(default)]
    pub prompt_eval_count: Option<u64>,
}

/// The most texts `POST /api/embed` accepts in one call.
const MAX_DOCUMENTS: usize = 1024;

impl OllamaConfig {
    /// Build an embedding wire reporting the supplied width, known model width,
    /// or zero if unknown. The width is metadata and is not sent to the daemon.
    pub(crate) fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        let model = model.into();
        let ndims = ndims
            .or_else(|| model_dimensions_from_identifier(&model))
            .unwrap_or_default();
        Embeddings {
            provider: self.clone(),
            model,
            ndims,
        }
    }
}

/// The embedding wire: `POST /api/embed`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embeddings {
    /// The daemon this wire speaks to.
    pub provider: OllamaConfig,
    /// The model to address.
    pub model: String,
    /// The width this wire reports, from the caller or the model's published
    /// dimensions. `0` means neither named one.
    pub ndims: usize,
}

impl Wire for Embeddings {
    type Op = Embedding;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = EmbeddingsDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .capabilities(Capabilities::embedding(MAX_DOCUMENTS, self.ndims))
    }

    fn encode(&self, texts: Vec<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        let body = serde_json::json!({ "model": self.model, "input": texts });
        let request = self
            .provider
            .request(http::Method::POST, "/api/embed")
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        EmbeddingsDecoder
    }
}

/// Decodes one `/api/embed` reply.
pub struct EmbeddingsDecoder;

impl<'id> Decoder<'id, Embedding> for EmbeddingsDecoder {
    type Event = EmbeddingResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_marker_keyed_frame(
            &frame.as_str(),
            &["embeddings"],
        )
    }

    fn decode(
        &mut self,
        reply: Self::Event,
        out: Out<'id, Embedding>,
    ) -> Result<Flow, ProviderError> {
        // Ollama counts the prompt it embedded and nothing else: an
        // embedding bills input only, so output is zero.
        let usage = crate::completion::Usage {
            input_tokens: reply.prompt_eval_count,
            output_tokens: reply.prompt_eval_count.map(|_| 0),
            total_tokens: reply.prompt_eval_count,
            ..Default::default()
        };
        Ok(out.end(crate::embeddings::EmbeddingResponse {
            model: Some(reply.model),
            usage,
            ..crate::embeddings::EmbeddingResponse::from_vectors(reply.embeddings)
        }))
    }
}

#[cfg(test)]
mod tests;
