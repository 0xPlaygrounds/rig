//! Ollama configuration, model identifiers, and its embedding and
//! model-listing wires.
//!
//! Chat goes through the daemon's OpenAI-compatible `/v1/chat/completions`
//! on the shared Chat Completions wire, as the
//! [`OLLAMA`](crate::providers::openai::wire::OLLAMA) dialect. Embeddings
//! use `/api/embed`, and the model listing `/api/tags`.
//!
//! ```no_run
//! use rig_core::providers::ollama;
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = ollama::Ollama::new();
//!
//! let qwen = provider.completion("qwen2.5:14b");
//! let embeddings = provider.embedding(ollama::ALL_MINILM, Some(384));
//! # Ok(())
//! # }
//! ```
//!
//! `Ollama::from_env` reads `OLLAMA_API_BASE_URL` and `OLLAMA_API_KEY` for
//! remote or authenticated daemons.
use crate::model::ModelInfo;
use serde::{Deserialize, Serialize};

pub mod wire;

pub use crate::client::ollama::Ollama;
pub use wire::{Embeddings, Models, OllamaConfig};

/// The address of a local daemon.
const OLLAMA_API_BASE_URL: &str = "http://localhost:11434";

/// Stable descriptor name recorded on normalized responses, streams, and
/// telemetry spans for this provider.
const PROVIDER_NAME: &str = "ollama";

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

/// The `llama3.2` model.
pub const LLAMA3_2: &str = "llama3.2";
/// The `llama3.1` model.
pub const LLAMA3_1: &str = "llama3.1";
/// The `llama3.3` model.
pub const LLAMA3_3: &str = "llama3.3";
/// The `llama4` multimodal model.
pub const LLAMA4: &str = "llama4";
/// The `llava` vision model.
pub const LLAVA: &str = "llava";
/// The `mistral` model.
pub const MISTRAL: &str = "mistral";
/// The `mistral-small3.2` model.
pub const MISTRAL_SMALL3_2: &str = "mistral-small3.2";
/// The `gemma3` model.
pub const GEMMA3: &str = "gemma3";
/// The `gemma4` model.
pub const GEMMA4: &str = "gemma4";
/// The `qwen3` model.
pub const QWEN3: &str = "qwen3";
/// The `qwen3.5` model.
pub const QWEN3_5: &str = "qwen3.5";
/// The `qwen3.6` model.
pub const QWEN3_6: &str = "qwen3.6";
/// The `qwen3.8` model.
pub const QWEN3_8: &str = "qwen3.8";
/// The `qwen3-coder` model.
pub const QWEN3_CODER: &str = "qwen3-coder";
/// The `deepseek-r1` reasoning model.
pub const DEEPSEEK_R1: &str = "deepseek-r1";
/// The `deepseek-v3.1` model.
pub const DEEPSEEK_V3_1: &str = "deepseek-v3.1";
/// The `gpt-oss` model.
pub const GPT_OSS: &str = "gpt-oss";
/// The `phi4` model.
pub const PHI4: &str = "phi4";

/// The reply of `GET /api/tags`: every model the daemon has pulled.
#[derive(Debug, Deserialize)]
pub struct ListModelsResponse {
    /// The installed models, in the daemon's own order.
    pub models: Vec<ListModelEntry>,
}

/// One installed model.
#[derive(Debug, Deserialize)]
pub struct ListModelEntry {
    /// The tag as the daemon displays it (`qwen3:4b`).
    pub name: String,
    /// The identifier a request addresses.
    pub model: String,
}

impl From<ListModelEntry> for ModelInfo {
    fn from(value: ListModelEntry) -> Self {
        ModelInfo::new(value.model, value.name)
    }
}

#[cfg(test)]
mod tests;

#[cfg(test)]
mod history_tests;
