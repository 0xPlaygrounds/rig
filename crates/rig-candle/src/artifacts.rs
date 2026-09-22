//! Owned and borrowed model artifact buffers.
//!
//! ```
//! use rig_candle::ModelData;
//!
//! fn artifacts(config: Vec<u8>, tokenizer: Vec<u8>, weights: Vec<u8>) -> ModelData {
//!     ModelData { config, tokenizer, weights }
//! }
//! ```

use crate::CandleError;

/// Owned model artifacts for exactly one unsharded checkpoint.
#[derive(Debug)]
pub struct ModelData {
    /// Contents of `config.json`.
    pub config: Vec<u8>,
    /// Contents of `tokenizer.json`.
    pub tokenizer: Vec<u8>,
    /// Contents of one safetensors or GGUF checkpoint, as identified by [`ModelArtifacts`].
    pub weights: Vec<u8>,
}

/// Borrowed GGUF artifacts for loading without copying the input buffers.
#[derive(Debug, Clone, Copy)]
pub struct GgufModelData<'a> {
    /// Contents of `config.json`.
    pub config: &'a [u8],
    /// Contents of `tokenizer.json`.
    pub tokenizer: &'a [u8],
    /// Contents of one GGUF checkpoint.
    pub weights: &'a [u8],
}

/// Byte-backed checkpoint format supplied to [`crate::CandleModel`].
#[derive(Debug)]
pub enum ModelArtifacts {
    /// One unsharded Hugging Face safetensors checkpoint.
    Safetensors(ModelData),
    /// A SmolLM2 or Qwen3 Q4_K_M GGUF checkpoint, validated during model loading.
    Gguf(ModelData),
}

pub(crate) fn require_nonempty(bytes: &[u8], artifact: &'static str) -> Result<(), CandleError> {
    if bytes.is_empty() {
        Err(CandleError::EmptyBuffer { artifact })
    } else {
        Ok(())
    }
}
