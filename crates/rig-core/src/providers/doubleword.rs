//! Doubleword's model identifiers.
//!
//! Configure realtime chat and embedding requests with
//! [`crate::providers::openai::wire::DOUBLEWORD`]. Async polling and batch
//! submission are not modeled by these operations.
//!
//! ```no_run
//! use rig_core::providers::doubleword;
//! use rig_core::providers::openai::wire::{DOUBLEWORD, OpenAI};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let doubleword = OpenAI::from_env_with(&DOUBLEWORD)?;
//! let chat = doubleword.chat(doubleword::QWEN3_5_9B);
//! let embedding = doubleword.embeddings(doubleword::QWEN3_EMBEDDING_8B, None);
//! # let _ = (chat, embedding);
//! # Ok(())
//! # }
//! ```

pub const QWEN3_5_4B: &str = "Qwen/Qwen3.5-4B";
pub const QWEN3_5_9B: &str = "Qwen/Qwen3.5-9B";
pub const QWEN3_5_397B_A17B: &str = "Qwen/Qwen3.5-397B-A17B-FP8";
pub const QWEN3_6_35B_A3B: &str = "Qwen/Qwen3.6-35B-A3B-FP8";
pub const GPT_OSS_20B: &str = "openai/gpt-oss-20b";
pub const GPT_OSS_120B: &str = "openai/gpt-oss-120b";
pub const DEEPSEEK_V4_PRO: &str = "deepseek-ai/DeepSeek-V4-Pro";
pub const DEEPSEEK_V4_FLASH: &str = "deepseek-ai/DeepSeek-V4-Flash";
pub const KIMI_K2_6: &str = "moonshotai/Kimi-K2.6";
pub const GLM_5_2: &str = "zai-org/GLM-5.2-FP8";
pub const QWEN3_VL_30B: &str = "Qwen/Qwen3-VL-30B-A3B-Instruct-FP8";
pub const QWEN3_VL_235B: &str = "Qwen/Qwen3-VL-235B-A22B-Instruct-FP8";

/// Identifier for the Qwen3 Embedding 8B model. Supported output dimensions are
/// defined by [`crate::providers::openai::wire::DOUBLEWORD`].
pub const QWEN3_EMBEDDING_8B: &str = "Qwen/Qwen3-Embedding-8B";
