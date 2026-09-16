//! Doubleword's model identifiers.
//!
//! [Doubleword](https://docs.doubleword.ai) is an OpenAI chat-completions
//! dialect, so it has no client and no models of its own:
//! [`openai::wire::DOUBLEWORD`](crate::providers::openai::wire::DOUBLEWORD)
//! carries the base URL, the `DOUBLEWORD_API_KEY` and `DOUBLEWORD_BASE_URL`
//! variables, and the embeddings quirks (no `encoding_format`, no `user`,
//! usage not guaranteed). What lives here is the model identifiers.
//!
//! This is the **realtime** tier: synchronous chat completions, streaming,
//! and embeddings on the same host. Doubleword's two cheaper tiers are
//! separate mechanisms and Rig models neither — the **async** tier is the
//! same endpoints with `service_tier: "flex"` (plus `background: true` to
//! poll), while only the **batch** tier uses the OpenAI-compatible Batch API
//! (`/v1/batches`) with a JSONL upload.
//!
//! # Example
//! ```ignore
//! use rig_core::prelude::*;
//! use rig_core::providers::doubleword;
//! use rig_core::providers::openai::wire::{DOUBLEWORD, OpenAI};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let doubleword = OpenAI::from_env_with(&DOUBLEWORD)?.bound()?;
//! let chat = doubleword.completion(doubleword::QWEN3_5_9B);
//! let embedding = doubleword.embedding(doubleword::QWEN3_EMBEDDING_8B, None);
//! # let _ = (chat, embedding);
//! # Ok(())
//! # }
//! ```

// ================================================================
// Doubleword Completion Models
// ================================================================
// A non-exhaustive selection; the authoritative list is `GET /v1/models`.
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

// ================================================================
// Doubleword Embedding Models
// ================================================================
/// Doubleword's only embedding model. Its model page
/// (<https://docs.doubleword.ai/inference-api/models/qwen-qwen3-embedding-8b>)
/// documents "Output Dimensions: 32-4096 Configurable", and 4096 is also the
/// width it returns when a request names none.
pub const QWEN3_EMBEDDING_8B: &str = "Qwen/Qwen3-Embedding-8B";
