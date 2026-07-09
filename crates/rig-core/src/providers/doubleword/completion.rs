//! Doubleword completion models.
//!
//! Completions run through the shared OpenAI-compatible
//! [`GenericCompletionModel`](openai::completion::GenericCompletionModel); the
//! dialect is declared by the `OpenAICompatibleProvider` impl on
//! [`DoublewordExt`](super::client::DoublewordExt) in `client.rs`.

use crate::providers::openai;

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

/// Doubleword completion model — the shared OpenAI-compatible
/// [`GenericCompletionModel`](openai::completion::GenericCompletionModel)
/// specialized to Doubleword.
pub type CompletionModel<H = reqwest::Client> =
    openai::completion::GenericCompletionModel<super::client::DoublewordExt, H>;
