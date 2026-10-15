//! Doubleword's chat, Responses, and embedding models.
//!
//! [`from_env`] and [`new`] build a client on the
//! [`DOUBLEWORD`](crate::providers::openai::wire::DOUBLEWORD) dialect.
//! Select `.responses(model)` and [`ServiceTier::Flex`](crate::completion::ServiceTier::Flex)
//! for flex processing. The request waits for its completed response when
//! `background` is omitted. Polling and batch submission are outside these operations.
//!
//! ```no_run
//! use rig_core::completion::{CompletionRequest, GenerationOptions, ServiceTier};
//! use rig_core::providers::doubleword;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let model = doubleword::from_env()?.responses(doubleword::QWEN3_5_9B);
//! let options = GenerationOptions::default().service_tier(ServiceTier::Flex);
//! let response = model
//!     .call(CompletionRequest::new("Summarize this report.").options(options))
//!     .await?;
//! # let _ = response;
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

/// The provider key: the dialect name, a reply's `Origin::provider` and
/// the typed provider-options key.
pub const PROVIDER_NAME: &str = "doubleword";

crate::client::macros::openai_vendor!(crate::providers::openai::wire::DOUBLEWORD, "Doubleword");
