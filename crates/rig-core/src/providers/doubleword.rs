//! Doubleword's model identifiers.
//!
//! [`from_env`] and [`new`] build a client on the [`DOUBLEWORD`](crate::providers::openai::wire::DOUBLEWORD) dialect for
//! realtime chat, embeddings and its Responses endpoint. On `.responses(model)`,
//! [`ServiceTier::Flex`](crate::completion::ServiceTier::Flex) runs a request
//! at the flex tier and a unary call waits for its result; a long wait can
//! time out the connection. Background polling and batch submission are not
//! modeled by these operations.
//!
//! ```no_run
//! use rig_core::completion::{CompletionRequest, GenerationOptions, ServiceTier};
//! use rig_core::providers::doubleword;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let doubleword = doubleword::from_env()?;
//! let flex = GenerationOptions::default().service_tier(ServiceTier::Flex);
//! let reply = doubleword
//!     .responses(doubleword::QWEN3_5_9B)
//!     .call(CompletionRequest::new("Summarize this report.").options(flex))
//!     .await?;
//! let embedding = doubleword.embedding(doubleword::QWEN3_EMBEDDING_8B, None);
//! # let _ = (reply, embedding);
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
