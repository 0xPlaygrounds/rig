//! Hyperbolic's model identifiers.
//!
//! Hyperbolic is an OpenAI chat-completions dialect, so it has no client and
//! no models of its own:
//! [`openai::wire::HYPERBOLIC`](crate::providers::openai::wire::HYPERBOLIC)
//! carries the base URL (the bare host — every Hyperbolic path carries its
//! own `/v1`), the `HYPERBOLIC_API_KEY` variable, and the quirks that make
//! it Hyperbolic: no tool calling, no `response_format`, and tool-exchange
//! remnants stripped from a shared history while content-part arrays are
//! kept for its vision models.
//!
//! What remains here is data: the chat model identifiers, and the image
//! model identifiers Hyperbolic addresses its diffusion endpoint by. Its
//! speech endpoint is addressed by *language* rather than by model, so the
//! "model" a speech request names is a language tag such as `EN`.
//!
//! # Example
//! A wire is the config plus a model; `.bind(transport)` (or `.bound()` from
//! `rig-reqwest`) turns it into the model.
//! ```no_run
//! use rig_core::providers::hyperbolic;
//! use rig_core::providers::openai::wire::{HYPERBOLIC, OpenAI};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let provider = OpenAI::from_env_with(&HYPERBOLIC)?;
//! let llama_3_1_8b = provider.chat(hyperbolic::LLAMA_3_1_8B);
//! // Image generation needs the `image` feature.
//! # #[cfg(feature = "image")]
//! let sdxl = provider.images(hyperbolic::SDXL_TURBO);
//! # Ok(())
//! # }
//! ```

// ================================================================
// Hyperbolic Completion API
// ================================================================

/// Meta Llama 3.1b Instruct model with 8B parameters.
pub const LLAMA_3_1_8B: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct";
/// Meta Llama 3.3b Instruct model with 70B parameters.
pub const LLAMA_3_3_70B: &str = "meta-llama/Llama-3.3-70B-Instruct";
/// Meta Llama 3.1b Instruct model with 70B parameters.
pub const LLAMA_3_1_70B: &str = "meta-llama/Meta-Llama-3.1-70B-Instruct";
/// Meta Llama 3 Instruct model with 70B parameters.
pub const LLAMA_3_70B: &str = "meta-llama/Meta-Llama-3-70B-Instruct";
/// Hermes 3 Instruct model with 70B parameters.
pub const HERMES_3_70B: &str = "NousResearch/Hermes-3-Llama-3.1-70b";
/// Deepseek v2.5 model.
pub const DEEPSEEK_2_5: &str = "deepseek-ai/DeepSeek-V2.5";
/// Qwen 2.5 model with 72B parameters.
pub const QWEN_2_5_72B: &str = "Qwen/Qwen2.5-72B-Instruct";
/// Meta Llama 3.2b Instruct model with 3B parameters.
pub const LLAMA_3_2_3B: &str = "meta-llama/Llama-3.2-3B-Instruct";
/// Qwen 2.5 Coder Instruct model with 32B parameters.
pub const QWEN_2_5_CODER_32B: &str = "Qwen/Qwen2.5-Coder-32B-Instruct";
/// Preview (latest) version of Qwen model with 32B parameters.
pub const QWEN_QWQ_PREVIEW_32B: &str = "Qwen/QwQ-32B-Preview";
/// Deepseek R1 Zero model.
pub const DEEPSEEK_R1_ZERO: &str = "deepseek-ai/DeepSeek-R1-Zero";
/// Deepseek R1 model.
pub const DEEPSEEK_R1: &str = "deepseek-ai/DeepSeek-R1";

// =======================================
// Hyperbolic Image Generation API
// =======================================

pub const SDXL1_0_BASE: &str = "SDXL1.0-base";
pub const SD2: &str = "SD2";
pub const SD1_5: &str = "SD1.5";
pub const SSD: &str = "SSD";
pub const SDXL_TURBO: &str = "SDXL-turbo";
pub const SDXL_CONTROLNET: &str = "SDXL-ControlNet";
pub const SD1_5_CONTROLNET: &str = "SD1.5-ControlNet";
