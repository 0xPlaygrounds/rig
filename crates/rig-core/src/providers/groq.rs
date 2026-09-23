//! Groq's model identifiers.
//!
//! Configure requests with [`crate::providers::openai::wire::GROQ`]. Reasoning
//! options such as `reasoning_format` belong in request `additional_params`.
//!
//! ```no_run
//! use rig_core::providers::groq;
//! use rig_core::providers::openai::wire::{GROQ, OpenAI};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let gpt_oss = OpenAI::from_env_with(&GROQ)?.chat(groq::GPT_OSS_120B);
//! # let _ = gpt_oss;
//! # Ok(())
//! # }
//! ```

/// Identifier for the `llama-3.1-8b-instant` chat model.
pub const LLAMA_3_1_8B_INSTANT: &str = "llama-3.1-8b-instant";
/// Identifier for the `llama-3.3-70b-versatile` chat model.
pub const LLAMA_3_3_70B_VERSATILE: &str = "llama-3.3-70b-versatile";
/// The `openai/gpt-oss-120b` model. Used for chat completion.
pub const GPT_OSS_120B: &str = "openai/gpt-oss-120b";
/// The `openai/gpt-oss-20b` model. Used for chat completion.
pub const GPT_OSS_20B: &str = "openai/gpt-oss-20b";
/// The `openai/gpt-oss-safeguard-20b` model (preview). Used for chat completion.
pub const GPT_OSS_SAFEGUARD_20B: &str = "openai/gpt-oss-safeguard-20b";
/// The `qwen/qwen3.8-27b` model (preview). Used for chat completion.
pub const QWEN3_8_27B: &str = "qwen/qwen3.8-27b";
/// The `minimaxai/minimax-m2.7` model (preview, enterprise). Used for chat completion.
pub const MINIMAX_M2_7: &str = "minimaxai/minimax-m2.7";

/// The `whisper-large-v3` transcription model.
pub const WHISPER_LARGE_V3: &str = "whisper-large-v3";
/// The `whisper-large-v3-turbo` transcription model.
pub const WHISPER_LARGE_V3_TURBO: &str = "whisper-large-v3-turbo";
