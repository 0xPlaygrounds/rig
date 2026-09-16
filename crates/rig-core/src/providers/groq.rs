//! Groq's model identifiers.
//!
//! Groq is an OpenAI chat-completions dialect, so it has no client and no
//! models of its own:
//! [`openai::wire::GROQ`](crate::providers::openai::wire::GROQ) carries the
//! base URL, the `GROQ_API_KEY` variable, the `x-request-id` header Groq
//! reports its transport request id on, and the one rewrite Groq needs —
//! folding its compound-system native tools from `additional_params.tools`
//! into `compound_custom.enabled_tools` so they do not clobber the
//! function-tool array. The same dialect serves transcription
//! (`/audio/transcriptions`) and the model listing, whose entries carry
//! Groq's `context_window` and `max_completion_tokens`.
//!
//! What remains here is data: the model identifiers. Groq's reasoning options
//! (`reasoning_format`, `include_reasoning`) ride on the request's
//! `additional_params` as plain JSON.
//!
//! # Example
//! A wire is the config plus a model; `.bind(transport)` (or `.bound()` from
//! `rig-reqwest`) turns it into the model.
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

// ================================================================
// Groq Completion API
// ================================================================

// Model IDs follow <https://console.groq.com/docs/models>. Retired IDs are listed at
// <https://console.groq.com/docs/deprecations>.

/// The `llama-3.1-8b-instant` model. Used for chat completion. Retired from free and
/// developer tiers on 2026-08-16; still served to enterprise contracts.
pub const LLAMA_3_1_8B_INSTANT: &str = "llama-3.1-8b-instant";
/// The `llama-3.3-70b-versatile` model. Used for chat completion. Retired from free and
/// developer tiers on 2026-08-16; still served to enterprise contracts.
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
/// The `groq/compound` agentic system (built-in web search and code execution).
pub const COMPOUND: &str = "groq/compound";
/// The `groq/compound-mini` agentic system (built-in web search and code execution).
pub const COMPOUND_MINI: &str = "groq/compound-mini";

// ================================================================
// Groq Transcription API
// ================================================================

/// The `whisper-large-v3` transcription model.
pub const WHISPER_LARGE_V3: &str = "whisper-large-v3";
/// The `whisper-large-v3-turbo` transcription model.
pub const WHISPER_LARGE_V3_TURBO: &str = "whisper-large-v3-turbo";
