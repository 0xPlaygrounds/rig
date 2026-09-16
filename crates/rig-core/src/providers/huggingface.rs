//! Hugging Face's model identifiers.
//!
//! The [Hugging Face Inference Providers
//! router](https://huggingface.co/docs/inference-providers) is an OpenAI
//! chat-completions dialect, so it has no client and no models of its own:
//! [`openai::wire::HUGGINGFACE`](crate::providers::openai::wire::HUGGINGFACE)
//! carries the base URL, the `HUGGINGFACE_API_KEY` variable, and the two
//! routing quirks that make the router unlike a plain OpenAI host — chat
//! lives under `/v1` while transcription and image generation address
//! `/{model}` at the root.
//!
//! The router is one host in front of many backends, and which backend
//! answers is
//! [`openai::wire::SubRoute`](crate::providers::openai::wire::SubRoute) — the
//! former `SubProvider`, same variant names and same route slugs. The choice
//! is observable three ways: `Fireworks` addresses models by a
//! fully-qualified id, and only `HFInference` serves transcription and image
//! generation.
//!
//! What lives here is the model identifiers.
//!
//! # Example
//! A wire is the config plus a model; `.bind(transport)` (or `.bound()` from
//! `rig-reqwest`) turns it into the model.
//! ```no_run
//! use rig_core::providers::huggingface;
//! use rig_core::providers::openai::wire::{HUGGINGFACE, OpenAI, SubRoute};
//!
//! # fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let router = OpenAI::from_env_with(&HUGGINGFACE)?;
//! let chat = router.chat(huggingface::GEMMA_2);
//! let transcription = router.transcriptions(huggingface::WHISPER_LARGE_V3);
//!
//! // Routed to a different backend, which addresses models by a
//! // fully-qualified id.
//! let fireworks = OpenAI::from_env_with(&HUGGINGFACE)?
//!     .with_sub_route(SubRoute::Fireworks)
//!     .chat(huggingface::META_LLAMA_3_1);
//! # let _ = (chat, transcription, fireworks);
//! # Ok(())
//! # }
//! ```

// ================================================================
// Hugging Face Completion Models
// ================================================================

// Conversational LLMs
/// `google/gemma-2-2b-it` completion model
pub const GEMMA_2: &str = "google/gemma-2-2b-it";
/// `meta-llama/Meta-Llama-3.1-8B-Instruct` completion model
pub const META_LLAMA_3_1: &str = "meta-llama/Meta-Llama-3.1-8B-Instruct";
/// `PowerInfer/SmallThinker-3B-Preview` completion model
pub const SMALLTHINKER_PREVIEW: &str = "PowerInfer/SmallThinker-3B-Preview";
/// `Qwen/Qwen2.5-7B-Instruct` completion model
pub const QWEN2_5: &str = "Qwen/Qwen2.5-7B-Instruct";
/// `Qwen/Qwen2.5-Coder-32B-Instruct` completion model
pub const QWEN2_5_CODER: &str = "Qwen/Qwen2.5-Coder-32B-Instruct";

// Conversational VLMs

/// `Qwen/Qwen2-VL-7B-Instruct` visual-language completion model
pub const QWEN2_VL: &str = "Qwen/Qwen2-VL-7B-Instruct";
/// `Qwen/QVQ-72B-Preview` visual-language completion model
pub const QWEN_QVQ_PREVIEW: &str = "Qwen/QVQ-72B-Preview";

// ================================================================
// Hugging Face Transcription Models
// ================================================================
// Served only by the router's own backend (`SubRoute::HFInference`), which
// addresses them as `/{model}` at the root.
pub const WHISPER_LARGE_V3: &str = "openai/whisper-large-v3";
pub const WHISPER_LARGE_V3_TURBO: &str = "openai/whisper-large-v3-turbo";

pub const WHISPER_SMALL: &str = "openai/whisper-small";

// ================================================================
// Hugging Face Image Generation Models
// ================================================================
/// Image-generation model identifiers, named as the model cards spell them.
#[cfg(feature = "image")]
#[cfg_attr(docsrs, doc(cfg(feature = "image")))]
#[allow(non_upper_case_globals)]
pub mod image_generation_models {
    pub const Flux1: &str = "black-forest-labs/FLUX.1-dev";
    pub const Kolors: &str = "Kwai-Kolors/Kolors";
    pub const StableDiffusion3: &str = "stabilityai/stable-diffusion-3-medium-diffusers";
}
#[cfg(feature = "image")]
pub use image_generation_models::*;
