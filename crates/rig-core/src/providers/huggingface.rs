//! Hugging Face's model identifiers.
//!
//! [`from_env`] and [`new`] build a client on the [`HUGGINGFACE`](crate::providers::openai::wire::HUGGINGFACE) dialect.
//! [`OpenAIConfig::with_sub_route`](crate::providers::openai::OpenAIConfig::with_sub_route)
//! selects a backend; transcription
//! and image generation require `HFInference`.
//!
//! ```no_run
//! use rig_core::providers::huggingface;
//!
//! # fn run() -> Result<(), rig_core::RigError> {
//! let router = huggingface::from_env()?;
//! let chat = router.chat(huggingface::GEMMA_2);
//! # let _ = chat;
//! # Ok(())
//! # }
//! ```

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

/// `Qwen/Qwen2-VL-7B-Instruct` visual-language completion model
pub const QWEN2_VL: &str = "Qwen/Qwen2-VL-7B-Instruct";
/// `Qwen/QVQ-72B-Preview` visual-language completion model
pub const QWEN_QVQ_PREVIEW: &str = "Qwen/QVQ-72B-Preview";

pub const WHISPER_LARGE_V3: &str = "openai/whisper-large-v3";
pub const WHISPER_LARGE_V3_TURBO: &str = "openai/whisper-large-v3-turbo";

pub const WHISPER_SMALL: &str = "openai/whisper-small";

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

crate::client::macros::openai_vendor!(crate::providers::openai::wire::HUGGINGFACE, "Hugging Face");
