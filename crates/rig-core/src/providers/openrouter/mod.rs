//! OpenRouter Inference API integration
//!
//! # Example
//! ```no_run
//! use rig_core::completion::CompletionRequest;
//! use rig_core::http_runtime::HttpRuntime;
//! use rig_core::providers::openrouter;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let cfg = openrouter::functions::Config::from_env(openrouter::PERPLEXITY_SONAR_PRO)?;
//! let rt = HttpRuntime::new();
//!
//! let request = CompletionRequest::from_prompt("Who are you?");
//! let response = openrouter::functions::complete(&cfg, &rt, request).await?;
//! # Ok(())
//! # }
//! ```

#[cfg(feature = "audio")]
#[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
pub mod audio_generation;
pub mod completion;
pub mod functions;
pub mod model_listing;
pub mod transcription;

crate::providers::client::define_http_client! {
    config = functions::Config,
    default_base_url = functions::DEFAULT_BASE_URL,
    api_key_required = true,
}
crate::providers::client::impl_http_embedding_config_factory!(Client, functions::EmbeddingConfig);

impl Client {
    /// Materialize transcription configuration sharing this connection.
    pub fn transcription_config(&self, model: impl Into<String>) -> functions::Config {
        self.config(model)
    }

    /// Materialize audio-generation configuration sharing this connection.
    #[cfg(feature = "audio")]
    pub fn audio_generation_config(&self, model: impl Into<String>) -> functions::Config {
        self.config(model)
    }
}

#[cfg(feature = "audio")]
pub use audio_generation::{GPT_4O_MINI_TTS, KOKORO_82M, VOXTRAL_MINI_TTS};
pub use completion::*;
// The shared chat-completions stream state machine calls this at exactly this
// path when a dialect enables reasoning-detail decoration.
pub(crate) use completion::decorate_streaming_tool_call;
pub use transcription::{
    CHIRP_3, GPT_4O_MINI_TRANSCRIBE, GPT_4O_TRANSCRIBE, TranscriptionResponse, WHISPER_1,
    WHISPER_LARGE_V3, WHISPER_LARGE_V3_TURBO,
};
