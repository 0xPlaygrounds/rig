//! Type-position aliases for every rig-core provider type that is generic over the
//! transport, defaulted to the bundled [`crate::ReqwestClient`]: `rig::providers::openai::CompletionModel`
//! means `…::CompletionModel<ReqwestClient>` again, so `Agent<openai::CompletionModel>` and
//! `let c: openai::Client = …` read as before rig-core lost its default transport.
//!
//! Each module re-exports everything from the rig-core provider module and then shadows the
//! transport-generic names with defaulted aliases (nested rig-core paths stay generic).
//!
//! Construction goes through [`crate::client::DefaultTransportClient`] /
//! [`crate::client::DefaultTransportBuilder`]: type-alias defaults do not apply in expression
//! position, so `openai::Client::new(..)` needs those traits, not these aliases.
//!
//! # Generated file — do not edit
//!
//! Regenerate with `cargo xtask generate-provider-aliases`; CI runs the same
//! command with `--check`. The source of truth is rig-core's own rustdoc
//! output, so a type that is generic over the transport gets an alias here
//! whether it was written by hand or produced by a macro, and a type whose
//! parameter already has a default (`ClientBuilder<H = Missing>`) is left
//! alone. See `xtask/src/aliases.rs`.

/// The bundled transport every alias here defaults to.
pub type DefaultHttp = crate::ReqwestClient;

pub mod anthropic {
    pub use rig_core::providers::anthropic::*;
    pub type AnthropicModelLister<H = super::DefaultHttp> =
        rig_core::providers::anthropic::model_listing::AnthropicModelLister<H>;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::anthropic::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::anthropic::completion::CompletionModel<H>;
}

pub mod azure {
    pub use rig_core::providers::azure::*;
    #[cfg(feature = "audio")]
    #[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
    pub type AudioGenerationModel<H = super::DefaultHttp> =
        rig_core::providers::azure::AudioGenerationModel<H>;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::azure::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::azure::CompletionModel<H>;
    pub type EmbeddingModel<H = super::DefaultHttp> = rig_core::providers::azure::EmbeddingModel<H>;
    #[cfg(feature = "image")]
    #[cfg_attr(docsrs, doc(cfg(feature = "image")))]
    pub type ImageGenerationModel<H = super::DefaultHttp> =
        rig_core::providers::azure::ImageGenerationModel<H>;
    pub type TranscriptionModel<H = super::DefaultHttp> =
        rig_core::providers::azure::TranscriptionModel<H>;
}

pub mod chatgpt {
    pub use rig_core::providers::chatgpt::*;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::chatgpt::Client<H>;
    pub type ResponsesCompletionModel<H = super::DefaultHttp> =
        rig_core::providers::chatgpt::ResponsesCompletionModel<H>;
}

pub mod cohere {
    pub use rig_core::providers::cohere::*;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::cohere::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::cohere::CompletionModel<H>;
    pub type EmbeddingModel<H = super::DefaultHttp> =
        rig_core::providers::cohere::EmbeddingModel<H>;
    pub type ImageEmbeddingModel<H = super::DefaultHttp> =
        rig_core::providers::cohere::ImageEmbeddingModel<H>;
}

pub mod copilot {
    pub use rig_core::providers::copilot::*;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::copilot::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::copilot::CompletionModel<H>;
    pub type CopilotModelLister<H = super::DefaultHttp> =
        rig_core::providers::copilot::CopilotModelLister<H>;
    pub type EmbeddingModel<H = super::DefaultHttp> =
        rig_core::providers::copilot::EmbeddingModel<H>;
}

pub mod deepseek {
    pub use rig_core::providers::deepseek::*;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::deepseek::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::deepseek::CompletionModel<H>;
    pub type DeepSeekModelLister<H = super::DefaultHttp> =
        rig_core::providers::deepseek::DeepSeekModelLister<H>;
}

pub mod doubleword {
    pub use rig_core::providers::doubleword::*;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::doubleword::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::doubleword::CompletionModel<H>;
    pub type EmbeddingModel<H = super::DefaultHttp> =
        rig_core::providers::doubleword::EmbeddingModel<H>;
}

pub mod gemini {
    pub use rig_core::providers::gemini::*;
    pub type CachedContentClient<H = super::DefaultHttp> =
        rig_core::providers::gemini::CachedContentClient<H>;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::gemini::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::gemini::CompletionModel<H>;
    pub type EmbeddingModel<H = super::DefaultHttp> =
        rig_core::providers::gemini::EmbeddingModel<H>;
    pub type GeminiInteractionsModelLister<H = super::DefaultHttp> =
        rig_core::providers::gemini::GeminiInteractionsModelLister<H>;
    pub type GeminiModelLister<H = super::DefaultHttp> =
        rig_core::providers::gemini::GeminiModelLister<H>;
    #[cfg(feature = "image")]
    #[cfg_attr(docsrs, doc(cfg(feature = "image")))]
    pub type ImageGenerationModel<H = super::DefaultHttp> =
        rig_core::providers::gemini::ImageGenerationModel<H>;
    pub type InteractionsClient<H = super::DefaultHttp> =
        rig_core::providers::gemini::InteractionsClient<H>;
    pub type InteractionsCompletionModel<H = super::DefaultHttp> =
        rig_core::providers::gemini::interactions_api::InteractionsCompletionModel<H>;
    pub type TranscriptionModel<H = super::DefaultHttp> =
        rig_core::providers::gemini::transcription::TranscriptionModel<H>;
}

pub mod groq {
    pub use rig_core::providers::groq::*;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::groq::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::groq::CompletionModel<H>;
    pub type GroqModelLister<H = super::DefaultHttp> =
        rig_core::providers::groq::GroqModelLister<H>;
    pub type TranscriptionModel<H = super::DefaultHttp> =
        rig_core::providers::groq::TranscriptionModel<H>;
}

pub mod huggingface {
    pub use rig_core::providers::huggingface::*;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::huggingface::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::huggingface::completion::CompletionModel<H>;
    #[cfg(feature = "image")]
    #[cfg_attr(docsrs, doc(cfg(feature = "image")))]
    pub type ImageGenerationModel<H = super::DefaultHttp> =
        rig_core::providers::huggingface::image_generation::ImageGenerationModel<H>;
    pub type TranscriptionModel<H = super::DefaultHttp> =
        rig_core::providers::huggingface::transcription::TranscriptionModel<H>;
}

pub mod hyperbolic {
    pub use rig_core::providers::hyperbolic::*;
    #[cfg(feature = "audio")]
    #[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
    pub type AudioGenerationModel<H = super::DefaultHttp> =
        rig_core::providers::hyperbolic::AudioGenerationModel<H>;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::hyperbolic::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::hyperbolic::CompletionModel<H>;
    #[cfg(feature = "image")]
    #[cfg_attr(docsrs, doc(cfg(feature = "image")))]
    pub type ImageGenerationModel<H = super::DefaultHttp> =
        rig_core::providers::hyperbolic::ImageGenerationModel<H>;
}

pub mod llamacpp {
    pub use rig_core::providers::llamacpp::*;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::llamacpp::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::llamacpp::CompletionModel<H>;
    pub type EmbeddingModel<H = super::DefaultHttp> =
        rig_core::providers::llamacpp::EmbeddingModel<H>;
    pub type LlamacppModelLister<H = super::DefaultHttp> =
        rig_core::providers::llamacpp::client::LlamacppModelLister<H>;
    pub type RerankModel<H = super::DefaultHttp> = rig_core::providers::llamacpp::RerankModel<H>;
}

pub mod minimax {
    pub use rig_core::providers::minimax::*;
    pub type AnthropicClient<H = super::DefaultHttp> =
        rig_core::providers::minimax::AnthropicClient<H>;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::minimax::Client<H>;
    pub type MiniMaxModelLister<H = super::DefaultHttp> =
        rig_core::providers::minimax::MiniMaxModelLister<H>;
}

pub mod mira {
    pub use rig_core::providers::mira::*;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::mira::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::mira::CompletionModel<H>;
    pub type MiraModelLister<H = super::DefaultHttp> =
        rig_core::providers::mira::MiraModelLister<H>;
}

pub mod mistral {
    pub use rig_core::providers::mistral::*;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::mistral::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::mistral::CompletionModel<H>;
    pub type EmbeddingModel<H = super::DefaultHttp> =
        rig_core::providers::mistral::EmbeddingModel<H>;
    pub type MistralModelLister<H = super::DefaultHttp> =
        rig_core::providers::mistral::MistralModelLister<H>;
    pub type TranscriptionModel<H = super::DefaultHttp> =
        rig_core::providers::mistral::TranscriptionModel<H>;
}

pub mod moonshot {
    pub use rig_core::providers::moonshot::*;
    pub type AnthropicClient<H = super::DefaultHttp> =
        rig_core::providers::moonshot::AnthropicClient<H>;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::moonshot::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::moonshot::CompletionModel<H>;
    pub type MoonshotModelLister<H = super::DefaultHttp> =
        rig_core::providers::moonshot::MoonshotModelLister<H>;
}

pub mod ollama {
    pub use rig_core::providers::ollama::*;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::ollama::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::ollama::CompletionModel<H>;
    pub type EmbeddingModel<H = super::DefaultHttp> =
        rig_core::providers::ollama::EmbeddingModel<H>;
    pub type OllamaModelLister<H = super::DefaultHttp> =
        rig_core::providers::ollama::OllamaModelLister<H>;
}

pub mod openai {
    pub use rig_core::providers::openai::*;
    #[cfg(feature = "audio")]
    #[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
    pub type AudioGenerationModel<H = super::DefaultHttp> =
        rig_core::providers::openai::audio_generation::AudioGenerationModel<H>;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::openai::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::openai::CompletionModel<H>;
    #[cfg(feature = "audio")]
    #[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
    pub type CompletionsAudioGenerationModel<H = super::DefaultHttp> =
        rig_core::providers::openai::audio_generation::CompletionsAudioGenerationModel<H>;
    pub type CompletionsClient<H = super::DefaultHttp> =
        rig_core::providers::openai::CompletionsClient<H>;
    #[cfg(feature = "image")]
    #[cfg_attr(docsrs, doc(cfg(feature = "image")))]
    pub type CompletionsImageGenerationModel<H = super::DefaultHttp> =
        rig_core::providers::openai::CompletionsImageGenerationModel<H>;
    pub type CompletionsTranscriptionModel<H = super::DefaultHttp> =
        rig_core::providers::openai::CompletionsTranscriptionModel<H>;
    pub type EmbeddingModel<H = super::DefaultHttp> =
        rig_core::providers::openai::EmbeddingModel<H>;
    #[cfg(feature = "image")]
    #[cfg_attr(docsrs, doc(cfg(feature = "image")))]
    pub type ImageGenerationModel<H = super::DefaultHttp> =
        rig_core::providers::openai::ImageGenerationModel<H>;
    pub type OpenAICompletionsModelLister<H = super::DefaultHttp> =
        rig_core::providers::openai::OpenAICompletionsModelLister<H>;
    pub type OpenAIModelLister<H = super::DefaultHttp> =
        rig_core::providers::openai::OpenAIModelLister<H>;
    pub type ResponsesCompletionModel<H = super::DefaultHttp> =
        rig_core::providers::openai::responses_api::ResponsesCompletionModel<H>;
    pub type TranscriptionModel<H = super::DefaultHttp> =
        rig_core::providers::openai::TranscriptionModel<H>;
}

pub mod openrouter {
    pub use rig_core::providers::openrouter::*;
    #[cfg(feature = "audio")]
    #[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
    pub type AudioGenerationModel<H = super::DefaultHttp> =
        rig_core::providers::openrouter::AudioGenerationModel<H>;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::openrouter::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::openrouter::CompletionModel<H>;
    pub type EmbeddingModel<H = super::DefaultHttp> =
        rig_core::providers::openrouter::EmbeddingModel<H>;
    pub type OpenRouterModelLister<H = super::DefaultHttp> =
        rig_core::providers::openrouter::OpenRouterModelLister<H>;
    pub type TranscriptionModel<H = super::DefaultHttp> =
        rig_core::providers::openrouter::TranscriptionModel<H>;
}

pub mod perplexity {
    pub use rig_core::providers::perplexity::*;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::perplexity::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::perplexity::CompletionModel<H>;
}

pub mod together {
    pub use rig_core::providers::together::*;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::together::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::together::CompletionModel<H>;
    pub type EmbeddingModel<H = super::DefaultHttp> =
        rig_core::providers::together::EmbeddingModel<H>;
}

pub mod venice {
    pub use rig_core::providers::venice::*;
    #[cfg(feature = "audio")]
    #[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
    pub type AudioGenerationModel<H = super::DefaultHttp> =
        rig_core::providers::venice::AudioGenerationModel<H>;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::venice::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> =
        rig_core::providers::venice::CompletionModel<H>;
    pub type EmbeddingModel<H = super::DefaultHttp> =
        rig_core::providers::venice::EmbeddingModel<H>;
    #[cfg(feature = "image")]
    #[cfg_attr(docsrs, doc(cfg(feature = "image")))]
    pub type ImageGenerationModel<H = super::DefaultHttp> =
        rig_core::providers::venice::ImageGenerationModel<H>;
    pub type TranscriptionModel<H = super::DefaultHttp> =
        rig_core::providers::venice::TranscriptionModel<H>;
    pub type VeniceModelLister<H = super::DefaultHttp> =
        rig_core::providers::venice::VeniceModelLister<H>;
}

pub mod voyageai {
    pub use rig_core::providers::voyageai::*;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::voyageai::Client<H>;
    pub type EmbeddingModel<H = super::DefaultHttp> =
        rig_core::providers::voyageai::EmbeddingModel<H>;
    pub type RerankModel<H = super::DefaultHttp> = rig_core::providers::voyageai::RerankModel<H>;
}

pub mod xai {
    pub use rig_core::providers::xai::*;
    #[cfg(feature = "audio")]
    #[cfg_attr(docsrs, doc(cfg(feature = "audio")))]
    pub type AudioGenerationModel<H = super::DefaultHttp> =
        rig_core::providers::xai::AudioGenerationModel<H>;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::xai::Client<H>;
    pub type CompletionModel<H = super::DefaultHttp> = rig_core::providers::xai::CompletionModel<H>;
    #[cfg(feature = "image")]
    #[cfg_attr(docsrs, doc(cfg(feature = "image")))]
    pub type ImageGenerationModel<H = super::DefaultHttp> =
        rig_core::providers::xai::ImageGenerationModel<H>;
}

pub mod xiaomimimo {
    pub use rig_core::providers::xiaomimimo::*;
    pub type AnthropicClient<H = super::DefaultHttp> =
        rig_core::providers::xiaomimimo::AnthropicClient<H>;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::xiaomimimo::Client<H>;
    pub type XiaomiMimoModelLister<H = super::DefaultHttp> =
        rig_core::providers::xiaomimimo::XiaomiMimoModelLister<H>;
}

pub mod zai {
    pub use rig_core::providers::zai::*;
    pub type AnthropicClient<H = super::DefaultHttp> = rig_core::providers::zai::AnthropicClient<H>;
    pub type Client<H = super::DefaultHttp> = rig_core::providers::zai::Client<H>;
}
