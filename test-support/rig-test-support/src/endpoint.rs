//! A provider configuration paired with the transport its models send
//! through, for tests that build several models from one recorded provider.
//! Each constructor pairs the provider's wire with a clone of the transport.

use rig_agent::agent::AgentBuilder;
use rig_agent::extractor::ExtractorBuilder;
use rig_core::Model;
use rig_core::embeddings::EmbeddingsBuilder;
use rig_core::http_client::BoxedHttpClient;
use rig_core::providers::{anthropic, cohere, copilot, gemini, ollama, openai, voyageai};

/// A provider configuration and its transport.
#[derive(Clone, Debug)]
pub struct Endpoint<P, H = BoxedHttpClient> {
    /// The provider configuration the wires are built from.
    pub wire: P,
    /// The transport every model built here sends through.
    pub http: H,
}

impl<P, H> Endpoint<P, H> {
    /// Pair `wire` with `http`.
    pub fn new(wire: P, http: H) -> Self {
        Self { wire, http }
    }
}

/// Generates the constructors an [`Endpoint`] over one provider offers:
/// each wraps the provider's own wire constructor in a [`Model`].
macro_rules! endpoint {
    ($provider:ty {
        $($method:ident($($arg:ident: $ty:ty),*) -> $wire:ty;)*
    } $(completion: $completion:ty;)? $(embedding: $embedding:ty;)?) => {
        impl<H: Clone> Endpoint<$provider, H> {
            $(
                #[doc = concat!("The provider's `", stringify!($method), "` model.")]
                pub fn $method(&self $(, $arg: $ty)*) -> Model<$wire, H> {
                    Model::new(self.wire.$method($($arg),*), self.http.clone())
                }
            )*
        }

        $(
            impl<H> Endpoint<$provider, H>
            where
                H: rig_core::driver::Transport<$completion>,
            {
                /// An agent over the provider's completion model.
                pub fn agent(&self, model: impl Into<String>) -> AgentBuilder {
                    AgentBuilder::new(Model::new(self.wire.completion(model), self.http.clone()))
                }

                /// A typed extractor over the provider's completion model.
                pub fn extractor<T>(&self, model: impl Into<String>) -> ExtractorBuilder<T>
                where
                    T: schemars::JsonSchema
                        + serde::de::DeserializeOwned
                        + serde::Serialize
                        + Send
                        + Sync
                        + 'static,
                {
                    ExtractorBuilder::new(Model::new(self.wire.completion(model), self.http.clone()))
                }
            }
        )?

        $(
            impl<H> Endpoint<$provider, H>
            where
                H: rig_core::driver::Transport<$embedding>,
            {
                /// An embeddings builder over the provider's `model`.
                pub fn embeddings<D: rig_core::Embed>(
                    &self,
                    model: impl Into<String>,
                ) -> EmbeddingsBuilder<Model<$embedding, H>, D> {
                    EmbeddingsBuilder::new(self.embedding(model, None))
                }

                /// An embeddings builder over the provider's `model` at `ndims`.
                pub fn embeddings_with_ndims<D: rig_core::Embed>(
                    &self,
                    model: impl Into<String>,
                    ndims: usize,
                ) -> EmbeddingsBuilder<Model<$embedding, H>, D> {
                    EmbeddingsBuilder::new(self.embedding(model, Some(ndims)))
                }
            }
        )?
    };
}

endpoint!(openai::wire::OpenAI {
    completion(model: impl Into<String>) -> openai::wire::OpenAiWire;
    responses(model: impl Into<String>) -> openai::responses_api::wire::Responses;
    chat(model: impl Into<String>) -> openai::wire::Chat;
    embedding(model: impl Into<String>, ndims: Option<usize>) -> openai::wire::Embeddings;
    rerank(model: impl Into<String>) -> openai::wire::Rerank;
    transcription(model: impl Into<String>) -> openai::wire::Transcriptions;
    models() -> openai::wire::Models;
    verify() -> openai::wire::Verify;
    image_generation(model: impl Into<String>) -> openai::wire::Images;
    audio_generation(model: impl Into<String>) -> openai::wire::Speech;
} completion: openai::wire::OpenAiWire; embedding: openai::wire::Embeddings;);

endpoint!(anthropic::wire::Anthropic {
    completion(model: impl Into<String>) -> anthropic::wire::Messages;
    models() -> anthropic::wire::Models;
    verify() -> anthropic::wire::Verify;
} completion: anthropic::wire::Messages;);

endpoint!(gemini::Gemini {
    completion(model: impl Into<String>) -> gemini::completion::GenerateContent;
    interactions(model: impl Into<String>) -> gemini::interactions_api::Interactions;
    embedding(model: impl Into<String>, ndims: Option<usize>) -> gemini::embedding::Embeddings;
    transcription(model: impl Into<String>) -> gemini::transcription::Transcriptions;
    image_generation(model: impl Into<String>) -> gemini::image_generation::Images;
    models() -> gemini::model_listing::Models;
    verify() -> gemini::model_listing::VerifyKey;
    interactions_models() -> gemini::model_listing::InteractionsModels;
    cached_contents() -> gemini::cached_content::CachedContents;
} completion: gemini::completion::GenerateContent; embedding: gemini::embedding::Embeddings;);

endpoint!(cohere::wire::Cohere {
    completion(model: impl Into<String>) -> cohere::Chat;
    embedding(model: impl Into<String>, ndims: Option<usize>) -> cohere::wire::Embeddings;
    image_embedding() -> cohere::wire::ImageEmbeddings;
} completion: cohere::Chat; embedding: cohere::wire::Embeddings;);

endpoint!(ollama::wire::Ollama {
    completion(model: impl Into<String>) -> ollama::wire::Chat;
    embedding(model: impl Into<String>, ndims: Option<usize>) -> ollama::wire::Embeddings;
    models() -> ollama::wire::Models;
} completion: ollama::wire::Chat; embedding: ollama::wire::Embeddings;);

endpoint!(copilot::wire::Copilot {
    completion(model: impl Into<String>) -> copilot::wire::CopilotWire;
    embedding(model: impl Into<String>, ndims: Option<usize>) -> copilot::wire::Embeddings;
    models() -> copilot::wire::Models;
} completion: copilot::wire::CopilotWire; embedding: copilot::wire::Embeddings;);

endpoint!(voyageai::wire::VoyageAi {
    embedding(model: impl Into<String>, ndims: Option<usize>) -> voyageai::wire::Embeddings;
    rerank(model: impl Into<String>) -> voyageai::wire::Rerank;
} embedding: voyageai::wire::Embeddings;);
