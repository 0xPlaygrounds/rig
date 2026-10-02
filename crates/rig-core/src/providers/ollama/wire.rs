//! Ollama daemon configuration and chat, embedding, and model-listing wires.
//!
//! ```
//! use rig_core::providers::ollama::OllamaConfig;
//! let ollama = OllamaConfig::new().client();
//! assert_eq!(ollama.completion("qwen3").wire.model, "qwen3");
//! ```

use crate::client::env::{self, EnvError};
use crate::completion::CompletionRequest;
use crate::error::EncodeError;
use crate::error::ProviderError;
use crate::model::{ModelInfo, ModelList};
use crate::operation::{Completion, Embedding, ModelListing, ModelPage};
use crate::wire::Flow;
use crate::wire::{
    Body, Capabilities, Decoder, Descriptor, Encoded, Framing, Mode, Out, Secret, Wire, WireEvent,
    WireFrame,
};
use serde::{Deserialize, Serialize};

use super::{
    EmbeddingResponse as OllamaEmbeddingResponse, ListModelsResponse, OLLAMA_API_BASE_URL,
    OllamaCompletionRequest, PROVIDER_NAME, model_dimensions_from_identifier,
};

/// The environment variable overriding the daemon's address.
const BASE_URL_ENV: &str = "OLLAMA_API_BASE_URL";

/// The environment variable carrying the bearer token a proxied daemon
/// requires. A local daemon needs none.
const API_KEY_ENV: &str = "OLLAMA_API_KEY";

/// The most texts `POST /api/embed` accepts in one call.
const MAX_DOCUMENTS: usize = 1024;

/// The settings of an Ollama daemon: serializable, and the credential is
/// never serialized. [`connect`](Self::connect) puts it on a transport as an
/// [`Ollama`](super::Ollama) client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct OllamaConfig {
    /// The daemon's address.
    pub base_url: String,
    /// Bearer token for a secured daemon. Empty by default; no credential is
    /// sent when empty.
    pub api_key: Secret,
}

impl Default for OllamaConfig {
    fn default() -> Self {
        Self::new()
    }
}

impl OllamaConfig {
    /// The local daemon, unauthenticated.
    pub fn new() -> Self {
        Self {
            base_url: OLLAMA_API_BASE_URL.to_owned(),
            api_key: Secret::default(),
        }
    }

    /// The daemon `OLLAMA_API_BASE_URL` names, with the token
    /// `OLLAMA_API_KEY` carries. Both are optional: an unset base URL is the
    /// local daemon and an unset key is no credential.
    pub fn from_env() -> Result<Self, EnvError> {
        let mut provider = Self::new();
        if let Some(base_url) = env::optional(BASE_URL_ENV)? {
            provider = provider.with_base_url(base_url);
        }
        if let Some(api_key) = env::optional(API_KEY_ENV)? {
            provider.api_key = api_key.into();
        }
        Ok(provider)
    }

    /// Point the wires at another daemon.
    pub fn with_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.base_url = base_url.as_ref().trim_end_matches('/').to_owned();
        self
    }

    /// Authenticate against a proxied daemon.
    pub fn with_api_key(mut self, api_key: impl Into<Secret>) -> Self {
        self.api_key = api_key.into();
        self
    }

    /// The chat wire for `model`.
    pub(crate) fn completion(&self, model: impl Into<String>) -> Chat {
        Chat {
            provider: self.clone(),
            model: model.into(),
        }
    }

    /// Build an embedding wire reporting the supplied width, known model width,
    /// or zero if unknown. The width is metadata and is not sent to the daemon.
    pub(crate) fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        let model = model.into();
        let ndims = ndims
            .or_else(|| model_dimensions_from_identifier(&model))
            .unwrap_or_default();
        Embeddings {
            provider: self.clone(),
            model,
            ndims,
        }
    }

    /// The model-listing wire.
    pub(crate) fn models(&self) -> Models {
        Models {
            provider: self.clone(),
        }
    }

    /// One request to `path`, with the credential only when there is one.
    fn request(&self, method: http::Method, path: &str) -> http::request::Builder {
        let builder = http::Request::builder()
            .method(method)
            .uri(format!("{}{path}", self.base_url))
            .header(http::header::CONTENT_TYPE, "application/json");
        if self.api_key.is_empty() {
            builder
        } else {
            builder.header(
                http::header::AUTHORIZATION,
                format!("Bearer {}", self.api_key.expose()),
            )
        }
    }
}

/// The chat wire: `POST /api/chat`, NDJSON when streamed.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Chat {
    /// The daemon this wire speaks to.
    pub provider: OllamaConfig,
    /// The model to address.
    pub model: String,
}

impl Wire for Chat {
    type Op = Completion;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = super::OllamaDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .replay(self)
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        let mut body = OllamaCompletionRequest::try_from((self.model.as_str(), request))?;
        body.stream = mode == Mode::Streaming;
        crate::providers::internal::trace_json(
            crate::providers::internal::LogTarget::Completions,
            "Ollama completion request",
            &body,
        );
        let request = self
            .provider
            .request(http::Method::POST, "/api/chat")
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
        // Both modes decode the same record shape; streaming needs NDJSON framing.
        Ok(Encoded::new(
            request,
            match mode {
                Mode::Unary => Framing::Whole,
                Mode::Streaming => Framing::Ndjson,
            },
        ))
    }
    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        super::OllamaDecoder::default()
    }
}

impl crate::completion::ReplayTarget for Chat {
    fn api(&self) -> crate::message::Api {
        crate::message::Api::from_static("ollama.chat")
    }

    fn provider(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.model
    }
}

/// The embedding wire: `POST /api/embed`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embeddings {
    /// The daemon this wire speaks to.
    pub provider: OllamaConfig,
    /// The model to address.
    pub model: String,
    /// The width this wire reports, from the caller or the model's published
    /// dimensions. `0` means neither named one.
    pub ndims: usize,
}

impl Wire for Embeddings {
    type Op = Embedding;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = EmbeddingsDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .capabilities(Capabilities::embedding(MAX_DOCUMENTS, self.ndims))
    }

    fn encode(&self, texts: Vec<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        let body = serde_json::json!({ "model": self.model, "input": texts });
        let request = self
            .provider
            .request(http::Method::POST, "/api/embed")
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        EmbeddingsDecoder
    }
}

/// Decodes one `/api/embed` reply.
pub struct EmbeddingsDecoder;

impl<'id> Decoder<'id, Embedding> for EmbeddingsDecoder {
    type Event = OllamaEmbeddingResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_marker_keyed_frame(
            &frame.as_str(),
            &["embeddings"],
        )
    }

    fn decode(
        &mut self,
        reply: Self::Event,
        out: Out<'id, Embedding>,
    ) -> Result<Flow, ProviderError> {
        // Ollama counts the prompt it embedded and nothing else: every token
        // of an embedding is input.
        let usage = crate::completion::Usage {
            input_tokens: reply.prompt_eval_count,
            total_tokens: reply.prompt_eval_count,
            ..Default::default()
        };
        Ok(out.end(crate::embeddings::EmbeddingResponse {
            model: Some(reply.model),
            usage,
            ..crate::embeddings::EmbeddingResponse::from_vectors(reply.embeddings)
        }))
    }
}

/// The model-listing wire: `GET /api/tags`, unpaged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Models {
    /// The daemon this wire speaks to.
    pub provider: OllamaConfig,
}

impl Wire for Models {
    type Op = ModelListing;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ModelsDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
    }

    fn encode(&self, _cursor: Option<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        let request = self
            .provider
            .request(http::Method::GET, "/api/tags")
            .body(Body::empty())?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ModelsDecoder
    }
}

/// Decodes `GET /api/tags`. The daemon answers with every installed model at
/// once, so there is no cursor to follow.
pub struct ModelsDecoder;

impl<'id> Decoder<'id, ModelListing> for ModelsDecoder {
    type Event = ListModelsResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_marker_keyed_frame(&frame.as_str(), &["models"])
    }

    fn decode(
        &mut self,
        reply: Self::Event,
        out: Out<'id, ModelListing>,
    ) -> Result<Flow, ProviderError> {
        Ok(out.end(ModelPage {
            models: ModelList::new(reply.models.into_iter().map(ModelInfo::from).collect()),
            next: None,
        }))
    }
}

#[cfg(test)]
mod tests;
