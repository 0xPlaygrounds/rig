//! Ollama daemon configuration and chat, embedding, and model-listing wires.
//!
//! ```
//! use rig_core::providers::ollama::Ollama;
//! let chat = Ollama::new().chat("qwen3");
//! assert_eq!(chat.model, "qwen3");
//! ```

use crate::client::env::{self, EnvError};
use crate::completion::CompletionRequest;
use crate::driver::{HasEmbedding, HasModelListing};
use crate::embeddings::Embedding as Vector;
use crate::error::EncodeError;
use crate::model::{Model, ModelList};
use crate::operation::{Completion, Embedding, EmbeddingCapabilities, ModelListing};
use crate::wire::{
    Body, Decoder, Encoded, Framing, HasCompletion, Mode, Output, Secret, Sink, Wire, WireEvent,
    WireFrame,
};
use serde::{Deserialize, Serialize};

use super::{
    EmbeddingResponse as OllamaEmbeddingResponse, ListModelsResponse, OLLAMA_API_BASE_URL,
    OllamaCompletionRequest, OllamaDecoder, PROVIDER_NAME, model_dimensions_from_identifier,
};

/// The environment variable overriding the daemon's address.
const BASE_URL_ENV: &str = "OLLAMA_API_BASE_URL";

/// The environment variable carrying the bearer token a proxied daemon
/// requires. A local daemon needs none.
const API_KEY_ENV: &str = "OLLAMA_API_KEY";

/// The most texts `POST /api/embed` accepts in one call.
const MAX_DOCUMENTS: usize = 1024;

/// The shared configuration of an Ollama daemon.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Ollama {
    /// The daemon's address.
    pub base_url: String,
    /// Bearer token for a secured daemon. Empty by default; no credential is
    /// sent when empty.
    pub api_key: Secret,
}

impl Default for Ollama {
    fn default() -> Self {
        Self::new()
    }
}

impl Ollama {
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
    pub fn chat(&self, model: impl Into<String>) -> Chat {
        Chat {
            provider: self.clone(),
            model: model.into(),
        }
    }

    /// Build an embedding wire reporting the supplied width, known model width,
    /// or zero if unknown. The width is metadata and is not sent to the daemon.
    pub fn embeddings(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
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
    pub fn models(&self) -> Models {
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
    pub provider: Ollama,
    /// The model to address.
    pub model: String,
}

impl Wire for Chat {
    type Op = Completion;
    type Decoder = OllamaDecoder;

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
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

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        OllamaDecoder::default()
    }
}

/// The embedding wire: `POST /api/embed`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Embeddings {
    /// The daemon this wire speaks to.
    pub provider: Ollama,
    /// The model to address.
    pub model: String,
    /// The width this wire reports, from the caller or the model's published
    /// dimensions. `0` means neither named one.
    pub ndims: usize,
}

impl Wire for Embeddings {
    type Op = Embedding;
    type Decoder = EmbeddingsDecoder;

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn encode(&self, texts: Vec<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        let body = serde_json::json!({ "model": self.model, "input": texts });
        let request = self
            .provider
            .request(http::Method::POST, "/api/embed")
            .body(Body::Bytes(serde_json::to_vec(&body)?))?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        EmbeddingsDecoder
    }

    fn capabilities(&self) -> EmbeddingCapabilities {
        EmbeddingCapabilities::new(MAX_DOCUMENTS, self.ndims)
    }
}

/// Decodes one `/api/embed` reply.
pub struct EmbeddingsDecoder;

impl Decoder<Embedding> for EmbeddingsDecoder {
    type Event = OllamaEmbeddingResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_marker_keyed_frame(
            &frame.as_str(),
            &["embeddings"],
        )
    }

    fn interpret(&mut self, reply: Self::Event, out: &mut Output<Embedding>) {
        let raw = match serde_json::to_value(&reply) {
            Ok(raw) => raw,
            Err(error) => {
                out.push(Err(error.into()));
                return;
            }
        };
        // Ollama counts the prompt it embedded and nothing else: every token
        // of an embedding is input.
        let usage = crate::completion::Usage {
            input_tokens: reply.prompt_eval_count,
            total_tokens: reply.prompt_eval_count,
            ..Default::default()
        };
        // The vectors only; the operation's fold pairs them with the texts
        // that were sent, which `/api/embed` does not echo back.
        let vectors = reply
            .embeddings
            .into_iter()
            .map(|vec| Vector {
                document: String::new(),
                vec,
            })
            .collect();
        out.push(Ok(crate::embeddings::EmbeddingResponse::new(
            vectors,
            PROVIDER_NAME,
        )
        .with_model(reply.model)
        .with_usage(usage)
        .with_raw(raw)));
    }
}

/// The model-listing wire: `GET /api/tags`, unpaged.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Models {
    /// The daemon this wire speaks to.
    pub provider: Ollama,
}

impl Wire for Models {
    type Op = ModelListing;
    type Decoder = ModelsDecoder;

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn encode(&self, _request: (), _mode: Mode) -> Result<Encoded, EncodeError> {
        let request = self
            .provider
            .request(http::Method::GET, "/api/tags")
            .body(Body::empty())?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        ModelsDecoder
    }
}

/// Decodes `GET /api/tags`. The daemon answers with every installed model at
/// once, so there is no cursor to follow.
pub struct ModelsDecoder;

impl Decoder<ModelListing> for ModelsDecoder {
    type Event = ListModelsResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        crate::providers::internal::wire::classify_marker_keyed_frame(&frame.as_str(), &["models"])
    }

    fn interpret(&mut self, reply: Self::Event, out: &mut Output<ModelListing>) {
        out.push(Ok(ModelList::new(
            reply.models.into_iter().map(Model::from).collect(),
        )));
    }
}

impl HasCompletion for Ollama {
    type Wire = Chat;

    fn completion(&self, model: impl Into<String>) -> Chat {
        self.chat(model)
    }
}

impl HasEmbedding for Ollama {
    type Wire = Embeddings;

    fn embedding(&self, model: impl Into<String>, ndims: Option<usize>) -> Embeddings {
        self.embeddings(model, ndims)
    }
}

impl HasModelListing for Ollama {
    type Wire = Models;

    fn model_listing(&self) -> Models {
        self.models()
    }
}

#[cfg(test)]
mod tests;
