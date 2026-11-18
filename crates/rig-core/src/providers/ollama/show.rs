//! The context window an Ollama daemon serves a model with, read from its
//! `POST /api/show`.

use serde::Deserialize;
use serde_json::{Map, Value};

use crate::error::{EncodeError, ProviderError};
use crate::operation::ContextWindow;
use crate::providers::openai::wire::OpenAIConfig;
use crate::providers::openai::wire::json_post;
use crate::wire::{Descriptor, DocumentDecoder, Encoded, Mode, Wire};

/// The daemon's route for a model's details, at its root.
pub(crate) const SHOW_PATH: &str = "/api/show";

/// The reply of `POST /api/show`, as far as the window goes.
#[derive(Debug, Deserialize)]
pub(crate) struct ShowResponse {
    /// The model's parameters, one `key value` per line, such as
    /// `num_ctx 8192`.
    #[serde(default)]
    parameters: String,
    /// What the weights say of the model, such as `qwen3.context_length`
    /// under `general.architecture` `qwen3`.
    #[serde(default)]
    model_info: Map<String, Value>,
}

impl ShowResponse {
    /// The model's window: the `num_ctx` it is set to, which the daemon
    /// cuts a prompt to, else the context length its weights were trained
    /// for, which can be larger than the daemon's own default window.
    pub(crate) fn window(&self) -> Option<u32> {
        let set = self.parameters.lines().find_map(|line| {
            let mut words = line.split_whitespace();
            (words.next() == Some("num_ctx")).then(|| words.next())?
        });
        if let Some(set) = set {
            return set.parse().ok().filter(|window| *window > 0);
        }
        let architecture = self.model_info.get("general.architecture")?.as_str()?;
        let length = self
            .model_info
            .get(&format!("{architecture}.context_length"))?;
        u32::try_from(length.as_u64()?)
            .ok()
            .filter(|window| *window > 0)
    }
}

/// The wire asking the daemon behind an [`OLLAMA`] configuration for the
/// window it serves `model` with.
///
/// [`OLLAMA`]: crate::providers::openai::wire::OLLAMA
#[derive(Debug, Clone)]
pub(crate) struct Show {
    /// The daemon, as its OpenAI-compatible API is configured.
    pub(crate) provider: OpenAIConfig,
    /// The model asked about.
    pub(crate) model: String,
}

impl Wire for Show {
    type Op = ContextWindow;
    type Payload = Encoded;
    type Frame = crate::wire::WireFrame;
    type Event = ShowResponse;
    type Decoder<'id> = DocumentDecoder<ContextWindow, ShowResponse>;
    type Reassembler = crate::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME)
    }

    fn encode(&self, _request: (), _mode: Mode) -> Result<Encoded, EncodeError> {
        let body = serde_json::json!({ "model": self.model });
        json_post(&self.provider, SHOW_PATH, None, &body)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        DocumentDecoder::new(
            |body| {
                crate::providers::internal::wire::classify_marker_keyed_frame(
                    body,
                    &["parameters", "model_info"],
                )
            },
            |reply: ShowResponse| {
                reply.window().ok_or_else(|| {
                    ProviderError::Response("/api/show names no context window".to_owned())
                })
            },
        )
    }
}

#[cfg(test)]
mod tests;
