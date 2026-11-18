//! The context window a llama.cpp server serves its model with, read from
//! its `GET /props`.

use serde::Deserialize;

use crate::error::{EncodeError, ProviderError};
use crate::operation::ContextWindow;
use crate::providers::openai::wire::{OpenAIConfig, get};
use crate::wire::{Descriptor, DocumentDecoder, Encoded, Mode, Wire};

/// The server's properties route, at its root.
const PROPS_PATH: &str = "/props";

/// The reply of `GET /props`, as far as the window goes.
#[derive(Debug, Deserialize)]
pub(crate) struct PropsResponse {
    /// The settings each request gets.
    default_generation_settings: GenerationSettings,
}

/// A server's settings for each request.
#[derive(Debug, Deserialize)]
struct GenerationSettings {
    /// The tokens a request and its reply may hold: one slot's share of
    /// the server's context.
    n_ctx: u32,
}

/// The wire asking the server behind a [`LLAMACPP`] configuration for its
/// window.
///
/// [`LLAMACPP`]: crate::providers::openai::wire::LLAMACPP
#[derive(Debug, Clone)]
pub(crate) struct Props {
    /// The server, as its OpenAI-compatible API is configured.
    pub(crate) provider: OpenAIConfig,
}

impl Wire for Props {
    type Op = ContextWindow;
    type Payload = Encoded;
    type Frame = crate::wire::WireFrame;
    type Event = PropsResponse;
    type Decoder<'id> = DocumentDecoder<ContextWindow, PropsResponse>;
    type Reassembler = crate::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME)
    }

    fn encode(&self, _request: (), _mode: Mode) -> Result<Encoded, EncodeError> {
        get(&self.provider, PROPS_PATH)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        DocumentDecoder::new(
            |body| {
                crate::providers::internal::wire::classify_marker_keyed_frame(
                    body,
                    &["default_generation_settings"],
                )
            },
            |reply: PropsResponse| match reply.default_generation_settings.n_ctx {
                0 => Err(ProviderError::Response(
                    "/props names no context window".to_owned(),
                )),
                window => Ok(window),
            },
        )
    }
}
