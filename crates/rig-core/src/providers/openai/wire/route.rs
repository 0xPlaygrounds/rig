//! The completion wire a dialect builds when asked for "a completion":
//! whichever of its two endpoints is its flagship.
//!
//! OpenAI, xAI and ChatGPT serve `/responses` as their primary completion
//! API and keep `/chat/completions` beside it; every OpenAI-*compatible*
//! gateway serves only the latter. So a [`Dialect`](super::Dialect) names
//! its [`Route`], and [`OpenAiWire`] is that route's wire — a wire choosing
//! a wire, with every [`Wire`] and [`Decoder`] method dispatching on the
//! variant. There is no second request conversion, no second decoder and no
//! second observation projection: the two arms are the two wires that
//! already exist. Naming a route explicitly is
//! [`OpenAI::chat`](super::OpenAI::chat) or
//! [`OpenAI::responses`](super::OpenAI::responses).

use serde::{Deserialize, Serialize};

use crate::completion::{CompletionError, CompletionRequest, ProviderCapabilities};
use crate::operation::Completion;
use crate::providers::openai::responses_api::streaming::{ResponsesDecoder, ResponsesEvent};
use crate::providers::openai::responses_api::wire::Responses;
use crate::telemetry::CompletionOperation;
use crate::wire::{
    Body, Decoder, Encoded, Mode, ObservationSink, Output, Wire, WireEvent, WireFrame,
};

use super::OpenAI;
use super::chat::{Chat, ChatDecoder, ChatEvent};

/// Which completion endpoint a dialect serves as its default.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum Route {
    /// `POST /chat/completions`: the one endpoint every dialect serves.
    Chat,
    /// `POST /responses`: the flagship of OpenAI, xAI and ChatGPT.
    Responses,
}

/// The dialect's default completion wire: its [`Route`]'s wire.
///
/// Both variants are the shared wire types on the same configuration, so
/// this is what [`Bound<OpenAI>::completion`](crate::driver::Bound) — and
/// the agent sugar on top of it — builds. A caller who wants a specific
/// endpoint names it and gets the concrete wire back; a caller who wants a
/// route-specific option on this one matches the variant.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OpenAiWire {
    /// The chat-completions wire.
    Chat(Chat),
    /// The Responses wire.
    Responses(Responses),
}

impl OpenAiWire {
    /// The wire for `model` on `provider`'s default route.
    pub fn new(provider: OpenAI, model: impl Into<String>) -> Self {
        match provider.dialect.quirks.completion_route {
            Route::Chat => Self::Chat(Chat::new(provider, model)),
            Route::Responses => Self::Responses(Responses::new(provider, model)),
        }
    }

    /// The configuration this wire speaks to.
    pub fn provider(&self) -> &OpenAI {
        match self {
            Self::Chat(wire) => &wire.provider,
            Self::Responses(wire) => &wire.provider,
        }
    }

    /// Sanitize tool schemas for OpenAI's strict mode on whichever route
    /// this is.
    pub fn with_strict_tools(self) -> Self {
        match self {
            Self::Chat(wire) => Self::Chat(wire.with_strict_tools()),
            Self::Responses(wire) => Self::Responses(wire.with_strict_tools()),
        }
    }
}

impl From<Chat> for OpenAiWire {
    fn from(wire: Chat) -> Self {
        Self::Chat(wire)
    }
}

impl From<Responses> for OpenAiWire {
    fn from(wire: Responses) -> Self {
        Self::Responses(wire)
    }
}

impl Wire for OpenAiWire {
    type Op = Completion;
    type Decoder = OpenAiDecoder;

    fn name(&self) -> &str {
        match self {
            Self::Chat(wire) => wire.name(),
            Self::Responses(wire) => wire.name(),
        }
    }

    fn model(&self) -> Option<&str> {
        match self {
            Self::Chat(wire) => wire.model(),
            Self::Responses(wire) => wire.model(),
        }
    }

    fn route(&self) -> Option<&str> {
        match self {
            Self::Chat(wire) => wire.route(),
            Self::Responses(wire) => wire.route(),
        }
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, CompletionError> {
        match self {
            Self::Chat(wire) => wire.encode(request, mode),
            Self::Responses(wire) => wire.encode(request, mode),
        }
    }

    fn decoder(&self) -> OpenAiDecoder {
        match self {
            Self::Chat(wire) => OpenAiDecoder::Chat(wire.decoder()),
            Self::Responses(wire) => OpenAiDecoder::Responses(wire.decoder()),
        }
    }

    fn capabilities(&self) -> ProviderCapabilities {
        match self {
            Self::Chat(wire) => wire.capabilities(),
            Self::Responses(wire) => wire.capabilities(),
        }
    }

    fn telemetry(&self, streaming: bool) -> CompletionOperation {
        match self {
            Self::Chat(wire) => wire.telemetry(streaming),
            Self::Responses(wire) => wire.telemetry(streaming),
        }
    }
}

/// One classified frame of whichever route is answering.
pub enum OpenAiEvent {
    /// A chat-completions frame.
    Chat(ChatEvent),
    /// A Responses frame.
    Responses(ResponsesEvent),
}

/// The chosen route's decoder.
pub enum OpenAiDecoder {
    /// The chat-completions state machine.
    Chat(ChatDecoder),
    /// The Responses state machine.
    Responses(ResponsesDecoder),
}

impl Decoder<Completion> for OpenAiDecoder {
    type Event = OpenAiEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<OpenAiEvent> {
        match self {
            Self::Chat(decoder) => decoder.classify(frame).map(OpenAiEvent::Chat),
            Self::Responses(decoder) => decoder.classify(frame).map(OpenAiEvent::Responses),
        }
    }

    fn interpret(&mut self, event: OpenAiEvent, out: &mut Output<Completion>) {
        match (self, event) {
            (Self::Chat(decoder), OpenAiEvent::Chat(event)) => decoder.interpret(event, out),
            (Self::Responses(decoder), OpenAiEvent::Responses(event)) => {
                decoder.interpret(event, out);
            }
            // A decoder is built fresh per reply and only ever sees the
            // events its own `classify` produced, so the routes cannot
            // cross; there is nothing for the other route's state machine to
            // do with a frame it never decoded.
            (Self::Chat(_), OpenAiEvent::Responses(_))
            | (Self::Responses(_), OpenAiEvent::Chat(_)) => {}
        }
    }

    fn finish(&mut self, out: &mut Output<Completion>) {
        match self {
            Self::Chat(decoder) => decoder.finish(out),
            Self::Responses(decoder) => decoder.finish(out),
        }
    }

    fn whole_reply(&mut self) {
        match self {
            Self::Chat(decoder) => decoder.whole_reply(),
            Self::Responses(decoder) => decoder.whole_reply(),
        }
    }

    fn flush_before_terminal_error(&mut self, out: &mut Output<Completion>) {
        match self {
            Self::Chat(decoder) => decoder.flush_before_terminal_error(out),
            Self::Responses(decoder) => decoder.flush_before_terminal_error(out),
        }
    }

    fn project(&self, payload: &[u8], sink: &mut dyn ObservationSink) {
        match self {
            Self::Chat(decoder) => decoder.project(payload, sink),
            Self::Responses(decoder) => decoder.project(payload, sink),
        }
    }

    fn document(&self) -> Option<serde_json::Value> {
        match self {
            Self::Chat(decoder) => decoder.document(),
            Self::Responses(decoder) => decoder.document(),
        }
    }

    fn continuation(&self) -> Option<http::Request<Body>> {
        match self {
            Self::Chat(decoder) => decoder.continuation(),
            Self::Responses(decoder) => decoder.continuation(),
        }
    }

    fn is_analysis_only(&self, frame: &WireFrame) -> bool {
        match self {
            Self::Chat(decoder) => decoder.is_analysis_only(frame),
            Self::Responses(decoder) => decoder.is_analysis_only(frame),
        }
    }

    fn is_finished(&self) -> bool {
        match self {
            Self::Chat(decoder) => decoder.is_finished(),
            Self::Responses(decoder) => decoder.is_finished(),
        }
    }
}
