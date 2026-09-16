//! The completion wire a configuration builds when asked for "a
//! completion": whichever of its two endpoints it is routed to.
//!
//! OpenAI, xAI and ChatGPT serve `/responses` as their primary completion
//! API and keep `/chat/completions` beside it; every OpenAI-*compatible*
//! gateway serves only the latter. So a [`Dialect`](super::Dialect) names
//! its flagship [`Route`], a configuration may pick the other one once with
//! [`OpenAI::with_route`](super::OpenAI::with_route), and [`OpenAiWire`] is
//! that route's wire — a wire choosing a wire, with every [`Wire`] and
//! [`Decoder`] method dispatching on the variant. There is no second
//! request conversion, no second decoder and no second observation
//! projection: the two arms are the two wires that already exist. Naming a
//! route for one model is [`OpenAI::chat`](super::OpenAI::chat) or
//! [`OpenAI::responses`](super::OpenAI::responses).

use serde::{Deserialize, Serialize};

use crate::completion::{CompletionError, CompletionRequest, ProviderCapabilities};
use crate::operation::Completion;
use crate::providers::openai::responses_api::streaming::{ResponsesDecoder, ResponsesEvent};
use crate::providers::openai::responses_api::wire::Responses;
use crate::providers::openai::responses_api::{
    ResponsesToolDefinition, SystemInstructionsPlacement,
};
use crate::telemetry::CompletionOperation;
use crate::wire::{
    Body, Decoder, Encoded, Mode, ObservationSink, Output, Wire, WireEvent, WireFrame,
};

use super::OpenAI;
use super::chat::{Chat, ChatDecoder, ChatEvent};

/// Ask whichever route this is.
///
/// Every [`Wire`] and [`Decoder`] method below is the same two-arm match on
/// the variant — the enum chooses a wire, and the method asks that wire — so
/// the match is written once here instead of fourteen times. The two methods
/// whose arms genuinely differ, `decoder` and `classify`, wrap their result
/// in the matching variant and are written out.
macro_rules! on_route {
    ($chosen:expr, $wire:ident => $ask:expr) => {
        match $chosen {
            Self::Chat($wire) => $ask,
            Self::Responses($wire) => $ask,
        }
    };
}

/// Which completion endpoint a configuration serves.
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
/// endpoint names it and gets the concrete wire back; every option either
/// route takes is also forwarded here, and is a no-op on the route that
/// has no such option, so `map_wire` reaches all of them.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum OpenAiWire {
    /// The chat-completions wire.
    Chat(Chat),
    /// The Responses wire.
    Responses(Responses),
}

impl OpenAiWire {
    /// The wire for `model` on `provider`'s
    /// [`completion_route`](OpenAI::completion_route).
    pub fn new(provider: OpenAI, model: impl Into<String>) -> Self {
        match provider.completion_route() {
            Route::Chat => Self::Chat(Chat::new(provider, model)),
            Route::Responses => Self::Responses(Responses::new(provider, model)),
        }
    }

    /// The configuration this wire speaks to.
    pub fn provider(&self) -> &OpenAI {
        on_route!(self, wire => &wire.provider)
    }

    /// Sanitize tool schemas for OpenAI's strict mode on whichever route
    /// this is.
    pub fn with_strict_tools(self) -> Self {
        self.on_chat(Chat::with_strict_tools)
            .on_responses(Responses::with_strict_tools)
    }

    /// Serialize tool-result content as arrays: a chat-completions shape,
    /// so a no-op on the Responses route, whose request has one content
    /// encoding.
    pub fn with_tool_result_array_content(self) -> Self {
        self.on_chat(Chat::with_tool_result_array_content)
    }

    /// Ask the provider to cache the prompt: an OpenRouter `cache_control`
    /// on the chat body, so a no-op on the Responses route.
    pub fn with_prompt_caching(self) -> Self {
        self.on_chat(Chat::with_prompt_caching)
    }

    /// Add a provider-side tool to every request: a Responses shape, so a
    /// no-op on the chat route, which carries no wire-level tools.
    pub fn with_tool(self, tool: impl Into<ResponsesToolDefinition>) -> Self {
        self.on_responses(|wire| wire.with_tool(tool))
    }

    /// Add provider-side tools to every request: a no-op on the chat
    /// route, which carries no wire-level tools.
    pub fn with_tools<I, Tool>(self, tools: I) -> Self
    where
        I: IntoIterator<Item = Tool>,
        Tool: Into<ResponsesToolDefinition>,
    {
        self.on_responses(|wire| wire.with_tools(tools))
    }

    /// Put Rig's system instructions somewhere other than the dialect's
    /// default placement: a no-op on the chat route, where a system message
    /// has one place to go.
    pub fn with_system_instructions_placement(
        self,
        placement: SystemInstructionsPlacement,
    ) -> Self {
        self.on_responses(|wire| wire.with_system_instructions_placement(placement))
    }

    /// Send Rig's system instructions as `system` messages in `input`: a
    /// no-op on the chat route, which sends them that way already.
    pub fn with_system_instructions_as_messages(self) -> Self {
        self.on_responses(Responses::with_system_instructions_as_messages)
    }

    /// Apply a chat-route option; the Responses route is left as it is.
    fn on_chat(self, option: impl FnOnce(Chat) -> Chat) -> Self {
        match self {
            Self::Chat(wire) => Self::Chat(option(wire)),
            responses => responses,
        }
    }

    /// Apply a Responses-route option; the chat route is left as it is.
    fn on_responses(self, option: impl FnOnce(Responses) -> Responses) -> Self {
        match self {
            Self::Responses(wire) => Self::Responses(option(wire)),
            chat => chat,
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
        on_route!(self, wire => wire.name())
    }

    fn model(&self) -> Option<&str> {
        on_route!(self, wire => wire.model())
    }

    fn route(&self) -> Option<&str> {
        on_route!(self, wire => wire.route())
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, CompletionError> {
        on_route!(self, wire => wire.encode(request, mode))
    }

    fn decoder(&self, mode: Mode) -> OpenAiDecoder {
        match self {
            Self::Chat(wire) => OpenAiDecoder::Chat(wire.decoder(mode)),
            Self::Responses(wire) => OpenAiDecoder::Responses(wire.decoder(mode)),
        }
    }

    fn capabilities(&self) -> ProviderCapabilities {
        on_route!(self, wire => wire.capabilities())
    }

    fn telemetry(&self, streaming: bool) -> CompletionOperation {
        on_route!(self, wire => wire.telemetry(streaming))
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
        on_route!(self, decoder => decoder.finish(out))
    }

    fn flush_before_terminal_error(&mut self, out: &mut Output<Completion>) {
        on_route!(self, decoder => decoder.flush_before_terminal_error(out))
    }

    fn project(&self, payload: &[u8], sink: &mut dyn ObservationSink) {
        on_route!(self, decoder => decoder.project(payload, sink))
    }

    fn document(&self) -> Option<serde_json::Value> {
        on_route!(self, decoder => decoder.document())
    }

    fn continuation(&self) -> Option<http::Request<Body>> {
        on_route!(self, decoder => decoder.continuation())
    }

    fn is_analysis_only(&self, frame: &WireFrame) -> bool {
        on_route!(self, decoder => decoder.is_analysis_only(frame))
    }

    fn is_finished(&self) -> bool {
        on_route!(self, decoder => decoder.is_finished())
    }
}

#[cfg(test)]
mod tests;
