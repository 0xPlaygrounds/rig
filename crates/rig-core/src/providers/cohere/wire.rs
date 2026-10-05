//! Cohere's configuration, and its chat wire: the OpenAI Compatibility API,
//! or the native chat API when a request carries documents or the wire's
//! route asks for it.
//!
//! ```
//! use rig_core::providers::cohere::{ChatRoute, CohereConfig};
//! let config = CohereConfig::new("key").with_base_url("https://api.cohere.ai/");
//! assert_eq!(config.base_url, "https://api.cohere.ai");
//! let chat = config.completion("command-a-03-2025").with_route(ChatRoute::Native);
//! assert_eq!(chat.route, ChatRoute::Native);
//! ```

use crate::client::env::{self, EnvError};
use crate::completion::{CompletionRequest, ReplayTarget};
use crate::error::{EncodeError, ProviderError};
use crate::operation::Completion;
use crate::providers::openai::wire::{COHERE, Chat, OpenAIConfig};
use crate::wire::{
    Decoder, Descriptor, Encoded, Flow, Mode, Out, Secret, Wire, WireEvent, WireFrame,
};
use serde::{Deserialize, Serialize};

use super::NativeChat;

/// Cohere's API root.
const BASE_URL: &str = "https://api.cohere.ai";

/// Where Cohere's OpenAI Compatibility API sits under the API root.
const COMPATIBILITY_PATH: &str = "/compatibility/v1";

/// The environment variable carrying the API key.
const API_KEY_ENV: &str = "COHERE_API_KEY";

/// The settings of a Cohere provider: serializable, and the credential is
/// never serialized. [`connect`](Self::connect) puts it on a transport as a
/// [`Cohere`](super::Cohere) client.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CohereConfig {
    /// The API key, sent as `Authorization: Bearer`.
    pub api_key: Secret,
    /// The API root, without a trailing slash.
    pub base_url: String,
}

impl CohereConfig {
    /// Cohere with default settings.
    pub fn new(api_key: impl Into<Secret>) -> Self {
        Self {
            api_key: api_key.into(),
            base_url: BASE_URL.to_owned(),
        }
    }

    /// Cohere from `COHERE_API_KEY`.
    pub fn from_env() -> Result<Self, EnvError> {
        Ok(Self::new(env::required(API_KEY_ENV)?))
    }

    /// Point the wires at another API root.
    pub fn with_base_url(mut self, base_url: impl AsRef<str>) -> Self {
        self.base_url = base_url.as_ref().trim_end_matches('/').to_owned();
        self
    }

    /// The chat wire for `model` under this API root, on the
    /// [`ChatRoute::Auto`] route.
    pub fn completion(&self, model: impl Into<String>) -> CohereChat {
        let model = model.into();
        CohereChat {
            compatibility_api: OpenAIConfig::with_key(&COHERE, self.api_key.clone())
                .with_base_url(format!("{}{COMPATIBILITY_PATH}", self.base_url))
                .chat(model.clone()),
            native_api: NativeChat::new(self.clone(), model),
            route: ChatRoute::Auto,
        }
    }
}

/// Which Cohere API a chat request goes to.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRoute {
    /// The native API for a request that carries documents, so Cohere
    /// grounds the answer in them and cites them, and the Compatibility
    /// API for any other.
    #[default]
    Auto,
    /// The native API for every request.
    Native,
    /// The Compatibility API for every request, with documents sent as
    /// text in the history.
    Compatibility,
}

/// Cohere chat: each request goes to the OpenAI Compatibility API or the
/// native chat API, as [`ChatRoute`] says.
///
/// The two are different APIs to history replay. A turn made on one route
/// replays on the other from its canonical fields only, so its citations
/// and tool plan stay behind. [`ChatRoute::Native`] keeps a whole
/// conversation on the native API.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CohereChat {
    /// The Compatibility API's wire.
    pub compatibility_api: Chat,
    /// The native API's wire.
    pub native_api: NativeChat,
    /// Which API each request goes to.
    pub route: ChatRoute,
}

impl CohereChat {
    /// Send requests by `route`.
    pub fn with_route(mut self, route: ChatRoute) -> Self {
        self.route = route;
        self
    }

    /// Ask both APIs to hold every tool call to its tool's schema.
    pub fn with_strict_tools(mut self) -> Self {
        self.compatibility_api = self.compatibility_api.with_strict_tools();
        self.native_api = self.native_api.with_strict_tools();
        self
    }

    /// Whether `request` goes to the native API.
    fn native_for(&self, request: &CompletionRequest) -> bool {
        match self.route {
            ChatRoute::Auto => !request.documents.is_empty(),
            ChatRoute::Native => true,
            ChatRoute::Compatibility => false,
        }
    }
}

impl Wire for CohereChat {
    type Op = Completion;
    type Payload = Encoded;
    type Frame = WireFrame;
    type Decoder<'id> = CohereDecoder;

    /// The Compatibility API's description, naming this wire as the replay
    /// target that routes each request.
    fn describe(&self) -> Descriptor<'_> {
        self.compatibility_api.describe().replay(self)
    }

    fn encode(&self, request: CompletionRequest, mode: Mode) -> Result<Encoded, EncodeError> {
        if self.native_for(&request) {
            self.native_api.encode(request, mode)
        } else {
            self.compatibility_api.encode(request, mode)
        }
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        CohereDecoder {
            native_api: Wire::decoder(&self.native_api),
            compatibility_api: Wire::decoder(&self.compatibility_api),
            native_seen: false,
        }
    }
}

/// Every request names its route, so these are only what the replay of a
/// history adapted to the wire as a whole reads: the Compatibility API's.
impl ReplayTarget for CohereChat {
    fn api(&self) -> crate::message::Api {
        self.compatibility_api.api()
    }

    fn provider(&self) -> &str {
        ReplayTarget::provider(&self.compatibility_api)
    }

    fn model(&self) -> &str {
        ReplayTarget::model(&self.compatibility_api)
    }

    fn accepts(&self, model: &str) -> crate::completion::Accepts {
        self.compatibility_api.accepts(model)
    }

    fn route(&self, request: &CompletionRequest) -> Option<&dyn ReplayTarget> {
        Some(if self.native_for(request) {
            &self.native_api
        } else {
            &self.compatibility_api
        })
    }
}

/// One frame of a Cohere chat reply, from whichever API answered.
pub enum CohereEvent {
    /// A native chat event or reply.
    Native(super::streaming::ChatEvent),
    /// A Compatibility API chunk, reply or signal.
    Compatibility(crate::providers::openai::wire::chat::ChatEvent),
}

/// Decodes a Cohere chat reply from either API. The decoder is not told
/// which API a request went to, so each frame is read by its shape: a
/// native event carries a `type`, and a native reply a `message`.
pub struct CohereDecoder {
    native_api: super::streaming::ChatDecoder,
    compatibility_api: crate::providers::openai::wire::chat::ChatDecoder,
    /// Whether a native frame arrived, so the end of the frames is read as
    /// the native API's.
    native_seen: bool,
}

/// Whether `data` is a native chat frame.
fn is_native(data: &str) -> bool {
    let Ok(serde_json::Value::Object(frame)) = serde_json::from_str(data) else {
        return false;
    };
    frame.get("type").is_some_and(serde_json::Value::is_string)
        || (frame.contains_key("message") && !frame.contains_key("choices"))
}

impl<'id> Decoder<'id, Completion> for CohereDecoder {
    type Event = CohereEvent;

    fn classify(&self, frame: WireFrame) -> WireEvent<CohereEvent> {
        if is_native(&frame.as_str()) {
            self.native_api.classify(frame).map(CohereEvent::Native)
        } else {
            self.compatibility_api
                .classify(frame)
                .map(CohereEvent::Compatibility)
        }
    }

    fn decode(
        &mut self,
        event: CohereEvent,
        out: Out<'id, Completion>,
    ) -> Result<Flow, ProviderError> {
        match event {
            CohereEvent::Native(event) => {
                self.native_seen = true;
                self.native_api.decode(event, out)
            }
            CohereEvent::Compatibility(event) => self.compatibility_api.decode(event, out),
        }
    }

    fn eof(&mut self, out: Out<'id, Completion>) -> Result<Flow, ProviderError> {
        if self.native_seen {
            self.native_api.eof(out)
        } else {
            self.compatibility_api.eof(out)
        }
    }
}

#[cfg(test)]
mod tests;
