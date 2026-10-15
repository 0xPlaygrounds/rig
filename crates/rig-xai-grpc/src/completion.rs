//! The xAI completion wire over gRPC: `GetCompletion` for a unary call,
//! `GetCompletionChunk` for a streamed one.
//!
//! The request is rig-core's xAI Chat Completions body, read into the
//! `GetCompletionsRequest` it describes, and replies are read by the Chat
//! Completions decoder. The wire reports as provider `xai` on the
//! `openai.chat` API, so it shares xAI's catalog entries, typed options,
//! reply extras and history items with the REST Chat wire.
//!
//! ```no_run
//! use rig_xai_grpc::XaiGrpc;
//!
//! # async fn example() -> Result<(), rig_xai_grpc::XaiGrpcError> {
//! let model = XaiGrpc::new("API_KEY").await?.completion("grok-4.3");
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

use futures::StreamExt;
use rig_core::catalog::ModelFacts;
use rig_core::completion::options::{Mapping, OptionFields, OptionMap};
use rig_core::completion::{Accepts, CompletionRequest, LaterSystem, Media, Pairing, ReplayTarget};
use rig_core::driver::{Exchange, Opened, Opening, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::message::DocumentSourceKind as Source;
use rig_core::message::{self, AssistantContent, DocumentData, DocumentMediaType};
use rig_core::operation::Completion;
use rig_core::providers::openai::OpenAIConfig;
use rig_core::providers::openai::wire::Chat;
use rig_core::providers::xai;
use rig_core::wire::{Descriptor, Mode, Wire};
use serde_json::{Map, Value};

use crate::XaiGrpc;
use crate::proto::{GetChatCompletionChunk, GetChatCompletionResponse, GetCompletionsRequest};

/// One frame of a reply: a unary call's whole response, or one chunk of a
/// stream.
#[derive(Clone, Debug, PartialEq)]
pub enum XaiFrame {
    /// The `GetCompletion` response.
    Whole(GetChatCompletionResponse),
    /// One `GetCompletionChunk` message.
    Chunk(GetChatCompletionChunk),
}

/// The xAI chat endpoint for one model over gRPC.
#[derive(Clone, Debug, PartialEq)]
pub struct GetCompletion {
    /// The xAI Chat Completions wire whose body the request is.
    chat: Chat,
}

impl GetCompletion {
    /// The wire for `model`.
    pub fn new(model: impl Into<String>) -> Self {
        // The key is never read: the gRPC transport authenticates the call.
        let provider = OpenAIConfig::with_key(&xai::DIALECT, "");
        Self {
            chat: Chat::new(provider, model),
        }
    }

    /// The same wire, encoding with `facts` and pricing its replies by
    /// them.
    pub fn with_facts(mut self, facts: ModelFacts) -> Self {
        self.chat = self.chat.with_facts(facts);
        self
    }
}

impl Wire for GetCompletion {
    type Op = Completion;
    type Payload = GetCompletionsRequest;
    type Frame = XaiFrame;
    type Decoder<'id> = crate::streaming::GrpcAdapter;
    type Reassembler = crate::streaming::document::XaiDocument;

    fn describe(&self) -> Descriptor<'_> {
        self.chat.describe().replay(self)
    }

    /// The xAI Chat Completions body, read into the request it describes:
    /// a field the proto does not declare is an error rather than dropped.
    fn encode(
        &self,
        request: CompletionRequest,
        _mode: Mode,
    ) -> Result<GetCompletionsRequest, EncodeError> {
        let body = self.chat.request_body(&request, self, Mode::Unary)?;
        crate::chat::request(body.deserialize::<crate::chat::ChatBody>()?)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        crate::streaming::GrpcAdapter::new(self.chat.decoder())
    }
}

impl ReplayTarget for GetCompletion {
    /// xAI's Chat Completions mapping. The gRPC request has no prompt cache
    /// key.
    fn map_options(&self, request: &CompletionRequest, fields: OptionFields<'_>) -> OptionMap {
        let cache_key = fields.cache_key;
        let mut map = self.chat.map_options(request, fields);
        map.cache_key = Mapping::of(cache_key, |_| {
            Mapping::unsupported("the xAI gRPC request has no prompt cache key")
        });
        map
    }

    fn facts(&self) -> Option<&ModelFacts> {
        self.chat.facts()
    }

    fn api(&self) -> message::Api {
        self.chat.api()
    }

    fn provider(&self) -> &str {
        self.chat.provider()
    }

    fn model(&self) -> &str {
        self.chat.model()
    }

    fn accepts(&self, model: &str) -> Accepts {
        self.chat.accepts(model)
    }

    /// Images as on the Chat wire; a text document, a file id, and a PDF as
    /// data or a URL; no audio or video, which the proto has no part for.
    fn encodes(&self, model: &str, media: Media<'_>) -> bool {
        match media {
            Media::Image(..) => self.chat.encodes(model, media),
            Media::Document(document) => match &document.data {
                DocumentData::Text(_) | DocumentData::File(Source::FileId(_)) => true,
                DocumentData::File(Source::Base64(_) | Source::Url(_)) => {
                    document.media_type == Some(DocumentMediaType::PDF)
                }
                DocumentData::File(_) => false,
            },
            Media::Audio(_) | Media::Video(_) => false,
        }
    }

    fn normalize_tool_call_id(
        &self,
        id: &str,
        model: &str,
        source: Option<&message::Origin>,
    ) -> String {
        self.chat.normalize_tool_call_id(id, model, source)
    }

    fn continues_stored(&self, request: &CompletionRequest) -> bool {
        self.chat.continues_stored(request)
    }

    fn declares_tools(&self, request: &CompletionRequest) -> bool {
        self.chat.declares_tools(request)
    }

    fn identity(&self, item: &Value) -> Map<String, Value> {
        self.chat.identity(item)
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        self.chat.call_id_slot()
    }

    fn states_finish_reason(&self) -> bool {
        self.chat.states_finish_reason()
    }

    fn binds_context(&self, model: &str) -> bool {
        self.chat.binds_context(model)
    }

    fn drops_unbound_items(&self, request: &CompletionRequest) -> bool {
        self.chat.drops_unbound_items(request)
    }

    fn takes_documents(&self) -> bool {
        self.chat.takes_documents()
    }

    fn needs_next(&self, item: &Value) -> bool {
        self.chat.needs_next(item)
    }

    fn hosted_pair(&self, item: &Value) -> Option<(Pairing, String)> {
        self.chat.hosted_pair(item)
    }

    fn starts_with_user(&self) -> bool {
        self.chat.starts_with_user()
    }

    fn later_system(&self, model: &str) -> LaterSystem {
        self.chat.later_system(model)
    }

    fn alternates_roles(&self) -> bool {
        self.chat.alternates_roles()
    }

    fn hosted_needs_tools(&self) -> bool {
        self.chat.hosted_needs_tools()
    }

    fn result_parts(&self, model: &str) -> bool {
        self.chat.result_parts(model)
    }

    fn sends_alone(&self, block: &AssistantContent) -> bool {
        self.chat.sends_alone(block)
    }
}

impl Transport<GetCompletion> for XaiGrpc {
    fn send(&self, request: GetCompletionsRequest, exchange: Exchange) -> Opening<XaiFrame> {
        let mode = exchange.mode;
        let mut client = self.grpc_client();
        Opening::new(async move {
            Ok(match mode {
                Mode::Unary => match client.get_completion(request).await {
                    Ok(response) => {
                        let response = response.into_inner();
                        let id = Some(response.id.clone());
                        Opened::new(futures::stream::iter([Ok(XaiFrame::Whole(response))]))
                            .with_request_id(id)
                    }
                    Err(status) => Opened::failed(rpc_error(&status)),
                },
                Mode::Streaming => match client.get_completion_chunk(request).await {
                    Ok(response) => {
                        let mut chunks = response.into_inner();
                        // Stop receiving after a tonic failure.
                        Opened::new(async_stream::stream! {
                            while let Some(item) = chunks.next().await {
                                match item {
                                    Ok(chunk) => yield Ok(XaiFrame::Chunk(chunk)),
                                    Err(status) => {
                                        yield Err(rpc_error(&status));
                                        break;
                                    }
                                }
                            }
                        })
                    }
                    Err(status) => Opened::failed(rpc_error(&status)),
                },
            })
        })
    }
}

/// The tonic status as a provider failure, its gRPC code kept as the
/// provider code.
pub(crate) fn rpc_error(status: &tonic::Status) -> ProviderError {
    ProviderError::from_provider_body(status.to_string())
        .with_provider_code(Some(grpc_code_name(status.code())))
}

/// The gRPC status code's canonical name (`UNAVAILABLE`).
pub(crate) fn grpc_code_name(code: tonic::Code) -> String {
    format!("{code:?}")
        .chars()
        .fold(String::new(), |mut name, c| {
            if c.is_ascii_uppercase() && !name.is_empty() {
                name.push('_');
            }
            name.push(c.to_ascii_uppercase());
            name
        })
}

#[cfg(test)]
#[allow(clippy::expect_used, clippy::panic)]
mod tests;
