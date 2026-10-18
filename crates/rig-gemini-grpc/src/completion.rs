//! The Gemini `GenerateContent` completion wire over gRPC: `GenerateContent`
//! for a unary call, `StreamGenerateContent` for a streamed one.
//!
//! ```no_run
//! use rig_core::Model;
//! use rig_gemini_grpc::{GeminiGrpc, completion::{GEMINI_2_5_FLASH, GenerateContent}};
//!
//! # async fn example() -> Result<(), rig_gemini_grpc::GeminiGrpcError> {
//! let model = GeminiGrpc::new("API_KEY").await?.completion(GEMINI_2_5_FLASH);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

/// `gemini-2.5-flash` completion model
pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
/// `gemini-2.0-flash-lite` completion model
pub const GEMINI_2_0_FLASH_LITE: &str = "gemini-2.0-flash-lite";
/// `gemini-2.0-flash` completion model
pub const GEMINI_2_0_FLASH: &str = "gemini-2.0-flash";

use futures::StreamExt;
use rig_core::completion::CompletionRequest;
use rig_core::driver::{Exchange, Opened, Opening, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::message;
use rig_core::operation::Completion;
use rig_core::providers::gemini::completion as rest;
use rig_core::providers::gemini::streaming::document::GenerateContentResponse as Document;
use rig_core::providers::gemini::streaming::{GenerateContentChunk, GenerateContentDecoder};
use rig_core::telemetry::GenAiOperation;
use rig_core::wire::{Descriptor, Mode, Wire};
use serde_json::Value;

use super::GeminiGrpc;
use super::proto::{GenerateContentRequest, GenerateContentResponse};

/// Stable descriptor name reported on normalized responses from this provider.
pub const PROVIDER_NAME: &str = "gemini-grpc";

/// The `GenerateContent` endpoint for one model.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerateContent {
    pub model: String,
    /// The model facts the encoder reads and replies are priced by.
    pub facts: rig_core::catalog::ModelFacts,
}

impl GenerateContent {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
            facts: rig_core::catalog::ModelFacts::default(),
        }
    }

    /// The same wire, encoding with `facts` and pricing its replies by
    /// them.
    pub fn with_facts(mut self, facts: rig_core::catalog::ModelFacts) -> Self {
        self.facts = facts;
        self
    }
}

impl Wire for GenerateContent {
    type Op = Completion;
    type Payload = GenerateContentRequest;
    /// Each reply's REST chunk, transcoded once by the transport
    /// ([`rest_chunk`]) and read by the Gemini API's decoder.
    type Frame = GenerateContentChunk;
    type Event = GenerateContentChunk;
    type Decoder<'id> = GenerateContentDecoder;
    type Reassembler = Document;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .facts(&self.facts)
            .telemetry(|_| GenAiOperation::GenerateContent)
            .replay(self)
    }

    /// The REST wire's request, transcoded: every field the shared encoder
    /// builds reaches the protobuf request, and one the proto does not
    /// declare is an error rather than dropped.
    fn encode(
        &self,
        request: CompletionRequest,
        _mode: Mode,
    ) -> Result<GenerateContentRequest, EncodeError> {
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let body = rest::request_body(&request, self, &model, None, |body| {
            body.insert("model".to_owned(), format!("models/{model}").into());
        })?;
        Ok(crate::rest::from_rest(serde_json::to_value(&body)?)?)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        GenerateContentDecoder::default()
    }
}

impl rig_core::completion::ReplayTarget for GenerateContent {
    /// The GenerateContent mapping, for what the gRPC request declares.
    fn map_options(
        &self,
        request: &CompletionRequest,
        fields: rig_core::completion::options::OptionFields<'_>,
    ) -> rig_core::completion::options::OptionMap {
        let model = request.model.as_deref().unwrap_or(&self.model);
        rest::generate_content_options(&self.facts, model, rest::Route::Grpc, fields)
    }

    fn facts(&self) -> Option<&rig_core::catalog::ModelFacts> {
        Some(&self.facts)
    }

    fn api(&self) -> message::Api {
        message::Api::from_static("gemini.generate_content")
    }

    fn provider(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> &str {
        &self.model
    }

    /// What the model reads, as on every GenerateContent wire.
    fn accepts(&self, model: &str) -> rig_core::completion::Accepts {
        rest::accepts(model)
    }

    /// The media the Gemini API takes, as on the REST wire.
    fn encodes(&self, _model: &str, media: rig_core::completion::Media<'_>) -> bool {
        rest::encodes(media, false)
    }

    fn normalize_tool_call_id(
        &self,
        id: &str,
        model: &str,
        _source: Option<&message::Origin>,
    ) -> String {
        rest::normalize_tool_call_id(model, id)
    }

    /// Gemini takes system text only in `systemInstruction`: later system
    /// messages fold into the leading one, as pi's `collapseSystemMessages`.
    fn later_system(&self, _model: &str) -> rig_core::completion::LaterSystem {
        rig_core::completion::LaterSystem::Leading
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        rest::CALL_ID_SLOT
    }

    /// Tools in `additional_params` or a cached content count, as on the
    /// REST wire.
    fn declares_tools(&self, request: &CompletionRequest) -> bool {
        rest::declares_tools(self, request)
    }
}

impl Transport<GenerateContent> for GeminiGrpc {
    fn send(
        &self,
        request: GenerateContentRequest,
        exchange: Exchange,
    ) -> Opening<GenerateContentChunk> {
        let mode = exchange.mode;
        let mut client = self.grpc_client();
        Opening::new(async move {
            Ok(match mode {
                Mode::Unary => match client.generate_content(request).await {
                    Ok(response) => unary(&response.into_inner())?,
                    Err(status) => Opened::failed(rpc_error(&status)),
                },
                Mode::Streaming => match client.stream_generate_content(request).await {
                    // Each message is transcoded as it arrives; the first
                    // failure ends the stream.
                    Ok(response) => Opened::new(futures::stream::unfold(
                        Some(response.into_inner()),
                        |chunks| async move {
                            let mut chunks = chunks?;
                            let chunk = match chunks.next().await? {
                                Ok(response) => rest_chunk(&response),
                                Err(status) => Err(rpc_error(&status)),
                            };
                            let rest = chunk.is_ok().then_some(chunks);
                            Some((chunk, rest))
                        },
                    )),
                    Err(status) => Opened::failed(rpc_error(&status)),
                },
            })
        })
    }
}

/// `response` as the REST chunk the Gemini API's decoder and reassembler
/// read: its canonical proto3 JSON. A transport of the wire transcodes each
/// reply once, with this; a message that does not transcode fails the call.
pub fn rest_chunk(
    response: &GenerateContentResponse,
) -> Result<GenerateContentChunk, ProviderError> {
    Ok(match crate::rest::to_rest(response)? {
        Value::Object(chunk) => GenerateContentChunk(chunk),
        _ => GenerateContentChunk::default(),
    })
}

/// A unary reply: its one message, whose REST JSON is the response's `raw`.
pub(crate) fn unary(
    response: &GenerateContentResponse,
) -> Result<Opened<GenerateContentChunk>, ProviderError> {
    let chunk = rest_chunk(response)?;
    let document = Value::Object(chunk.0.clone());
    Ok(Opened::new(futures::stream::iter([Ok(chunk)])).with_document(document))
}

/// Preserves tonic status display text with the RPC code, which decides
/// retryability. Transport failures use the same provider-body
/// representation.
pub(crate) fn rpc_error(status: &tonic::Status) -> ProviderError {
    ProviderError::from_provider_body(status.to_string())
        .with_provider_code(Some(grpc_code_name(status.code())))
}

/// The gRPC status code's canonical name (`UNAVAILABLE`): the code the
/// provider answered with, kept apart from the message so a report can
/// key on it.
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
pub(crate) mod tests;
