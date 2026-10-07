//! The Gemini `GenerateContent` completion wire over gRPC.
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
use rig_core::error::EncodeError;
use rig_core::error::ProviderError;
use rig_core::message;
use rig_core::operation::Completion;
use rig_core::providers::gemini::completion as rest;
use rig_core::wire::{Descriptor, Mode, Wire};

use super::GeminiGrpc;
use super::proto::{GenerateContentRequest, GenerateContentResponse};

/// The `GenerateContent` endpoint for one model: `GenerateContent` for a
/// unary call, `StreamGenerateContent` for a streamed one.
#[derive(Clone, Debug, PartialEq)]
pub struct GenerateContent {
    pub model: String,
}

impl GenerateContent {
    pub fn new(model: impl Into<String>) -> Self {
        Self {
            model: model.into(),
        }
    }
}

impl Wire for GenerateContent {
    type Op = Completion;
    type Payload = GenerateContentRequest;
    type Frame = GenerateContentResponse;
    type Decoder<'id> = crate::streaming::GrpcAdapter;
    type Reassembler = rig_core::providers::gemini::streaming::document::GenerateContentResponse;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
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
        crate::streaming::GrpcAdapter::default()
    }
}

impl rig_core::completion::ReplayTarget for GenerateContent {
    /// The GenerateContent mapping, for what the gRPC request declares.
    fn map_options(
        &self,
        request: &rig_core::completion::CompletionRequest,
        fields: rig_core::completion::options::OptionFields<'_>,
    ) -> rig_core::completion::options::OptionMap {
        let model = request.model.as_deref().unwrap_or(&self.model);
        rest::generate_content_options(model, rest::Route::Grpc, fields)
    }

    fn api(&self) -> rig_core::message::Api {
        rig_core::message::Api::from_static("gemini.generate_content")
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
    fn declares_tools(&self, request: &rig_core::completion::CompletionRequest) -> bool {
        rest::declares_tools(self, request)
    }
}

impl Transport<GenerateContent> for GeminiGrpc {
    fn send(
        &self,
        request: GenerateContentRequest,
        exchange: Exchange,
    ) -> Opening<GenerateContentResponse> {
        let mode = exchange.mode;
        let mut client = self.grpc_client();
        Opening::new(async move {
            Ok(match mode {
                Mode::Unary => match client.generate_content(request).await {
                    Ok(response) => unary(response.into_inner())?,
                    Err(status) => Opened::failed(rpc_error(&status)),
                },
                Mode::Streaming => match client.stream_generate_content(request).await {
                    Ok(response) => {
                        let mut chunks = response.into_inner();
                        // Stop receiving after a tonic failure.
                        Opened::new(async_stream::stream! {
                            while let Some(item) = chunks.next().await {
                                match item {
                                    Ok(chunk) => yield Ok(chunk),
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

/// A unary reply: its one message, whose REST JSON is the response's `raw`.
/// A message that does not transcode fails as its decoding would.
pub(crate) fn unary(
    response: GenerateContentResponse,
) -> Result<Opened<GenerateContentResponse>, ProviderError> {
    let document = crate::streaming::document::rest_document(&response)?;
    Ok(Opened::new(futures::stream::iter([Ok(response)])).with_document(document))
}

/// Stable descriptor name reported on normalized responses from this provider.
pub const PROVIDER_NAME: &str = "gemini-grpc";

/// Preserves tonic status display text with RPC code and retry classification.
/// Transport failures use the same provider-body representation.
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
#[allow(
    clippy::expect_used,
    clippy::panic,
    clippy::unwrap_used,
    clippy::indexing_slicing
)]
pub(crate) mod tests;
