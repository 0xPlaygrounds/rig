//! The Vertex AI `GenerateContent` completion wire and model identifiers.
//! A streamed call re-emits the unary reply: this integration has no
//! streaming RPC.
//!
//! ```no_run
//! use rig_core::Model;
//! use rig_vertexai::{VertexAi, completion::{GEMINI_2_5_FLASH, GenerateContent}};
//!
//! # async fn example() -> Result<(), rig_vertexai::client::VertexAiClientError> {
//! let model = VertexAi::from_env()?.completion(GEMINI_2_5_FLASH);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

use super::VertexAi;
use crate::types::completion_response::{PROVIDER_NAME, VertexDecoder};
use base64::Engine as _;
use base64::engine::general_purpose::{STANDARD as BASE64, URL_SAFE};
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::CompletionRequest;
use rig_core::driver::{Exchange, Opened, Opening, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::operation::Completion;
use rig_core::providers::gemini::completion as rest;
use rig_core::wire::{Descriptor, Mode, Wire};

pub use rig_core::providers::gemini::completion::GEMINI_2_5_FLASH;

/// The `GenerateContent` endpoint for one model.
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
    /// The SDK request. Its `model` is the model id the request addresses;
    /// the transport qualifies it with the project and location.
    type Payload = vertexai::model::GenerateContentRequest;
    type Frame = vertexai::model::GenerateContentResponse;
    type Decoder<'id> = VertexDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .replay(self)
    }

    /// The REST wire's request, read into the SDK's types: every field the
    /// shared encoder builds reaches Vertex AI.
    fn encode(
        &self,
        request: CompletionRequest,
        _mode: Mode,
    ) -> Result<vertexai::model::GenerateContentRequest, EncodeError> {
        let model = request.model.clone().unwrap_or_else(|| self.model.clone());
        let mut body = rest::request_body(request, self, &model)?;
        let contents = body
            .get_mut("contents")
            .and_then(serde_json::Value::as_array_mut);
        for content in contents.into_iter().flatten() {
            standard_signatures(content);
        }
        body.insert("model".to_owned(), model.into());
        Ok(serde_json::from_value(serde_json::Value::Object(body))?)
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        VertexDecoder::default()
    }
}

/// Respell every thought signature in `content` in standard base64, the
/// only alphabet the SDK's byte fields read. Google's placeholder signature
/// is URL-safe base64; Vertex AI reads the bytes either spelling decodes
/// to, so both name the same signature.
fn standard_signatures(content: &mut serde_json::Value) {
    let parts = content
        .get_mut("parts")
        .and_then(serde_json::Value::as_array_mut);
    for part in parts.into_iter().flatten() {
        if let Some(signature) = part.get_mut("thoughtSignature")
            && let Some(text) = signature.as_str()
            && BASE64.decode(text).is_err()
            && let Ok(bytes) = URL_SAFE.decode(text)
        {
            *signature = serde_json::Value::String(BASE64.encode(bytes));
        }
    }
}

impl rig_core::completion::ReplayTarget for GenerateContent {
    fn api(&self) -> rig_core::message::Api {
        rig_core::message::Api::from_static("vertexai.generate_content")
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

    /// The media Vertex AI takes: what the Gemini API takes, and image URLs inside
    /// function responses too.
    fn encodes(&self, _model: &str, media: rig_core::completion::Media<'_>) -> bool {
        rest::encodes(media, true)
    }

    fn normalize_tool_call_id(
        &self,
        id: &str,
        model: &str,
        _source: Option<&rig_core::message::Origin>,
    ) -> String {
        rest::normalize_tool_call_id(model, id)
    }

    fn call_id_slot(&self) -> Option<&'static str> {
        rest::CALL_ID_SLOT
    }

    /// Tools in `additional_params` or a cached content count, as on the
    /// REST wire.
    fn declares_tools(&self, request: &rig_core::completion::CompletionRequest) -> bool {
        rest::declares_tools(request)
    }
}

/// Both modes send the unary RPC; a streamed call re-emits its reply.
impl Transport<GenerateContent> for VertexAi {
    fn send(
        &self,
        mut request: vertexai::model::GenerateContentRequest,
        _exchange: Exchange,
    ) -> Opening<vertexai::model::GenerateContentResponse> {
        request.model = format!(
            "projects/{}/locations/{}/publishers/google/models/{}",
            self.project(),
            self.location(),
            request.model
        );
        let client = self.clone();
        Opening::new(async move {
            let service = match client.inner().await {
                Ok(service) => service,
                Err(error) => return Err(ProviderError::request(error)),
            };
            match service
                .generate_content()
                .with_request(request)
                .send()
                .await
            {
                Ok(response) => {
                    tracing::debug!(
                        target: "rig_core::vertexai",
                        "Vertex AI completion response: {response:?}"
                    );
                    Ok(Opened::new(futures::stream::iter([Ok(response)])))
                }
                Err(error) => Ok(Opened::failed(rpc_error(&error))),
            }
        })
    }
}

/// Preserves SDK error display text, HTTP status, RPC code, and retry hints.
/// Transport errors also use the provider-body representation.
fn rpc_error(error: &google_cloud_aiplatform_v1::Error) -> ProviderError {
    let status = error
        .http_status_code()
        .and_then(|code| rig_core::http_client::StatusCode::from_u16(code).ok());
    let code = error.status().map(|status| status.code.name());
    // SDK transport classifications supply retry hints when no RPC code exists.
    let transient = code.map(transient_rpc_code).or_else(|| {
        (error.is_transport() || error.is_io() || error.is_timeout() || error.is_connect())
            .then_some(true)
    });
    ProviderError::from_provider_body(error.to_string())
        .with_provider_status(status)
        .with_provider_code(code.map(str::to_owned))
        .with_transient(transient)
}

/// Recognizes transient RPC codes; unlisted codes are non-transient.
fn transient_rpc_code(code: &str) -> bool {
    matches!(
        code,
        "UNAVAILABLE" | "RESOURCE_EXHAUSTED" | "DEADLINE_EXCEEDED" | "ABORTED"
    )
}

#[cfg(test)]
#[allow(clippy::expect_used)]
mod tests;
