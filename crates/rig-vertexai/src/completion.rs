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
use crate::types::completion_request::VertexCompletionRequest;
use crate::types::completion_response::{PROVIDER_NAME, VertexDecoder};
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::CompletionRequest;
use rig_core::driver::{Exchange, Opened, Opening, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::operation::Completion;
use rig_core::wire::{Descriptor, Mode, Wire};

/// `gemini-1.5-pro`
pub const GEMINI_1_5_PRO: &str = "gemini-1.5-pro";
/// `gemini-1.5-flash`
pub const GEMINI_1_5_FLASH: &str = "gemini-1.5-flash";
/// `gemini-1.5-pro-latest`
pub const GEMINI_1_5_PRO_LATEST: &str = "gemini-1.5-pro-latest";
/// `gemini-1.5-flash-latest`
pub const GEMINI_1_5_FLASH_LATEST: &str = "gemini-1.5-flash-latest";
/// `gemini-2.0-flash-exp`
pub const GEMINI_2_0_FLASH_EXP: &str = "gemini-2.0-flash-exp";
/// `gemini-2.5-flash-lite`
pub const GEMINI_2_5_FLASH_LITE: &str = "gemini-2.5-flash-lite";
/// `gemini-2.5-flash`
pub const GEMINI_2_5_FLASH: &str = "gemini-2.5-flash";
/// `gemini-2.5-pro`
pub const GEMINI_2_5_PRO: &str = "gemini-2.5-pro";

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

/// One `GenerateContent` request: the model it addresses and the request.
pub struct VertexRequest {
    model: String,
    request: VertexCompletionRequest,
}

impl Wire for GenerateContent {
    type Op = Completion;
    type Payload = VertexRequest;
    type Frame = vertexai::model::GenerateContentResponse;
    type Decoder<'id> = VertexDecoder;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(PROVIDER_NAME)
            .model(self.model.as_str())
            .replay(self)
    }

    fn encode(
        &self,
        request: CompletionRequest,
        _mode: Mode,
    ) -> Result<VertexRequest, EncodeError> {
        tracing::debug!(
            target: "rig_core::vertexai",
            "Vertex AI completion request: {request:?}"
        );
        Ok(VertexRequest {
            model: self.model.clone(),
            request: VertexCompletionRequest(request),
        })
    }
    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        VertexDecoder::default()
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

    /// Gemini on Vertex reads images in every role.
    fn accepts(&self, _model: &str) -> rig_core::completion::Accepts {
        rig_core::completion::Accepts::ALL
    }

    fn normalize_tool_call_id(
        &self,
        id: &str,
        model: &str,
        _source: Option<&rig_core::message::Origin>,
    ) -> String {
        rig_core::providers::gemini::completion::normalize_tool_call_id(model, id)
    }
}

/// Both modes send the unary RPC; a streamed call re-emits its reply.
impl Transport<GenerateContent> for VertexAi {
    fn send(
        &self,
        payload: VertexRequest,
        _exchange: Exchange,
    ) -> Opening<vertexai::model::GenerateContentResponse> {
        let VertexRequest { model, request } = payload;
        let generation_config = match request.generation_config() {
            Ok(config) => config,
            Err(error) => return Opening::failed(error),
        };
        let system_instruction = request.system_instruction();
        let tools = request.tools();
        let tool_config = request.tool_config();
        let contents = match request.contents() {
            Ok(contents) => contents,
            Err(error) => return Opening::failed(error),
        };
        let model_path = format!(
            "projects/{}/locations/{}/publishers/google/models/{model}",
            self.project(),
            self.location()
        );
        let client = self.clone();
        Opening::new(async move {
            let service = match client.inner().await {
                Ok(service) => service,
                Err(error) => return Err(ProviderError::request(error)),
            };
            let mut request_builder = service
                .generate_content()
                .set_model(&model_path)
                .set_contents(contents);
            if let Some(config) = generation_config {
                request_builder = request_builder.set_generation_config(config);
            }
            if let Some(system_instruction) = system_instruction {
                request_builder = request_builder.set_system_instruction(system_instruction);
            }
            if let Some(tools) = tools {
                request_builder = request_builder.set_tools([tools]);
            }
            if let Some(tool_config) = tool_config {
                request_builder = request_builder.set_tool_config(tool_config);
            }
            match request_builder.send().await {
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
