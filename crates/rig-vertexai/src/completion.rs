//! The Vertex AI `GenerateContent` completion wire and model identifiers.
//! A streamed call re-emits the unary reply: this integration has no
//! streaming RPC.
//!
//! ```no_run
//! use rig_core::wire::Wire as _;
//! use rig_vertexai::{VertexAi, completion::{GEMINI_2_5_FLASH, GenerateContent}};
//!
//! # async fn example() -> Result<(), rig_vertexai::client::VertexAiClientError> {
//! let model = GenerateContent::new(GEMINI_2_5_FLASH).on(VertexAi::from_env()?);
//! # let _ = model;
//! # Ok(())
//! # }
//! ```

use super::VertexAi;
use crate::types::completion_request::VertexCompletionRequest;
use crate::types::completion_response::{PROVIDER_NAME, VertexDecoder};
use google_cloud_aiplatform_v1 as vertexai;
use rig_core::completion::CompletionRequest;
use rig_core::driver::{Observation, Opened, Transport};
use rig_core::error::{EncodeError, ProviderError};
use rig_core::operation::Completion;
use rig_core::wire::{Mode, Wire};

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
    type Decoder = VertexDecoder;

    fn name(&self) -> &str {
        PROVIDER_NAME
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
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

    fn decoder(&self, _mode: Mode) -> VertexDecoder {
        VertexDecoder::default()
    }
}

/// Both modes send the unary RPC; a streamed call re-emits its reply.
impl Transport<GenerateContent> for VertexAi {
    fn send(
        &self,
        payload: VertexRequest,
        _mode: Mode,
        _observation: Option<Observation>,
    ) -> Result<
        impl Future<Output = Opened<VertexRequest, vertexai::model::GenerateContentResponse>>
        + Send
        + 'static
        + use<>,
        ProviderError,
    > {
        let VertexRequest { model, request } = payload;
        let generation_config = request.generation_config()?;
        let system_instruction = request.system_instruction();
        let tools = request.tools();
        let tool_config = request.tool_config();
        let contents = request.contents()?;
        let model_path = format!(
            "projects/{}/locations/{}/publishers/google/models/{model}",
            self.project(),
            self.location()
        );
        let client = self.clone();
        Ok(async move {
            let service = match client.inner().await {
                Ok(service) => service,
                Err(error) => return Opened::failed(ProviderError::Provider(error.to_string())),
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
                    Opened::new(futures::stream::iter([Ok(response)]))
                }
                Err(error) => Opened::failed(rpc_error(&error)),
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
