//! Shared request driver for OpenAI-style image generation endpoints.
//!
//! OpenAI, Azure OpenAI, xAI and Hyperbolic each build a provider-specific
//! JSON body, but the tail is identical: POST the body, classify the response
//! through the provider's success-or-error envelope, and convert the payload.
//! The body and path stay with each provider — those are the real wire
//! differences — while this driver owns the shared send/decode tail.

use base64::Engine;
use base64::prelude::BASE64_STANDARD;
use bytes::Bytes;
use serde::de::DeserializeOwned;

use super::envelope::ProviderEnvelope;
use crate::client::{Client, Provider};
use crate::http_client::{self, HttpClientExt};
use crate::image_generation::{
    self, ImageGenerationError, ImageGenerationRequest, NormalizeImageGenerationResponse,
};
use crate::wasm_compat::{WasmCompatSend, WasmCompatSync};

/// Decodes the first base64 image selected from a provider response while
/// retaining that response in Rig's normalized wrapper.
/// Decodes the base64 image `select` picks out of a provider payload, with
/// the provider's own wording for the missing-image and decode-failure
/// errors preserved.
pub(crate) fn decode_base64_image<T>(
    response: &T,
    select: fn(&T) -> Option<&str>,
    missing_message: &'static str,
    decode_error_prefix: Option<&'static str>,
) -> Result<Vec<u8>, ImageGenerationError> {
    let encoded = select(response)
        .ok_or_else(|| ImageGenerationError::ResponseError(missing_message.to_owned()))?;
    BASE64_STANDARD.decode(encoded).map_err(|error| {
        ImageGenerationError::ResponseError(match decode_error_prefix {
            Some(prefix) => format!("{prefix}{error}"),
            None => error.to_string(),
        })
    })
}

#[doc(hidden)]
pub trait JsonImageGenerationProvider: Provider {
    const IMAGE_GENERATION_PATH: &'static str;

    /// Stable descriptor name of the provider, stamped on every normalized
    /// response — an input to normalization, never hardcoded in the shared
    /// conversion.
    const PROVIDER_NAME: &'static str;

    /// The provider's transport request-id response header, when it has one.
    const REQUEST_ID_HEADER: Option<&'static str> = None;

    /// The provider's own image-generation payload: what the model's inherent
    /// `raw_image_generation` returns, and what normalizes onto
    /// [`image_generation::ImageGenerationResponse`].
    type Response: DeserializeOwned
        + serde::Serialize
        + WasmCompatSend
        + WasmCompatSync
        + NormalizeImageGenerationResponse;
    fn image_generation_request_builder<H>(
        client: &Client<Self, H>,
        _model: &str,
    ) -> Result<http_client::Builder, ImageGenerationError>
    where
        H: HttpClientExt,
    {
        Ok(client.post(Self::IMAGE_GENERATION_PATH)?)
    }

    fn image_generation_request_body(
        model: &str,
        request: ImageGenerationRequest,
    ) -> Result<serde_json::Value, ImageGenerationError>;
}

#[doc(hidden)]
#[derive(Clone)]
pub struct GenericImageGenerationModel<Ext, H> {
    client: Client<Ext, H>,
    /// Name of the image generation model.
    pub model: String,
}

impl<Ext, H> GenericImageGenerationModel<Ext, H> {
    /// Creates an image generation model backed by `client`.
    pub fn new(client: Client<Ext, H>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }

    /// Creates an image generation model from a borrowed model name.
    pub fn with_model(client: Client<Ext, H>, model: &str) -> Self {
        Self::new(client, model)
    }
}

impl<Ext, H> GenericImageGenerationModel<Ext, H>
where
    Ext: JsonImageGenerationProvider + Clone + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    /// Perform the generation and return the provider's native response
    /// instead of the normalized [`image_generation::ImageGenerationResponse`].
    /// Same request, transport, parser, and error path as
    /// [`image_generation::ImageGenerationModel::image_generation`].
    pub async fn raw_image_generation(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<Ext::Response, ImageGenerationError> {
        self.raw_image_generation_with_request_id(request)
            .await
            .map(|(response, _)| response)
    }

    /// [`Self::raw_image_generation`] plus the transport request id from the
    /// provider's request-id response header, when it carries one.
    pub async fn raw_image_generation_with_request_id(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<(Ext::Response, Option<String>), ImageGenerationError> {
        let builder = Ext::image_generation_request_builder(&self.client, &self.model)?;
        let body = Ext::image_generation_request_body(&self.model, request)?;
        send_image_generation::<_, crate::providers::openai::client::ApiResponse<Ext::Response>>(
            &self.client,
            builder,
            body,
            Ext::REQUEST_ID_HEADER,
        )
        .await
    }
}

impl<Ext, H> image_generation::ImageGenerationModel for GenericImageGenerationModel<Ext, H>
where
    Ext: JsonImageGenerationProvider + Clone + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    async fn image_generation(
        &self,
        request: ImageGenerationRequest,
    ) -> Result<image_generation::ImageGenerationResponse, ImageGenerationError> {
        crate::telemetry::instrument_modality(
            Ext::PROVIDER_NAME,
            &self.model,
            crate::telemetry::ModalityOperation::ImageGeneration,
            async {
                let (response, provider_request_id) =
                    self.raw_image_generation_with_request_id(request).await?;
                let captured = serde_json::to_value(&response)?;
                Ok(response
                    .normalize(Ext::PROVIDER_NAME)?
                    .with_optional_provider_request_id(provider_request_id)
                    .with_raw(captured))
            },
        )
        .await
    }
}

impl<Ext, H> crate::client::ConstructImageGenerationModel<Client<Ext, H>>
    for GenericImageGenerationModel<Ext, H>
where
    Ext: JsonImageGenerationProvider + Clone + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    fn construct(client: &Client<Ext, H>, model: String) -> Self {
        Self::new(client.clone(), model)
    }
}

pub(crate) async fn send_image_generation<C, A>(
    client: &C,
    builder: http_client::Builder,
    body: serde_json::Value,
    request_id_header: Option<&str>,
) -> Result<(A::Payload, Option<String>), ImageGenerationError>
where
    C: HttpClientExt,
    A: DeserializeOwned + ProviderEnvelope,
{
    let body = serde_json::to_vec(&body)?;

    let req = builder
        .body(body)
        .map_err(|e| ImageGenerationError::HttpError(e.into()))?;

    let response = client.send::<_, Bytes>(req).await?;

    // Taking the response apart hands the headers over already owned, so both
    // failure paths keep their rate-limit metadata at no cost to the success
    // path (rig#2210).
    let (parts, body) = response.into_parts();
    let status = parts.status;
    let provider_request_id =
        super::transcription::request_id_from_headers(&parts.headers, request_id_header);
    let headers = Box::new(parts.headers);
    let response_body = body.into_future().await?;

    if !status.is_success() {
        return Err(ImageGenerationError::from_http_response(
            status,
            String::from_utf8_lossy(&response_body).into_owned(),
        )
        .with_response_headers(Some(headers)));
    }

    match serde_json::from_slice::<A>(&response_body)?.into_payload() {
        Ok(response) => Ok((response, provider_request_id)),
        Err(message) => {
            tracing::warn!(message = %message, "provider returned an error response");
            Err(ImageGenerationError::from_http_response(
                status,
                String::from_utf8_lossy(&response_body).into_owned(),
            )
            .with_response_headers(Some(headers)))
        }
    }
}

#[cfg(test)]
mod header_preservation_tests;
