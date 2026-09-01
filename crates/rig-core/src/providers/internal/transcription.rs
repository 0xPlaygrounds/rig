//! Shared request plumbing for OpenAI-style `/audio/transcriptions` endpoints.
//!
//! OpenAI, Groq and Azure OpenAI all accept the same multipart body — the
//! audio as a `file` part plus optional `language`, `prompt` and `temperature`
//! fields and a flattened `additional_params` object — and answer with the
//! same `{ text }` payload and OpenAI-style error envelope. The per-provider
//! differences are limited to request routing and whether the model is sent as
//! a form field.

use bytes::Bytes;
use serde::de::DeserializeOwned;

use super::envelope::ProviderEnvelope;
use crate::client::Client;
use crate::http_client::multipart::Part;
use crate::http_client::{self, HttpClientExt, MultipartForm};
use crate::transcription::{
    self, NormalizeTranscriptionResponse, TranscriptionError, TranscriptionRequest,
};

/// Provider-specific request routing for the shared OpenAI-style model.
#[doc(hidden)]
pub trait OpenAiTranscriptionClient: HttpClientExt + Clone {
    /// Whether the model is a multipart form field. Azure addresses the model
    /// as a deployment in the request URL instead.
    const MODEL_IN_FORM: bool;

    /// Stable descriptor name of the provider, stamped on every normalized
    /// response. An input to normalization, never hardcoded in the shared
    /// conversion: this wire shape is shared by several providers.
    const PROVIDER_NAME: &'static str;

    /// The provider's transport request-id response header, when it has one
    /// (OpenAI `x-request-id`). `None` means the provider reports none.
    const REQUEST_ID_HEADER: Option<&'static str>;

    fn transcription_request(&self, model: &str) -> http_client::Result<http_client::Builder>;
}

/// The common model shell for OpenAI, Groq, and Azure OpenAI transcription.
/// Their response and multipart wire formats are identical; the client trait
/// above retains the only variation, request routing.
#[derive(Clone)]
pub struct OpenAiTranscriptionModel<C> {
    client: C,
    /// Name of the transcription model or, for Azure OpenAI, deployment.
    pub model: String,
}

impl<C> OpenAiTranscriptionModel<C> {
    /// Create a transcription model backed by `client`.
    pub fn new(client: C, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

impl<C> OpenAiTranscriptionModel<C>
where
    C: OpenAiTranscriptionClient + 'static,
{
    /// Perform the transcription and return the provider's native response
    /// instead of the normalized [`transcription::TranscriptionResponse`].
    /// Same request, transport, parser, and error path as
    /// [`transcription::TranscriptionModel::transcription`].
    pub async fn raw_transcription(
        &self,
        request: TranscriptionRequest,
    ) -> Result<crate::providers::openai::TranscriptionResponse, TranscriptionError> {
        self.raw_transcription_with_request_id(request)
            .await
            .map(|(response, _)| response)
    }

    /// [`Self::raw_transcription`] plus the transport request id from the
    /// provider's request-id response header, when it carries one.
    pub async fn raw_transcription_with_request_id(
        &self,
        request: TranscriptionRequest,
    ) -> Result<
        (
            crate::providers::openai::TranscriptionResponse,
            Option<String>,
        ),
        TranscriptionError,
    > {
        let form = transcription_form(
            request,
            TranscriptionFields {
                model: C::MODEL_IN_FORM.then_some(self.model.as_str()),
            },
        )?;

        send_transcription::<
            _,
            crate::providers::openai::client::ApiResponse<
                crate::providers::openai::TranscriptionResponse,
            >,
        >(
            &self.client,
            self.client.transcription_request(&self.model)?,
            form,
            C::REQUEST_ID_HEADER,
        )
        .await
    }
}

impl<C> transcription::TranscriptionModel for OpenAiTranscriptionModel<C>
where
    C: OpenAiTranscriptionClient + 'static,
{
    async fn transcription(
        &self,
        request: TranscriptionRequest,
    ) -> Result<transcription::TranscriptionResponse, TranscriptionError> {
        crate::telemetry::instrument_modality(
            C::PROVIDER_NAME,
            &self.model,
            crate::telemetry::ModalityOperation::Transcription,
            async {
                let (response, provider_request_id) =
                    self.raw_transcription_with_request_id(request).await?;
                let captured = serde_json::to_value(&response)?;
                Ok(response
                    .normalize(C::PROVIDER_NAME)?
                    .with_optional_provider_request_id(provider_request_id)
                    .with_raw(captured))
            },
        )
        .await
    }
}

impl<C> crate::client::ConstructTranscriptionModel<C> for OpenAiTranscriptionModel<C>
where
    C: OpenAiTranscriptionClient + 'static,
{
    fn construct(client: &C, model: String) -> Self {
        Self::new(client.clone(), model)
    }
}

/// The client-plus-model wrapper behind each provider's public
/// `TranscriptionModel` alias. Only the transcription conversation itself is
/// provider-specific, so the storage and constructor live here once; each
/// provider keeps its own [`TranscriptionModel`](transcription::TranscriptionModel)
/// impl on its alias.
#[derive(Clone)]
pub struct GenericTranscriptionModel<Ext, H> {
    pub(crate) client: Client<Ext, H>,
    /// Name of the model (e.g.: `whisper-1`)
    pub model: String,
}

impl<Ext, H> GenericTranscriptionModel<Ext, H> {
    pub fn new(client: Client<Ext, H>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

/// The per-provider parts of an OpenAI-style transcription request.
#[derive(Debug, Clone, Copy)]
pub(crate) struct TranscriptionFields<'a> {
    /// Model to send as the `model` form field, or `None` when the provider
    /// addresses the model through the URL instead — Azure targets a
    /// deployment, not a model name.
    pub model: Option<&'a str>,
}

/// Builds the multipart body shared by OpenAI-style transcription endpoints.
///
/// Field order matches the order these providers previously built by hand, so
/// recorded requests stay byte-comparable.
pub(crate) fn transcription_form(
    request: TranscriptionRequest,
    fields: TranscriptionFields<'_>,
) -> Result<MultipartForm, TranscriptionError> {
    let mut body = MultipartForm::new();

    if let Some(model) = fields.model {
        body = body.text("model", model.to_owned());
    }

    body = body.part(Part::bytes("file", request.data).filename(request.filename));

    if let Some(language) = request.language {
        body = body.text("language", language);
    }

    if let Some(prompt) = request.prompt {
        body = body.text("prompt", prompt);
    }

    if let Some(temperature) = request.temperature {
        body = body.text("temperature", temperature.to_string());
    }

    if let Some(additional_params) = request.additional_params {
        let params = additional_params.as_object().ok_or_else(|| {
            TranscriptionError::RequestError(Box::new(std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "additional transcription parameters must be a JSON object",
            )))
        })?;

        for (key, value) in params {
            // String values go on the form verbatim — `Value::to_string`
            // would send them JSON-quoted (`"verbose_json"`), which providers
            // reject or ignore. Non-string values stay JSON-encoded.
            let value = match value {
                serde_json::Value::String(text) => text.clone(),
                other => other.to_string(),
            };
            body = body.text(key.to_owned(), value);
        }
    }

    Ok(body)
}

/// Sends an OpenAI-style transcription request and decodes the shared
/// success-or-error envelope, returning the provider's typed payload plus the
/// transport request id read from `request_id_header`, when the provider has
/// one and the response carried it.
///
/// `builder` is the provider's already-path-built POST request; `A` is the
/// provider's own response envelope so error-body classification is unchanged.
/// Provider error bodies are preserved raw via
/// [`TranscriptionError::from_http_response`].
pub(crate) async fn send_transcription<C, A>(
    client: &C,
    builder: http_client::Builder,
    form: MultipartForm,
    request_id_header: Option<&str>,
) -> Result<(A::Payload, Option<String>), TranscriptionError>
where
    C: HttpClientExt,
    A: DeserializeOwned + ProviderEnvelope,
{
    let req = builder
        .body(form)
        .map_err(|e| TranscriptionError::HttpError(e.into()))?;

    let response = client.send_multipart::<Bytes>(req).await?;

    // Taking the response apart hands the headers over already owned, so both
    // failure paths keep their rate-limit metadata at no cost to the success
    // path (rig#2210).
    let (parts, body) = response.into_parts();
    let status = parts.status;
    let provider_request_id = request_id_from_headers(&parts.headers, request_id_header);
    let headers = Box::new(parts.headers);
    let response_body = body.into_future().await?;

    if status.is_success() {
        match serde_json::from_slice::<A>(&response_body)?.into_payload() {
            Ok(response) => Ok((response, provider_request_id)),
            Err(message) => {
                tracing::warn!(message = %message, "provider returned an error response");
                Err(TranscriptionError::from_http_response(
                    status,
                    String::from_utf8_lossy(&response_body).into_owned(),
                )
                .with_response_headers(Some(headers)))
            }
        }
    } else {
        Err(TranscriptionError::from_http_response(
            status,
            String::from_utf8_lossy(&response_body).into_owned(),
        )
        .with_response_headers(Some(headers)))
    }
}

/// Reads the provider's transport request id off a response's headers, when
/// the provider names such a header and the response carries a non-empty
/// value. `None` is the documented "not reported" outcome.
pub(crate) fn request_id_from_headers(
    headers: &http::HeaderMap,
    request_id_header: Option<&str>,
) -> Option<String> {
    request_id_header.and_then(|header| {
        headers
            .get(header)
            .and_then(|value| value.to_str().ok())
            .filter(|value| !value.is_empty())
            .map(str::to_string)
    })
}

/// Sends a JSON-bodied transcription request and splits the response on
/// status, mirroring [`send_transcription`] for providers whose transcription
/// endpoint takes JSON instead of multipart.
///
/// `builder` is the provider's already-path-built POST request (including any
/// provider-specific headers) and `body` the serialized JSON payload. On a
/// 2xx status the raw body is handed to `decode` together with the status so
/// each provider keeps its own payload decoding, logging and error-envelope
/// classification; the decoded payload is returned with the transport request
/// id read from `request_id_header`. Non-2xx statuses preserve the raw body
/// via [`TranscriptionError::from_http_response`].
pub(crate) async fn send_json_transcription<C, R>(
    client: &C,
    builder: http_client::Builder,
    body: Vec<u8>,
    request_id_header: Option<&str>,
    decode: impl FnOnce(http::StatusCode, &[u8]) -> Result<R, TranscriptionError>,
) -> Result<(R, Option<String>), TranscriptionError>
where
    C: HttpClientExt,
{
    let req = builder
        .body(body)
        .map_err(|e| TranscriptionError::HttpError(e.into()))?;

    let response = client.send::<_, Vec<u8>>(req).await?;
    let (parts, body) = response.into_parts();
    let status = parts.status;
    let provider_request_id = request_id_from_headers(&parts.headers, request_id_header);
    let headers = Box::new(parts.headers);
    let body = body.await?;

    if status.is_success() {
        Ok((decode(status, &body)?, provider_request_id))
    } else {
        Err(TranscriptionError::from_http_response(
            status,
            String::from_utf8_lossy(&body).into_owned(),
        )
        .with_response_headers(Some(headers)))
    }
}

#[cfg(test)]
mod tests;

/// rig#2210: a failed transcription response keeps its headers on both shared
/// drivers, so the capability error's `provider_response_headers()` is not a
/// promise the driver quietly breaks.
#[cfg(test)]
mod header_preservation_tests;
