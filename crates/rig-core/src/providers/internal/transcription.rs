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
use crate::transcription::{self, TranscriptionError, TranscriptionRequest};

/// Provider-specific request routing for the shared OpenAI-style model.
pub(crate) trait OpenAiTranscriptionClient: HttpClientExt + Clone {
    /// Whether the model is a multipart form field. Azure addresses the model
    /// as a deployment in the request URL instead.
    const MODEL_IN_FORM: bool;

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

impl<C> transcription::TranscriptionModel for OpenAiTranscriptionModel<C>
where
    C: OpenAiTranscriptionClient + 'static,
{
    type Response = crate::providers::openai::TranscriptionResponse;
    type Client = C;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn transcription(
        &self,
        request: TranscriptionRequest,
    ) -> Result<transcription::TranscriptionResponse<Self::Response>, TranscriptionError> {
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
        )
        .await
    }
}

/// The client-plus-model wrapper behind each provider's public
/// `TranscriptionModel` alias. Only the transcription conversation itself is
/// provider-specific, so the storage and constructor live here once; each
/// provider keeps its own [`TranscriptionModel`](transcription::TranscriptionModel)
/// impl on its alias.
#[derive(Clone)]
pub struct GenericTranscriptionModel<Ext, H = reqwest::Client> {
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
/// success-or-error envelope.
///
/// `builder` is the provider's already-path-built POST request; `A` is the
/// provider's own response envelope so error-body classification is unchanged.
/// Provider error bodies are preserved raw via
/// [`TranscriptionError::from_http_response`].
pub(crate) async fn send_transcription<C, A>(
    client: &C,
    builder: http_client::Builder,
    form: MultipartForm,
) -> Result<transcription::TranscriptionResponse<A::Payload>, TranscriptionError>
where
    C: HttpClientExt,
    A: DeserializeOwned + ProviderEnvelope,
    A::Payload:
        TryInto<transcription::TranscriptionResponse<A::Payload>, Error = TranscriptionError>,
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
    let headers = Box::new(parts.headers);
    let response_body = body.into_future().await?;

    if status.is_success() {
        match serde_json::from_slice::<A>(&response_body)?.into_payload() {
            Ok(response) => response.try_into(),
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

/// Sends a JSON-bodied transcription request and splits the response on
/// status, mirroring [`send_transcription`] for providers whose transcription
/// endpoint takes JSON instead of multipart.
///
/// `builder` is the provider's already-path-built POST request (including any
/// provider-specific headers) and `body` the serialized JSON payload. On a
/// 2xx status the raw body is handed to `decode` together with the status so
/// each provider keeps its own payload decoding, logging and error-envelope
/// classification; non-2xx statuses preserve the raw body via
/// [`TranscriptionError::from_http_response`].
pub(crate) async fn send_json_transcription<C, R>(
    client: &C,
    builder: http_client::Builder,
    body: Vec<u8>,
    decode: impl FnOnce(
        http::StatusCode,
        &[u8],
    ) -> Result<transcription::TranscriptionResponse<R>, TranscriptionError>,
) -> Result<transcription::TranscriptionResponse<R>, TranscriptionError>
where
    C: HttpClientExt,
{
    let req = builder
        .body(body)
        .map_err(|e| TranscriptionError::HttpError(e.into()))?;

    let response = client.send::<_, Vec<u8>>(req).await?;
    let (parts, body) = response.into_parts();
    let status = parts.status;
    let headers = Box::new(parts.headers);
    let body = body.await?;

    if status.is_success() {
        decode(status, &body)
    } else {
        Err(TranscriptionError::from_http_response(
            status,
            String::from_utf8_lossy(&body).into_owned(),
        )
        .with_response_headers(Some(headers)))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn request() -> TranscriptionRequest {
        TranscriptionRequest {
            data: vec![1, 2, 3],
            filename: "audio.mp3".to_owned(),
            language: Some("en".to_owned()),
            prompt: Some("a prompt".to_owned()),
            temperature: Some(0.5),
            additional_params: None,
        }
    }

    /// Form field names, in the order they are written to the wire.
    fn field_names(form: &MultipartForm) -> Vec<&str> {
        form.parts().iter().map(Part::name).collect()
    }

    /// The encoded body, so assertions cover what is actually sent rather
    /// than the builder's internal state.
    fn encoded(form: MultipartForm) -> String {
        let (_, body) = form.boundary("BOUNDARY").encode();
        String::from_utf8_lossy(&body).into_owned()
    }

    /// The wire representation of a text field.
    fn text_field(name: &str, value: &str) -> String {
        format!("Content-Disposition: form-data; name=\"{name}\"\r\n\r\n{value}\r\n")
    }

    /// Each provider's form shape, as a table so a new provider is one row.
    #[test]
    fn form_field_shape_per_provider() {
        let cases = [
            (
                "openai/groq: model in body",
                TranscriptionFields {
                    model: Some("whisper-1"),
                },
                &["model", "file", "language", "prompt", "temperature"][..],
            ),
            (
                "azure: model addressed through the URL",
                TranscriptionFields { model: None },
                &["file", "language", "prompt", "temperature"][..],
            ),
        ];

        for (case, fields, expected) in cases {
            let form = transcription_form(request(), fields).expect(case);
            assert_eq!(field_names(&form), expected, "{case}");
        }
    }

    #[test]
    fn sends_field_values_on_the_wire() {
        let form = transcription_form(
            request(),
            TranscriptionFields {
                model: Some("whisper-1"),
            },
        )
        .expect("form should build");

        let body = encoded(form);
        for (name, value) in [
            ("model", "whisper-1"),
            ("language", "en"),
            ("prompt", "a prompt"),
            ("temperature", "0.5"),
        ] {
            assert!(body.contains(&text_field(name, value)), "{name}: {body}");
        }
        assert!(
            body.contains("name=\"file\"; filename=\"audio.mp3\""),
            "{body}"
        );
    }

    #[test]
    fn omits_unset_optional_fields() {
        let request = TranscriptionRequest {
            data: vec![1, 2, 3],
            filename: "audio.mp3".to_owned(),
            language: None,
            prompt: None,
            temperature: None,
            additional_params: None,
        };

        let form = transcription_form(
            request,
            TranscriptionFields {
                model: Some("whisper-1"),
            },
        )
        .expect("form should build");

        assert_eq!(field_names(&form), ["model", "file"]);
    }

    #[test]
    fn flattens_additional_params_onto_the_form() {
        let mut request = request();
        request.additional_params = Some(serde_json::json!({
            "response_format": "verbose_json",
            "timestamp_granularities": ["word"],
        }));

        let form = transcription_form(
            request,
            TranscriptionFields {
                model: Some("whisper-1"),
            },
        )
        .expect("form should build");

        // String values go on the form verbatim (a JSON-quoted
        // `"verbose_json"` would be rejected or ignored by the provider);
        // non-string values stay JSON-encoded.
        let body = encoded(form);
        assert!(
            body.contains(&text_field("response_format", "verbose_json")),
            "{body}"
        );
        assert!(
            body.contains(&text_field("timestamp_granularities", "[\"word\"]")),
            "{body}"
        );
    }

    #[test]
    fn rejects_non_object_additional_params() {
        let mut request = request();
        request.additional_params = Some(serde_json::json!("not an object"));

        let error = transcription_form(
            request,
            TranscriptionFields {
                model: Some("whisper-1"),
            },
        )
        .expect_err("non-object additional params should be rejected");

        assert!(matches!(error, TranscriptionError::RequestError(_)));
    }
}

/// rig#2210: a failed transcription response keeps its headers on both shared
/// drivers, so the capability error's `provider_response_headers()` is not a
/// promise the driver quietly breaks.
#[cfg(test)]
mod header_preservation_tests {
    use super::*;
    use crate::test_utils::RecordingHttpClient;

    fn rate_limited() -> RecordingHttpClient {
        let mut headers = http::HeaderMap::new();
        headers.insert(http::header::RETRY_AFTER, "20".parse().expect("value"));
        RecordingHttpClient::with_error_response_headers(
            http::StatusCode::TOO_MANY_REQUESTS,
            r#"{"error":"slow down"}"#,
            headers,
        )
    }

    fn assert_retry_after(error: &TranscriptionError, driver: &str) {
        assert_eq!(
            error
                .provider_response_headers()
                .and_then(|headers| headers.get(http::header::RETRY_AFTER))
                .and_then(|value| value.to_str().ok()),
            Some("20"),
            "{driver}: Retry-After not recoverable",
        );
        assert_eq!(
            error.provider_response_status(),
            Some(http::StatusCode::TOO_MANY_REQUESTS),
            "{driver}: status lost",
        );
    }

    #[tokio::test]
    async fn json_driver_non_success_response_preserves_headers() {
        let error = send_json_transcription::<_, serde_json::Value>(
            &rate_limited(),
            http_client::Request::builder()
                .method(http::Method::POST)
                .uri("https://example.test/v1/audio/transcriptions"),
            b"{}".to_vec(),
            |_, _| unreachable!("a 429 never reaches the decoder"),
        )
        .await
        .err()
        .expect("a 429 should fail");

        assert_retry_after(&error, "send_json_transcription");
    }

    #[tokio::test]
    async fn multipart_driver_non_success_response_preserves_headers() {
        let error = send_transcription::<
            _,
            crate::providers::openai::client::ApiResponse<
                crate::providers::openai::TranscriptionResponse,
            >,
        >(
            &rate_limited(),
            http_client::Request::builder()
                .method(http::Method::POST)
                .uri("https://example.test/v1/audio/transcriptions"),
            MultipartForm::default(),
        )
        .await
        .err()
        .expect("a 429 should fail");

        assert_retry_after(&error, "send_transcription");
    }
}
