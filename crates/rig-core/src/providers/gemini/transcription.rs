use std::path::Path;

use base64::{Engine, prelude::BASE64_STANDARD};
use mime_guess;
use serde_json::{Map, Value};

use crate::{
    completion::Usage,
    operation::Transcription,
    providers::gemini::completion::gemini_api_types::{
        Blob, Content, GenerateContentRequest, GenerationConfig, Part, PartKind, Role,
        visible_text_parts,
    },
    providers::internal::wire::classify_marker_keyed_frame,
    transcription::{self, NormalizeTranscriptionResponse, TranscriptionError},
    wire::{Body, Decoder, Encoded, Framing, Mode, Output, Sink, Wire, WireEvent, WireFrame},
};

use super::completion::gemini_api_types::GenerateContentResponse;

const TRANSCRIPTION_PREAMBLE: &str =
    "Translate the provided audio exactly. Do not add additional information.";

/// The `generateContent` body one transcription sends: the audio inline as
/// base64, under the preamble's system instruction.
///
/// The one place the bytes Gemini receives are stated: [`Transcriptions`]
/// encodes through it.
fn transcription_body(
    request: transcription::TranscriptionRequest,
) -> Result<Vec<u8>, TranscriptionError> {
    let additional_params = request
        .additional_params
        .unwrap_or_else(|| Value::Object(Map::new()));
    let mut generation_config = serde_json::from_value::<GenerationConfig>(additional_params)?;

    // A temperature named on the request outranks one carried inside
    // `additional_params`.
    if let Some(temp) = request.temperature {
        generation_config.temperature = Some(temp);
    }

    let system_instruction = Some(Content {
        parts: vec![TRANSCRIPTION_PREAMBLE.into()],
        role: Some(Role::Model),
    });

    // Gemini needs a mime type for the blob and the filename is the only
    // evidence the request carries; mp3 is the format the endpoint documents.
    let mime_type = mime_guess::from_path(Path::new(&request.filename))
        .first()
        .map_or_else(|| "audio/mpeg".to_string(), |mime| mime.to_string());

    let body = GenerateContentRequest {
        contents: vec![Content {
            parts: vec![Part {
                thought: Some(false),
                thought_signature: None,
                part: PartKind::InlineData(Blob {
                    mime_type,
                    data: BASE64_STANDARD.encode(request.data),
                }),
                additional_params: None,
            }],
            role: Some(Role::User),
        }],
        generation_config: Some(generation_config),
        safety_settings: None,
        tools: None,
        tool_config: None,
        system_instruction,
        cached_content: None,
        additional_params: None,
    };

    tracing::trace!(
        target: "rig::transcription",
        "Sending completion request to Gemini API {}",
        serde_json::to_string_pretty(&body)?
    );

    Ok(serde_json::to_vec(&body)?)
}

/// The transcription wire: `POST /v1beta/models/{model}:generateContent`.
///
/// Gemini has no dedicated transcription endpoint. The audio rides inline as
/// base64 inside an ordinary `generateContent` request, so this wire sends
/// JSON rather than a multipart form, and a transcript never arrives in
/// pieces: both [`Mode`]s send the same request and read one whole reply.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Transcriptions {
    /// The provider this wire speaks to.
    pub provider: super::Gemini,
    /// The model transcribing, for example
    /// [`GEMINI_2_0_FLASH`](super::completion::GEMINI_2_0_FLASH).
    pub model: String,
}

impl Transcriptions {
    /// The transcription wire for `model`.
    pub fn new(provider: super::Gemini, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

impl Wire for Transcriptions {
    type Op = Transcription;
    type Decoder = TranscriptionsDecoder;

    fn name(&self) -> &str {
        super::PROVIDER_NAME
    }

    fn model(&self) -> Option<&str> {
        Some(&self.model)
    }

    fn encode(
        &self,
        request: transcription::TranscriptionRequest,
        _mode: Mode,
    ) -> Result<Encoded, TranscriptionError> {
        let body = transcription_body(request)?;
        // The GenerateContent family authenticates through the `key` query
        // parameter, appended last.
        let request = http::Request::post(format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.provider.base_url,
            self.model,
            self.provider.api_key.expose()
        ))
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::Bytes(body))
        .map_err(|error| TranscriptionError::HttpError(error.into()))?;
        // Gemini reports no transport request-id header.
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        TranscriptionsDecoder
    }
}

/// Decodes one `generateContent` reply into a transcript.
///
/// `project` stays at its default: this reply carries nothing beyond what the
/// GenerateContent decoder's own projection already observes.
#[derive(Default)]
pub struct TranscriptionsDecoder;

impl Decoder<Transcription> for TranscriptionsDecoder {
    type Event = GenerateContentResponse;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_marker_keyed_frame(
            &frame.as_str(),
            &["candidates", "promptFeedback", "usageMetadata"],
        )
    }

    /// The mapping — the no-candidates and no-text errors, and the usage,
    /// model and response-id carry — is [`NormalizeTranscriptionResponse`]'s,
    /// so this wire holds no second reading of the same reply.
    fn interpret(&mut self, event: Self::Event, out: &mut Output<Transcription>) {
        out.push(event.normalize(super::PROVIDER_NAME));
    }
}

impl NormalizeTranscriptionResponse for GenerateContentResponse {
    fn normalize(
        self,
        provider: &str,
    ) -> Result<transcription::TranscriptionResponse, TranscriptionError> {
        let candidate = self.candidates.first().ok_or_else(|| {
            TranscriptionError::ResponseError("No response candidates in response".into())
        })?;

        let mut parts = candidate
            .content
            .as_ref()
            .map(visible_text_parts)
            .into_iter()
            .flatten()
            .peekable();
        if parts.peek().is_none() {
            return Err(TranscriptionError::ResponseError(
                "Response content contains no text".to_string(),
            ));
        }
        let text = parts.collect::<String>();

        let usage = self
            .usage_metadata
            .as_ref()
            .map(Usage::from)
            .unwrap_or_default();

        Ok(transcription::TranscriptionResponse::new(text, provider)
            .with_optional_model(self.model_version)
            .with_response_id(self.response_id)
            .with_usage(usage))
    }
}

#[cfg(test)]
mod tests;
