use std::path::Path;

use base64::{Engine, prelude::BASE64_STANDARD};
use serde_json::{Map, Value, json};

use super::completion::usage_of;
use crate::error::{EncodeError, ProviderError};
use crate::json_utils::Lenient;
use crate::operation::Transcription;
use crate::providers::internal::wire::classify_marker_keyed_frame;
use crate::transcription;
use crate::wire::{
    Body, Decoder, Descriptor, Encoded, Flow, Framing, Mode, Out, Wire, WireEvent, WireFrame,
};

const TRANSCRIPTION_PREAMBLE: &str =
    "Translate the provided audio exactly. Do not add additional information.";

/// Encode audio as inline base64 with a transcription system instruction.
/// Reject invalid generation parameters or JSON serialization failures.
fn transcription_body(
    request: transcription::TranscriptionRequest,
) -> Result<Vec<u8>, EncodeError> {
    let mut generation_config = match request.additional_params {
        None | Some(Value::Null) => Map::new(),
        Some(Value::Object(config)) => config,
        Some(other) => {
            return Err(EncodeError::request(format!(
                "Gemini transcription `additional_params` should be an object, got {other}"
            )));
        }
    };
    // A temperature named on the request outranks one carried inside
    // `additional_params`.
    if let Some(temp) = request.temperature {
        generation_config.insert("temperature".to_owned(), Value::from(temp));
    }
    // The request supplies no explicit MIME type, so infer it from the filename.
    let mime_type = mime_guess::from_path(Path::new(&request.filename))
        .first()
        .map_or_else(|| "audio/mpeg".to_string(), |mime| mime.to_string());
    let data = BASE64_STANDARD.encode(request.data);
    let body = json!({
        "contents": [{
            "parts": [{ "inlineData": { "mimeType": mime_type, "data": data }, "thought": false }],
            "role": "user",
        }],
        "generationConfig": generation_config,
        "safetySettings": null,
        "toolConfig": null,
        "systemInstruction": {
            "parts": [{ "text": TRANSCRIPTION_PREAMBLE, "thought": false }],
            "role": "model",
        },
    });
    tracing::trace!(
        target: "rig::transcription",
        "Sending completion request to Gemini API {}",
        serde_json::to_string_pretty(&body)?
    );
    Ok(serde_json::to_vec(&body)?)
}

/// The transcription wire: `POST /v1beta/models/{model}:generateContent`.
///
/// Both [`Mode`]s send inline audio in JSON and read a whole response.
#[derive(Clone, Debug, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct Transcriptions {
    /// The provider this wire speaks to.
    pub provider: super::GeminiConfig,
    /// The model transcribing, for example
    /// [`GEMINI_2_0_FLASH`](super::completion::GEMINI_2_0_FLASH).
    pub model: String,
}

impl Transcriptions {
    /// The transcription wire for `model`.
    pub fn new(provider: super::GeminiConfig, model: impl Into<String>) -> Self {
        Self {
            provider,
            model: model.into(),
        }
    }
}

impl Wire for Transcriptions {
    type Op = Transcription;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = TranscriptionsDecoder;
    type Reassembler = crate::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME).model(self.model.as_str())
    }

    fn encode(
        &self,
        request: transcription::TranscriptionRequest,
        _mode: Mode,
    ) -> Result<Encoded, EncodeError> {
        let body = transcription_body(request)?;
        let request = http::Request::post(format!(
            "{}/v1beta/models/{}:generateContent?key={}",
            self.provider.base_url,
            self.model,
            self.provider.api_key.expose()
        ))
        .header(http::header::CONTENT_TYPE, "application/json")
        .body(Body::Bytes(body))?;
        // Gemini reports no transport request-id header.
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        TranscriptionsDecoder
    }
}

/// Decode visible text from the first `generateContent` candidate.
/// Missing candidates or visible text produce response errors.
#[derive(Default)]
pub struct TranscriptionsDecoder;

impl<'id> Decoder<'id, Transcription> for TranscriptionsDecoder {
    type Event = Value;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_marker_keyed_frame(
            &frame.as_str(),
            &["candidates", "promptFeedback", "usageMetadata"],
        )
    }

    fn decode(
        &mut self,
        event: Self::Event,
        out: Out<'id, Transcription>,
    ) -> Result<Flow, ProviderError> {
        Ok(out.end(transcript_of(&event)?))
    }
}

/// The first candidate's visible text parts, those not marked `thought`, as
/// a transcript of `reply`. Errors when there is no candidate or no visible
/// text part.
pub fn transcript_of(reply: &Value) -> Result<transcription::TranscriptionResponse, ProviderError> {
    let candidate = reply
        .arr("candidates")
        .first()
        .ok_or_else(|| ProviderError::Response("No response candidates in response".into()))?;
    let parts: Vec<&str> = candidate
        .get("content")
        .map(|content| content.arr("parts"))
        .unwrap_or_default()
        .iter()
        .filter(|part| part.bool("thought") != Some(true))
        .filter_map(|part| part.str("text"))
        .collect();
    if parts.is_empty() {
        return Err(ProviderError::Response(
            "Response content contains no text".to_string(),
        ));
    }
    Ok(transcription::TranscriptionResponse {
        model: reply.str("modelVersion").map(str::to_owned),
        response_id: Some(reply.str("responseId").unwrap_or_default().to_owned()),
        usage: reply.get("usageMetadata").map(usage_of).unwrap_or_default(),
        ..transcription::TranscriptionResponse::new(parts.concat())
    })
}

impl super::GeminiConfig {
    /// The audio transcription wire.
    pub(crate) fn transcription(&self, model: impl Into<String>) -> Transcriptions {
        Transcriptions::new(self.clone(), model)
    }
}

#[cfg(test)]
mod tests;
