use crate::http_client::HttpClientExt;
use crate::providers::openrouter::Client;
use crate::transcription;
use crate::transcription::TranscriptionError;
use crate::wasm_compat::WasmCompatSend;
use base64::Engine;
use base64::engine::general_purpose::STANDARD;
use bytes::Bytes;
use serde::{Deserialize, Serialize};

// ================================================================
// Model constants
// ================================================================

/// The `openai/whisper-1` model.
pub const WHISPER_1: &str = "openai/whisper-1";
/// The `openai/whisper-large-v3-turbo` model.
pub const WHISPER_LARGE_V3_TURBO: &str = "openai/whisper-large-v3-turbo";
/// The `openai/whisper-large-v3` model.
pub const WHISPER_LARGE_V3: &str = "openai/whisper-large-v3";
/// The `openai/gpt-4o-transcribe` model.
pub const GPT_4O_TRANSCRIBE: &str = "openai/gpt-4o-transcribe";
/// The `openai/gpt-4o-mini-transcribe` model.
pub const GPT_4O_MINI_TRANSCRIBE: &str = "openai/gpt-4o-mini-transcribe";
/// The `google/chirp-3` model.
pub const CHIRP_3: &str = "google/chirp-3";

// ================================================================
// Request/Response types
// ================================================================

#[allow(dead_code)]
#[derive(Debug, Serialize)]
struct InputAudio {
    data: String,
    format: String,
}

#[allow(dead_code)]
#[derive(Debug, Serialize)]
struct TranscriptionRequestInput {
    model: String,
    input_audio: InputAudio,
    #[serde(skip_serializing_if = "Option::is_none")]
    language: Option<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    temperature: Option<f64>,
}

#[derive(Debug, Deserialize)]
pub struct TranscriptionResponse {
    pub text: String,
    #[serde(default)]
    pub usage: Option<TranscriptionUsage>,
}

#[derive(Debug, Deserialize)]
pub struct TranscriptionUsage {
    #[serde(default)]
    pub seconds: Option<f64>,
    #[serde(default)]
    pub total_tokens: Option<usize>,
    #[serde(default)]
    pub input_tokens: Option<usize>,
    #[serde(default)]
    pub output_tokens: Option<usize>,
    #[serde(default)]
    pub cost: Option<f64>,
}

impl TryFrom<TranscriptionResponse>
    for transcription::TranscriptionResponse<TranscriptionResponse>
{
    type Error = TranscriptionError;

    fn try_from(value: TranscriptionResponse) -> Result<Self, Self::Error> {
        Ok(transcription::TranscriptionResponse {
            text: value.text.clone(),
            response: value,
        })
    }
}

// ================================================================
// Model
// ================================================================

#[derive(Clone)]
pub struct TranscriptionModel<T = reqwest::Client> {
    client: Client<T>,
    pub model: String,
}

impl<T> TranscriptionModel<T> {
    pub fn new(client: Client<T>, model: impl Into<String>) -> Self {
        Self {
            client,
            model: model.into(),
        }
    }
}

fn infer_format_from_filename(filename: &str) -> String {
    std::path::Path::new(filename)
        .extension()
        .and_then(|e| e.to_str())
        .and_then(|ext| match ext.to_lowercase().as_str() {
            "wav" => Some("wav"),
            "mp3" => Some("mp3"),
            "flac" => Some("flac"),
            "m4a" => Some("m4a"),
            "ogg" => Some("ogg"),
            "webm" => Some("webm"),
            "aac" => Some("aac"),
            _ => None,
        })
        .unwrap_or("wav")
        .to_string()
}

impl<T> transcription::TranscriptionModel for TranscriptionModel<T>
where
    T: HttpClientExt + Clone + std::fmt::Debug + Default + WasmCompatSend + 'static,
{
    type Response = TranscriptionResponse;
    type Client = Client<T>;

    fn make(client: &Self::Client, model: impl Into<String>) -> Self {
        Self::new(client.clone(), model)
    }

    async fn transcription(
        &self,
        request: transcription::TranscriptionRequest,
    ) -> Result<transcription::TranscriptionResponse<Self::Response>, TranscriptionError> {
        if let Some(_prompt) = request.prompt {
            return Err(TranscriptionError::RequestError(Box::new(
                std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "OpenRouter STT does not support a top-level prompt field. \
                     Provider-specific prompt options can be passed via `additional_params`. \
                     Example: {\"provider\": {\"options\": {\"<provider>\": {\"prompt\": \"<text>\"}}}}",
                ),
            )));
        }

        let audio_b64 = STANDARD.encode(&request.data);
        let format = infer_format_from_filename(&request.filename);

        let mut body_map: serde_json::Map<String, serde_json::Value> = [
            ("model".to_string(), serde_json::json!(self.model)),
            (
                "input_audio".to_string(),
                serde_json::json!({
                    "data": audio_b64,
                    "format": format,
                }),
            ),
        ]
        .into_iter()
        .collect();

        if let Some(language) = request.language {
            body_map.insert("language".to_string(), serde_json::json!(language));
        }
        if let Some(temperature) = request.temperature {
            body_map.insert("temperature".to_string(), serde_json::json!(temperature));
        }

        if let Some(ref additional_params) = request.additional_params {
            let params = additional_params.as_object().ok_or_else(|| {
                TranscriptionError::RequestError(Box::new(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    "additional transcription parameters must be a JSON object",
                )))
            })?;
            for (k, v) in params {
                body_map.insert(k.clone(), v.clone());
            }
        }

        let body = serde_json::to_vec(&serde_json::Value::Object(body_map))?;

        let req = self
            .client
            .post("/audio/transcriptions")?
            .header("Content-Type", "application/json")
            .body(body)
            .map_err(|e| TranscriptionError::HttpError(e.into()))?;

        let response = self.client.send::<_, Bytes>(req).await?;
        let status = response.status();
        let body_bytes = response.into_body().await?;

        if status.is_success() {
            let resp: TranscriptionResponse = serde_json::from_slice(&body_bytes)?;
            resp.try_into()
        } else {
            let text = String::from_utf8_lossy(&body_bytes).to_string();
            Err(TranscriptionError::ProviderError(text))
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_infer_format_from_filename() {
        assert_eq!(infer_format_from_filename("audio.wav"), "wav");
        assert_eq!(infer_format_from_filename("audio.mp3"), "mp3");
        assert_eq!(infer_format_from_filename("audio.flac"), "flac");
        assert_eq!(infer_format_from_filename("audio.m4a"), "m4a");
        assert_eq!(infer_format_from_filename("audio.ogg"), "ogg");
        assert_eq!(infer_format_from_filename("audio.webm"), "webm");
        assert_eq!(infer_format_from_filename("audio.aac"), "aac");
        assert_eq!(infer_format_from_filename("audio.WAV"), "wav");
        assert_eq!(infer_format_from_filename("audio.MP3"), "mp3");
        assert_eq!(infer_format_from_filename("unknown"), "wav");
        assert_eq!(infer_format_from_filename("noextension"), "wav");
        assert_eq!(infer_format_from_filename("meeting.final.mp3"), "mp3");
        assert_eq!(infer_format_from_filename("audio.tar.gz"), "wav");
    }

    #[test]
    fn test_transcription_request_serialization() {
        let audio_b64 = STANDARD.encode(b"test audio data");
        let req = TranscriptionRequestInput {
            model: "openai/whisper-1".to_string(),
            input_audio: InputAudio {
                data: audio_b64,
                format: "mp3".to_string(),
            },
            language: Some("en".to_string()),
            temperature: None,
        };
        let json = serde_json::to_string(&req).unwrap();
        assert!(json.contains("\"model\":\"openai/whisper-1\""));
        assert!(json.contains("\"input_audio\""));
        assert!(json.contains("\"language\":\"en\""));
    }

    #[test]
    fn test_transcription_response_deserialization() {
        let json = r#"{"text": "Hello world", "usage": {"seconds": 1.5, "cost": 0.001}}"#;
        let resp: TranscriptionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.text, "Hello world");
        let usage = resp.usage.unwrap();
        assert_eq!(usage.seconds, Some(1.5));
    }

    #[test]
    fn test_transcription_response_without_usage() {
        let json = r#"{"text": "Hello world"}"#;
        let resp: TranscriptionResponse = serde_json::from_str(json).unwrap();
        assert_eq!(resp.text, "Hello world");
        assert!(resp.usage.is_none());
    }
}
