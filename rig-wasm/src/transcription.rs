use rig::transcription::TranscriptionRequest as CoreTranscriptionRequest;
use serde::{Deserialize, Serialize};
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Deserialize, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct TranscriptionRequest {
    data: Vec<u8>,
    filename: String,
    language: String,
    prompt: Option<String>,
    temperature: Option<f64>,
    #[serde(flatten)]
    additional_params: serde_json::Map<String, serde_json::Value>,
}

impl From<TranscriptionRequest> for CoreTranscriptionRequest {
    fn from(value: TranscriptionRequest) -> Self {
        let TranscriptionRequest {
            data,
            filename,
            language,
            prompt,
            temperature,
            additional_params,
        } = value;
        CoreTranscriptionRequest {
            data,
            filename,
            language,
            prompt,
            temperature,
            additional_params: Some(serde_json::Value::Object(additional_params)),
        }
    }
}
