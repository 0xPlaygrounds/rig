use rig::transcription::TranscriptionRequest as CoreTranscriptionRequest;
use wasm_bindgen::prelude::*;

use crate::{JsResult, JsonObject};
#[wasm_bindgen]
pub struct TranscriptionRequest(CoreTranscriptionRequest);

#[wasm_bindgen]
impl TranscriptionRequest {
    #[wasm_bindgen(constructor)]
    pub fn new(arr: wasm_bindgen_futures::js_sys::Uint8Array) -> Self {
        let data = arr.to_vec();
        let req = CoreTranscriptionRequest {
            data,
            filename: String::new(),
            language: String::new(),
            prompt: None,
            temperature: None,
            additional_params: None,
        };

        Self(req)
    }

    #[wasm_bindgen(js_name = "setFilename")]
    pub fn filename(mut self, filename: &str) -> Self {
        self.0.filename = filename.to_string();
        self
    }

    #[wasm_bindgen(js_name = "setLanguage")]
    pub fn language(mut self, language: &str) -> Self {
        self.0.language = language.to_string();
        self
    }

    #[wasm_bindgen(js_name = "setPrompt")]
    pub fn prompt(mut self, prompt: &str) -> Self {
        self.0.prompt = Some(prompt.to_string());
        self
    }

    #[wasm_bindgen(js_name = "setTemperature")]
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.0.temperature = Some(temperature);
        self
    }

    #[wasm_bindgen(js_name = "setAdditionalParams")]
    pub fn additional_params(mut self, additional_params: JsonObject) -> JsResult<Self> {
        let value: serde_json::Value = serde_wasm_bindgen::from_value(additional_params.obj)
            .map_err(|x| JsError::new(x.to_string().as_ref()))?;

        self.0.additional_params = Some(value);
        Ok(self)
    }
}

impl From<TranscriptionRequest> for CoreTranscriptionRequest {
    fn from(value: TranscriptionRequest) -> Self {
        value.0
    }
}

impl From<CoreTranscriptionRequest> for TranscriptionRequest {
    fn from(value: CoreTranscriptionRequest) -> Self {
        Self(value)
    }
}
