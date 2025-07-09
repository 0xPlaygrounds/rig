use crate::JsonObject;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct ImageGenerationRequest(rig::image_generation::ImageGenerationRequest);

#[wasm_bindgen]
impl ImageGenerationRequest {
    #[wasm_bindgen]
    pub fn new(prompt: &str) -> Self {
        let req = rig::image_generation::ImageGenerationRequest {
            prompt: prompt.to_string(),
            height: 256,
            width: 256,
            additional_params: None,
        };

        Self(req)
    }

    #[wasm_bindgen(js_name = "setPrompt")]
    pub fn prompt(mut self, prompt: &str) -> Self {
        self.0.prompt = prompt.to_string();
        self
    }

    #[wasm_bindgen(js_name = "setHeight")]
    pub fn height(mut self, height: u32) -> Self {
        self.0.height = height;
        self
    }

    #[wasm_bindgen(js_name = "setWidth")]
    pub fn width(mut self, width: u32) -> Self {
        self.0.width = width;
        self
    }

    #[wasm_bindgen(js_name = "setAdditionalParameters")]
    pub fn additional_parameters(mut self, json: JsonObject) -> Self {
        let params: serde_json::Value = serde_wasm_bindgen::from_value(json.obj).unwrap();
        self.0.additional_params = Some(params);
        self
    }
}

impl From<ImageGenerationRequest> for rig::image_generation::ImageGenerationRequest {
    fn from(value: ImageGenerationRequest) -> Self {
        value.0
    }
}

impl From<rig::image_generation::ImageGenerationRequest> for ImageGenerationRequest {
    fn from(value: rig::image_generation::ImageGenerationRequest) -> Self {
        Self(value)
    }
}
