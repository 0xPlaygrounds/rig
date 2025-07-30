use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ImageGenerationRequest {
    prompt: String,
    height: u32,
    width: u32,
    #[serde(flatten)]
    additional_params: serde_json::Map<String, serde_json::Value>,
}

impl From<ImageGenerationRequest> for rig::image_generation::ImageGenerationRequest {
    fn from(value: ImageGenerationRequest) -> Self {
        let ImageGenerationRequest {
            prompt,
            height,
            width,
            additional_params,
        } = value;

        Self {
            prompt,
            height,
            width,
            additional_params: Some(serde_json::Value::Object(additional_params)),
        }
    }
}
