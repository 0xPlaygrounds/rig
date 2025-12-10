use crate::embedding::Embedding;
use crate::{JsModelOpts, JsResult, ModelOpts, StringIterable};
use rig::client::embeddings::EmbeddingsClient;
use rig::embeddings::EmbeddingModel;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::js_sys::{self};

#[wasm_bindgen]
pub struct VoyageAIEmbeddingModel(rig::providers::voyageai::EmbeddingModel);

#[wasm_bindgen]
impl VoyageAIEmbeddingModel {
    #[wasm_bindgen(constructor)]
    pub fn new(opts: JsModelOpts) -> JsResult<Self> {
        let model_opts: ModelOpts = serde_wasm_bindgen::from_value(opts.obj)
            .map_err(|x| JsError::new(format!("Failed to create model options: {x}").as_ref()))?;

        let client = rig::providers::voyageai::Client::new(&model_opts.api_key);
        let model = client.embedding_model(&model_opts.model_name);
        Ok(Self(model))
    }

    #[wasm_bindgen(js_name = "embedText")]
    pub async fn embed_text(&self, text: String) -> JsResult<Embedding> {
        let res = self
            .0
            .embed_text(&text)
            .await
            .map_err(|e| JsError::new(e.to_string().as_ref()))?;

        Ok(Embedding::from(res))
    }

    #[wasm_bindgen(js_name = "embedTexts")]
    pub async fn embed_texts(&self, iter: StringIterable) -> JsResult<Vec<Embedding>> {
        let arr = js_sys::Array::from(&iter.obj);

        let val = arr
            .into_iter()
            .map(|x| {
                x.as_string()
                    .ok_or_else(|| JsError::new(format!("Expected string, got {x:?}").as_ref()))
            })
            .collect::<Result<Vec<String>, JsError>>()?;

        Ok(self
            .0
            .embed_texts(val)
            .await
            .map_err(|x| JsError::new(x.to_string().as_ref()))?
            .into_iter()
            .map(crate::embedding::Embedding::from)
            .collect())
    }
}
