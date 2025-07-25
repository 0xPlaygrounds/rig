use wasm_bindgen::prelude::*;
use wasm_bindgen::{JsError, JsValue};
use wasm_bindgen_futures::js_sys::Reflect;

pub mod audio_generation;
pub mod completion;
pub mod embedding;
pub mod image_generation;
pub mod providers;
pub mod tool;
pub mod transcription;
pub mod vector_store;

pub type JsResult<T> = Result<T, JsError>;

#[wasm_bindgen]
unsafe extern "C" {
    /// `console.log()` from Rust.
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);

    /// A TS interface that implements all of the `rig::tool::Tool` trait functions.
    #[wasm_bindgen(typescript_type = "JsToolObject")]
    pub type JsToolObject;
    #[wasm_bindgen(method)]
    fn name(this: &JsToolObject) -> String;
    #[wasm_bindgen(method)]
    fn definition(this: &JsToolObject, prompt: String) -> JsValue;
    #[wasm_bindgen(method)]
    fn call(this: &JsToolObject, input: JsValue) -> JsValue;

    /// A Rust conversion type for the JS equivalent of `impl Iterator<Item = String>`.
    #[wasm_bindgen(typescript_type = "Iterable<string>")]
    pub type StringIterable;

    #[wasm_bindgen(typescript_type = "JsVectorStore")]
    pub type JsVectorStoreShim;
    #[wasm_bindgen(method)]
    fn top_n(this: &JsVectorStoreShim, query: &str, n: u32) -> JsValue;
    #[wasm_bindgen(method)]
    fn top_n_ids(this: &JsVectorStoreShim, query: &str, n: u32) -> JsValue;

    #[wasm_bindgen(typescript_type = "JSONObject")]
    pub type JsonObject;

    #[wasm_bindgen(typescript_type = "AgentOpts")]
    pub type JsAgentOpts;

    #[wasm_bindgen(typescript_type = "ModelOpts")]
    pub type JsModelOpts;

    #[wasm_bindgen(typescript_type = "CompletionOpts")]
    pub type JsCompletionOpts;
    #[wasm_bindgen(typescript_type = "TranscriptionOpts")]
    pub type JsTranscriptionOpts;
    #[wasm_bindgen(typescript_type = "ImageGenerationOpts")]
    pub type JsImageGenerationOpts;

    #[wasm_bindgen(typescript_type = "InMemoryVectorStoreOpts")]
    pub type JsInMemoryVectorStoreOpts;

    #[wasm_bindgen(typescript_type = "CanEmbed")]
    pub type ImplementsVectorStoreIndexTrait;

    #[wasm_bindgen(typescript_type = "AudioGenerationOpts")]
    pub type JsAudioGenerationOpts;
}

#[derive(serde::Deserialize)]
#[serde(rename_all = "camelCase")]
pub struct ModelOpts {
    api_key: String,
    model_name: String,
    #[serde(flatten)]
    additional_params: serde_json::Map<String, serde_json::Value>,
}

#[wasm_bindgen(js_name = "initPanicHook")]
pub fn init_panic_hook() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
}

pub fn ensure_type_implements_functions(ty: &JsValue, funcs: Vec<&str>) -> JsResult<()> {
    for func in funcs {
        let name = Reflect::get(ty, &JsValue::from_str(func)).map_err(|_| {
            JsError::new(&format!(
                "The '{func}' function is required to be on value {ty:?}"
            ))
        })?;
        if !name.is_function() {
            let typename = name
                .js_typeof()
                .as_string()
                .expect("converting a JS type to a string shouldn't panic");

            return Err(JsError::new(
                format!("expected {name:?} to be a function, got type: {typename}").as_ref(),
            ));
        }
    }

    Ok(())
}
