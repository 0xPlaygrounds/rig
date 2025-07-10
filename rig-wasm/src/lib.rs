use wasm_bindgen::prelude::*;
use wasm_bindgen::{JsError, JsValue};

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
}

#[wasm_bindgen(js_name = "initPanicHook")]
pub fn init_panic_hook() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
}
