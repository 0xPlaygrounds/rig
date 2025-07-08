use wasm_bindgen::prelude::*;
use wasm_bindgen::{JsError, JsValue};
pub mod completion;
pub mod providers;
pub mod tool;

pub type JsResult<T> = Result<T, JsError>;

#[wasm_bindgen]
unsafe extern "C" {
    #[wasm_bindgen(typescript_type = "JsToolObject")]
    pub type JsToolObject;
    #[wasm_bindgen(method)]
    fn name(this: &JsToolObject) -> String;
    #[wasm_bindgen(method)]
    fn definition(this: &JsToolObject, prompt: String) -> JsValue;
    #[wasm_bindgen(method)]
    fn call(this: &JsToolObject, input: JsValue) -> JsValue;
}

#[wasm_bindgen(js_name = "initPanicHook")]
pub fn init_panic_hook() {
    std::panic::set_hook(Box::new(console_error_panic_hook::hook));
}
