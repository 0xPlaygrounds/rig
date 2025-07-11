use crate::{JsResult, JsToolObject, ensure_type_implements_functions};
use rig::tool::ToolError;
use send_wrapper::SendWrapper;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{JsFuture, js_sys, spawn_local};

/// A tool that can take a JavaScript function.
/// Generally speaking, any class that implements the `JsToolObject` TS interface will work when creating this.
#[wasm_bindgen]
pub struct JsTool {
    inner: SendWrapper<JsValue>,
}

#[wasm_bindgen]
impl JsTool {
    #[wasm_bindgen(constructor)]
    pub fn new(tool: JsValue) -> JsResult<Self> {
        let required_fns = vec!["name", "definition", "call"];
        ensure_type_implements_functions(&tool, required_fns)?;
        let inner = SendWrapper::new(tool);

        Ok(Self { inner })
    }
}

impl rig::tool::Tool for JsTool {
    type Args = serde_json::Value;
    type Error = rig::tool::ToolError;
    type Output = serde_json::Value;

    const NAME: &str = "JS_TOOL";

    fn name(&self) -> String {
        let res: &JsToolObject = self.inner.unchecked_ref();
        res.name()
    }

    fn definition(
        &self,
        prompt: String,
    ) -> impl Future<Output = rig::completion::ToolDefinition> + Send + Sync {
        let func = js_sys::Reflect::get(&self.inner, &JsValue::from_str("definition"))
            .expect("tool must have a definition method")
            .unchecked_into::<js_sys::Function>();

        let this = &self.inner;
        let prompt = JsValue::from_str(&prompt);
        let res = func.call1(this, &prompt).expect("definition call failed");

        let value: rig::completion::ToolDefinition = serde_wasm_bindgen::from_value(res)
            .inspect_err(|x| println!("Error: {x}"))
            .unwrap();

        async { value }
    }

    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send + Sync {
        let (tx, rx) = futures::channel::oneshot::channel();
        let js_args = serde_wasm_bindgen::to_value(&args).expect("This should be a JSON object!");

        let func = self.inner.clone();

        spawn_local(async move {
            let call_fn = js_sys::Reflect::get(&func, &JsValue::from_str("call"))
                .map_err(|_| ToolError::ToolCallError("tool.call missing".into()))
                .expect("Call function doesn't exist!")
                .unchecked_into::<js_sys::Function>();

            let promise = call_fn
                .call1(&func, &js_args)
                .map_err(|_| ToolError::ToolCallError("tool.call failed".into()))
                .expect("tool.call should succeed")
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| ToolError::ToolCallError("tool.call did not return a Promise".into()))
                .expect("This should return a promise");

            let res = match JsFuture::from(promise).await {
                Ok(res) => res,
                Err(_) => panic!(
                    "Couldn't get a JsFuture from the promise! This shouldn't normally panic."
                ),
            };

            let value: serde_json::Value = serde_wasm_bindgen::from_value(res)
                .inspect_err(|x| println!("Error: {x}"))
                .unwrap();

            let _ = tx
                .send(value)
                .map_err(|x| ToolError::ToolCallError(x.to_string().into()));
        });

        async {
            let res = rx
                .await
                .map_err(|x| ToolError::ToolCallError(x.to_string().into()))?;

            Ok(res)
        }
    }
}
