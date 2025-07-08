use crate::JsToolObject;
use rig::tool::ToolError;
use send_wrapper::SendWrapper;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::{JsFuture, js_sys, spawn_local};

/// A tool that uses JavaScript.
/// Unfortunately, JavaScript functions are *mut u8 at their core (when it comes to how they're typed in Rust).
/// This means that we need to use `send_wrapper::SendWrapper` which automatically makes it Send.
/// However, if it gets dropped from outside of the thread where it was created, it will panic.
#[wasm_bindgen]
pub struct JsTool {
    inner: SendWrapper<JsToolObject>,
}

#[wasm_bindgen]
impl JsTool {
    pub fn new(tool: JsToolObject) -> Self {
        let inner = SendWrapper::new(tool);
        Self { inner }
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
        let res = self.inner.definition(prompt);

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
        let value = serde_wasm_bindgen::to_value(&args)
            .inspect_err(|x| println!("Error: {x}"))
            .map_err(|x| ToolError::ToolCallError(x.to_string().into()))
            .unwrap();

        let func: JsToolObject = self.inner.clone().unchecked_into::<JsToolObject>();

        spawn_local(async move {
            let res: js_sys::Promise = func
                .call(value)
                .dyn_into()
                .expect("This method call should return a promise!");

            let res = match JsFuture::from(res).await {
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
