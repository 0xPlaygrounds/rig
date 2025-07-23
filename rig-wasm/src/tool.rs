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
        let (result_tx, result_rx) = futures::channel::oneshot::channel();
        let (error_tx, error_rx) = futures::channel::oneshot::channel();
        let js_args = serde_wasm_bindgen::to_value(&args).expect("This should be a JSON object!");

        let func = self.inner.clone();

        spawn_local(async move {
            let call_fn = match js_sys::Reflect::get(&func, &JsValue::from_str("call"))
                .map_err(|_| ToolError::ToolCallError("tool.call missing".into()))
            {
                Ok(res) => res,
                Err(e) => {
                    error_tx
                        .send(e)
                        .expect("sending a message to a oneshot channel shouldn't fail");

                    return;
                }
            };

            let call_fn = call_fn.unchecked_into::<js_sys::Function>();

            if !call_fn.is_function() {
                error_tx
                    .send(ToolError::ToolCallError(
                        "tool.call is not a function".into(),
                    ))
                    .expect("sending a message to a oneshot channel shouldn't fail");

                return;
            }

            let promise = match call_fn
                .call1(&func, &js_args)
                .map_err(|_| ToolError::ToolCallError("tool.call failed".into()))
            {
                Ok(res) => res,
                Err(e) => {
                    error_tx
                        .send(e)
                        .expect("sending a message to a oneshot channel shouldn't fail");

                    return;
                }
            };

            let promise = match promise
                .dyn_into::<js_sys::Promise>()
                .map_err(|_| ToolError::ToolCallError("tool.call did not return a Promise".into()))
            {
                Ok(res) => res,
                Err(e) => {
                    error_tx
                        .send(e)
                        .expect("sending a message to a oneshot channel shouldn't fail");

                    return;
                }
            };

            let res = match JsFuture::from(promise).await.map_err(|x| {
                ToolError::ToolCallError(
                    format!(
                        "unable to turn JS promise back into Future: {x}",
                        x = x.as_string().unwrap()
                    )
                    .into(),
                )
            }) {
                Ok(res) => res,
                Err(e) => {
                    error_tx
                        .send(e)
                        .expect("sending a message to a oneshot channel shouldn't fail");

                    return;
                }
            };

            let res: serde_json::Value = match serde_wasm_bindgen::from_value(res).map_err(|x| {
                ToolError::ToolCallError(
                    format!(
                        "result from LLM provider was unable to be deserialized back to JSON: {x}",
                    )
                    .into(),
                )
            }) {
                Ok(res) => res,
                Err(e) => {
                    error_tx
                        .send(e)
                        .expect("sending a message to a oneshot channel shouldn't fail");

                    return;
                }
            };

            result_tx
                .send(res)
                .expect("sending a message to a oneshot channel shouldn't fail");
        });

        async {
            tokio::select! {
                res = result_rx => {
                    Ok(res.unwrap())
                },
                err = error_rx => {
                    Err(ToolError::ToolCallError(err.inspect_err(|x| println!("Future was cancelled: {x}")).unwrap().to_string().into()))
                }
            }
        }
    }
}
