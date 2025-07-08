use crate::JsToolObject;
use send_wrapper::SendWrapper;
use wasm_bindgen::prelude::*;

// #[wasm_bindgen]
// pub struct JsTool {
//     inner: JsValue,
// }

// #[wasm_bindgen]
// impl JsTool {
//     #[wasm_bindgen(constructor)]
//     pub fn new(inner: JsToolObject) -> Self {
//         let js = JsValue::from(inner);
//         Self {
//             inner: inner.clone().into(),
//         }
//     }
// }

// impl rig::tool::Tool for JsTool {
//     type Args = JsValue;
//     type Error = String;
//     type Output = JsValue;

//     const NAME: &str = self.inner.name().as_ref();

//     fn name(&self) -> String {
//         self.inner.name()
//     }

//     async fn definition(&self, prompt: String) -> rig::completion::ToolDefinition {
//         self.definition(prompt)
//     }

//     async fn call(&self, args: Self::Args) -> Result<Self::Output, Self::Error> {
//         self.inner.call(args).await
//     }
// }

/// A tool that uses JavaScript.
/// Unfortunately, JavaScript functions are *mut u8 at their core (when it comes to how they're typed in Rust).
/// This means that we need to use `send_wrapper::SendWrapper` which automatically makes it Send.
/// However, if it gets dropped from outside of the thread where it was created, it will panic.
#[wasm_bindgen]
pub struct JsTool {
    inner: SendWrapper<JsToolObject>, // name: String,
                                      // definition: serde_json::Value,
                                      // function: SendWrapper<js_sys::Function>,
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
        println!("{res:?}");
        let value: rig::completion::ToolDefinition = serde_wasm_bindgen::from_value(res)
            .inspect_err(|x| println!("Error: {x}"))
            .unwrap();

        async { value }
    }

    fn call(
        &self,
        args: Self::Args,
    ) -> impl Future<Output = Result<Self::Output, Self::Error>> + Send + Sync {
        let value = serde_wasm_bindgen::to_value(&args)
            .inspect_err(|x| println!("Error: {x}"))
            .unwrap();
        let res = self.inner.call(value);
        let value: serde_json::Value = serde_wasm_bindgen::from_value(res)
            .inspect_err(|x| println!("Error: {x}"))
            .unwrap();

        async { Ok(value) }
    }
}
