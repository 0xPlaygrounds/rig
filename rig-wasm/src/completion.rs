use crate::JsResult;
use rig::message::ToolCall;
use std::collections::HashMap;
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
#[derive(Clone)]
pub struct Message(rig::message::Message);

impl From<Message> for rig::message::Message {
    fn from(value: Message) -> Self {
        value.0
    }
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct Document(rig::completion::Document);

impl From<Document> for rig::completion::Document {
    fn from(value: Document) -> Self {
        value.0
    }
}

#[wasm_bindgen]
impl Document {
    #[wasm_bindgen(constructor)]
    pub fn new(id: &str, text: &str) -> Self {
        Self(rig::completion::Document {
            id: id.to_string(),
            text: text.to_string(),
            additional_props: HashMap::new(),
        })
    }

    #[wasm_bindgen(js_name = "setAdditionalProps")]
    pub fn set_additional_props(mut self, additional_props: JsValue) -> JsResult<Self> {
        let value: HashMap<String, String> = serde_wasm_bindgen::from_value(additional_props)
            .map_err(|x| JsError::new(x.to_string().as_ref()))?;
        self.0.additional_props = value;
        Ok(self)
    }
}

#[wasm_bindgen]
#[derive(Clone)]
pub struct ToolDefinition(rig::completion::ToolDefinition);

impl From<ToolDefinition> for rig::completion::ToolDefinition {
    fn from(value: ToolDefinition) -> Self {
        value.0
    }
}

#[wasm_bindgen]
#[allow(dead_code)]
pub struct AssistantContent(rig::completion::AssistantContent);

impl From<AssistantContent> for rig::completion::AssistantContent {
    fn from(value: AssistantContent) -> Self {
        value.0
    }
}

#[wasm_bindgen]
impl AssistantContent {
    pub fn text(text: &str) -> Self {
        Self(rig::completion::AssistantContent::Text(
            rig::message::Text {
                text: text.to_string(),
            },
        ))
    }

    pub fn tool_call(id: &str, function: ToolFunction) -> Self {
        Self(rig::completion::AssistantContent::ToolCall(ToolCall {
            id: id.to_string(),
            call_id: None,
            function: function.0,
        }))
    }

    pub fn tool_call_with_call_id(id: &str, call_id: &str, function: ToolFunction) -> Self {
        Self(rig::completion::AssistantContent::ToolCall(ToolCall {
            id: id.to_string(),
            call_id: Some(call_id.to_string()),
            function: function.0,
        }))
    }
}

#[wasm_bindgen]
pub struct ToolFunction(rig::completion::message::ToolFunction);

impl From<ToolFunction> for rig::completion::message::ToolFunction {
    fn from(value: ToolFunction) -> Self {
        value.0
    }
}

#[wasm_bindgen]
impl ToolFunction {
    pub fn name(&self) -> String {
        self.0.name.clone()
    }

    pub fn args(&self) -> JsResult<JsValue> {
        serde_wasm_bindgen::to_value(&self.0.arguments)
            .map_err(|x| JsError::new(x.to_string().as_ref()))
    }
}

impl From<rig::completion::AssistantContent> for AssistantContent {
    fn from(value: rig::completion::AssistantContent) -> Self {
        Self(value)
    }
}
