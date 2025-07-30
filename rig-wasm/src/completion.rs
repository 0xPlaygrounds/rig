use crate::{JsCompletionOpts, JsResult};
use rig::{OneOrMany, message::ToolCall};
use std::collections::HashMap;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::js_sys::Reflect;

#[wasm_bindgen]
#[derive(Clone)]
pub struct Message(rig::message::Message);

impl From<Message> for rig::message::Message {
    fn from(value: Message) -> Self {
        value.0
    }
}

impl From<rig::message::Message> for Message {
    fn from(value: rig::message::Message) -> Self {
        Self(value)
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

impl From<rig::completion::Document> for Document {
    fn from(value: rig::completion::Document) -> Self {
        Self(value)
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

impl From<rig::completion::ToolDefinition> for ToolDefinition {
    fn from(value: rig::completion::ToolDefinition) -> Self {
        Self(value)
    }
}

#[wasm_bindgen]
impl ToolDefinition {
    #[wasm_bindgen(constructor)]
    pub fn new(args: JsValue) -> JsResult<Self> {
        let args: rig::completion::ToolDefinition =
            serde_wasm_bindgen::from_value(args).map_err(|err| {
                JsError::new(format!("ToolDefinition creation error: {err}").as_ref())
            })?;

        Ok(Self(args))
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

#[wasm_bindgen]
pub struct CompletionRequest {
    preamble: Option<String>,
    messages: Vec<Message>,
    documents: Vec<Document>,
    tools: Vec<ToolDefinition>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    additional_params: Option<serde_json::Value>,
}

impl From<CompletionRequest> for rig::completion::CompletionRequest {
    fn from(value: CompletionRequest) -> Self {
        let CompletionRequest {
            preamble,
            messages,
            documents,
            tools,
            temperature,
            max_tokens,
            additional_params,
        } = value;
        let messages: Vec<rig::message::Message> = messages
            .into_iter()
            .map(rig::message::Message::from)
            .collect();
        Self {
            preamble,
            chat_history: OneOrMany::many(messages)
                .expect("This should never panic as we already assert that there is >=1 messages"),
            documents: documents
                .into_iter()
                .map(rig::completion::Document::from)
                .collect(),
            tools: tools
                .into_iter()
                .map(rig::completion::ToolDefinition::from)
                .collect(),
            temperature,
            max_tokens,
            additional_params,
        }
    }
}

#[wasm_bindgen]
impl CompletionRequest {
    #[wasm_bindgen(constructor)]
    pub fn new(opts: JsCompletionOpts) -> JsResult<Self> {
        let preamble = Reflect::get(&opts, &JsValue::from_str("preamble"))
            .ok()
            .and_then(|x| x.as_string());

        let messages = Reflect::get(&opts, &JsValue::from_str("messages")).map_err(|_| {
            JsError::new("completion_request.messages should be an array of messages")
        })?;
        let messages = convert_messages_from_jsvalue(messages)?;

        let tools = if let Ok(jsvalue) = Reflect::get(&opts, &JsValue::from_str("tools")) {
            if jsvalue.is_array() {
                convert_tooldefs_from_jsvalue(jsvalue)?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let documents = if let Ok(jsvalue) = Reflect::get(&opts, &JsValue::from_str("documents")) {
            if jsvalue.is_array() {
                convert_documents_from_jsvalue(jsvalue)?
            } else {
                Vec::new()
            }
        } else {
            Vec::new()
        };

        let temperature = Reflect::get(&opts, &JsValue::from_str("temperature"))
            .ok()
            .and_then(|x| x.as_f64());

        let max_tokens = Reflect::get(&opts, &JsValue::from_str("maxTokens"))
            .ok()
            .and_then(|x| x.as_f64().map(|x| x as u64));

        let additional_params = if let Ok(res) =
            Reflect::get(&opts, &JsValue::from_str("additionalParams"))
            && res.is_object()
        {
            let map: serde_json::Map<String, serde_json::Value> =
                serde_wasm_bindgen::from_value(res).map_err(|x| {
                    JsError::new(
                        format!("Error when deserializing additional parameters: {x}").as_ref(),
                    )
                })?;

            Some(serde_json::Value::Object(map))
        } else {
            None
        };

        Ok(Self {
            preamble,
            messages,
            documents,
            tools,
            temperature,
            max_tokens,
            additional_params,
        })
    }
}

pub fn convert_messages_from_jsvalue(value: JsValue) -> JsResult<Vec<Message>> {
    let val: Vec<Message> = serde_wasm_bindgen::from_value::<Vec<rig::message::Message>>(value)
        .map_err(|x| {
            JsError::new(
                format!("Deserialization error while converting JS value to messages: {x}")
                    .as_ref(),
            )
        })?
        .into_iter()
        .map(Message::from)
        .collect();

    if val.is_empty() {
        return Err(JsError::new(
            "Completion requests need at least one message!",
        ));
    }

    Ok(val)
}

pub fn convert_tooldefs_from_jsvalue(value: JsValue) -> JsResult<Vec<ToolDefinition>> {
    let val: Vec<ToolDefinition> = serde_wasm_bindgen::from_value::<
        Vec<rig::completion::ToolDefinition>,
    >(value)
    .map_err(|x| {
        JsError::new(
            format!("Deserialization error while converting JS value to tools: {x}").as_ref(),
        )
    })?
    .into_iter()
    .map(ToolDefinition::from)
    .collect();

    Ok(val)
}

pub fn convert_documents_from_jsvalue(value: JsValue) -> JsResult<Vec<Document>> {
    let val: Vec<Document> =
        serde_wasm_bindgen::from_value::<Vec<rig::completion::Document>>(value)
            .map_err(|x| {
                JsError::new(
                    format!("Deserialization error while converting JS value to documents: {x}")
                        .as_ref(),
                )
            })?
            .into_iter()
            .map(Document::from)
            .collect();

    Ok(val)
}

#[derive(serde::Serialize)]
pub struct CompletionResponse<T> {
    choice: OneOrMany<rig::completion::AssistantContent>,
    raw_response: T,
}

impl<T> From<rig::completion::CompletionResponse<T>> for CompletionResponse<T>
where
    T: serde::Serialize,
{
    fn from(value: rig::completion::CompletionResponse<T>) -> Self {
        Self {
            choice: value.choice,
            raw_response: value.raw_response,
        }
    }
}
