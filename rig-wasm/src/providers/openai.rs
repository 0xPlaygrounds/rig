use crate::completion::{AssistantContent, Document, Message, ToolDefinition};
use crate::embedding::Embedding;
use crate::tool::JsTool;
use crate::{JsResult, JsToolObject, StringIterable};
use rig::OneOrMany;
use rig::agent::{Agent, AgentBuilder};
use rig::client::CompletionClient;
use rig::client::embeddings::EmbeddingsClient;
use rig::completion::{Chat, CompletionModel, Prompt};
use rig::embeddings::EmbeddingModel;
use serde_json::Map;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::js_sys;

#[wasm_bindgen]
#[derive(Clone)]
pub struct OpenAIClient(rig::providers::openai::Client);

#[wasm_bindgen]
impl OpenAIClient {
    #[wasm_bindgen(constructor)]
    pub fn new(api_key: &str) -> Self {
        Self(rig::providers::openai::Client::new(api_key))
    }

    pub fn from_url(api_key: &str, base_url: &str) -> Self {
        Self(rig::providers::openai::Client::from_url(api_key, base_url))
    }

    pub fn completion_model(&self, model_name: &str) -> OpenAICompletionModel {
        OpenAICompletionModel::new(self, model_name)
    }

    pub fn agent(&self, model_name: &str) -> OpenAIAgentBuilder {
        OpenAIAgentBuilder::new(self, model_name)
    }

    pub fn embedding_model(&self, model_name: &str) -> OpenAIEmbeddingModel {
        OpenAIEmbeddingModel::new(self, model_name)
    }
}

#[wasm_bindgen]
pub struct OpenAIAgentBuilder {
    builder: AgentBuilder<rig::providers::openai::responses_api::ResponsesCompletionModel>,
}

#[wasm_bindgen]
impl OpenAIAgentBuilder {
    #[wasm_bindgen(constructor)]
    pub fn new(client: &OpenAIClient, model_name: &str) -> Self {
        let builder = client.0.agent(model_name);
        Self { builder }
    }

    #[wasm_bindgen(js_name = "setPreamble")]
    pub fn preamble(mut self, preamble: &str) -> Self {
        self.builder = self.builder.preamble(preamble);
        self
    }

    #[wasm_bindgen(js_name = "addTool")]
    pub fn add_tool(mut self, tool: JsToolObject) -> Self {
        let tool = JsTool::new(tool);
        self.builder = self.builder.tool(tool);
        self
    }

    pub fn build(self) -> OpenAIAgent {
        let agent = self.builder.build();

        OpenAIAgent(agent)
    }
}

#[wasm_bindgen]
pub struct OpenAIAgent(Agent<rig::providers::openai::responses_api::ResponsesCompletionModel>);

#[wasm_bindgen]
impl OpenAIAgent {
    pub async fn prompt(&self, prompt: &str) -> JsResult<String> {
        self.0
            .prompt(prompt)
            .await
            .map_err(|x| JsError::new(x.to_string().as_ref()))
    }

    pub async fn prompt_multi_turn(&self, prompt: &str, turns: u32) -> JsResult<String> {
        self.0
            .prompt(prompt)
            .multi_turn(turns as usize)
            .await
            .map_err(|x| JsError::new(x.to_string().as_ref()))
    }

    pub async fn chat(&self, prompt: &str, messages: Vec<Message>) -> JsResult<String> {
        let messages: Vec<rig::message::Message> = messages
            .into_iter()
            .map(rig::message::Message::from)
            .collect();
        self.0
            .chat(prompt, messages)
            .await
            .map_err(|x| JsError::new(x.to_string().as_ref()))
    }
}

#[wasm_bindgen]
pub struct OpenAICompletionModel {
    model: rig::providers::openai::responses_api::ResponsesCompletionModel,
}

#[wasm_bindgen]
impl OpenAICompletionModel {
    #[wasm_bindgen(constructor)]
    pub fn new(client: &OpenAIClient, model_name: &str) -> Self {
        let model = client.0.completion_model(model_name);
        Self { model }
    }
}

#[wasm_bindgen]
pub struct OpenAICompletionRequest {
    model: OpenAICompletionModel,
    prompt: Message,
    preamble: Option<String>,
    chat_history: Vec<Message>,
    documents: Vec<Document>,
    tools: Vec<ToolDefinition>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    additional_params: Option<serde_json::Value>,
}

#[wasm_bindgen]
impl OpenAICompletionRequest {
    #[wasm_bindgen(constructor)]
    pub fn new(model: OpenAICompletionModel, prompt: Message) -> Self {
        Self {
            model,
            prompt,
            preamble: None,
            chat_history: Vec::new(),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            max_tokens: None,
            additional_params: None,
        }
    }

    #[wasm_bindgen(js_name = "setPreamble")]
    pub fn set_preamble(mut self, preamble: &str) -> Self {
        self.preamble = Some(preamble.to_string());
        self
    }

    #[wasm_bindgen(js_name = "setChatHistory")]
    pub fn set_chat_history(mut self, chat_history: Vec<Message>) -> Self {
        self.chat_history = chat_history;
        self
    }

    #[wasm_bindgen(js_name = "setDocuments")]
    pub fn set_documents(mut self, documents: Vec<Document>) -> Self {
        self.documents = documents;
        self
    }

    #[wasm_bindgen(js_name = "setTools")]
    pub fn set_tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;
        self
    }

    #[wasm_bindgen(js_name = "setTemperature")]
    pub fn set_temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    #[wasm_bindgen(js_name = "setMaxTokens")]
    pub fn set_max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    #[wasm_bindgen(js_name = "setAdditionalParams")]
    pub fn additional_params(mut self, obj: JsValue) -> JsResult<Self> {
        let value: Map<String, serde_json::Value> = serde_wasm_bindgen::from_value(obj)?;
        let value = serde_json::Value::Object(value);
        self.additional_params = Some(value);
        Ok(self)
    }

    pub async fn send(self) -> JsResult<Vec<AssistantContent>> {
        let request = rig::completion::CompletionRequest::from(&self);
        let res = self
            .model
            .model
            .completion(request)
            .await
            .map_err(|x| JsError::new(format!("{x}").as_ref()))?
            .choice
            .into_iter()
            .map(AssistantContent::from)
            .collect();

        Ok(res)
    }
}

impl From<&OpenAICompletionRequest> for rig::completion::CompletionRequest {
    fn from(value: &OpenAICompletionRequest) -> Self {
        let mut chat_history: Vec<rig::message::Message> = value
            .chat_history
            .clone()
            .into_iter()
            .map(rig::message::Message::from)
            .collect();
        chat_history.push(rig::message::Message::from(value.prompt.clone()));

        rig::completion::CompletionRequest {
            preamble: value.preamble.clone(),
            chat_history: OneOrMany::many(chat_history).unwrap(),
            documents: value
                .documents
                .clone()
                .into_iter()
                .map(rig::completion::Document::from)
                .collect(),
            tools: value
                .tools
                .clone()
                .into_iter()
                .map(rig::completion::ToolDefinition::from)
                .collect(),
            temperature: value.temperature,
            max_tokens: value.max_tokens,
            additional_params: value.additional_params.clone(),
        }
    }
}

#[wasm_bindgen]
pub struct OpenAIEmbeddingModel(rig::providers::openai::embedding::EmbeddingModel);

#[wasm_bindgen]
impl OpenAIEmbeddingModel {
    #[wasm_bindgen(constructor)]
    pub fn new(model: &OpenAIClient, model_name: &str) -> Self {
        let model = model.0.embedding_model(model_name);
        Self(model)
    }

    pub async fn embed_text(&self, text: String) -> JsResult<Embedding> {
        let res = self
            .0
            .embed_text(&text)
            .await
            .map_err(|e| JsError::new(e.to_string().as_ref()))?;

        Ok(Embedding::from(res))
    }

    pub async fn embed_texts(&self, iter: StringIterable) -> JsResult<Vec<Embedding>> {
        let iterable: JsValue = iter.unchecked_into();
        let arr = js_sys::Array::from(&iterable);

        let val = arr
            .into_iter()
            .map(|x| x.as_string().ok_or_else(|| JsError::new("Expected string")))
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
