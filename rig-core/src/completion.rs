use std::collections::HashMap;

use anyhow::Result;
use serde::{Deserialize, Serialize};

use crate::json_utils;

// ================================================================
// Request models
// ================================================================
#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Message {
    /// "system", "user", or "assistant"
    pub role: String,
    pub content: String,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct Document {
    pub id: String,
    pub text: String,
    #[serde(flatten)]
    pub additional_props: HashMap<String, String>,
}

#[derive(Clone, Debug, Deserialize, Serialize)]
pub struct ToolDefinition {
    pub name: String,
    pub description: String,
    pub parameters: serde_json::Value,
}

// ================================================================
// Implementations
// ================================================================
/// Trait defining a high-level LLM chat interface (i.e.: prompt in, response out).
pub trait Prompt {
    /// Send a prompt to the completion endpoint along with a chat history.
    /// If the response is a message, then it is returned
    /// as a string. If the response is a tool call, then the tool is called
    /// and the result is returned as a string.
    fn prompt(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> impl std::future::Future<Output = Result<String>>;
}

/// Trait defininig a low-level LLM completion interface
pub trait Completion<M: CompletionModel> {
    /// Generates a completion request builder for the given `prompt` and `chat_history`.
    /// This function is meant to be called by the user to further customize the
    /// request at prompt time before sending it.
    ///
    /// IMPORTANT: The CompletionModel that implements this trait will already
    /// populate fields (the exact fields depend on the model) in the builder.
    /// For fields that have already been set by the model, calling the corresponding
    /// method on the builder will overwrite the value set by the model.
    fn completion(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> impl std::future::Future<Output = Result<CompletionRequestBuilder<M>>> + Send;
}

#[derive(Debug)]
pub struct CompletionResponse<T> {
    pub choice: ModelChoice,
    pub raw_response: T,
}

#[derive(Debug)]
pub enum ModelChoice {
    Message(String),
    ToolCall(String, serde_json::Value),
}

pub trait CompletionModel: Clone + Send + Sync {
    type T;

    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl std::future::Future<Output = Result<CompletionResponse<Self::T>>> + Send;

    fn completion_request(&self, prompt: &str) -> CompletionRequestBuilder<Self> {
        CompletionRequestBuilder::new(self.clone(), prompt.to_string())
    }

    fn simple_completion(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> impl std::future::Future<Output = Result<CompletionResponse<Self::T>>> + Send {
        async move {
            self.completion_request(prompt)
                .messages(chat_history)
                .send()
                .await
        }
    }
}

pub struct CompletionRequest {
    pub temperature: Option<f64>,
    pub prompt: String,
    pub preamble: Option<String>,
    pub chat_history: Vec<Message>,
    pub documents: Vec<Document>,
    pub tools: Vec<ToolDefinition>,
    pub additional_params: Option<serde_json::Value>,
}

pub struct CompletionRequestBuilder<M: CompletionModel> {
    model: M,
    prompt: String,
    preamble: Option<String>,
    chat_history: Vec<Message>,
    documents: Vec<Document>,
    tools: Vec<ToolDefinition>,
    temperature: Option<f64>,
    additional_params: Option<serde_json::Value>,
}

impl<M: CompletionModel> CompletionRequestBuilder<M> {
    pub fn new(model: M, prompt: String) -> Self {
        Self {
            model,
            prompt,
            preamble: None,
            chat_history: Vec::new(),
            documents: Vec::new(),
            tools: Vec::new(),
            temperature: None,
            additional_params: None,
        }
    }

    pub fn preamble(mut self, preamble: String) -> Self {
        self.preamble = Some(preamble);
        self
    }

    pub fn message(mut self, message: Message) -> Self {
        self.chat_history.push(message);
        self
    }

    pub fn messages(self, messages: Vec<Message>) -> Self {
        messages
            .into_iter()
            .fold(self, |builder, msg| builder.message(msg))
    }

    pub fn document(mut self, document: Document) -> Self {
        self.documents.push(document);
        self
    }

    pub fn documents(self, documents: Vec<Document>) -> Self {
        documents
            .into_iter()
            .fold(self, |builder, doc| builder.document(doc))
    }

    pub fn tool(mut self, tool: ToolDefinition) -> Self {
        self.tools.push(tool);
        self
    }

    pub fn tools(self, tools: Vec<ToolDefinition>) -> Self {
        tools
            .into_iter()
            .fold(self, |builder, tool| builder.tool(tool))
    }

    pub fn additional_params(mut self, additional_params: serde_json::Value) -> Self {
        match self.additional_params {
            Some(params) => {
                self.additional_params = Some(json_utils::merge(params, additional_params));
            }
            None => {
                self.additional_params = Some(additional_params);
            }
        }
        self
    }

    pub fn additional_params_opt(mut self, additional_params: Option<serde_json::Value>) -> Self {
        self.additional_params = additional_params;
        self
    }

    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    pub fn temperature_opt(mut self, temperature: Option<f64>) -> Self {
        self.temperature = temperature;
        self
    }

    pub fn build(self) -> CompletionRequest {
        CompletionRequest {
            prompt: self.prompt,
            preamble: self.preamble,
            chat_history: self.chat_history,
            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            additional_params: self.additional_params,
        }
    }

    pub async fn send(self) -> Result<CompletionResponse<M::T>> {
        let model = self.model.clone();
        model.completion(self.build()).await
    }
}
