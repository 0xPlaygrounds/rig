use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use tower::{Layer, Service};

use crate::{
    completion::{
        CompletionError, CompletionModel, CompletionRequest, CompletionRequestBuilder,
        CompletionResponse, Document, ToolDefinition,
    },
    message::{Message, ToolResultContent, UserContent},
    OneOrMany,
};

#[derive(Clone)]
pub struct CompletionLayer<M> {
    model: M,
    preamble: Option<String>,
    documents: Vec<Document>,
    tools: Vec<ToolDefinition>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    additional_params: Option<serde_json::Value>,
}

impl<M> CompletionLayer<M>
where
    M: CompletionModel,
{
    pub fn builder(model: M) -> CompletionLayerBuilder<M> {
        CompletionLayerBuilder::new(model)
    }
}

#[derive(Default)]
pub struct CompletionLayerBuilder<M> {
    model: M,
    preamble: Option<String>,
    documents: Vec<Document>,
    tools: Vec<ToolDefinition>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    additional_params: Option<serde_json::Value>,
}

impl<M> CompletionLayerBuilder<M>
where
    M: CompletionModel,
{
    pub fn new(model: M) -> Self {
        Self {
            model,
            preamble: None,
            documents: vec![],
            tools: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
        }
    }

    pub fn preamble(mut self, preamble: String) -> Self {
        self.preamble = Some(preamble);

        self
    }

    pub fn preamble_opt(mut self, preamble: Option<String>) -> Self {
        self.preamble = preamble;

        self
    }

    pub fn documents(mut self, documents: Vec<Document>) -> Self {
        self.documents = documents;

        self
    }

    pub fn tools(mut self, tools: Vec<ToolDefinition>) -> Self {
        self.tools = tools;

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

    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);

        self
    }

    pub fn max_tokens_opt(mut self, max_tokens: Option<u64>) -> Self {
        self.max_tokens = max_tokens;

        self
    }

    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.additional_params = Some(params);

        self
    }

    pub fn additional_params_opt(mut self, params: Option<serde_json::Value>) -> Self {
        self.additional_params = params;

        self
    }

    pub fn build(self) -> CompletionLayer<M> {
        CompletionLayer {
            model: self.model,
            preamble: self.preamble,

            documents: self.documents,
            tools: self.tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params,
        }
    }
}

impl<M, S> Layer<S> for CompletionLayer<M>
where
    M: CompletionModel,
{
    type Service = CompletionLayerService<M, S>;
    fn layer(&self, inner: S) -> Self::Service {
        CompletionLayerService {
            inner,
            model: self.model.clone(),
            preamble: self.preamble.clone(),

            documents: self.documents.clone(),
            tools: self.tools.clone(),
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params.clone(),
        }
    }
}

#[derive(Clone)]
pub struct CompletionLayerService<M, S> {
    inner: S,
    model: M,
    preamble: Option<String>,
    documents: Vec<Document>,
    tools: Vec<ToolDefinition>,
    temperature: Option<f64>,
    max_tokens: Option<u64>,
    additional_params: Option<serde_json::Value>,
}

impl<M, S> Service<CompletionRequest> for CompletionLayerService<M, S>
where
    M: CompletionModel + 'static,
    S: Service<CompletionRequest, Response = (OneOrMany<Message>, String, ToolResultContent)>
        + Clone
        + Send
        + 'static,
    S::Future: Send,
{
    type Response = CompletionResponse<M::Response>;
    type Error = CompletionError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CompletionRequest) -> Self::Future {
        let mut inner = self.inner.clone();
        let model = self.model.clone();
        let preamble = self.preamble.clone();
        let documents = self.documents.clone();
        let temperature = self.temperature;
        let tools = self.tools.clone();
        let max_tokens = self.max_tokens;
        let additional_params = self.additional_params.clone();

        Box::pin(async move {
            let Ok((messages, id, tool_content)) = inner.call(req).await else {
                todo!("Handle error properly");
            };

            let tool_result_message = Message::User {
                content: OneOrMany::one(UserContent::tool_result(id, OneOrMany::one(tool_content))),
            };

            let messages: Vec<Message> = messages.into_iter().collect();

            let mut req = CompletionRequestBuilder::new(model.clone(), tool_result_message)
                .documents(documents.clone())
                .tools(tools.clone())
                .messages(messages)
                .temperature_opt(temperature)
                .max_tokens_opt(max_tokens)
                .additional_params_opt(additional_params.clone());

            if let Some(preamble) = preamble.clone() {
                req = req.preamble(preamble);
            }

            let req = req.build();

            model.completion(req).await
        })
    }
}

/// A completion model as a Tower service.
///
/// This allows you to use an LLM model (or client) essentially anywhere you'd use a regular Tower layer, like in an Axum web service.
#[derive(Clone)]
pub struct CompletionService<M> {
    /// The model itself.
    model: M,
}

impl<M> CompletionService<M>
where
    M: CompletionModel,
{
    pub fn new(model: M) -> Self {
        Self { model }
    }
}

impl<M> Service<CompletionRequest> for CompletionService<M>
where
    M: CompletionModel + 'static,
{
    type Response = CompletionResponse<M::Response>;
    type Error = CompletionError;

    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CompletionRequest) -> Self::Future {
        let model = self.model.clone();

        Box::pin(async move { model.completion(req).await })
    }
}
