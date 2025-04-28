use std::{future::Future, pin::Pin, task::Poll};
use tower::{Layer, Service};

use crate::{
    completion::{
        CompletionModel, CompletionRequest, CompletionRequestBuilder, Document, ToolDefinition,
    },
    message::Message,
};

/// A Tower layer to finish building your `CompletionRequestBuilder`.
/// Intended to be used with [`CompletionRequestBuilderService`].
///
/// See [`CompletionRequestBuilderService`] for usage.
pub struct FinishBuilding;

impl<S> Layer<S> for FinishBuilding {
    type Service = FinishBuildingService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        FinishBuildingService { inner }
    }
}

/// A Tower layer to finish building your `CompletionRequestBuilder`.
/// Not intended to be used directly. Use [`FinishBuilding`] instead.
///
/// See [`CompletionRequestBuilderService`] for usage.
pub struct FinishBuildingService<S> {
    inner: S,
}

impl<M, Msg, S> Service<(M, Msg)> for FinishBuildingService<S>
where
    M: CompletionModel + Send + 'static,
    Msg: Into<Message> + Send + 'static,
    S: Service<(M, Msg), Response = CompletionRequestBuilder<M>> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = CompletionRequest;
    type Error = ();
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut std::task::Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: (M, Msg)) -> Self::Future {
        let mut inner = self.inner.clone();
        Box::pin(async move {
            let Ok(res) = inner.call(req).await else {
                todo!("Handle error properly");
            };

            Ok(res.build())
        })
    }
}

/// A Tower layer to add documents to your `CompletionRequestBuilder`.
/// Intended to be used with [`CompletionRequestBuilderService`].
///
/// See [`CompletionRequestBuilderService`] for usage.
pub struct DocumentsLayer {
    documents: Vec<Document>,
}

impl DocumentsLayer {
    pub fn new(documents: Vec<Document>) -> Self {
        Self { documents }
    }
}

impl<S> Layer<S> for DocumentsLayer {
    type Service = DocumentsLayerService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        DocumentsLayerService {
            inner,
            documents: self.documents.clone(),
        }
    }
}

/// A Tower service to add documents to your `CompletionRequestBuilder`.
/// Not intended to be used directly - use [`DocumentsLayer`] instead.
///
/// See [`CompletionRequestBuilderService`] for usage.
pub struct DocumentsLayerService<S> {
    inner: S,
    documents: Vec<Document>,
}

impl<M, Msg, S> Service<(M, Msg)> for DocumentsLayerService<S>
where
    M: CompletionModel + Send + 'static,
    Msg: Into<Message> + Send + 'static,
    S: Service<(M, Msg), Response = CompletionRequestBuilder<M>> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = CompletionRequestBuilder<M>;
    type Error = ();
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut std::task::Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: (M, Msg)) -> Self::Future {
        let mut inner = self.inner.clone();
        let documents = self.documents.clone();
        Box::pin(async move {
            let Ok(res) = inner.call(req).await else {
                todo!("Handle error properly");
            };

            Ok(res.documents(documents))
        })
    }
}

/// A Tower layer to add a temperature value to your `CompletionRequestBuilder`.
/// Intended to be used with [`CompletionRequestBuilderService`].
///
/// See [`CompletionRequestBuilderService`] for usage.
pub struct TemperatureLayer {
    temperature: f64,
}

impl TemperatureLayer {
    pub fn new(temperature: f64) -> Self {
        Self { temperature }
    }
}

impl<S> Layer<S> for TemperatureLayer {
    type Service = TemperatureLayerService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        TemperatureLayerService {
            inner,
            temperature: self.temperature,
        }
    }
}

/// A Tower service to add a temperature value to your `CompletionRequestBuilder`.
/// Not intended to be used directly - use [`TemperatureLayer`] instead.
///
/// See [`CompletionRequestBuilderService`] for usage.
pub struct TemperatureLayerService<S> {
    inner: S,
    temperature: f64,
}

impl<M, Msg, S> Service<(M, Msg)> for TemperatureLayerService<S>
where
    M: CompletionModel + Send + 'static,
    Msg: Into<Message> + Send + 'static,
    S: Service<(M, Msg), Response = CompletionRequestBuilder<M>> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = CompletionRequestBuilder<M>;
    type Error = ();
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut std::task::Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: (M, Msg)) -> Self::Future {
        let mut inner = self.inner.clone();
        let temperature = self.temperature;
        Box::pin(async move {
            let Ok(res) = inner.call(req).await else {
                todo!("Handle error properly");
            };

            Ok(res.temperature(temperature))
        })
    }
}

/// A Tower service to add tools to your `CompletionRequestBuilder`.
/// Intended to be used with [`CompletionRequestBuilderService`].
///
/// See [`CompletionRequestBuilderService`] for usage.
pub struct ToolsLayer {
    tools: Vec<ToolDefinition>,
}

impl ToolsLayer {
    pub fn new(tools: Vec<ToolDefinition>) -> Self {
        Self { tools }
    }
}

impl<S> Layer<S> for ToolsLayer {
    type Service = ToolsLayerService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ToolsLayerService {
            inner,
            tools: self.tools.clone(),
        }
    }
}

/// A Tower service to add tools to your `CompletionRequestBuilder`.
/// Not intended to be used directly - use [`ToolsLayer`] instead.
///
/// See [`CompletionRequestBuilderService`] for usage.
pub struct ToolsLayerService<S> {
    inner: S,
    tools: Vec<ToolDefinition>,
}

impl<M, Msg, S> Service<(M, Msg)> for ToolsLayerService<S>
where
    M: CompletionModel + Send + 'static,
    Msg: Into<Message> + Send + 'static,
    S: Service<(M, Msg), Response = CompletionRequestBuilder<M>> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = CompletionRequestBuilder<M>;
    type Error = ();
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut std::task::Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: (M, Msg)) -> Self::Future {
        let mut inner = self.inner.clone();
        let tools = self.tools.clone();
        Box::pin(async move {
            let Ok(res) = inner.call(req).await else {
                todo!("Handle error properly");
            };

            Ok(res.tools(tools))
        })
    }
}

/// A Tower layer to add a preamble ("system message") to your `CompletionRequestBuilder`.
/// Intended to be used with [`CompletionRequestBuilderService`].
///
/// See [`CompletionRequestBuilderService`] for usage.
pub struct PreambleLayer {
    preamble: String,
}

impl PreambleLayer {
    pub fn new(preamble: String) -> Self {
        Self { preamble }
    }
}

impl<S> Layer<S> for PreambleLayer {
    type Service = PreambleLayerService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        PreambleLayerService {
            inner,
            preamble: self.preamble.clone(),
        }
    }
}

/// A Tower service to add a preamble ("system message") to your `CompletionRequestBuilder`.
/// Not intended to be used directly - use [`PreambleLayer`] instead.
///
/// See [`CompletionRequestBuilderService`] for usage.
pub struct PreambleLayerService<S> {
    inner: S,
    preamble: String,
}

impl<M, Msg, S> Service<(M, Msg)> for PreambleLayerService<S>
where
    M: CompletionModel + Send + 'static,
    Msg: Into<Message> + Send + 'static,
    S: Service<(M, Msg), Response = CompletionRequestBuilder<M>> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = CompletionRequestBuilder<M>;
    type Error = ();
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut std::task::Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: (M, Msg)) -> Self::Future {
        let mut inner = self.inner.clone();
        let preamble = self.preamble.clone();
        Box::pin(async move {
            let Ok(res) = inner.call(req).await else {
                todo!("Handle error properly");
            };

            Ok(res.preamble(preamble))
        })
    }
}

/// A Tower layer to add additional parameters to your `CompletionRequestBuilder` (which are not already covered by the other parameters).
/// Intended to be used with [`CompletionRequestBuilderService`].
///
/// See [`CompletionRequestBuilderService`] for usage.
pub struct AdditionalParamsLayer {
    additional_params: serde_json::Value,
}

impl AdditionalParamsLayer {
    pub fn new(additional_params: serde_json::Value) -> Self {
        Self { additional_params }
    }
}

impl<S> Layer<S> for AdditionalParamsLayer {
    type Service = AdditionalParamsLayerService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        AdditionalParamsLayerService {
            inner,
            additional_params: self.additional_params.clone(),
        }
    }
}

/// A Tower layer to add additional parameters to your `CompletionRequestBuilder` (which are not already covered by the other parameters).
/// Not intended to be used directly - use [`AdditionalParamsLayer`] instead.
///
/// See [`CompletionRequestBuilderService`] for usage.
pub struct AdditionalParamsLayerService<S> {
    inner: S,
    additional_params: serde_json::Value,
}

impl<M, Msg, S> Service<(M, Msg)> for AdditionalParamsLayerService<S>
where
    M: CompletionModel + Send + 'static,
    Msg: Into<Message> + Send + 'static,
    S: Service<(M, Msg), Response = CompletionRequestBuilder<M>> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = CompletionRequestBuilder<M>;
    type Error = ();
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut std::task::Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: (M, Msg)) -> Self::Future {
        let mut inner = self.inner.clone();
        let additional_params = self.additional_params.clone();
        Box::pin(async move {
            let Ok(res) = inner.call(req).await else {
                todo!("Handle error properly");
            };

            Ok(res.additional_params(additional_params))
        })
    }
}

/// A Tower layer to add a maximum tokens parameter to your `CompletionRequestBuilder`.
/// Intended to be used with [`CompletionRequestBuilderService`].
///
/// See [`CompletionRequestBuilderService`] for usage.
pub struct MaxTokensLayer {
    max_tokens: u64,
}

impl MaxTokensLayer {
    pub fn new(max_tokens: u64) -> Self {
        Self { max_tokens }
    }
}

impl<S> Layer<S> for MaxTokensLayer {
    type Service = MaxTokensLayerService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        MaxTokensLayerService {
            inner,
            max_tokens: self.max_tokens,
        }
    }
}

/// A Tower service to add a maximum tokens parameter to your `CompletionRequestBuilder`.
/// Not intended to be used directly - use [`MaxTokensLayer`] instead.
///
/// See [`CompletionRequestBuilderService`] for usage.
pub struct MaxTokensLayerService<S> {
    inner: S,
    max_tokens: u64,
}

impl<M, Msg, S> Service<(M, Msg)> for MaxTokensLayerService<S>
where
    M: CompletionModel + Send + 'static,
    Msg: Into<Message> + Send + 'static,
    S: Service<(M, Msg), Response = CompletionRequestBuilder<M>> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = CompletionRequestBuilder<M>;
    type Error = ();
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut std::task::Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: (M, Msg)) -> Self::Future {
        let mut inner = self.inner.clone();
        let max_tokens = self.max_tokens;
        Box::pin(async move {
            let Ok(res) = inner.call(req).await else {
                todo!("Handle error properly");
            };

            Ok(res.max_tokens(max_tokens))
        })
    }
}

/// A Tower layer to add a chat history to your `CompletionRequestBuilder`.
/// Intended to be used with [`CompletionRequestBuilderService`].
///
/// See [`CompletionRequestBuilderService`] for usage.
pub struct ChatHistoryLayer {
    chat_history: Vec<Message>,
}

impl ChatHistoryLayer {
    pub fn new(chat_history: Vec<Message>) -> Self {
        Self { chat_history }
    }
}

impl<S> Layer<S> for ChatHistoryLayer {
    type Service = ChatHistoryLayerService<S>;

    fn layer(&self, inner: S) -> Self::Service {
        ChatHistoryLayerService {
            inner,
            chat_history: self.chat_history.clone(),
        }
    }
}

/// A Tower service to add a chat history to your `CompletionRequestBuilder`.
/// Not intended to be used directly - use [`ChatHistoryLayer`] instead.
///
/// See [`CompletionRequestBuilderService`] for usage.
pub struct ChatHistoryLayerService<S> {
    inner: S,
    chat_history: Vec<Message>,
}

impl<M, Msg, S> Service<(M, Msg)> for ChatHistoryLayerService<S>
where
    M: CompletionModel + Send + 'static,
    Msg: Into<Message> + Send + 'static,
    S: Service<(M, Msg), Response = CompletionRequestBuilder<M>> + Clone + Send + 'static,
    S::Future: Send + 'static,
{
    type Response = CompletionRequestBuilder<M>;
    type Error = ();
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut std::task::Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: (M, Msg)) -> Self::Future {
        let mut inner = self.inner.clone();
        let chat_history = self.chat_history.clone();
        Box::pin(async move {
            let Ok(res) = inner.call(req).await else {
                todo!("Handle error properly");
            };

            Ok(res.messages(chat_history))
        })
    }
}

/// A `tower` service for building a completion request.
/// Note that the last layer to be applied goes first and the core service is placed last (so it gets executedd from bottom to top).
///
/// Usage:
/// ```rust
/// let agent = rig::providers::openai::Client::from_env();
/// let model = agent.completion_model("gpt-4o");
///
/// let service = tower::ServiceBuilder::new()
///     .layer(FinishBuilding)
///     .layer(TemperatureLayer::new(0.0))
///     .layer(PreambleLayer::new("You are a helpful assistant"))
///     .service(CompletionRequestBuilderService);
///
/// let request = service.call((model, "Hello world!".to_string())).await.unwrap();
///
/// let res = request.send().await.unwrap();
///
/// println!("{res:?}");
/// ```
pub struct CompletionRequestBuilderService;

impl<M, Msg> Service<(M, Msg)> for CompletionRequestBuilderService
where
    M: CompletionModel + Send + 'static,
    Msg: Into<Message> + Send + 'static,
{
    type Response = CompletionRequestBuilder<M>;
    type Error = ();
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut std::task::Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, (model, prompt): (M, Msg)) -> Self::Future {
        Box::pin(async move { Ok(CompletionRequestBuilder::new(model, prompt)) })
    }
}
