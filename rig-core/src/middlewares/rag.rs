use std::{
    future::Future,
    pin::Pin,
    task::{Context, Poll},
};

use serde::{Deserialize, Serialize};
use tower::{Layer, Service};

use crate::{
    completion::{CompletionRequest, CompletionResponse},
    message::{AssistantContent, Text},
};

use super::ServiceError;

pub struct RagLayer<F> {
    _fn: F,
}

impl<F> RagLayer<F> {
    pub fn new(_fn: F) -> Self {
        Self { _fn }
    }
}

pub struct RagLayerService<S, F> {
    inner: S,
    _fn: F,
}

impl<S, F> Layer<S> for RagLayer<F>
where
    F: Clone,
{
    type Service = RagLayerService<S, F>;

    fn layer(&self, inner: S) -> Self::Service {
        RagLayerService {
            inner,
            _fn: self._fn.clone(),
        }
    }
}

impl<S, T, Fut, F, M, E> Service<CompletionRequest> for RagLayerService<S, F>
where
    S: Service<CompletionRequest, Response = CompletionResponse<M>, Error = ServiceError>
        + Clone
        + Send
        + 'static,
    S::Future: Send + 'static,
    M: Send + 'static,
    F: Fn(String) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = Result<Vec<T>, E>> + Send,
    E: std::error::Error,
    T: Send + Serialize + for<'a> Deserialize<'a>,
{
    type Response = Vec<T>;
    type Error = ServiceError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CompletionRequest) -> Self::Future {
        let func = self._fn.clone();
        let mut inner = self.inner.clone();
        Box::pin(async move {
            let res = inner.call(req).await?;

            let AssistantContent::Text(Text { text }) = res.choice.first() else {
                return Err(ServiceError::required_option_not_exists("rag_text"));
            };

            let Ok(res) = (func)(text).await else {
                todo!("Handle error properly");
            };

            Ok(res)
        })
    }
}

pub struct RagService<F> {
    _fn: F,
}

impl<F> RagService<F> {
    pub fn new(_fn: F) -> Self {
        Self { _fn }
    }
}

impl<T, Fut, F, E> Service<CompletionRequest> for RagService<F>
where
    F: Fn(String) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = Result<Vec<T>, E>> + Send,
    T: Send + Serialize + for<'a> Deserialize<'a>,
    E: std::error::Error,
{
    type Response = Vec<T>;
    type Error = ServiceError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(&mut self, _cx: &mut Context<'_>) -> Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CompletionRequest) -> Self::Future {
        let func = self._fn.clone();
        Box::pin(async move {
            let Some(prompt) = req.chat_history.last().rag_text() else {
                return Err(ServiceError::required_option_not_exists("rag_text"));
            };

            let Ok(res) = (func)(prompt).await else {
                todo!("Handle error properly");
            };
            Ok(res)
        })
    }
}

pub trait RagSource: Clone + Send + Sync + 'static {
    type Output: Send + Serialize + for<'a> Deserialize<'a>;
    type Error: std::error::Error;

    fn query(
        &self,
        prompt: String,
    ) -> impl Future<Output = Result<Vec<Self::Output>, Self::Error>> + Send;
}

impl<F, Fut, T, E> RagSource for F
where
    F: Fn(String) -> Fut + Clone + Send + Sync + 'static,
    Fut: Future<Output = Result<Vec<T>, E>> + Send,
    T: Send + Serialize + for<'a> Deserialize<'a>,
    E: std::error::Error,
{
    type Output = T;
    type Error = E;

    fn query(
        &self,
        prompt: String,
    ) -> impl Future<Output = Result<Vec<Self::Output>, Self::Error>> + Send {
        (self)(prompt)
    }
}
