use serde::Deserialize;
use std::{future::Future, marker::PhantomData, pin::Pin, task::Poll};
use tower::{Layer, Service};

use crate::{
    completion::{CompletionRequest, CompletionResponse},
    message::{AssistantContent, Text},
};

use super::ServiceError;

pub struct ExtractorLayer<T> {
    _t: PhantomData<T>,
}

impl<T> ExtractorLayer<T>
where
    T: for<'a> Deserialize<'a>,
{
    pub fn new() -> Self {
        Self { _t: PhantomData }
    }
}

impl<T> Default for ExtractorLayer<T>
where
    T: for<'a> Deserialize<'a>,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<S, T> Layer<S> for ExtractorLayer<T>
where
    T: for<'a> Deserialize<'a>,
{
    type Service = ExtractorLayerService<S, T>;
    fn layer(&self, inner: S) -> Self::Service {
        ExtractorLayerService { inner, _t: self._t }
    }
}

pub struct ExtractorLayerService<S, T> {
    inner: S,
    _t: PhantomData<T>,
}

impl<S, F, T> Service<CompletionRequest> for ExtractorLayerService<S, T>
where
    S: Service<CompletionRequest, Response = CompletionResponse<F>, Error = ServiceError>
        + Clone
        + Send
        + 'static,
    S::Future: Send,
    F: 'static,
    T: for<'a> Deserialize<'a> + 'static,
{
    type Response = T;
    type Error = ServiceError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CompletionRequest) -> Self::Future {
        let mut inner = self.inner.clone();

        Box::pin(async move {
            let res = inner.call(req).await?;

            let AssistantContent::Text(Text { text }) = res.choice.first() else {
                todo!("Handle errors properly");
            };

            let obj = serde_json::from_str::<T>(&text)?;

            Ok(obj)
        })
    }
}
