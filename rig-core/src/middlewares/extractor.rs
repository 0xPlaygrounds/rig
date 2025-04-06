use std::{future::Future, pin::Pin, sync::Arc, task::Poll};

use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use tower::{Layer, Service};

use crate::{
    completion::{CompletionModel, CompletionRequest, CompletionResponse},
    extractor::{ExtractionError, Extractor},
    message::{AssistantContent, Text},
};

pub struct ExtractorLayer<M, T>
where
    M: CompletionModel,
    T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync,
{
    ext: Arc<Extractor<M, T>>,
}

impl<M, T> ExtractorLayer<M, T>
where
    M: CompletionModel,
    T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync,
{
    pub fn new(ext: Extractor<M, T>) -> Self {
        Self { ext: Arc::new(ext) }
    }
}

impl<S, M, T> Layer<S> for ExtractorLayer<M, T>
where
    M: CompletionModel + 'static,
    T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync + 'static,
{
    type Service = ExtractorLayerService<S, M, T>;
    fn layer(&self, inner: S) -> Self::Service {
        ExtractorLayerService {
            inner,
            ext: Arc::clone(&self.ext),
        }
    }
}

pub struct ExtractorLayerService<S, M, T>
where
    M: CompletionModel + 'static,
    T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync + 'static,
{
    inner: S,
    ext: Arc<Extractor<M, T>>,
}

impl<S, M, T> Service<CompletionRequest> for ExtractorLayerService<S, M, T>
where
    S: Service<CompletionRequest, Response = CompletionResponse<M::Response>>
        + Clone
        + Send
        + 'static,
    S::Future: Send,
    M: CompletionModel + 'static,
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync,
{
    type Response = T;
    type Error = ExtractionError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CompletionRequest) -> Self::Future {
        let ext = self.ext.clone();
        let mut inner = self.inner.clone();

        Box::pin(async move {
            let Ok(res) = inner.call(req).await else {
                todo!("Properly handle error");
            };

            let AssistantContent::Text(Text { text }) = res.choice.first() else {
                todo!("Handle errors properly");
            };

            ext.extract(&text).await
        })
    }
}

pub struct ExtractorService<M, T>
where
    M: CompletionModel + 'static,
    T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync + 'static,
{
    ext: Arc<Extractor<M, T>>,
}

impl<M, T> ExtractorService<M, T>
where
    M: CompletionModel + 'static,
    T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync + 'static,
{
    pub fn new(ext: Extractor<M, T>) -> Self {
        Self { ext: Arc::new(ext) }
    }
}

impl<M, T> Service<CompletionRequest> for ExtractorService<M, T>
where
    M: CompletionModel,
    T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync,
{
    type Response = T;
    type Error = ExtractionError;
    type Future = Pin<Box<dyn Future<Output = Result<Self::Response, Self::Error>> + Send>>;

    fn poll_ready(
        &mut self,
        _cx: &mut std::task::Context<'_>,
    ) -> std::task::Poll<Result<(), Self::Error>> {
        Poll::Ready(Ok(()))
    }

    fn call(&mut self, req: CompletionRequest) -> Self::Future {
        let ext = self.ext.clone();
        let Some(req) = req.prompt.rag_text() else {
            todo!("Handle error properly");
        };

        Box::pin(async move { ext.extract(&req).await })
    }
}
