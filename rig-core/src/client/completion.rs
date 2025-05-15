use crate::agent::AgentBuilder;
use crate::client::{AsCompletion, ProviderClient};
use crate::completion::{
    CompletionError, CompletionModel, CompletionModelDyn, CompletionRequest, CompletionResponse,
};
use crate::extractor::ExtractorBuilder;
use crate::streaming::StreamingCompletionResponse;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::sync::Arc;

pub trait CompletionClient: ProviderClient {
    type CompletionModel: CompletionModel;

    fn completion_model(&self, model: &str) -> Self::CompletionModel;

    fn agent(&self, model: &str) -> AgentBuilder<Self::CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    /// Create an extractor builder with the given completion model.
    fn extractor<T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync>(
        &self,
        model: &str,
    ) -> ExtractorBuilder<T, Self::CompletionModel> {
        ExtractorBuilder::new(self.completion_model(model))
    }
}

#[derive(Clone)]
pub struct CompletionModelHandle<'a> {
    pub inner: Arc<dyn CompletionModelDyn + 'a>,
}

impl<'a> CompletionModel for CompletionModelHandle<'a> {
    type Response = ();
    type StreamingResponse = ();

    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionResponse<Self::Response>, CompletionError>> + Send
    {
        self.inner.completion(request)
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Future<
        Output = Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>,
    > + Send {
        self.inner.stream(request)
    }
}

pub trait CompletionClientDyn: ProviderClient {
    fn completion_model<'a>(&'a self, model: &'a str) -> Box<dyn CompletionModelDyn + 'a>;

    fn agent<'a>(&'a self, model: &'a str) -> AgentBuilder<CompletionModelHandle<'a>>;
}

impl<
        T: CompletionClient<CompletionModel = M>,
        M: CompletionModel<StreamingResponse = R> + 'static,
        R: Clone + Unpin + 'static,
    > CompletionClientDyn for T
{
    fn completion_model<'a>(&'a self, model: &'a str) -> Box<dyn CompletionModelDyn + 'a> {
        Box::new(self.completion_model(model))
    }

    fn agent<'a>(&'a self, model: &'a str) -> AgentBuilder<CompletionModelHandle<'a>> {
        AgentBuilder::new(CompletionModelHandle {
            inner: Arc::new(self.completion_model(model)),
        })
    }
}

impl<T: CompletionClientDyn> AsCompletion for T {
    fn as_completion(&self) -> Option<Box<&dyn CompletionClientDyn>> {
        Some(Box::new(self))
    }
}
