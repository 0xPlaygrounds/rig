use crate::agent::AgentBuilder;
use crate::client::builder::FinalCompletionResponse;
use crate::client::{AsCompletion, ProviderClient};
use crate::completion::{
    CompletionError, CompletionModel, CompletionModelDyn, CompletionRequest, CompletionResponse,
    GetTokenUsage,
};
use crate::extractor::ExtractorBuilder;
use crate::streaming::StreamingCompletionResponse;
use crate::wasm_compat::WasmCompatSend;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::sync::Arc;

/// A provider client with completion capabilities.
/// Clone is required for conversions between client types.
pub trait CompletionClient: ProviderClient + Clone {
    /// The type of CompletionModel used by the client.
    type CompletionModel: CompletionModel;

    /// Create a completion model with the given name.
    ///
    /// # Example with OpenAI
    /// ```
    /// use rig::prelude::*;
    /// use rig::providers::openai::{Client, self};
    ///
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key");
    ///
    /// let gpt4 = openai.completion_model(openai::GPT_4);
    /// ```
    fn completion_model(&self, model: &str) -> Self::CompletionModel;

    /// Create an agent builder with the given completion model.
    ///
    /// # Example with OpenAI
    /// ```
    /// use rig::prelude::*;
    /// use rig::providers::openai::{Client, self};
    ///
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key");
    ///
    /// let agent = openai.agent(openai::GPT_4)
    ///    .preamble("You are comedian AI with a mission to make people laugh.")
    ///    .temperature(0.0)
    ///    .build();
    /// ```
    fn agent(&self, model: &str) -> AgentBuilder<Self::CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    /// Create an extractor builder with the given completion model.
    fn extractor<T>(&self, model: &str) -> ExtractorBuilder<Self::CompletionModel, T>
    where
        T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync,
    {
        ExtractorBuilder::new(self.completion_model(model))
    }
}

/// Wraps a CompletionModel in a dyn-compatible way for AgentBuilder.
#[derive(Clone)]
pub struct CompletionModelHandle<'a> {
    pub inner: Arc<dyn CompletionModelDyn + 'a>,
}

impl CompletionModel for CompletionModelHandle<'_> {
    type Response = ();
    type StreamingResponse = FinalCompletionResponse;

    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionResponse<Self::Response>, CompletionError>> + WasmCompatSend
    {
        self.inner.completion(request)
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Future<
        Output = Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>,
    > + WasmCompatSend {
        self.inner.stream(request)
    }
}

pub trait CompletionClientDyn: ProviderClient {
    /// Create a completion model with the given name.
    fn completion_model<'a>(&self, model: &str) -> Box<dyn CompletionModelDyn + 'a>;

    /// Create an agent builder with the given completion model.
    fn agent<'a>(&self, model: &str) -> AgentBuilder<CompletionModelHandle<'a>>;
}

impl<T, M, R> CompletionClientDyn for T
where
    T: CompletionClient<CompletionModel = M>,
    M: CompletionModel<StreamingResponse = R> + 'static,
    R: Clone + Unpin + GetTokenUsage + 'static,
{
    fn completion_model<'a>(&self, model: &str) -> Box<dyn CompletionModelDyn + 'a> {
        Box::new(self.completion_model(model))
    }

    fn agent<'a>(&self, model: &str) -> AgentBuilder<CompletionModelHandle<'a>> {
        AgentBuilder::new(CompletionModelHandle {
            inner: Arc::new(self.completion_model(model)),
        })
    }
}

impl<T> AsCompletion for T
where
    T: CompletionClientDyn + Clone + 'static,
{
    fn as_completion(&self) -> Option<Box<dyn CompletionClientDyn>> {
        Some(Box::new(self.clone()))
    }
}
