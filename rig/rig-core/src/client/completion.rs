use crate::agent::AgentBuilder;
use crate::client::FinalCompletionResponse;

#[allow(deprecated)]
use crate::completion::CompletionModelDyn;
use crate::completion::{
    CompletionError, CompletionModel, CompletionRequest, CompletionResponse, GetTokenUsage,
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
pub trait CompletionClient {
    /// The type of CompletionModel used by the client.
    type CompletionModel: CompletionModel<Client = Self>;

    /// Create a completion model with the given model.
    ///
    /// # Example with OpenAI
    /// ```
    /// use rig::prelude::*;
    /// use rig::providers::openai::{Client, self};
    ///
    /// // Initialize the OpenAI client
    /// let openai = Client::new("your-open-ai-api-key");
    ///
    /// let gpt4 = openai.completion_model(openai::GPT4);
    /// ```
    fn completion_model(&self, model: impl Into<String>) -> Self::CompletionModel {
        Self::CompletionModel::make(self, model)
    }

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
    fn agent(&self, model: impl Into<String>) -> AgentBuilder<Self::CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    /// Create an extractor builder with the given completion model.
    fn extractor<T>(&self, model: impl Into<String>) -> ExtractorBuilder<Self::CompletionModel, T>
    where
        T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync,
    {
        ExtractorBuilder::new(self.completion_model(model))
    }
}

#[allow(deprecated)]
#[deprecated(
    since = "0.25.0",
    note = "`DynClientBuilder` and related features have been deprecated and will be removed in a future release."
)]
/// Wraps a CompletionModel in a dyn-compatible way for AgentBuilder.
#[derive(Clone)]
pub struct CompletionModelHandle<'a>(Arc<dyn CompletionModelDyn + 'a>);

#[allow(deprecated)]
impl<'a> CompletionModelHandle<'a> {
    pub fn new(handle: Arc<dyn CompletionModelDyn + 'a>) -> Self {
        Self(handle)
    }
}

#[allow(deprecated)]
impl CompletionModel for CompletionModelHandle<'_> {
    type Response = ();
    type StreamingResponse = FinalCompletionResponse;
    type Client = ();

    /// **PANICS**: We are deprecating DynClientBuilder and related functionality, in the meantime
    /// there may be some invalid methods which panic when called, such as this one
    fn make(_: &Self::Client, _: impl Into<String>) -> Self {
        panic!("Cannot create a completion model handle from a client")
    }

    fn completion(
        &self,
        request: CompletionRequest,
    ) -> impl Future<Output = Result<CompletionResponse<Self::Response>, CompletionError>> + WasmCompatSend
    {
        self.0.completion(request)
    }

    fn stream(
        &self,
        request: CompletionRequest,
    ) -> impl Future<
        Output = Result<StreamingCompletionResponse<Self::StreamingResponse>, CompletionError>,
    > + WasmCompatSend {
        self.0.stream(request)
    }
}

#[allow(deprecated)]
#[deprecated(
    since = "0.25.0",
    note = "`DynClientBuilder` and related features have been deprecated and will be removed in a future release. In this case, use `CompletionClient` instead."
)]
pub trait CompletionClientDyn {
    /// Create a completion model with the given name.
    fn completion_model<'a>(&self, model: &str) -> Box<dyn CompletionModelDyn + 'a>;

    /// Create an agent builder with the given completion model.
    fn agent<'a>(&self, model: &str) -> AgentBuilder<CompletionModelHandle<'a>>;
}

#[allow(deprecated)]
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
        AgentBuilder::new(CompletionModelHandle(Arc::new(
            self.completion_model(model),
        )))
    }
}
