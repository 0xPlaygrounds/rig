use crate::agent::AgentBuilder;
use crate::client::{AsCompletion, ProviderClient};
use crate::completion::{
    CompletionError, CompletionModel, CompletionModelDyn, CompletionRequest, CompletionResponse,
    GetTokenUsage,
};
use crate::extractor::ExtractorBuilder;
use crate::streaming::StreamingCompletionResponse;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};
use std::future::Future;
use std::sync::Arc;

/// A provider client with text completion capabilities.
///
/// This trait extends [`ProviderClient`] to provide text generation functionality.
/// Providers that implement this trait can create completion models and agents.
///
/// # When to Implement
///
/// Implement this trait for provider clients that support:
/// - Text generation (chat completions, prompts)
/// - Multi-turn conversations
/// - Tool/function calling
/// - Streaming responses
///
/// # Examples
///
/// ```no_run
/// use rig::prelude::*;
/// use rig::providers::openai::{Client, self};
/// use rig::completion::Prompt;
///
/// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
/// let client = Client::new("api-key");
///
/// // Create a completion model
/// let model = client.completion_model(openai::GPT_4O);
///
/// // Or create an agent with configuration
/// let agent = client.agent(openai::GPT_4O)
///     .preamble("You are a helpful assistant")
///     .temperature(0.7)
///     .max_tokens(1000)
///     .build();
///
/// let response = agent.prompt("Hello!").await?;
/// # Ok(())
/// # }
/// ```
///
/// # See Also
///
/// - [`crate::completion::CompletionModel`] - The model trait for making completion requests
/// - [`crate::agent::Agent`] - High-level agent abstraction built on completion models
/// - [`CompletionClientDyn`] - Dynamic dispatch version for runtime polymorphism
pub trait CompletionClient: ProviderClient + Clone {
    /// The type of CompletionModel used by the client.
    type CompletionModel: CompletionModel;

    /// Creates a completion model with the specified model identifier.
    ///
    /// This method constructs a completion model that can be used to generate
    /// text completions directly or as part of an agent.
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier (e.g., "gpt-4o", "claude-3-7-sonnet")
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rig::prelude::*;
    /// use rig::providers::openai::{Client, self};
    /// use rig::completion::CompletionModel;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new("your-api-key");
    /// let model = client.completion_model(openai::GPT_4O);
    ///
    /// // Use the model to generate a completion
    /// let response = model
    ///     .completion_request("What is Rust?")
    ///     .send()
    ///     .await?;
    /// # Ok(())
    /// # }
    /// ```
    fn completion_model(&self, model: &str) -> Self::CompletionModel;

    /// Creates an agent builder configured with the specified completion model.
    ///
    /// Agents provide a higher-level abstraction over completion models, adding
    /// features like conversation management, tool integration, and persistent state.
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier (e.g., "gpt-4o", "claude-3-7-sonnet")
    ///
    /// # Returns
    ///
    /// An [`AgentBuilder`] that can be configured with preamble, tools, and other options.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rig::prelude::*;
    /// use rig::providers::openai::{Client, self};
    /// use rig::completion::Prompt;
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new("your-api-key");
    ///
    /// let agent = client.agent(openai::GPT_4O)
    ///     .preamble("You are a helpful assistant")
    ///     .temperature(0.7)
    ///     .build();
    ///
    /// let response = agent.prompt("Hello!").await?;
    /// # Ok(())
    /// # }
    /// ```
    fn agent(&self, model: &str) -> AgentBuilder<Self::CompletionModel> {
        AgentBuilder::new(self.completion_model(model))
    }

    /// Creates an extractor builder for structured data extraction.
    ///
    /// Extractors use the completion model to extract structured data from text,
    /// automatically generating the appropriate schema and parsing responses.
    ///
    /// # Type Parameters
    ///
    /// * `T` - The type to extract, must implement `JsonSchema`, `Deserialize`, and `Serialize`
    ///
    /// # Arguments
    ///
    /// * `model` - The model identifier (e.g., "gpt-4o", "claude-3-7-sonnet")
    ///
    /// # Returns
    ///
    /// An [`ExtractorBuilder`] that can be configured and used to extract structured data.
    ///
    /// # Examples
    ///
    /// ```no_run
    /// use rig::prelude::*;
    /// use rig::providers::openai::{Client, self};
    /// use serde::{Deserialize, Serialize};
    /// use schemars::JsonSchema;
    ///
    /// #[derive(Debug, Deserialize, Serialize, JsonSchema)]
    /// struct Person {
    ///     name: String,
    ///     age: u32,
    /// }
    ///
    /// # async fn example() -> Result<(), Box<dyn std::error::Error>> {
    /// let client = Client::new("api-key");
    ///
    /// let extractor = client.extractor::<Person>(openai::GPT_4O)
    ///     .build();
    ///
    /// let person = extractor.extract("John Doe is 30 years old").await?;
    /// # Ok(())
    /// # }
    /// ```
    fn extractor<T>(&self, model: &str) -> ExtractorBuilder<Self::CompletionModel, T>
    where
        T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync,
    {
        ExtractorBuilder::new(self.completion_model(model))
    }
}

/// A dynamic handle for completion models enabling trait object usage.
///
/// This struct wraps a [`CompletionModel`] in a way that allows it to be used
/// as a trait object with [`AgentBuilder`] and other generic contexts.
/// It uses `Arc` internally for efficient cloning.
///
/// # Examples
///
/// This type is primarily used internally by the dynamic client builder,
/// but can be useful when you need to store completion models of different types:
///
/// ```no_run
/// use rig::client::completion::CompletionModelHandle;
/// use rig::agent::AgentBuilder;
///
/// // CompletionModelHandle allows storing models from different providers
/// fn create_agent(model: CompletionModelHandle) -> AgentBuilder<CompletionModelHandle> {
///     AgentBuilder::new(model)
///         .preamble("You are a helpful assistant")
/// }
/// ```
#[derive(Clone)]
pub struct CompletionModelHandle<'a> {
    /// The inner dynamic completion model.
    pub inner: Arc<dyn CompletionModelDyn + 'a>,
}

impl CompletionModel for CompletionModelHandle<'_> {
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

/// Dynamic dispatch version of [`CompletionClient`].
///
/// This trait provides the same functionality as [`CompletionClient`] but returns
/// trait objects instead of associated types, enabling runtime polymorphism.
/// It is automatically implemented for all types that implement [`CompletionClient`].
///
/// # When to Use
///
/// Use this trait when you need to work with completion clients of different types
/// at runtime, such as in the [`DynClientBuilder`](crate::client::builder::DynClientBuilder).
pub trait CompletionClientDyn: ProviderClient {
    /// Creates a boxed completion model with the specified model identifier.
    ///
    /// Returns a trait object that can be used for dynamic dispatch.
    fn completion_model<'a>(&self, model: &str) -> Box<dyn CompletionModelDyn + 'a>;

    /// Creates an agent builder with a dynamically-dispatched completion model.
    ///
    /// Returns an agent builder using [`CompletionModelHandle`] for dynamic dispatch.
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
