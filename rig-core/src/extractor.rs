//! This module provides high-level abstractions for extracting structured data from text using LLMs.
//!
//! Note: The target structure must implement the `serde::Deserialize`, `serde::Serialize`,
//! and `schemars::JsonSchema` traits. Those can be easily derived using the `derive` macro.
//!
//! # Example
//! ```
//! use rig::providers::openai;
//!
//! // Initialize the OpenAI client
//! let openai = openai::Client::new("your-open-ai-api-key");
//!
//! // Define the structure of the data you want to extract
//! #[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
//! struct Person {
//!    name: Option<String>,
//!    age: Option<u8>,
//!    profession: Option<String>,
//! }
//!
//! // Create the extractor
//! let extractor = openai.extractor::<Person>(openai::GPT_4O)
//!     .build();
//!
//! // Extract structured data from text
//! let person = extractor.extract("John Doe is a 30 year old doctor.")
//!     .await
//!     .expect("Failed to extract data from text");
//! ```
//! # Hooks and Validation
//!
//! For advanced observability and validation, use the hook system:
//!
//! ```rust
//! use rig::extractor::{ExtractorWithHooks, ExtractorHook, ExtractorValidatorHook};
//!
//! // Create hooks and validators
//! let hooks: Vec<Box<dyn ExtractorHook>> = vec![Box::new(LoggingHook)];
//! let validators: Vec<Box<dyn ExtractorValidatorHook<Person>>> = vec![Box::new(AgeValidator)];
//!
//! // Use with any extractor - must import ExtractorWithHooks trait
//! let result = extractor.extract_with_hooks(text, hooks, validators).await?;
//!

use schemars::{JsonSchema, schema_for};
use serde::{Deserialize, Serialize};
use serde_json::{Value, json};
use std::boxed::Box;
use std::future::Future;
use std::marker::PhantomData;
use std::pin::Pin;

use crate::{
    agent::{Agent, AgentBuilder, AgentBuilderSimple},
    completion::{Completion, CompletionError, CompletionModel, ToolDefinition},
    message::{AssistantContent, Message, ToolCall, ToolChoice, ToolFunction},
    tool::Tool,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

const SUBMIT_TOOL_NAME: &str = "submit";

#[derive(Debug, thiserror::Error)]
pub enum ExtractionError {
    #[error("No data extracted")]
    NoData,

    #[error("Failed to deserialize the extracted data: {0}")]
    DeserializationError(#[from] serde_json::Error),

    #[error("CompletionError: {0}")]
    CompletionError(#[from] CompletionError),

    #[error("ValidationError: {0}")]
    ValidationError(String),
}
/// A trait for observing and reacting to extraction lifecycle events.
///
/// `ExtractorHook` provides a way to monitor the extraction process at key points,
/// enabling observability, logging, metrics collection, and debugging capabilities.
/// Hooks are called during extraction attempts and can be used to track progress,
/// measure performance, or implement custom monitoring logic.
/// To use hooks with extractors, import [`ExtractorWithHooks`] and call
/// extract_with_hooks
///
/// # Lifecycle Events
///
/// The hook methods are called in the following order during each extraction attempt:
/// 1. [`before_extract`] - Called before starting an extraction attempt
/// 2. [`after_parse`] - Called after successfully parsing JSON from the model response
/// 3. [`on_success`] OR [`on_error`] - Called when the attempt succeeds or fails
///
/// # Thread Safety
///
/// Implementations must be `Send + Sync + 'static` to be used across async boundaries
/// and stored in collections.
///
/// # Example
///
/// ```rust
/// use std::sync::{Arc, Mutex};
/// use std::pin::Pin;
/// use std::boxed::Box;
/// use std::future::Future;
/// use rig::extractor::ExtractorWithHooks;
///
/// #[derive(Clone)]
/// struct LoggingHook {
///     events: Arc<Mutex<Vec<String>>>,
/// }
///
/// impl ExtractorHook for LoggingHook {
///     fn before_extract(&self, attempt: u64, text: &Message) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
///         let events = Arc::clone(&self.events);
///         Box::pin(async move {
///             events.lock().unwrap().push(format!("Starting attempt {}", attempt));
///         })
///     }
///
///     fn after_parse(&self, attempt: u64, data: &Value) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
///         let events = Arc::clone(&self.events);
///         Box::pin(async move {
///             events.lock().unwrap().push(format!("Parsed data on attempt {}", attempt));
///         })
///     }
///
///     fn on_error(&self, attempt: u64, error: &ExtractionError) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
///         let events = Arc::clone(&self.events);
///         let error_msg = error.to_string();
///         Box::pin(async move {
///             events.lock().unwrap().push(format!("Error on attempt {}: {}", attempt, error_msg));
///         })
///     }
///
///     fn on_success(&self, attempt: u64, data: &Value) -> Pin<Box<dyn Future<Output = ()> + Send + '_>> {
///         let events = Arc::clone(&self.events);
///         Box::pin(async move {
///             events.lock().unwrap().push(format!("Success on attempt {}", attempt));
///         })
///     }
/// }
///
/// let hooks: Vec<Box<dyn ExtractorHook>> = vec![
///     Box::new(LoggingHook { events: Arc::new(Mutex::new(Vec::new())) }),
/// ];
///
/// let validators: Vec<Box<dyn ExtractorValidatorHook<Person>>> = vec![
///     Box::new(AgeValidator { min_age: 18, max_age: 120 }),
/// ];
///
/// let result = extractor.extract_with_hooks(text, hooks, validators).await?;
/// ```
///
/// [`before_extract`]: ExtractorHook::before_extract
/// [`after_parse`]: ExtractorHook::after_parse
/// [`on_success`]: ExtractorHook::on_success
/// [`on_error`]: ExtractorHook::on_error
pub trait ExtractorHook: Send + Sync + 'static {
    /// Called before each extraction attempt begins.
    ///
    /// This method is invoked at the start of each extraction attempt, before any
    /// communication with the language model. It provides an opportunity to perform
    /// setup operations, start timers, increment counters, or log the beginning
    /// of an extraction attempt.
    fn before_extract(
        &self,
        attempt: u64,
        text: &Message,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;

    /// Called after successfully parsing JSON from the model response.
    ///
    /// This method is invoked when the language model has returned a response
    /// and the JSON has been successfully parsed, but before any validation occurs.
    /// It's useful for inspecting the raw extracted data or logging successful
    /// parsing events.
    fn after_parse(
        &self,
        attempt: u64,
        data: &Value,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;

    /// Called when an extraction attempt fails.
    ///
    /// This method is invoked whenever an extraction attempt fails, whether due to
    /// model errors, parsing failures, validation errors, or other issues. It provides
    /// an opportunity to log errors, update failure metrics, or perform cleanup.
    fn on_error(
        &self,
        attempt: u64,
        error: &ExtractionError,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;

    /// Called when extraction succeeds completely.
    ///
    /// This method is invoked when an extraction attempt succeeds, meaning the model
    /// response was parsed successfully and all validation passed. It's called after
    /// all processing is complete and represents the final success of the extraction.
    fn on_success(
        &self,
        attempt: u64,
        data: &Value,
    ) -> Pin<Box<dyn Future<Output = ()> + Send + '_>>;
}

/// A trait for implementing custom validation logic on extracted data.
///
/// `ExtractorValidatorHook` allows you to define custom validation rules that are
/// applied to extracted data after JSON parsing but before the extraction is considered
/// successful. When validation fails, the extraction attempt is retried (if retries are
/// configured), giving the language model an opportunity to self-correct based on
/// the validation error feedback.
///
/// Validators are type-specific and work with the concrete extracted data structure,
/// enabling precise business rule validation, data quality checks, and domain-specific
/// constraints that go beyond basic JSON schema validation.
///
/// # Validation Flow
///
/// 1. Model extracts data and JSON is parsed successfully
/// 2. Each validator's [`validate`] method is called in sequence
/// 3. If any validator returns an error, the extraction attempt fails and may retry
/// 4. If all validators pass, the extraction succeeds
///
/// # Error Handling
///
/// Validation errors are converted to [`ExtractionError::ValidationError`] and fed back
/// into the retry loop, allowing the model to attempt self-correction on subsequent tries.
///
/// # Example
/// To use validators with extractors, import [`ExtractorWithHooks`] and call
/// extract_with_hooks
/// ```rust
/// use std::pin::Pin;
/// use std::boxed::Box;
/// use std::future::Future;
/// use rig::extractor::ExtractorWithHooks;
///
/// #[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
/// struct Person {
///     name: String,
///     age: u8,
///     email: Option<String>,
/// }
///
/// #[derive(Clone)]
/// struct AgeValidator {
///     min_age: u8,
///     max_age: u8,
/// }
///
/// impl ExtractorValidatorHook<Person> for AgeValidator {
///     fn validate(&self, person: &Person) -> Pin<Box<dyn Future<Output = Result<(), ExtractionError>> + Send + '_>> {
///         let min_age = self.min_age;
///         let max_age = self.max_age;
///         let age = person.age;
///         Box::pin(async move {
///             if age < min_age {
///                 return Err(ExtractionError::ValidationError(
///                     format!("Age {} is below minimum of {}", age, min_age)
///                 ));
///             }
///             if age > max_age {
///                 return Err(ExtractionError::ValidationError(
///                     format!("Age {} exceeds maximum of {}", age, max_age)
///                 ));
///             }
///             Ok(())
///         })
///     }
/// }
/// ```
/// You can chain multiple validators together. They are executed in order, and the first
/// validation failure will cause the extraction attempt to fail:
///
/// ```rust
/// let validators: Vec<Box<dyn ExtractorValidatorHook<Person>>> = vec![
///     Box::new(AgeValidator { min_age: 18, max_age: 120 }),
///     Box::new(EmailValidator),
///     Box::new(BusinessRuleValidator),
/// ];
///
/// let result = extractor.extract_with_hooks(text, vec![], validators).await?;
/// ```
///
/// [`validate`]: ExtractorValidatorHook::validate
pub trait ExtractorValidatorHook<T>: Send + Sync
where
    T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync,
{
    /// Validates the extracted data according to custom business rules.
    fn validate(
        &self,
        data: &T,
    ) -> Pin<Box<dyn Future<Output = Result<(), ExtractionError>> + Send + '_>>;
}

/// Extension trait that adds hook and validation capabilities to extractors.
///
/// `ExtractorWithHooks` extends any extractor with advanced observability and validation
/// features through the [`extract_with_hooks`] method.
///
/// # Usage
///
/// To use this functionality, you must import the trait:
///
/// ```rust
/// use rig::extractor::ExtractorWithHooks;
///
/// let hooks: Vec<Box<dyn ExtractorHook>> = vec![
///     Box::new(LoggingHook::new()),
/// ];
///
/// let validators: Vec<Box<dyn ExtractorValidatorHook<Person>>> = vec![
///     Box::new(AgeValidator { min_age: 18, max_age: 120 }),
/// ];
///
/// let result = extractor.extract_with_hooks(text, hooks, validators).await?;
/// ```
///
/// [`extract_with_hooks`]: ExtractorWithHooks::extract_with_hooks
pub trait ExtractorWithHooks<T>
where
    T: JsonSchema + for<'a> Deserialize<'a> + Send + Sync,
{
    /// Extracts structured data with observability hooks and custom validation.
    ///
    /// This method extends the basic extraction functionality with lifecycle hooks
    /// for monitoring and custom validators for data quality assurance. It provides
    /// the same extraction capabilities as [`Extractor::extract`] but with additional
    /// observability and validation features.
    fn extract_with_hooks(
        &self,
        text: impl Into<Message> + Send,
        hooks: Vec<Box<dyn ExtractorHook>>,
        validators: Vec<Box<dyn ExtractorValidatorHook<T>>>,
    ) -> Pin<Box<dyn Future<Output = Result<T, ExtractionError>> + Send + '_>>;
}

/// Extractor for structured data from text
pub struct Extractor<M, T>
where
    M: CompletionModel,
    T: JsonSchema + for<'a> Deserialize<'a> + WasmCompatSend + WasmCompatSync,
{
    agent: Agent<M>,
    _t: PhantomData<T>,
    retries: u64,
}

impl<M, T> ExtractorWithHooks<T> for Extractor<M, T>
where
    M: CompletionModel,
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + Send + Sync,
{
    /// This implementation creates the complete extraction lifecycle with
    /// observability and validation. It manages the retry loop, coordinates
    /// hook calls, and ensures validators are executed in the correct sequence.
    fn extract_with_hooks(
        &self,
        text: impl Into<Message> + Send,
        hooks: Vec<Box<dyn ExtractorHook>>,
        validators: Vec<Box<dyn ExtractorValidatorHook<T>>>,
    ) -> Pin<Box<dyn Future<Output = Result<T, ExtractionError>> + Send + '_>> {
        let text_msg = text.into();

        Box::pin(async move {
            let mut last_error = None;

            for i in 0..=self.retries {
                tracing::debug!(
                    "Attempting to extract Json. Retries left:{retries}",
                    retries = self.retries - i
                );

                for hook in &hooks {
                    hook.before_extract(i, &text_msg).await;
                }
                let attempt_t = text_msg.clone();
                match self
                    .extract_validated_json(attempt_t, i, &hooks, &validators)
                    .await
                {
                    Ok(data) => {
                        let data_value =
                            serde_json::to_value(&data).unwrap_or(serde_json::Value::Null);
                        for hook in &hooks {
                            hook.on_success(i, &data_value).await;
                        }
                        return Ok(data);
                    }
                    Err(e) => {
                        tracing::warn!(
                            "Attempt number {i} to extract Json failed: {e:?}. Retrying..."
                        );
                        for hook in &hooks {
                            hook.on_error(i, &e).await;
                        }
                        last_error = Some(e);
                    }
                }
            }
            Err(last_error.unwrap_or(ExtractionError::NoData))
        })
    }
}

impl<M, T> Extractor<M, T>
where
    M: CompletionModel,
    T: JsonSchema + for<'a> Deserialize<'a> + WasmCompatSend + WasmCompatSync,
{
    /// Attempts to extract data from the given text with a number of retries.
    ///
    /// The function will retry the extraction if the initial attempt fails or
    /// if the model does not call the `submit` tool.
    ///
    /// The number of retries is determined by the `retries` field on the Extractor struct.
    pub async fn extract(
        &self,
        text: impl Into<Message> + WasmCompatSend,
    ) -> Result<T, ExtractionError> {
        let mut last_error = None;
        let text_message = text.into();

        for i in 0..=self.retries {
            tracing::debug!(
                "Attempting to extract JSON. Retries left: {retries}",
                retries = self.retries - i
            );
            let attempt_text = text_message.clone();
            match self.extract_json(attempt_text, vec![]).await {
                Ok(data) => return Ok(data),
                Err(e) => {
                    tracing::warn!("Attempt {i} to extract JSON failed: {e:?}. Retrying...");
                    last_error = Some(e);
                }
            }
        }

        // If the loop finishes without a successful extraction, return the last error encountered.
        Err(last_error.unwrap_or(ExtractionError::NoData))
    }

    /// Attempts to extract data from the given text with a number of retries.
    ///
    /// The function will retry the extraction if the initial attempt fails or
    /// if the model does not call the `submit` tool.
    ///
    /// The number of retries is determined by the `retries` field on the Extractor struct.
    pub async fn extract_with_chat_history(
        &self,
        text: impl Into<Message> + WasmCompatSend,
        chat_history: Vec<Message>,
    ) -> Result<T, ExtractionError> {
        let mut last_error = None;
        let text_message = text.into();

        for i in 0..=self.retries {
            tracing::debug!(
                "Attempting to extract JSON. Retries left: {retries}",
                retries = self.retries - i
            );
            let attempt_text = text_message.clone();
            match self.extract_json(attempt_text, chat_history.clone()).await {
                Ok(data) => return Ok(data),
                Err(e) => {
                    tracing::warn!("Attempt {i} to extract JSON failed: {e:?}. Retrying...");
                    last_error = Some(e);
                }
            }
        }

        // If the loop finishes without a successful extraction, return the last error encountered.
        Err(last_error.unwrap_or(ExtractionError::NoData))
    }

    async fn extract_json(
        &self,
        text: impl Into<Message> + WasmCompatSend,
        messages: Vec<Message>,
    ) -> Result<T, ExtractionError> {
        let response = self.agent.completion(text, messages).await?.send().await?;

        if !response.choice.iter().any(|x| {
            let AssistantContent::ToolCall(ToolCall {
                function: ToolFunction { name, .. },
                ..
            }) = x
            else {
                return false;
            };

            name == SUBMIT_TOOL_NAME
        }) {
            tracing::warn!(
                "The submit tool was not called. If this happens more than once, please ensure the model you are using is powerful enough to reliably call tools."
            );
        }

        let arguments = response
            .choice
            .into_iter()
            // We filter tool calls to look for submit tool calls
            .filter_map(|content| {
                if let AssistantContent::ToolCall(ToolCall {
                    function: ToolFunction { arguments, name },
                    ..
                }) = content
                {
                    if name == SUBMIT_TOOL_NAME {
                        Some(arguments)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();

        if arguments.len() > 1 {
            tracing::warn!(
                "Multiple submit calls detected, using the last one. Providers / agents should only ensure one submit call."
            );
        }

        let raw_data = if let Some(arg) = arguments.into_iter().next() {
            arg
        } else {
            return Err(ExtractionError::NoData);
        };

        Ok(serde_json::from_value(raw_data)?)
    }

    async fn extract_validated_json(
        &self,
        text: impl Into<Message> + Send,
        attempt: u64,
        hooks: &[Box<dyn ExtractorHook>],
        validators: &[Box<dyn ExtractorValidatorHook<T>>],
    ) -> Result<T, ExtractionError> {
        let response = self.agent.completion(text, vec![]).await?.send().await?;

        let args = response
            .choice
            .into_iter()
            .filter_map(|content| {
                if let AssistantContent::ToolCall(ToolCall {
                    function: ToolFunction { arguments, name },
                    ..
                }) = content
                {
                    if name == SUBMIT_TOOL_NAME {
                        Some(arguments)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect::<Vec<_>>();
        let raw_data = args.into_iter().next().ok_or(ExtractionError::NoData)?;

        for hook in hooks {
            hook.after_parse(attempt, &raw_data).await;
        }

        let parsed_data: T = serde_json::from_value(raw_data.clone())?;

        for validator in validators {
            if let Err(val_error) = validator.validate(&parsed_data).await {
                return Err(ExtractionError::ValidationError(val_error.to_string()));
            }
        }
        Ok(parsed_data)
    }
    pub async fn get_inner(&self) -> &Agent<M> {
        &self.agent
    }

    pub async fn into_inner(self) -> Agent<M> {
        self.agent
    }
}

/// Builder for the Extractor
pub struct ExtractorBuilder<M, T>
where
    M: CompletionModel,
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + WasmCompatSend + WasmCompatSync + 'static,
{
    agent_builder: AgentBuilderSimple<M>,
    _t: PhantomData<T>,
    retries: Option<u64>,
}

impl<M, T> ExtractorBuilder<M, T>
where
    M: CompletionModel,
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + WasmCompatSend + WasmCompatSync + 'static,
{
    pub fn new(model: M) -> Self {
        Self {
            agent_builder: AgentBuilder::new(model)
                .preamble("\
                    You are an AI assistant whose purpose is to extract structured data from the provided text.\n\
                    You will have access to a `submit` function that defines the structure of the data to extract from the provided text.\n\
                    Use the `submit` function to submit the structured data.\n\
                    Be sure to fill out every field and ALWAYS CALL THE `submit` function, even with default values!!!.
                ")
                .tool(SubmitTool::<T> {_t: PhantomData})
                .tool_choice(ToolChoice::Required),
            retries: None,
            _t: PhantomData,
        }
    }

    /// Add additional preamble to the extractor
    pub fn preamble(mut self, preamble: &str) -> Self {
        self.agent_builder = self.agent_builder.append_preamble(&format!(
            "\n=============== ADDITIONAL INSTRUCTIONS ===============\n{preamble}"
        ));
        self
    }

    /// Add a context document to the extractor
    pub fn context(mut self, doc: &str) -> Self {
        self.agent_builder = self.agent_builder.context(doc);
        self
    }

    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.agent_builder = self.agent_builder.additional_params(params);
        self
    }

    /// Set the maximum number of tokens for the completion
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.agent_builder = self.agent_builder.max_tokens(max_tokens);
        self
    }

    /// Set the maximum number of retries for the extractor.
    pub fn retries(mut self, retries: u64) -> Self {
        self.retries = Some(retries);
        self
    }

    /// Set the `tool_choice` option for the inner Agent.
    pub fn tool_choice(mut self, choice: ToolChoice) -> Self {
        self.agent_builder = self.agent_builder.tool_choice(choice);
        self
    }

    /// Build the Extractor
    pub fn build(self) -> Extractor<M, T> {
        Extractor {
            agent: self.agent_builder.build(),
            _t: PhantomData,
            retries: self.retries.unwrap_or(0),
        }
    }
}

#[derive(Deserialize, Serialize)]
struct SubmitTool<T>
where
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + WasmCompatSend + WasmCompatSync,
{
    _t: PhantomData<T>,
}

#[derive(Debug, thiserror::Error)]
#[error("SubmitError")]
struct SubmitError;

impl<T> Tool for SubmitTool<T>
where
    T: JsonSchema + for<'a> Deserialize<'a> + Serialize + WasmCompatSend + WasmCompatSync,
{
    const NAME: &'static str = SUBMIT_TOOL_NAME;
    type Error = SubmitError;
    type Args = T;
    type Output = T;

    async fn definition(&self, _prompt: String) -> ToolDefinition {
        ToolDefinition {
            name: Self::NAME.to_string(),
            description: "Submit the structured data you extracted from the provided text."
                .to_string(),
            parameters: json!(schema_for!(T)),
        }
    }

    async fn call(&self, data: Self::Args) -> Result<Self::Output, Self::Error> {
        Ok(data)
    }
}
