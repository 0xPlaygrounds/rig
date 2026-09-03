//! Typed extraction: an [`Agent`] configured to answer through a `submit`
//! tool whose arguments are the value to extract, run as a
//! [`TypedRun`].
//!
//! The target type must implement `serde::Deserialize`, `serde::Serialize`,
//! and `schemars::JsonSchema`; all three derive.
//!
//! # Example
//! ```no_run
//! use rig_agent::prelude::*;
//! use rig_core::providers::openai;
//! use rig_reqwest::prelude::*;
//!
//! # async fn run() -> Result<(), Box<dyn std::error::Error>> {
//! let openai = openai::Client::new("your-open-ai-api-key")?;
//!
//! #[derive(serde::Deserialize, serde::Serialize, schemars::JsonSchema)]
//! struct Person {
//!    name: Option<String>,
//!    age: Option<u8>,
//!    profession: Option<String>,
//! }
//!
//! let extractor = openai.extractor::<Person>(openai::GPT_4O).retries(2).build();
//!
//! // `.await` gives a `TypedPromptResponse<Person>`; `.output` is the value.
//! let person = extractor.extract("John Doe is a 30 year old doctor.").await?.output;
//! # Ok(())
//! # }
//! ```

use std::marker::PhantomData;

use schemars::JsonSchema;
use serde::{Serialize, de::DeserializeOwned};

use rig_core::{
    message::{Message, ToolChoice},
    vector_store::{VectorStoreIndex, request::DynamicSearchFilter},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

use crate::{
    agent::{Agent, AgentBuilder, AgentHook, ModelRef, OutputMode, TypedRun},
    completion::CompletionModel,
};

const SUBMIT_TOOL_NAME: &str = "submit";

/// An agent configured for structured extraction: every [`extract`](Self::extract)
/// is a one-call [`TypedRun`] in output-tool mode with this extractor's retry
/// budget.
pub struct Extractor<T>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend + WasmCompatSync,
{
    agent: Agent,
    retries: u64,
    _t: PhantomData<T>,
}

impl<T> Extractor<T>
where
    T: JsonSchema + DeserializeOwned + WasmCompatSend + WasmCompatSync,
{
    /// Set a different default model for this extractor's subsequent runs.
    /// Use the model registered under `label` on the extractor's bus.
    pub fn with_model_ref(mut self, label: impl Into<ModelRef>) -> Self {
        self.agent.set_model_ref(label);
        self
    }

    /// Register `model` on the extractor's bus and use it.
    pub fn with_model<M>(mut self, model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        self.agent.set_model(model);
        self
    }

    /// The agent behind this extractor.
    pub fn agent(&self) -> &Agent {
        &self.agent
    }

    /// Extract structured data from `text`.
    ///
    /// The returned run can be configured further — `.history(..)` for chat
    /// context, `.using_model(..)` for a per-run default model, `.retries(..)`
    /// to override the extractor's budget — and `.await`ed for a
    /// [`TypedPromptResponse<T>`](crate::agent::TypedPromptResponse). The
    /// model must call the `submit` tool; a run in which it does not is an
    /// empty response and, within the retry budget, is retried from scratch.
    /// Usage accumulates across attempts, including attempts that received a
    /// billed response but failed extraction; attempts whose completion call
    /// itself errored contribute no usage.
    pub fn extract(&self, text: impl Into<Message>) -> TypedRun<T> {
        let runner = self.agent.runner(text).max_turns(1).output_tool(
            SUBMIT_TOOL_NAME,
            "Submit the structured data you extracted from the provided text.",
            false,
        );
        TypedRun::output_tool(runner).retries(self.retries)
    }
}

/// Builder for the Extractor
pub struct ExtractorBuilder<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + WasmCompatSend + WasmCompatSync + 'static,
{
    agent_builder: AgentBuilder,
    _t: PhantomData<T>,
    retries: Option<u64>,
}

/// Generate the `ExtractorBuilder` setters that forward verbatim to the inner
/// [`AgentBuilder`] method of the same name. Doc comments live at each
/// invocation; `preamble` (which wraps its argument) and `retries`
/// (builder-local) stay hand-written.
macro_rules! forward_agent_builder {
    ($( $(#[$attr:meta])* $name:ident $([$gen:ident : $($bound:tt)+])?
        ( $($arg:ident : $ty:ty),* );)+) => {$(
        $(#[$attr])*
        pub fn $name $(<$gen>)? (mut self, $($arg: $ty),*) -> Self
        $(where $gen: $($bound)+)?
        {
            self.agent_builder = self.agent_builder.$name($($arg),*);
            self
        }
    )+};
}

impl<T> ExtractorBuilder<T>
where
    T: JsonSchema + DeserializeOwned + Serialize + WasmCompatSend + WasmCompatSync + 'static,
{
    pub fn new<M>(model: M) -> Self
    where
        M: CompletionModel + 'static,
    {
        Self::from_agent_builder(AgentBuilder::new(model))
    }

    /// Create an extractor builder from an opaque runtime model handle.
    pub fn from_agent_builder(builder: AgentBuilder) -> Self {
        Self {
            agent_builder: builder
                .preamble("\
                    You are an AI assistant whose purpose is to extract structured data from the provided text.\n\
                    You will have access to a `submit` function that defines the structure of the data to extract from the provided text.\n\
                    Use the `submit` function to submit the structured data.\n\
                    Be sure to fill out every field and ALWAYS CALL THE `submit` function, even with default values!!!.
                ")
                .output_schema::<T>()
                .tool_choice(ToolChoice::Required)
                .output_mode(OutputMode::Tool),
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

    forward_agent_builder! {
        /// Add a context document to the extractor
        context(doc: &str);

        /// Add dynamic context retrieved from a vector store on every extraction attempt.
        ///
        /// This delegates to [`AgentBuilder::dynamic_context`] and therefore uses the
        /// same completion-call hook lifecycle as an agent.
        additional_params(params: serde_json::Value);

        /// Set the maximum number of tokens for the completion
        max_tokens(max_tokens: u64);

        /// Set the `tool_choice` option for the inner Agent.
        tool_choice(choice: ToolChoice);

        /// Add a provider-independent lifecycle hook to every extraction attempt.
        ///
        /// Completion-response hooks receive canonical Rig content, usage, prompt,
        /// and identity fields, just like hooks attached directly to an agent.
        add_hook[H: AgentHook + 'static](hook: H);
    }

    /// Retrieve `samples` documents from `index` for every extraction.
    pub fn dynamic_context<I, F>(mut self, samples: usize, index: I) -> Self
    where
        I: VectorStoreIndex<Filter = F> + 'static,
        F: DynamicSearchFilter + WasmCompatSend + WasmCompatSync + 'static,
    {
        self.agent_builder = self.agent_builder.dynamic_context(samples, index);
        self
    }

    /// Set the maximum number of retries for the extractor.
    pub fn retries(mut self, retries: u64) -> Self {
        self.retries = Some(retries);
        self
    }

    /// Build the Extractor
    pub fn build(self) -> Extractor<T> {
        Extractor {
            agent: self.agent_builder.build(),
            _t: PhantomData,
            retries: self.retries.unwrap_or(0),
        }
    }
}

#[cfg(test)]
mod tests;
