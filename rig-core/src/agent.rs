//! This module contains the implementation of the [Agent] struct and its builder.
//!
//! The [Agent] struct represents an LLM agent, which combines an LLM model with a preamble (system prompt),
//! a set of context documents, and a set of tools. Note: both context documents and tools can be either
//! static (i.e.: they are always provided) or dynamic (i.e.: they are RAGged at prompt-time).
//!
//! The [Agent] struct is highly configurable, allowing the user to define anything from
//! a simple bot with a specific system prompt to a complex RAG system with a set of dynamic
//! context documents and tools.
//!
//! The [Agent] struct implements the [Completion] and [Prompt] traits, allowing it to be used for generating
//! completions responses and prompts. The [Agent] struct also implements the [Chat] trait, which allows it to
//! be used for generating chat completions.
//!
//! The [AgentBuilder] implements the builder pattern for creating instances of [Agent].
//! It allows configuring the model, preamble, context documents, tools, temperature, and additional parameters
//! before building the agent.
//!
//! # Example
//! ```rust
//! use rig::{
//!     completion::{Chat, Completion, Prompt},
//!     providers::openai,
//! };
//!
//! let openai = openai::Client::from_env();
//!
//! // Configure the agent
//! let agent = openai.agent("gpt-4o")
//!     .preamble("System prompt")
//!     .context("Context document 1")
//!     .context("Context document 2")
//!     .tool(tool1)
//!     .tool(tool2)
//!     .temperature(0.8)
//!     .additional_params(json!({"foo": "bar"}))
//!     .build();
//!
//! // Use the agent for completions and prompts
//! // Generate a chat completion response from a prompt and chat history
//! let chat_response = agent.chat("Prompt", chat_history)
//!     .await
//!     .expect("Failed to chat with Agent");
//!
//! // Generate a prompt completion response from a simple prompt
//! let chat_response = agent.prompt("Prompt")
//!     .await
//!     .expect("Failed to prompt the Agent");
//!
//! // Generate a completion request builder from a prompt and chat history. The builder
//! // will contain the agent's configuration (i.e.: preamble, context documents, tools,
//! // model parameters, etc.), but these can be overwritten.
//! let completion_req_builder = agent.completion("Prompt", chat_history)
//!     .await
//!     .expect("Failed to create completion request builder");
//!
//! let response = completion_req_builder
//!     .temperature(0.9) // Overwrite the agent's temperature
//!     .send()
//!     .await
//!     .expect("Failed to send completion request");
//! ```
//!
//! RAG Agent example
//! ```rust
//! use rig::{
//!     completion::Prompt,
//!     embeddings::EmbeddingsBuilder,
//!     providers::openai,
//!     vector_store::{in_memory_store::InMemoryVectorStore, VectorStore},
//! };
//!
//! // Initialize OpenAI client
//! let openai = openai::Client::from_env();
//!
//! // Initialize OpenAI embedding model
//! let embedding_model = openai.embedding_model(openai::TEXT_EMBEDDING_ADA_002);
//!
//! // Create vector store, compute embeddings and load them in the store
//! let mut vector_store = InMemoryVectorStore::default();
//!
//! let embeddings = EmbeddingsBuilder::new(embedding_model.clone())
//!     .simple_document("doc0", "Definition of a *flurbo*: A flurbo is a green alien that lives on cold planets")
//!     .simple_document("doc1", "Definition of a *glarb-glarb*: A glarb-glarb is a ancient tool used by the ancestors of the inhabitants of planet Jiro to farm the land.")
//!     .simple_document("doc2", "Definition of a *linglingdong*: A term used by inhabitants of the far side of the moon to describe humans.")
//!     .build()
//!     .await
//!     .expect("Failed to build embeddings");
//!
//! vector_store.add_documents(embeddings)
//!     .await
//!     .expect("Failed to add documents");
//!
//! // Create vector store index
//! let index = vector_store.index(embedding_model);
//!
//! let agent = openai.agent(openai::GPT_4O)
//!     .preamble("
//!         You are a dictionary assistant here to assist the user in understanding the meaning of words.
//!         You will find additional non-standard word definitions that could be useful below.
//!     ")
//!     .dynamic_context(1, index)
//!     .build();
//!
//! // Prompt the agent and print the response
//! let response = agent.prompt("What does \"glarb-glarb\" mean?").await
//!     .expect("Failed to prompt the agent");
//! ```
use std::collections::HashMap;

use futures::{stream, StreamExt, TryStreamExt};

use crate::{
    completion::{
        Chat, Completion, CompletionError, CompletionModel, CompletionRequestBuilder, Document,
        Message, Prompt, PromptError,
    },
    message::AssistantContent,
    streaming::{
        StreamingChat, StreamingCompletion, StreamingCompletionModel, StreamingPrompt,
        StreamingResult,
    },
    tool::{Tool, ToolSet},
    vector_store::{VectorStoreError, VectorStoreIndexDyn},
};

/// Struct representing an LLM agent. An agent is an LLM model combined with a preamble
/// (i.e.: system prompt) and a static set of context documents and tools.
/// All context documents and tools are always provided to the agent when prompted.
///
/// # Example
/// ```
/// use rig::{completion::Prompt, providers::openai};
///
/// let openai = openai::Client::from_env();
///
/// let comedian_agent = openai
///     .agent("gpt-4o")
///     .preamble("You are a comedian here to entertain the user using humour and jokes.")
///     .temperature(0.9)
///     .build();
///
/// let response = comedian_agent.prompt("Entertain me!")
///     .await
///     .expect("Failed to prompt the agent");
/// ```
pub struct Agent<M: CompletionModel> {
    /// Completion model (e.g.: OpenAI's gpt-3.5-turbo-1106, Cohere's command-r)
    model: M,
    /// System prompt
    preamble: String,
    /// Context documents always available to the agent
    static_context: Vec<Document>,
    /// Tools that are always available to the agent (identified by their name)
    static_tools: Vec<String>,
    /// Temperature of the model
    temperature: Option<f64>,
    /// Maximum number of tokens for the completion
    max_tokens: Option<u64>,
    /// Additional parameters to be passed to the model
    additional_params: Option<serde_json::Value>,
    /// List of vector store, with the sample number
    dynamic_context: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
    /// Dynamic tools
    dynamic_tools: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
    /// Actual tool implementations
    pub tools: ToolSet,
}

impl<M: CompletionModel> Completion<M> for Agent<M> {
    async fn completion(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError> {
        let prompt = prompt.into();
        let rag_text = prompt.rag_text().clone();

        let completion_request = self
            .model
            .completion_request(prompt)
            .preamble(self.preamble.clone())
            .messages(chat_history)
            .temperature_opt(self.temperature)
            .max_tokens_opt(self.max_tokens)
            .additional_params_opt(self.additional_params.clone())
            .documents(self.static_context.clone());

        let agent = match &rag_text {
            Some(text) => {
                let dynamic_context = stream::iter(self.dynamic_context.iter())
                    .then(|(num_sample, index)| async {
                        Ok::<_, VectorStoreError>(
                            index
                                .top_n(text, *num_sample)
                                .await?
                                .into_iter()
                                .map(|(_, id, doc)| {
                                    // Pretty print the document if possible for better readability
                                    let text = serde_json::to_string_pretty(&doc)
                                        .unwrap_or_else(|_| doc.to_string());

                                    Document {
                                        id,
                                        text,
                                        additional_props: HashMap::new(),
                                    }
                                })
                                .collect::<Vec<_>>(),
                        )
                    })
                    .try_fold(vec![], |mut acc, docs| async {
                        acc.extend(docs);
                        Ok(acc)
                    })
                    .await
                    .map_err(|e| CompletionError::RequestError(Box::new(e)))?;

                let dynamic_tools = stream::iter(self.dynamic_tools.iter())
                    .then(|(num_sample, index)| async {
                        Ok::<_, VectorStoreError>(
                            index
                                .top_n_ids(text, *num_sample)
                                .await?
                                .into_iter()
                                .map(|(_, id)| id)
                                .collect::<Vec<_>>(),
                        )
                    })
                    .try_fold(vec![], |mut acc, docs| async {
                        for doc in docs {
                            if let Some(tool) = self.tools.get(&doc) {
                                acc.push(tool.definition(text.into()).await)
                            } else {
                                tracing::warn!("Tool implementation not found in toolset: {}", doc);
                            }
                        }
                        Ok(acc)
                    })
                    .await
                    .map_err(|e| CompletionError::RequestError(Box::new(e)))?;

                let static_tools = stream::iter(self.static_tools.iter())
                    .filter_map(|toolname| async move {
                        if let Some(tool) = self.tools.get(toolname) {
                            Some(tool.definition(text.into()).await)
                        } else {
                            tracing::warn!(
                                "Tool implementation not found in toolset: {}",
                                toolname
                            );
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .await;

                completion_request
                    .documents(dynamic_context)
                    .tools([static_tools.clone(), dynamic_tools].concat())
            }
            None => {
                let static_tools = stream::iter(self.static_tools.iter())
                    .filter_map(|toolname| async move {
                        if let Some(tool) = self.tools.get(toolname) {
                            // TODO: tool definitions should likely take an `Option<String>`
                            Some(tool.definition("".into()).await)
                        } else {
                            tracing::warn!(
                                "Tool implementation not found in toolset: {}",
                                toolname
                            );
                            None
                        }
                    })
                    .collect::<Vec<_>>()
                    .await;

                completion_request.tools(static_tools)
            }
        };

        Ok(agent)
    }
}

impl<M: CompletionModel> Prompt for Agent<M> {
    async fn prompt(&self, prompt: impl Into<Message> + Send) -> Result<String, PromptError> {
        self.chat(prompt, vec![]).await
    }
}

impl<M: CompletionModel> Prompt for &Agent<M> {
    async fn prompt(&self, prompt: impl Into<Message> + Send) -> Result<String, PromptError> {
        self.chat(prompt, vec![]).await
    }
}

impl<M: CompletionModel> Chat for Agent<M> {
    async fn chat(
        &self,
        prompt: impl Into<Message> + Send,
        chat_history: Vec<Message>,
    ) -> Result<String, PromptError> {
        let resp = self.completion(prompt, chat_history).await?.send().await?;

        // TODO: consider returning a `Message` instead of `String` for parallel responses / tool calls
        match resp.choice.first() {
            AssistantContent::Text(text) => Ok(text.text.clone()),
            AssistantContent::ToolCall(tool_call) => Ok(self
                .tools
                .call(
                    &tool_call.function.name,
                    tool_call.function.arguments.to_string(),
                )
                .await?),
        }
    }
}

/// A builder for creating an agent
///
/// # Example
/// ```
/// use rig::{providers::openai, agent::AgentBuilder};
///
/// let openai = openai::Client::from_env();
///
/// let gpt4o = openai.completion_model("gpt-4o");
///
/// // Configure the agent
/// let agent = AgentBuilder::new(model)
///     .preamble("System prompt")
///     .context("Context document 1")
///     .context("Context document 2")
///     .tool(tool1)
///     .tool(tool2)
///     .temperature(0.8)
///     .additional_params(json!({"foo": "bar"}))
///     .build();
/// ```
pub struct AgentBuilder<M: CompletionModel> {
    /// Completion model (e.g.: OpenAI's gpt-3.5-turbo-1106, Cohere's command-r)
    model: M,
    /// System prompt
    preamble: Option<String>,
    /// Context documents always available to the agent
    static_context: Vec<Document>,
    /// Tools that are always available to the agent (by name)
    static_tools: Vec<String>,
    /// Additional parameters to be passed to the model
    additional_params: Option<serde_json::Value>,
    /// Maximum number of tokens for the completion
    max_tokens: Option<u64>,
    /// List of vector store, with the sample number
    dynamic_context: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
    /// Dynamic tools
    dynamic_tools: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
    /// Temperature of the model
    temperature: Option<f64>,
    /// Actual tool implementations
    tools: ToolSet,
}

impl<M: CompletionModel> AgentBuilder<M> {
    pub fn new(model: M) -> Self {
        Self {
            model,
            preamble: None,
            static_context: vec![],
            static_tools: vec![],
            temperature: None,
            max_tokens: None,
            additional_params: None,
            dynamic_context: vec![],
            dynamic_tools: vec![],
            tools: ToolSet::default(),
        }
    }

    /// Set the system prompt
    pub fn preamble(mut self, preamble: &str) -> Self {
        self.preamble = Some(preamble.into());
        self
    }

    /// Append to the preamble of the agent
    pub fn append_preamble(mut self, doc: &str) -> Self {
        self.preamble = Some(format!(
            "{}\n{}",
            self.preamble.unwrap_or_else(|| "".into()),
            doc
        ));
        self
    }

    /// Add a static context document to the agent
    pub fn context(mut self, doc: &str) -> Self {
        self.static_context.push(Document {
            id: format!("static_doc_{}", self.static_context.len()),
            text: doc.into(),
            additional_props: HashMap::new(),
        });
        self
    }

    /// Add a static tool to the agent
    pub fn tool(mut self, tool: impl Tool + 'static) -> Self {
        let toolname = tool.name();
        self.tools.add_tool(tool);
        self.static_tools.push(toolname);
        self
    }

    /// Add some dynamic context to the agent. On each prompt, `sample` documents from the
    /// dynamic context will be inserted in the request.
    pub fn dynamic_context(
        mut self,
        sample: usize,
        dynamic_context: impl VectorStoreIndexDyn + 'static,
    ) -> Self {
        self.dynamic_context
            .push((sample, Box::new(dynamic_context)));
        self
    }

    /// Add some dynamic tools to the agent. On each prompt, `sample` tools from the
    /// dynamic toolset will be inserted in the request.
    pub fn dynamic_tools(
        mut self,
        sample: usize,
        dynamic_tools: impl VectorStoreIndexDyn + 'static,
        toolset: ToolSet,
    ) -> Self {
        self.dynamic_tools.push((sample, Box::new(dynamic_tools)));
        self.tools.add_tools(toolset);
        self
    }

    /// Set the temperature of the model
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Set the maximum number of tokens for the completion
    pub fn max_tokens(mut self, max_tokens: u64) -> Self {
        self.max_tokens = Some(max_tokens);
        self
    }

    /// Set additional parameters to be passed to the model
    pub fn additional_params(mut self, params: serde_json::Value) -> Self {
        self.additional_params = Some(params);
        self
    }

    /// Build the agent
    pub fn build(self) -> Agent<M> {
        Agent {
            model: self.model,
            preamble: self.preamble.unwrap_or_default(),
            static_context: self.static_context,
            static_tools: self.static_tools,
            temperature: self.temperature,
            max_tokens: self.max_tokens,
            additional_params: self.additional_params,
            dynamic_context: self.dynamic_context,
            dynamic_tools: self.dynamic_tools,
            tools: self.tools,
        }
    }
}

impl<M: StreamingCompletionModel> StreamingCompletion<M> for Agent<M> {
    async fn stream_completion(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError> {
        // Reuse the existing completion implementation to build the request
        // This ensures streaming and non-streaming use the same request building logic
        self.completion(prompt, chat_history).await
    }
}

impl<M: StreamingCompletionModel> StreamingPrompt for Agent<M> {
    async fn stream_prompt(&self, prompt: &str) -> Result<StreamingResult, CompletionError> {
        self.stream_chat(prompt, vec![]).await
    }
}

impl<M: StreamingCompletionModel> StreamingChat for Agent<M> {
    async fn stream_chat(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> Result<StreamingResult, CompletionError> {
        self.stream_completion(prompt, chat_history)
            .await?
            .stream()
            .await
    }
}
