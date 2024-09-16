//! This module contains the implementation of the [RagAgent] struct and its builder.
//!
//! The [RagAgent] struct defines a fully featured RAG system, combining and agent with
//! dynamic context documents and tools which are used to enhance the completion model at
//! prompt-time.
//!
//! The [RagAgentBuilder] implements the builder pattern for creating instances of [RagAgent].
//! It allows configuring the model, preamble, dynamic context documents and tools, as well
//! as all the other parameters that are available in the [crate::agent::AgentBuilder].
//!
//! # Example
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
//! let rag_agent = openai.context_rag_agent(openai::GPT_4O)
//!     .preamble("
//!         You are a dictionary assistant here to assist the user in understanding the meaning of words.
//!         You will find additional non-standard word definitions that could be useful below.
//!     ")
//!     .dynamic_context(1, index)
//!     .build();
//!
//! // Prompt the agent and print the response
//! let response = rag_agent.prompt("What does \"glarb-glarb\" mean?").await
//!     .expect("Failed to prompt the agent");
//! ```
use std::collections::HashMap;

use futures::{stream, StreamExt, TryStreamExt};

use crate::{
    completion::{
        Chat, Completion, CompletionError, CompletionModel, CompletionRequestBuilder,
        CompletionResponse, Document, Message, ModelChoice, Prompt, PromptError,
    },
    tool::{Tool, ToolSet, ToolSetError},
    vector_store::{VectorStoreError, VectorStoreIndexDyn},
};

/// Struct representing a RAG agent, i.e.: an agent enhanced with two collections of
/// vector store indices, one for context documents and one for tools.
/// The ragged context and tools are used to enhance the completion model at prompt-time.
/// Note: The type of the [VectorStoreIndex] must be the same for all the dynamic context
/// and tools indices (but can be different for context and tools).
/// If you need to use a more complex combination of vector store indices,
/// you should implement a custom agent.
pub struct RagAgent<M: CompletionModel> {
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
    /// Additional parameters to be passed to the model
    additional_params: Option<serde_json::Value>,
    /// List of vector store, with the sample number
    dynamic_context: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
    /// Dynamic tools
    dynamic_tools: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
    /// Actual tool implementations
    pub tools: ToolSet,
}

impl<M: CompletionModel> Completion<M> for RagAgent<M> {
    async fn completion(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError> {
        let dynamic_context = stream::iter(self.dynamic_context.iter())
            .then(|(num_sample, index)| async {
                Ok::<_, VectorStoreError>(
                    index
                        .top_n_from_query(prompt, *num_sample)
                        .await?
                        .into_iter()
                        .map(|(_, doc)| {
                            let doc_text = serde_json::to_string_pretty(&doc.document)
                                .unwrap_or_else(|_| doc.document.to_string());

                            Document {
                                id: doc.id,
                                text: doc_text,
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
                        .top_n_ids_from_query(prompt, *num_sample)
                        .await?
                        .into_iter()
                        .map(|(_, doc)| doc)
                        .collect::<Vec<_>>(),
                )
            })
            .try_fold(vec![], |mut acc, docs| async {
                for doc in docs {
                    if let Some(tool) = self.tools.get(&doc) {
                        acc.push(tool.definition(prompt.into()).await)
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
                    Some(tool.definition(prompt.into()).await)
                } else {
                    tracing::warn!("Tool implementation not found in toolset: {}", toolname);
                    None
                }
            })
            .collect::<Vec<_>>()
            .await;

        Ok(self
            .model
            .completion_request(prompt)
            .preamble(self.preamble.clone())
            .messages(chat_history)
            .documents([self.static_context.clone(), dynamic_context].concat())
            .tools([static_tools.clone(), dynamic_tools].concat())
            .temperature_opt(self.temperature)
            .additional_params_opt(self.additional_params.clone()))
    }
}

impl<M: CompletionModel> Prompt for RagAgent<M> {
    async fn prompt(&self, prompt: &str) -> Result<String, PromptError> {
        self.chat(prompt, vec![]).await
    }
}

impl<M: CompletionModel> Chat for RagAgent<M> {
    async fn chat(&self, prompt: &str, chat_history: Vec<Message>) -> Result<String, PromptError> {
        match self.completion(prompt, chat_history).await?.send().await? {
            CompletionResponse {
                choice: ModelChoice::Message(msg),
                ..
            } => Ok(msg),
            CompletionResponse {
                choice: ModelChoice::ToolCall(toolname, args),
                ..
            } => Ok(self.tools.call(&toolname, args.to_string()).await?),
        }
    }
}

impl<M: CompletionModel> RagAgent<M> {
    pub async fn call_tool(&self, toolname: &str, args: &str) -> Result<String, ToolSetError> {
        self.tools.call(toolname, args.to_string()).await
    }
}

/// Builder for creating a RAG agent
///
/// # Example
/// ```
/// use rig::{providers::openai, rag_agent::RagAgentBuilder};
/// use serde_json::json;
///
/// let openai_client = openai::Client::from_env();
///
/// let model = openai_client.completion_model("gpt-4");
///
/// // Configure the agent
/// let agent = RagAgentBuilder::new(model)
///     .preamble("System prompt")
///     .static_context("Context document 1")
///     .static_context("Context document 2")
///     .dynamic_context(2, vector_index)
///     .tool(tool1)
///     .tool(tool2)
///     .temperature(0.8)
///     .additional_params(json!({"foo": "bar"}))
///     .build();
/// ```
pub struct RagAgentBuilder<M: CompletionModel> {
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
    /// List of vector store, with the sample number
    dynamic_context: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
    /// Dynamic tools
    dynamic_tools: Vec<(usize, Box<dyn VectorStoreIndexDyn>)>,
    /// Temperature of the model
    temperature: Option<f64>,
    /// Actual tool implementations
    tools: ToolSet,
}

impl<M: CompletionModel> RagAgentBuilder<M> {
    pub fn new(model: M) -> Self {
        Self {
            model,
            preamble: None,
            static_context: vec![],
            static_tools: vec![],
            temperature: None,
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

    /// Add a static context document to the RAG agent
    pub fn static_context(mut self, doc: &str) -> Self {
        self.static_context.push(Document {
            id: format!("static_doc_{}", self.static_context.len()),
            text: doc.into(),
            additional_props: HashMap::new(),
        });
        self
    }

    /// Add a static tool to the RAG agent
    pub fn static_tool(mut self, tool: impl Tool + 'static) -> Self {
        let toolname = tool.name();
        self.tools.add_tool(tool);
        self.static_tools.push(toolname);
        self
    }

    /// Add some dynamic context to the RAG agent. On each prompt, `sample` documents from the
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

    /// Add some dynamic tools to the RAG agent. On each prompt, `sample` tools from the
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

    /// Build the RAG agent
    pub fn build(self) -> RagAgent<M> {
        RagAgent {
            model: self.model,
            preamble: self.preamble.unwrap_or_default(),
            static_context: self.static_context,
            static_tools: self.static_tools,
            temperature: self.temperature,
            additional_params: self.additional_params,
            dynamic_context: self.dynamic_context,
            dynamic_tools: self.dynamic_tools,
            tools: self.tools,
        }
    }
}
