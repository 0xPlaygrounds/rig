// pub mod context;
// pub mod tools;

use std::collections::HashMap;

use anyhow::Result;
use futures::{stream, StreamExt, TryStreamExt};

use crate::{
    completion::{
        Completion, CompletionError, CompletionModel, CompletionRequestBuilder, CompletionResponse,
        Document, Message, ModelChoice, Prompt, PromptError,
    },
    tool::{Tool, ToolSet},
    vector_store::{NoIndex, VectorStoreIndex},
};

/// Struct representing a RAG agent, i.e.: an agent enhanced with two collections of
/// vector store indices, one for context documents and one for tools.
/// The ragged context and tools are used to enhance the completion model at prompt-time.
/// Note: The type of the VectorStoreIndex must be the same for all the dynamic context
/// and tools indices (but can be different for context and tools).
/// If you need to use a more complex combination of vector store indices,
/// you should implement a custom agent.
pub struct RagAgent<M: CompletionModel, C: VectorStoreIndex, T: VectorStoreIndex> {
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
    dynamic_context: Vec<(usize, C)>,
    /// Dynamic tools
    dynamic_tools: Vec<(usize, T)>,
    /// Actual tool implementations
    pub tools: ToolSet,
}

pub type ToolRagAgent<M, T> = RagAgent<M, NoIndex, T>;
pub type ContextRagAgent<M, C> = RagAgent<M, C, NoIndex>;

impl<M: CompletionModel, C: VectorStoreIndex, T: VectorStoreIndex> Completion<M>
    for RagAgent<M, C, T>
{
    async fn completion(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> Result<CompletionRequestBuilder<M>, CompletionError> {
        let dynamic_context = stream::iter(self.dynamic_context.iter())
            .then(|(num_sample, index)| async {
                anyhow::Ok(
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
            .map_err(|e| CompletionError::RequestError(format!("Error ragging context documents: {}", e)))?;

        let dynamic_tools = stream::iter(self.dynamic_tools.iter())
            .then(|(num_sample, index)| async {
                anyhow::Ok(
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
            .map_err(|e| CompletionError::RequestError(format!("Error ragging tools: {}", e)))?;

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

impl<M: CompletionModel, C: VectorStoreIndex, T: VectorStoreIndex> Prompt for RagAgent<M, C, T> {
    async fn prompt(
        &self,
        prompt: &str,
        chat_history: Vec<Message>,
    ) -> Result<String, PromptError> {
        match self.completion(prompt, chat_history).await?.send().await? {
            CompletionResponse {
                choice: ModelChoice::Message(msg),
                ..
            } => Ok(msg),
            CompletionResponse {
                choice: ModelChoice::ToolCall(toolname, args),
                ..
            } => self
                .tools
                .call(&toolname, args.to_string())
                .await
                .map_err(|e| PromptError::ToolCallError(format!("{}", e))),
        }
    }
}

impl<M: CompletionModel, C: VectorStoreIndex, T: VectorStoreIndex> RagAgent<M, C, T> {
    pub async fn call_tool(&self, toolname: &str, args: &str) -> Result<String> {
        self.tools.call(toolname, args.to_string()).await
    }
}

pub struct RagAgentBuilder<M: CompletionModel, C: VectorStoreIndex, T: VectorStoreIndex> {
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
    dynamic_context: Vec<(usize, C)>,
    /// Dynamic tools
    dynamic_tools: Vec<(usize, T)>,
    /// Temperature of the model
    temperature: Option<f64>,
    /// Actual tool implementations
    tools: ToolSet,
}

impl<M: CompletionModel, C: VectorStoreIndex, T: VectorStoreIndex> RagAgentBuilder<M, C, T> {
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
    pub fn static_tool(mut self, tool: impl Tool + Sync + 'static) -> Self {
        let toolname = tool.name();
        self.tools.add_tool(tool);
        self.static_tools.push(toolname);
        self
    }

    /// Add some dynamic context to the RAG agent. On each prompt, `sample` documents from the
    /// dynamic context will be inserted in the request.
    pub fn dynamic_context(mut self, sample: usize, dynamic_context: C) -> Self {
        self.dynamic_context.push((sample, dynamic_context));
        self
    }

    /// Add some dynamic tools to the RAG agent. On each prompt, `sample` tools from the
    /// dynamic toolset will be inserted in the request.
    pub fn dynamic_tools(mut self, sample: usize, dynamic_tools: T, toolset: ToolSet) -> Self {
        self.dynamic_tools.push((sample, dynamic_tools));
        self.tools.add_tools(toolset);
        self
    }

    /// Set the temperature of the model
    pub fn temperature(mut self, temperature: f64) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Build the RAG agent
    pub fn build(self) -> RagAgent<M, C, T> {
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
