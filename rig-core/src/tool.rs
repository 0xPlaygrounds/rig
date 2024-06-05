use std::{collections::HashMap, pin::Pin};

use futures::{Future, TryFutureExt};
use serde::{Deserialize, Serialize};

use crate::{
    completion::{self, ToolDefinition},
    vector_store::VectorStore,
};

#[derive(Debug, thiserror::Error)]
pub enum ToolError {
    /// Error returned by the tool
    #[error("ToolCallError: {0}")]
    ToolCallError(#[from] Box<dyn std::error::Error + Send + Sync>),
}

/// Trait that represents a simple LLM tool
pub trait Tool: Sized + Send + Sync {
    /// The name of the tool. This name should be unique.
    const NAME: &'static str;

    type Error: std::error::Error + Send + Sync + 'static;

    /// A method returning the name of the tool.
    fn name(&self) -> String {
        Self::NAME.to_string()
    }

    /// A method returning the tool definition. The user prompt can be used to
    /// tailor the definition to the specific use case.
    fn definition(&self, _prompt: String) -> impl Future<Output = ToolDefinition> + Send + Sync;

    /// The tool execution method.
    /// Both the arguments and return value are a String since these values are meant to
    /// be the output and input of LLM models (respectively)
    fn call(&self, args: String)
        -> impl Future<Output = Result<String, Self::Error>> + Send + Sync;
}

/// Trait that represents an LLM tool that can be stored in a vector store and RAGged
pub trait ToolEmbedding: Tool {
    /// Type of the tool' context. This context will be saved and loaded from the
    /// vector store when ragging the tool.
    /// This context can be used to store the tool's static configuration and local
    /// context.
    type Context: for<'a> Deserialize<'a> + Serialize;

    /// Type of the tool's state. This state will be passed to the tool when initializing it.
    /// This state can be used to pass runtime arguments to the tool such as clients,
    /// API keys and other configuration.
    type State: Send;

    /// A method returning the documents that will be used as embeddings for the tool.
    /// This allows for a tool to be retrieved from multiple embedding "directions".
    /// If the tool will not be RAGged, this method should return an empty vector.
    fn embedding_docs(&self) -> Vec<String>;

    /// A method returning the context of the tool.
    fn context(&self) -> Self::Context;

    /// A method to initialize the tool from the context, and a state.
    fn init(state: Self::State, context: Self::Context) -> anyhow::Result<Self>;

    fn load(
        state: Self::State,
        store: &impl VectorStore,
    ) -> impl std::future::Future<Output = anyhow::Result<Self>> + Send {
        async {
            if let Some(context) = store.get_document::<Self::Context>(Self::NAME).await? {
                Self::init(state, context)
            } else {
                Err(anyhow::anyhow!("Context not found for tool {}", Self::NAME))
            }
        }
    }
}

/// Wrapper trait to allow for dynamic dispatch of simple tools
pub trait ToolDyn: Send + Sync {
    fn name(&self) -> String;

    fn definition(
        &self,
        prompt: String,
    ) -> Pin<Box<dyn Future<Output = ToolDefinition> + Send + Sync + '_>>;

    fn call(
        &self,
        args: String,
    ) -> Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send + Sync + '_>>;
}

impl<T: Tool> ToolDyn for T {
    fn name(&self) -> String {
        self.name()
    }

    fn definition(
        &self,
        prompt: String,
    ) -> Pin<Box<dyn Future<Output = ToolDefinition> + Send + Sync + '_>> {
        Box::pin(<Self as Tool>::definition(self, prompt))
    }

    fn call(
        &self,
        args: String,
    ) -> Pin<Box<dyn Future<Output = Result<String, ToolError>> + Send + Sync + '_>> {
        Box::pin(
            <Self as Tool>::call(self, args).map_err(|e| ToolError::ToolCallError(Box::new(e))),
        )
    }
}

/// Wrapper trait to allow for dynamic dispatch of raggable tools
pub trait ToolEmbeddingDyn: ToolDyn {
    fn context(&self) -> anyhow::Result<serde_json::Value>;

    fn embedding_docs(&self) -> Vec<String>;
}

impl<T: ToolEmbedding> ToolEmbeddingDyn for T {
    fn context(&self) -> anyhow::Result<serde_json::Value> {
        Ok(serde_json::to_value(&self.context())?)
    }

    fn embedding_docs(&self) -> Vec<String> {
        self.embedding_docs()
    }
}

pub(crate) enum ToolType {
    Simple(Box<dyn ToolDyn>),
    Embedding(Box<dyn ToolEmbeddingDyn>),
}

impl ToolType {
    pub fn name(&self) -> String {
        match self {
            ToolType::Simple(tool) => tool.name(),
            ToolType::Embedding(tool) => tool.name(),
        }
    }

    pub async fn definition(&self, prompt: String) -> ToolDefinition {
        match self {
            ToolType::Simple(tool) => tool.definition(prompt).await,
            ToolType::Embedding(tool) => tool.definition(prompt).await,
        }
    }

    pub async fn call(&self, args: String) -> Result<String, ToolError> {
        match self {
            ToolType::Simple(tool) => tool.call(args).await,
            ToolType::Embedding(tool) => tool.call(args).await,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ToolSetError {
    /// Error returned by the tool
    #[error("ToolCallError: {0}")]
    ToolCallError(#[from] ToolError),

    #[error("ToolNotFoundError: {0}")]
    ToolNotFoundError(String),

    // TODO: Revisit this
    #[error("JsonError: {0}")]
    JsonError(#[from] serde_json::Error),
}

/// A struct that holds a set of tools
#[derive(Default)]
pub struct ToolSet {
    pub(crate) tools: HashMap<String, ToolType>,
}

impl ToolSet {
    pub fn new(tools: Vec<impl ToolDyn + 'static>) -> Self {
        let mut toolset = Self::default();
        tools.into_iter().for_each(|tool| {
            toolset.add_tool(tool);
        });
        toolset
    }

    pub fn builder() -> ToolSetBuilder {
        ToolSetBuilder::default()
    }

    pub fn contains(&self, toolname: &str) -> bool {
        self.tools.contains_key(toolname)
    }

    pub fn add_tool(&mut self, tool: impl ToolDyn + 'static) {
        self.tools
            .insert(tool.name(), ToolType::Simple(Box::new(tool)));
    }

    pub fn add_tools(&mut self, toolset: ToolSet) {
        self.tools.extend(toolset.tools);
    }

    pub(crate) fn get(&self, toolname: &str) -> Option<&ToolType> {
        self.tools.get(toolname)
    }

    pub async fn call(&self, toolname: &str, args: String) -> Result<String, ToolSetError> {
        if let Some(tool) = self.tools.get(toolname) {
            tracing::info!(target: "ai",
                "Calling tool {toolname} with args:\n{}",
                serde_json::to_string_pretty(&args).unwrap_or_else(|_| args.clone())
            );
            Ok(tool.call(args).await?)
        } else {
            Err(ToolSetError::ToolNotFoundError(toolname.to_string()))
        }
    }

    pub async fn documents(&self) -> Result<Vec<completion::Document>, ToolSetError> {
        let mut docs = Vec::new();
        for tool in self.tools.values() {
            match tool {
                ToolType::Simple(tool) => {
                    docs.push(completion::Document {
                        id: tool.name(),
                        text: format!(
                            "\
                            Tool: {}\n\
                            Definition: \n\
                            {}\
                        ",
                            tool.name(),
                            serde_json::to_string_pretty(&tool.definition("".to_string()).await)?
                        ),
                        additional_props: HashMap::new(),
                    });
                }
                ToolType::Embedding(tool) => {
                    docs.push(completion::Document {
                        id: tool.name(),
                        text: format!(
                            "\
                            Tool: {}\n\
                            Definition: \n\
                            {}\
                        ",
                            tool.name(),
                            serde_json::to_string_pretty(&tool.definition("".to_string()).await)?
                        ),
                        additional_props: HashMap::new(),
                    });
                }
            }
        }
        Ok(docs)
    }
}

#[derive(Default)]
pub struct ToolSetBuilder {
    tools: Vec<ToolType>,
}

impl ToolSetBuilder {
    pub fn static_tool(mut self, tool: impl ToolDyn + 'static) -> Self {
        self.tools.push(ToolType::Simple(Box::new(tool)));
        self
    }

    pub fn dynamic_tool(mut self, tool: impl ToolEmbeddingDyn + 'static) -> Self {
        self.tools.push(ToolType::Embedding(Box::new(tool)));
        self
    }

    pub fn build(self) -> ToolSet {
        ToolSet {
            tools: self
                .tools
                .into_iter()
                .map(|tool| (tool.name(), tool))
                .collect(),
        }
    }
}
