//! The module defines the [ToolSchema] struct, which is used to embed an object that implements [crate::tool::ToolEmbedding]

use crate::{Embed, tool::ToolEmbedding};
use serde::Serialize;

use super::embed::EmbedError;

/// Embeddable document that is used as an intermediate representation of a tool when
/// RAGging tools.
#[derive(Clone, Serialize, Default, Eq, PartialEq)]
pub struct ToolSchema {
    pub name: String,
    pub context: serde_json::Value,
    pub embedding_docs: Vec<String>,
}

impl Embed for ToolSchema {
    fn embed(&self, embedder: &mut super::embed::TextEmbedder) -> Result<(), EmbedError> {
        for doc in &self.embedding_docs {
            embedder.embed(doc.clone());
        }
        Ok(())
    }
}

impl ToolSchema {
    /// Convert an embedding-backed tool to a [`ToolSchema`].
    ///
    /// # Example
    /// ```rust
    /// use rig_core::{
    ///     embeddings::ToolSchema,
    ///     tool::{Tool, ToolEmbedding},
    /// };
    ///
    /// #[derive(Debug, thiserror::Error)]
    /// #[error("Nothing error")]
    /// struct NothingError;
    ///
    /// #[derive(Debug, thiserror::Error)]
    /// #[error("Init error")]
    /// struct InitError;
    ///
    /// struct Nothing;
    /// impl Tool for Nothing {
    ///     const NAME: &'static str = "nothing";
    ///
    ///     type Args = ();
    ///     type Output = ();
    ///     type Error = NothingError;
    ///
    ///     fn description(&self) -> String {
    ///         "nothing".to_string()
    ///     }
    ///
    ///     fn parameters(&self) -> serde_json::Value {
    ///         serde_json::json!({})
    ///     }
    ///
    ///     async fn call(&self, _args: Self::Args) -> Result<Self::Output, Self::Error> {
    ///         Ok(())
    ///     }
    /// }
    ///
    /// impl ToolEmbedding for Nothing {
    ///     type InitError = InitError;
    ///     type Context = ();
    ///     type State = ();
    ///
    ///     fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
    ///         Ok(Nothing)
    ///     }
    ///
    ///     fn embedding_docs(&self) -> Vec<String> {
    ///         vec!["Do nothing.".into()]
    ///     }
    ///
    ///     fn context(&self) -> Self::Context {}
    /// }
    ///
    /// let tool = ToolSchema::try_from(&Nothing).unwrap();
    ///
    /// assert_eq!(tool.name, "nothing".to_string());
    /// assert_eq!(tool.embedding_docs, vec!["Do nothing.".to_string()]);
    /// ```
    pub fn try_from<T>(tool: &T) -> Result<Self, EmbedError>
    where
        T: ToolEmbedding + 'static,
    {
        Self::from_embedding_tool(T::NAME, tool)
    }

    fn from_embedding_tool<T>(name: impl Into<String>, tool: &T) -> Result<Self, EmbedError>
    where
        T: ToolEmbedding,
    {
        Ok(ToolSchema {
            name: name.into(),
            context: serde_json::to_value(tool.context()).map_err(EmbedError::new)?,
            embedding_docs: tool.embedding_docs(),
        })
    }
}

#[cfg(test)]
mod tests {
    use std::convert::Infallible;

    use super::ToolSchema;
    use crate::tool::{Tool, ToolEmbedding, ToolExecutionError};

    struct NamedTool;

    impl Tool for NamedTool {
        const NAME: &'static str = "static_name";

        type Error = rig::tool::ToolExecutionError;

        type Args = ();
        type Output = ();

        fn description(&self) -> String {
            "A statically named tool".to_string()
        }

        fn parameters(&self) -> serde_json::Value {
            serde_json::json!({})
        }

        async fn call(&self, _args: Self::Args) -> Result<Self::Output, ToolExecutionError> {
            Ok(())
        }
    }

    impl ToolEmbedding for NamedTool {
        type InitError = Infallible;
        type Context = ();
        type State = ();

        fn embedding_docs(&self) -> Vec<String> {
            vec!["named tool".to_string()]
        }

        fn context(&self) -> Self::Context {}

        fn init(_state: Self::State, _context: Self::Context) -> Result<Self, Self::InitError> {
            Ok(Self)
        }
    }

    #[test]
    fn try_from_uses_canonical_tool_name() {
        let schema = ToolSchema::try_from(&NamedTool).unwrap();

        assert_eq!(schema.name, NamedTool::NAME);
    }
}
