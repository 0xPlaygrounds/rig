//! The module defines the [ToolSchema] struct, which is used to embed an object that implements [crate::tool::ToolEmbedding]

use crate::{Embed, tool::ToolEmbeddingDyn};
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
    /// Convert item that implements [ToolEmbeddingDyn] to an [ToolSchema].
    ///
    /// # Example
    /// ```rust
    /// use rig_core::{
    ///     embeddings::ToolSchema,
    ///     tool::{Tool, ToolEmbedding, ToolEmbeddingDyn},
    /// };
    ///
    /// #[derive(Debug, thiserror::Error)]
    /// #[error("Math error")]
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
    ///     type Error = NothingError;
    ///     type Args = ();
    ///     type Output = ();
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
    pub fn try_from(tool: &dyn ToolEmbeddingDyn) -> Result<Self, EmbedError> {
        Ok(ToolSchema {
            name: tool.name(),
            context: tool.context().map_err(EmbedError::new)?,
            embedding_docs: tool.embedding_docs(),
        })
    }
}
