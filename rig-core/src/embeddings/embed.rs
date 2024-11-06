//! The module defines the [ExtractEmbeddingFields] trait, which must be implemented for types that can be embedded.

// use crate::one_or_many::OneOrMany;

use super::EmbeddingModel;

/// Error type used for when the `extract_embedding_fields` method fails.
/// Used by default implementations of `ExtractEmbeddingFields` for common types.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct EmbedError(#[from] Box<dyn std::error::Error + Send + Sync>);

impl EmbedError {
    pub fn new<E: std::error::Error + Send + Sync + 'static>(error: E) -> Self {
        EmbedError(Box::new(error))
    }
}

/// Derive this trait for structs whose fields need to be converted to vector embeddings.
/// The `extract_embedding_fields` method returns a `OneOrMany<String>`. This function extracts the fields that need to be embedded and returns them as a list of strings.
/// If there is an error generating the list of strings, the method should return an error that implements `std::error::Error`.
/// # Example
/// ```rust
/// use std::env;
///
/// use serde::{Deserialize, Serialize};
/// use rig::{OneOrMany, EmptyListError, ExtractEmbeddingFields};
///
/// struct FakeDefinition {
///     id: String,
///     word: String,
///     definitions: String,
/// }
///
/// let fake_definition = FakeDefinition {
///     id: "doc1".to_string(),
///     word: "rock".to_string(),
///     definitions: "the solid mineral material forming part of the surface of the earth, a precious stone".to_string()
/// };
///
/// impl ExtractEmbeddingFields for FakeDefinition {
///     type Error = EmptyListError;
///
///     fn extract_embedding_fields(&self) -> Result<OneOrMany<String>, Self::Error> {
///         // Embeddings only need to be generated for `definition` field.
///         // Split the definitions by comma and collect them into a vector of strings.
///         // That way, different embeddings can be generated for each definition in the definitions string.
///         let definitions = self.definitions.split(",").collect::<Vec<_>>().into_iter().map(|s| s.to_string()).collect();
///
///         OneOrMany::many(definitions)
///     }
/// }
/// ```
pub trait Embed {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError>;
}

#[derive(Default)]
pub struct Embedder {
    pub texts: Vec<String>,
}

impl Embedder {
    pub fn embed(&mut self, text: String) {
        self.texts.push(text);
    }
}

// ================================================================
// Implementations of ExtractEmbeddingFields for common types
// ================================================================
impl Embed for String {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
        embedder.embed(self.clone());
        Ok(())
    }
}

impl Embed for &str {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for i8 {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for i16 {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for i32 {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for i64 {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for i128 {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for f32 {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for f64 {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for bool {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for char {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for serde_json::Value {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
        embedder.embed(serde_json::to_string(self).map_err(EmbedError::new)?);
        Ok(())
    }
}

impl<T: Embed> Embed for Vec<T> {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
        for item in self {
            item.embed(embedder).map_err(EmbedError::new)?;
        }
        Ok(())
    }
}
