//! The module defines the [Embed] trait, which must be implemented for types that can be embedded by the `EmbeddingsBuilder`.

/// Error type used for when the `embed` method fo the `Embed` trait fails.
/// Used by default implementations of `Embed` for common types.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct EmbedError(#[from] Box<dyn std::error::Error + Send + Sync>);

impl EmbedError {
    pub fn new<E: std::error::Error + Send + Sync + 'static>(error: E) -> Self {
        EmbedError(Box::new(error))
    }
}

/// Derive this trait for objects that need to be converted to vector embeddings.
/// The `embed` method accumulates string values that need to be embedded by adding them to the `TextEmbedder`.
/// If an error occurs, the method should return `EmbedError`.
/// # Example
/// ```rust
/// use std::env;
///
/// use serde::{Deserialize, Serialize};
/// use rig::Embed;
///
/// struct FakeDefinition {
///     id: String,
///     word: String,
///     definitions: String,
/// }
///
/// impl Embed for FakeDefinition {
///     fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
///        // Embeddings only need to be generated for `definition` field.
///        // Split the definitions by comma and collect them into a vector of strings.
///        // That way, different embeddings can be generated for each definition in the definitions string.
///        self.definitions
///            .split(",")
///            .for_each(|s| {
///                embedder.embed(s.to_string());
///            });
///
///        Ok(())
///     }
/// }
/// ```
pub trait Embed {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError>;
}

/// Accumulates string values that need to be embedded.
/// Used by the `Embed` trait.
#[derive(Default)]
pub struct TextEmbedder {
    pub(crate) texts: Vec<String>,
}

impl TextEmbedder {
    pub fn embed(&mut self, text: String) {
        self.texts.push(text);
    }
}

/// Client-side function to convert an object that implements the `Embed` trait to a vector of strings.
/// Similar to `serde`'s `serde_json::to_string()` function
pub fn to_texts(item: impl Embed) -> Result<Vec<String>, EmbedError> {
    let mut embedder = TextEmbedder::default();
    item.embed(&mut embedder)?;
    Ok(embedder.texts)
}

// ================================================================
// Implementations of Embed for common types
// ================================================================

impl Embed for String {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.clone());
        Ok(())
    }
}

impl Embed for &str {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for i8 {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for i16 {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for i32 {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for i64 {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for i128 {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for f32 {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for f64 {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for bool {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for char {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.to_string());
        Ok(())
    }
}

impl Embed for serde_json::Value {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(serde_json::to_string(self).map_err(EmbedError::new)?);
        Ok(())
    }
}

impl<T: Embed> Embed for Vec<T> {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        for item in self {
            item.embed(embedder).map_err(EmbedError::new)?;
        }
        Ok(())
    }
}
