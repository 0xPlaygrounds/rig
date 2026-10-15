//! Text extraction for embedding builders, including implementations for common types.
//!
//! ```
//! use rig_core::embeddings::to_texts;
//!
//! assert_eq!(to_texts("document")?, vec!["document"]);
//! # Ok::<(), rig_core::embeddings::EmbedError>(())
//! ```

/// Failure extracting text for embedding.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct EmbedError(#[from] Box<dyn std::error::Error + Send + Sync>);

impl EmbedError {
    pub fn new<E: std::error::Error + Send + Sync + 'static>(error: E) -> Self {
        EmbedError(Box::new(error))
    }
}

/// Extracts text fragments for vector embedding in append order.
/// Implementations report extraction failures through [`EmbedError`].
///
/// ```
/// use rig_core::{Embed, embeddings::{EmbedError, TextEmbedder, to_texts}};
///
/// struct Definitions(String);
/// impl Embed for Definitions {
///     fn embed(&self, out: &mut TextEmbedder) -> Result<(), EmbedError> {
///         for definition in self.0.split(',') {
///             out.embed(definition.trim().to_owned());
///         }
///         Ok(())
///     }
/// }
/// assert_eq!(to_texts(Definitions("fruit, company".into()))?, vec!["fruit", "company"]);
/// # Ok::<(), EmbedError>(())
/// ```
pub trait Embed {
    /// Append all text fragments that should be embedded for this value.
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError>;
}

/// Accumulates string values that need to be embedded.
/// Used by the [Embed] trait.
#[derive(Default)]
pub struct TextEmbedder {
    pub(crate) texts: Vec<String>,
}

impl TextEmbedder {
    /// Appends a text fragment without filtering or deduplication.
    pub fn embed(&mut self, text: String) {
        self.texts.push(text);
    }
}

/// Extracts text fragments in append order, propagating extraction errors.
pub fn to_texts(item: impl Embed) -> Result<Vec<String>, EmbedError> {
    let mut embedder = TextEmbedder::default();
    item.embed(&mut embedder)?;
    Ok(embedder.texts)
}

impl Embed for String {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(self.clone());
        Ok(())
    }
}

/// Embeds each type as its `Display` text.
macro_rules! embed_as_text {
    ($($type:ty),*) => {$(
        impl Embed for $type {
            fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
                embedder.embed(self.to_string());
                Ok(())
            }
        }
    )*};
}

embed_as_text!(&str, i8, i16, i32, i64, i128, f32, f64, bool, char);

impl Embed for serde_json::Value {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        embedder.embed(serde_json::to_string(self).map_err(EmbedError::new)?);
        Ok(())
    }
}

impl<T: Embed> Embed for &T {
    fn embed(&self, embedder: &mut TextEmbedder) -> Result<(), EmbedError> {
        (*self).embed(embedder)
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
