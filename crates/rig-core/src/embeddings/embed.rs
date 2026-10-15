//! Input extraction for embedding builders, including implementations for common types.
//!
//! ```
//! use rig_core::embeddings::to_texts;
//!
//! assert_eq!(to_texts("document")?, vec!["document"]);
//! # Ok::<(), rig_core::embeddings::EmbedError>(())
//! ```

/// Failure extracting text or images for embedding.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct EmbedError(#[from] Box<dyn std::error::Error + Send + Sync>);

impl EmbedError {
    pub fn new<E: std::error::Error + Send + Sync + 'static>(error: E) -> Self {
        EmbedError(Box::new(error))
    }
}

/// Extracts the texts and images to embed for a value, in append order.
/// Implementations report extraction failures through [`EmbedError`].
///
/// ```
/// use rig_core::{Embed, embeddings::{EmbedError, Embedder, to_texts}};
///
/// struct Definitions(String);
/// impl Embed for Definitions {
///     fn embed(&self, out: &mut Embedder) -> Result<(), EmbedError> {
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
    /// Append every text and image that should be embedded for this value.
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError>;
}

/// One input to embed.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) enum EmbedInput {
    /// A text, for the text embedding model.
    Text(String),
    /// An image's encoded file bytes, for the image embedding model.
    Image(Vec<u8>),
}

/// Accumulates the texts and images a value embeds. Used by the [Embed]
/// trait.
#[derive(Default)]
pub struct Embedder {
    pub(crate) inputs: Vec<EmbedInput>,
}

impl Embedder {
    /// Appends a text without filtering or deduplication.
    pub fn embed(&mut self, text: String) {
        self.inputs.push(EmbedInput::Text(text));
    }

    /// Appends an image's encoded file bytes (PNG, JPEG, ...). Embedding it
    /// needs an image model on the builder.
    pub fn image(&mut self, bytes: Vec<u8>) {
        self.inputs.push(EmbedInput::Image(bytes));
    }
}

/// A value embedded an image where only texts were expected.
#[derive(Debug, thiserror::Error)]
#[error("the value embeds an image, which has no text form")]
struct ImageNotText;

/// Extracts texts in append order, propagating extraction errors. A value
/// that embeds an image is an error.
pub fn to_texts(item: impl Embed) -> Result<Vec<String>, EmbedError> {
    let mut embedder = Embedder::default();
    item.embed(&mut embedder)?;
    embedder
        .inputs
        .into_iter()
        .map(|input| match input {
            EmbedInput::Text(text) => Ok(text),
            EmbedInput::Image(_) => Err(EmbedError::new(ImageNotText)),
        })
        .collect()
}

/// An image that cannot be embedded from its bytes.
#[derive(Debug, thiserror::Error)]
enum ImageBytesError {
    /// The image is a reference rather than inline bytes.
    #[error("image {0} is not inline, so it has no bytes to embed")]
    NotInline(String),
    /// The inline base64 does not decode.
    #[error("image data is not valid base64: {0}")]
    Base64(#[from] base64::DecodeError),
}

/// The encoded file bytes an inline image carries.
fn image_bytes(image: &crate::message::Image) -> Result<Vec<u8>, ImageBytesError> {
    use crate::message::DocumentSourceKind;
    use base64::Engine as _;
    match &image.data {
        DocumentSourceKind::Base64(data) => {
            Ok(base64::engine::general_purpose::STANDARD.decode(data)?)
        }
        DocumentSourceKind::Raw(bytes) => Ok(bytes.clone()),
        other => Err(ImageBytesError::NotInline(other.to_string())),
    }
}

/// A context document embeds its text, when it has any, then each of its
/// images, which must carry their bytes inline.
impl Embed for crate::completion::Document {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
        if !self.text.is_empty() {
            embedder.embed(self.text.clone());
        }
        for image in &self.images {
            embedder.image(image_bytes(image).map_err(EmbedError::new)?);
        }
        Ok(())
    }
}

impl Embed for String {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
        embedder.embed(self.clone());
        Ok(())
    }
}

/// Embeds each type as its `Display` text.
macro_rules! embed_as_text {
    ($($type:ty),*) => {$(
        impl Embed for $type {
            fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
                embedder.embed(self.to_string());
                Ok(())
            }
        }
    )*};
}

embed_as_text!(&str, i8, i16, i32, i64, i128, f32, f64, bool, char);

impl Embed for serde_json::Value {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
        embedder.embed(serde_json::to_string(self).map_err(EmbedError::new)?);
        Ok(())
    }
}

impl<T: Embed> Embed for &T {
    fn embed(&self, embedder: &mut Embedder) -> Result<(), EmbedError> {
        (*self).embed(embedder)
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
