//! Provider-agnostic embedding abstractions.
//!
//! Embeddings are numerical representations of text or other inputs. Rig uses
//! an embedding [`Model`](crate::Model) to generate vectors, [`Embed`] to
//! select which text from a Rust value should be embedded, and
//! [`EmbeddingsBuilder`] to batch embedding requests for vector stores or
//! retrieval workflows.
//!
//! Every provider's `embedding(model, ndims)` reads `ndims` the same way.
//! `Some(n)` asks for `n`-wide vectors where the provider's API takes a
//! width, and a reply of any other width fails with
//! [`ProviderError::MismatchedDimensions`](crate::error::ProviderError::MismatchedDimensions).
//! `None` takes the model's default width, reported when Rig knows it and
//! zero otherwise.
//!
//! ```
//! use rig_core::embeddings::to_texts;
//!
//! assert_eq!(to_texts(vec!["first", "second"])?, vec!["first", "second"]);
//! # Ok::<(), rig_core::embeddings::EmbedError>(())
//! ```

pub mod builder;
pub mod embed;
pub mod embedding;
pub mod tool;

pub mod distance;
pub use builder::EmbeddingsBuilder;
pub use embed::{Embed, EmbedError, TextEmbedder, to_texts};
pub use embedding::*;
pub use tool::ToolSchema;
