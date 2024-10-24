//! The module defines the [ExtractEmbeddingFields] trait, which must be implemented for types that can be embedded.

use crate::one_or_many::OneOrMany;

/// Error type used for when the `extract_embedding_fields` method fails.
/// Used by default implementations of `ExtractEmbeddingFields` for common types.
#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ExtractEmbeddingFieldsError(#[from] Box<dyn std::error::Error + Send + Sync>);

impl ExtractEmbeddingFieldsError {
    pub fn new<E: std::error::Error + Send + Sync + 'static>(error: E) -> Self {
        ExtractEmbeddingFieldsError(Box::new(error))
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
pub trait ExtractEmbeddingFields {
    type Error: std::error::Error + Sync + Send + 'static;

    fn extract_embedding_fields(&self) -> Result<OneOrMany<String>, Self::Error>;
}

// ================================================================
// Implementations of ExtractEmbeddingFields for common types
// ================================================================
impl ExtractEmbeddingFields for String {
    type Error = ExtractEmbeddingFieldsError;

    fn extract_embedding_fields(&self) -> Result<OneOrMany<String>, Self::Error> {
        Ok(OneOrMany::one(self.clone()))
    }
}

impl ExtractEmbeddingFields for i8 {
    type Error = ExtractEmbeddingFieldsError;

    fn extract_embedding_fields(&self) -> Result<OneOrMany<String>, Self::Error> {
        Ok(OneOrMany::one(self.to_string()))
    }
}

impl ExtractEmbeddingFields for i16 {
    type Error = ExtractEmbeddingFieldsError;

    fn extract_embedding_fields(&self) -> Result<OneOrMany<String>, Self::Error> {
        Ok(OneOrMany::one(self.to_string()))
    }
}

impl ExtractEmbeddingFields for i32 {
    type Error = ExtractEmbeddingFieldsError;

    fn extract_embedding_fields(&self) -> Result<OneOrMany<String>, Self::Error> {
        Ok(OneOrMany::one(self.to_string()))
    }
}

impl ExtractEmbeddingFields for i64 {
    type Error = ExtractEmbeddingFieldsError;

    fn extract_embedding_fields(&self) -> Result<OneOrMany<String>, Self::Error> {
        Ok(OneOrMany::one(self.to_string()))
    }
}

impl ExtractEmbeddingFields for i128 {
    type Error = ExtractEmbeddingFieldsError;

    fn extract_embedding_fields(&self) -> Result<OneOrMany<String>, Self::Error> {
        Ok(OneOrMany::one(self.to_string()))
    }
}

impl ExtractEmbeddingFields for f32 {
    type Error = ExtractEmbeddingFieldsError;

    fn extract_embedding_fields(&self) -> Result<OneOrMany<String>, Self::Error> {
        Ok(OneOrMany::one(self.to_string()))
    }
}

impl ExtractEmbeddingFields for f64 {
    type Error = ExtractEmbeddingFieldsError;

    fn extract_embedding_fields(&self) -> Result<OneOrMany<String>, Self::Error> {
        Ok(OneOrMany::one(self.to_string()))
    }
}

impl ExtractEmbeddingFields for bool {
    type Error = ExtractEmbeddingFieldsError;

    fn extract_embedding_fields(&self) -> Result<OneOrMany<String>, Self::Error> {
        Ok(OneOrMany::one(self.to_string()))
    }
}

impl ExtractEmbeddingFields for char {
    type Error = ExtractEmbeddingFieldsError;

    fn extract_embedding_fields(&self) -> Result<OneOrMany<String>, Self::Error> {
        Ok(OneOrMany::one(self.to_string()))
    }
}

impl ExtractEmbeddingFields for serde_json::Value {
    type Error = ExtractEmbeddingFieldsError;

    fn extract_embedding_fields(&self) -> Result<OneOrMany<String>, Self::Error> {
        Ok(OneOrMany::one(
            serde_json::to_string(self).map_err(ExtractEmbeddingFieldsError::new)?,
        ))
    }
}

impl<T: ExtractEmbeddingFields> ExtractEmbeddingFields for Vec<T> {
    type Error = ExtractEmbeddingFieldsError;

    fn extract_embedding_fields(&self) -> Result<OneOrMany<String>, Self::Error> {
        let items = self
            .iter()
            .map(|item| item.extract_embedding_fields())
            .collect::<Result<Vec<_>, _>>()
            .map_err(ExtractEmbeddingFieldsError::new)?;

        OneOrMany::merge(items).map_err(ExtractEmbeddingFieldsError::new)
    }
}
