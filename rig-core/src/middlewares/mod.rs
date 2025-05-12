use thiserror::Error;

use crate::{
    completion::CompletionError, extractor::ExtractionError, tool::ToolSetError,
    vector_store::VectorStoreError,
};

pub mod build_completions;
pub mod completion;
pub mod components;
pub mod extractor;
pub mod parallel;
pub mod rag;
pub mod tools;

#[derive(Debug, Error)]
pub enum ServiceError {
    #[error("{0}")]
    ExtractionError(#[from] ExtractionError),
    #[error("{0}")]
    CompletionError(#[from] CompletionError),
    #[error("{0}")]
    ToolSetError(#[from] ToolSetError),
    #[error("{0}")]
    VectorStoreError(#[from] VectorStoreError),
    #[error("Value required but was null: {0}")]
    RequiredOptionNotFound(String),
    #[error("{0}")]
    Json(#[from] serde_json::Error),
    #[error("Custom error: {0}")]
    Other(#[from] Box<dyn std::error::Error + Send + Sync>),
}

impl ServiceError {
    pub fn required_option_not_exists<S: Into<String>>(val: S) -> Self {
        let val: String = val.into();
        Self::RequiredOptionNotFound(val)
    }
}
