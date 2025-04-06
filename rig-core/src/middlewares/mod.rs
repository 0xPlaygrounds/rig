use thiserror::Error;

use crate::{completion::CompletionError, extractor::ExtractionError, tool::ToolSetError};

pub mod completion;
pub mod components;
pub mod extractor;
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
}
