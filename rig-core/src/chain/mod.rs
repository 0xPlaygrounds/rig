pub mod chain;
pub mod try_chain;

pub use chain::Chain;
pub use try_chain::{Empty, TryChain};

use crate::{completion, vector_store};

#[derive(Debug, thiserror::Error)]
pub enum ChainError {
    #[error("Failed to prompt agent: {0}")]
    PromptError(#[from] completion::PromptError),

    #[error("Failed to lookup documents: {0}")]
    LookupError(#[from] vector_store::VectorStoreError),
}

pub fn new<T: Send + Sync>() -> Empty<T, ChainError> {
    Empty::default()
}

pub fn with_error<T: Send + Sync, E: Send + Sync>() -> Empty<T, E> {
    Empty::default()
}
