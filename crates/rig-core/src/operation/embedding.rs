use crate::embeddings::{EmbeddingError, EmbeddingResponse};

use super::{Operation, TakeOne};

/// The text embedding operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Embedding;

impl Operation for Embedding {
    type Request = Vec<String>;
    type Event = EmbeddingResponse;
    type Response = EmbeddingResponse;
    type Error = EmbeddingError;
    type Capabilities = ();
    type Fold = TakeOne<EmbeddingResponse, EmbeddingError>;
}
