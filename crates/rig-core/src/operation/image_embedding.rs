use crate::embeddings::{EmbeddingError, ImageEmbeddingResponse};

use super::{Operation, TakeOne};

/// The image embedding operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct ImageEmbedding;

impl Operation for ImageEmbedding {
    type Request = Vec<Vec<u8>>;
    type Event = ImageEmbeddingResponse;
    type Response = ImageEmbeddingResponse;
    type Error = EmbeddingError;
    type Capabilities = ();
    type Fold = TakeOne<ImageEmbeddingResponse, EmbeddingError>;
}
