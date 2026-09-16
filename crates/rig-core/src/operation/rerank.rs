use crate::rerank::{RerankError, RerankResponse};

use super::{Operation, TakeOne};

/// The input to a reranking operation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RerankRequest {
    pub query: String,
    pub documents: Vec<String>,
    pub top_n: Option<usize>,
}

/// The document reranking operation.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Rerank;

impl Operation for Rerank {
    type Request = RerankRequest;
    type Event = RerankResponse;
    type Response = RerankResponse;
    type Error = RerankError;
    type Capabilities = ();
    type Fold = TakeOne<RerankResponse, RerankError>;
}
