use crate::rerank::RerankModel;

/// A provider client with reranking capabilities.
pub trait RerankingClient {
    /// The type of [`RerankModel`] used by the Client.
    type RerankModel: RerankModel;

    /// Create a reranking model with the given model identifier.
    fn rerank_model(&self, model: impl Into<String>) -> Self::RerankModel;
}
