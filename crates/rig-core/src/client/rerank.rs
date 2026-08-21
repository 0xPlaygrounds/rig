use crate::rerank::RerankModel;

/// A provider client with reranking capabilities.
pub trait RerankingClient {
    /// The type of [`RerankModel`] used by the Client.
    type RerankModel: RerankModel;

    /// Create a reranking model with the given model identifier.
    fn rerank_model(&self, model: impl Into<String>) -> Self::RerankModel;
}

/// Construction hook for the blanket [`RerankingClient`] implementation over
/// [`crate::client::Client`] — the rerank twin of
/// [`crate::client::ConstructCompletionModel`].
///
/// Public for the same reason: an out-of-tree provider extension built on the
/// generic `Client<Ext, H>` cannot implement [`RerankingClient`] for that foreign
/// type (orphan rule), so it implements this trait on its own model type and
/// the blanket implementation supplies the constructor. Providers with their
/// own client type implement [`RerankingClient`] directly and never need this.
pub trait ConstructRerankModel<C>: Sized {
    /// Build this model from its provider client and a model identifier.
    fn construct(client: &C, model: String) -> Self;
}
