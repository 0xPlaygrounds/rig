use super::{Client, ModelTransport, Provider};
use crate::rerank::RerankModel;

/// A provider client with reranking capabilities.
pub trait RerankingClient {
    /// The type of [`RerankModel`] used by the Client.
    type RerankModel: RerankModel;

    /// Create a reranking model with the given model identifier.
    fn rerank_model(&self, model: impl Into<String>) -> Self::RerankModel;
}

/// A [`Provider`] that offers rerank models. Implementing this is what makes
/// [`RerankingClient`] available on `Client<Self, H>`.
pub trait HasRerank: Provider {
    /// The concrete rerank model built over transport `H`.
    type Model<H>: RerankModel
    where
        H: ModelTransport;

    /// Build the rerank model `model` from `client`.
    fn rerank_model<H>(client: &Client<Self, H>, model: String) -> Self::Model<H>
    where
        H: ModelTransport;
}

impl<P, H> RerankingClient for Client<P, H>
where
    P: HasRerank,
    H: ModelTransport,
{
    type RerankModel = P::Model<H>;

    fn rerank_model(&self, model: impl Into<String>) -> Self::RerankModel {
        P::rerank_model(self, model.into())
    }
}
