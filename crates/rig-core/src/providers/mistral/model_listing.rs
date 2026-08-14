use crate::providers::{
    internal::model_listing::{ListModelEntry, impl_model_lister},
    mistral::Client,
};

impl_model_lister!(
    /// [`ModelLister`](crate::client::ModelLister) implementation for the
    /// Mistral API (`GET /v1/models`).
    MistralModelLister,
    Client<H>,
    ListModelEntry,
    "Mistral",
    "/v1/models"
);
