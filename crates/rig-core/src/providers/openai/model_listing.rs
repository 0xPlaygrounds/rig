use crate::providers::{
    internal::model_listing::{ListModelEntry, impl_model_lister},
    openai::Client,
};

impl_model_lister!(
    /// [`ModelLister`](crate::client::ModelLister) implementation for the
    /// OpenAI API (`GET /models`).
    OpenAIModelLister,
    Client<H>,
    ListModelEntry,
    "OpenAI",
    "/models"
);
