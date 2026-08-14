use crate::providers::{
    internal::model_listing::{ListModelEntry, impl_model_lister},
    openai::{Client, CompletionsClient},
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

impl_model_lister!(
    /// OpenAI model lister for a client using the Chat Completions extension.
    OpenAICompletionsModelLister,
    CompletionsClient<H>,
    ListModelEntry,
    "OpenAI",
    "/models"
);
