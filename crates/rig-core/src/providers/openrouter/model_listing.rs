use crate::{
    model::Model,
    providers::{internal, openrouter::Client},
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ModelEntry {
    id: String,
    name: String,
    description: Option<String>,
    created: u64,
    context_length: Option<u32>,
}

impl From<ModelEntry> for Model {
    fn from(value: ModelEntry) -> Self {
        Model {
            id: value.id,
            name: Some(value.name),
            description: value.description,
            r#type: None,
            created_at: Some(value.created),
            owned_by: None,
            context_length: value.context_length,
            // OpenRouter reports an output ceiling under
            // `top_provider.max_completion_tokens`, which this entry does not
            // parse. Left unreported rather than guessed (rig#2322).
            max_output_tokens: None,
        }
    }
}

internal::model_listing::impl_model_lister!(
    /// [`ModelLister`](crate::client::ModelLister) implementation for the
    /// OpenRouter API (`GET /models`).
    OpenRouterModelLister,
    Client<H>,
    ModelEntry,
    "OpenRouter",
    "/models"
);
