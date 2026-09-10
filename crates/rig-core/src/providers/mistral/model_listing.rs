use crate::model::Model;
use crate::providers::{internal::model_listing::impl_model_lister, mistral::Client};
use serde::Deserialize;

/// One entry of Mistral's `GET /v1/models` response.
///
/// Mistral's listing carries more than the shared OpenAI-shaped entry models:
/// a human `description` and the model's `max_context_length`, both of which
/// [`Model`] has slots for. Using the shared entry here dropped them on the
/// floor, so this provider keeps its own.
#[derive(Debug, Deserialize)]
pub(crate) struct MistralModelEntry {
    id: String,
    #[serde(default)]
    name: Option<String>,
    #[serde(default)]
    description: Option<String>,
    #[serde(default)]
    created: Option<u64>,
    #[serde(default)]
    owned_by: Option<String>,
    /// Mistral's spelling of the context window, in tokens.
    #[serde(default)]
    max_context_length: Option<u32>,
    /// Mistral labels the model kind `type` (e.g. `base`, `fine-tuned`).
    #[serde(default, rename = "type")]
    kind: Option<String>,
}

impl From<MistralModelEntry> for Model {
    fn from(value: MistralModelEntry) -> Self {
        let mut model = Model::from_id(value.id);
        model.name = value.name;
        model.description = value.description;
        model.created_at = value.created;
        model.owned_by = value.owned_by;
        model.context_length = value.max_context_length;
        model.r#type = value.kind;
        model
    }
}

impl_model_lister!(
    /// [`ModelLister`](crate::client::ModelLister) implementation for the
    /// Mistral API (`GET /v1/models`).
    MistralModelLister,
    Client<H>,
    MistralModelEntry,
    "Mistral",
    "/v1/models"
);

#[cfg(test)]
mod tests;
