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
mod tests {
    use super::*;

    /// Derived from a live `GET /v1/models` entry: Mistral names the context
    /// window `max_context_length` and carries a `description`, neither of
    /// which the shared OpenAI-shaped entry models.
    #[test]
    fn entry_maps_mistrals_own_fields_onto_model() {
        let entry: MistralModelEntry = serde_json::from_value(serde_json::json!({
            "id": "mistral-small-latest",
            "object": "model",
            "created": 1786767624,
            "owned_by": "mistralai",
            "name": "mistral-small-2603",
            "description": "Mistral Small 4.",
            "max_context_length": 262144,
            "type": "base",
            "capabilities": {"completion_chat": true, "vision": true}
        }))
        .expect("a live listing entry should deserialize");

        let model = Model::from(entry);
        assert_eq!(model.id, "mistral-small-latest");
        assert_eq!(model.name.as_deref(), Some("mistral-small-2603"));
        assert_eq!(model.description.as_deref(), Some("Mistral Small 4."));
        assert_eq!(model.context_length, Some(262_144));
        assert_eq!(model.r#type.as_deref(), Some("base"));
        assert_eq!(model.owned_by.as_deref(), Some("mistralai"));
        assert_eq!(model.created_at, Some(1_786_767_624));
    }

    /// Every field but `id` is optional, so an entry that carries only what
    /// the shared shape carries still lists.
    #[test]
    fn entry_tolerates_a_minimal_listing() {
        let entry: MistralModelEntry =
            serde_json::from_value(serde_json::json!({"id": "mistral-embed"}))
                .expect("a minimal entry should deserialize");

        let model = Model::from(entry);
        assert_eq!(model.id, "mistral-embed");
        assert_eq!(model.context_length, None);
        assert_eq!(model.description, None);
    }
}
