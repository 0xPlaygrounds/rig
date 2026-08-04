use crate::model::{Model, ModelList, ModelListingError};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ListModelsResponse {
    data: Vec<ModelEntry>,
}

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
        }
    }
}

/// Path of the model-listing endpoint, relative to the API base URL.
pub(crate) const LIST_MODELS_PATH: &str = "/models";

/// Parse a `GET /models` response into a [`ModelList`]. Pure.
///
/// Used by [`functions::list_models`](super::functions::list_models).
pub(crate) fn parse_list_models_response(
    status: http::StatusCode,
    body: &[u8],
) -> Result<ModelList, ModelListingError> {
    if !status.is_success() {
        return Err(ModelListingError::api_error_with_context(
            "OpenRouter",
            LIST_MODELS_PATH,
            status.as_u16(),
            body,
        ));
    }
    let api_resp: ListModelsResponse = serde_json::from_slice(body).map_err(|error| {
        ModelListingError::parse_error_with_context("OpenRouter", LIST_MODELS_PATH, &error, body)
    })?;
    let models = api_resp.data.into_iter().map(Model::from).collect();
    Ok(ModelList::new(models))
}
