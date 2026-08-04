use crate::model::{Model, ModelListingError};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ListModelsResponse {
    data: Vec<ListModelEntry>,
    has_more: bool,
    last_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ListModelEntry {
    id: String,
    display_name: String,
}

impl From<ListModelEntry> for Model {
    fn from(value: ListModelEntry) -> Self {
        Model::new(value.id, value.display_name)
    }
}

/// Path of one model-listing page, relative to the API base URL. Pure.
pub(crate) fn list_models_path(after_id: Option<&str>) -> String {
    match after_id {
        Some(cursor) => format!("/v1/models?after_id={cursor}"),
        None => "/v1/models".to_string(),
    }
}

/// Parse one `GET /v1/models` page into its models plus the next-page
/// cursor (`None` when this is the last page). Pure.
///
/// Used by [`functions::list_models`](super::functions::list_models).
pub(crate) fn parse_models_page(
    path: &str,
    status: http::StatusCode,
    body: &[u8],
) -> Result<(Vec<Model>, Option<String>), ModelListingError> {
    if !status.is_success() {
        return Err(ModelListingError::api_error_with_context(
            "Anthropic",
            path,
            status.as_u16(),
            body,
        ));
    }
    let page: ListModelsResponse = serde_json::from_slice(body).map_err(|error| {
        ModelListingError::parse_error_with_context("Anthropic", path, &error, body)
    })?;
    let models = page.data.into_iter().map(Model::from).collect();
    let next_after_id = if page.has_more { page.last_id } else { None };
    Ok((models, next_after_id))
}
