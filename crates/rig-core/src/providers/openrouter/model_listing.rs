use crate::{
    client::ModelLister,
    http_client::{self, HttpClientExt},
    model::{Model, ModelList, ModelListingError},
    providers::openrouter::Client,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
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

#[derive(Clone)]
pub struct OpenRouterModelLister<H = reqwest::Client> {
    client: Client<H>,
}

impl<H> ModelLister<H> for OpenRouterModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    type Client = Client<H>;

    fn new(client: Self::Client) -> Self {
        Self { client }
    }

    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        let req = self
            .client
            .get(LIST_MODELS_PATH)?
            .body(http_client::NoBody)?;
        let response = self.client.send::<_, Vec<u8>>(req).await?;
        let status = response.status();
        let body = response.into_body().await?;
        parse_list_models_response(status, &body)
    }
}

/// Path of the model-listing endpoint, relative to the API base URL.
pub(crate) const LIST_MODELS_PATH: &str = "/models";

/// Parse a `GET /models` response into a [`ModelList`]. Pure.
///
/// Shared by the classic [`OpenRouterModelLister`] and
/// [`functions::list_models`](super::functions::list_models).
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
