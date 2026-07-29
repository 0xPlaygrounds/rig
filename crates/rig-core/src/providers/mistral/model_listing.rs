use crate::{
    client::ModelLister,
    http_client::{self, HttpClientExt},
    model::{Model, ModelList, ModelListingError},
    providers::mistral::Client,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ListModelsResponse {
    data: Vec<ListModelEntry>,
}

#[derive(Debug, Deserialize)]
struct ListModelEntry {
    id: String,
    name: Option<String>,
    created: u64,
    owned_by: String,
}

impl From<ListModelEntry> for Model {
    fn from(value: ListModelEntry) -> Self {
        let mut model = Model::from_id(value.id);
        model.name = value.name;
        model.created_at = Some(value.created);
        model.owned_by = Some(value.owned_by);
        model
    }
}

/// [`ModelLister`] implementation for the Mistral API (`GET /v1/models`).
#[derive(Clone)]
pub struct MistralModelLister<H = reqwest::Client> {
    client: Client<H>,
}

impl<H> ModelLister<H> for MistralModelLister<H>
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
pub(crate) const LIST_MODELS_PATH: &str = "/v1/models";

/// Parse a `GET /v1/models` response into a [`ModelList`]. Pure.
///
/// Shared by the classic [`MistralModelLister`] and
/// [`functions::list_models`](super::functions::list_models).
pub(crate) fn parse_list_models_response(
    status: http::StatusCode,
    body: &[u8],
) -> Result<ModelList, ModelListingError> {
    if !status.is_success() {
        return Err(ModelListingError::api_error_with_context(
            "Mistral",
            LIST_MODELS_PATH,
            status.as_u16(),
            body,
        ));
    }
    let api_resp: ListModelsResponse = serde_json::from_slice(body).map_err(|error| {
        ModelListingError::parse_error_with_context("Mistral", LIST_MODELS_PATH, &error, body)
    })?;
    let models = api_resp.data.into_iter().map(Model::from).collect();
    Ok(ModelList::new(models))
}
