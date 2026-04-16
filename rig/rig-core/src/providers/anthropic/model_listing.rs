use crate::{
    client::ModelLister,
    http_client::{self, HttpClientExt},
    model::{Model, ModelList, ModelListingError},
    providers::anthropic::Client,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
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

/// [`ModelLister`] implementation for the Anthropic API (`GET /v1/models`).
///
/// Automatically paginates through all pages using cursor-based pagination.
#[derive(Clone)]
pub struct AnthropicModelLister<H = reqwest::Client> {
    client: Client<H>,
}

impl<H> ModelLister<H> for AnthropicModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    type Client = Client<H>;

    fn new(client: Self::Client) -> Self {
        Self { client }
    }

    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        let mut all_models = Vec::new();
        let mut after_id: Option<String> = None;

        loop {
            let path = match &after_id {
                Some(cursor) => format!("/v1/models?after_id={cursor}"),
                None => "/v1/models".to_string(),
            };

            let req = self.client.get(&path)?.body(http_client::NoBody)?;
            let response = self.client.send::<_, Vec<u8>>(req).await?;

            if !response.status().is_success() {
                let status_code = response.status().as_u16();
                let body = response.into_body().await?;
                return Err(ModelListingError::api_error_with_context(
                    "Anthropic",
                    &path,
                    status_code,
                    &body,
                ));
            }

            let body = response.into_body().await?;
            let page: ListModelsResponse = serde_json::from_slice(&body).map_err(|error| {
                ModelListingError::parse_error_with_context("Anthropic", &path, &error, &body)
            })?;

            all_models.extend(page.data.into_iter().map(Model::from));

            if !page.has_more {
                break;
            }

            after_id = page.last_id;
        }

        Ok(ModelList::new(all_models))
    }
}
