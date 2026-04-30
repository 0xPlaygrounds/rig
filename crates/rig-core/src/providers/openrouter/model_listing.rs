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
        let path = "/models";
        let req = self.client.get(path)?.body(http_client::NoBody)?;
        let response = self.client.send::<_, Vec<u8>>(req).await?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let body = response.into_body().await?;
            return Err(ModelListingError::api_error_with_context(
                "OpenRouter",
                path,
                status_code,
                &body,
            ));
        }

        let body = response.into_body().await?;
        let api_resp: ListModelsResponse = serde_json::from_slice(&body).map_err(|error| {
            ModelListingError::parse_error_with_context("OpenRouter", path, &error, &body)
        })?;
        let models = api_resp.data.into_iter().map(Model::from).collect();

        Ok(ModelList::new(models))
    }
}
