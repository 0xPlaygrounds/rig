use crate::{
    client::ModelLister,
    http_client::{self, HttpClientExt},
    model::{Model, ModelList, ModelListingError},
    providers::mistral::Client,
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
        Model {
            id: value.id,
            name: value.name,
            created_at: Some(value.created),
            owned_by: Some(value.owned_by),
            ..Default::default()
        }
    }
}

/// [`ModelLister`] implementation for the Mistral API (`GET /v1/models`).
#[derive(Clone)]
pub struct MistralModelLister<H = reqwest::Client> {
    client: Client<H>,
}

impl<H> ModelLister<H> for MistralModelLister<H>
where
    H: HttpClientExt + Send + Sync + 'static,
{
    type Client = Client<H>;

    fn new(client: Self::Client) -> Self {
        Self { client }
    }

    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        let req = self.client.get("/v1/models")?.body(http_client::NoBody)?;
        let response = self.client.send(req).await?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let text = http_client::text(response).await?;
            return Err(ModelListingError::api_error(status_code, text));
        }

        let body = response.into_body().await?;
        let api_resp: ListModelsResponse = serde_json::from_slice(&body)?;
        let models = api_resp.data.into_iter().map(Model::from).collect();

        Ok(ModelList::new(models))
    }
}
