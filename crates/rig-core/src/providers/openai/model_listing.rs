use crate::{
    client::ModelLister,
    http_client::HttpClientExt,
    model::{Model, ModelList, ModelListingError},
    providers::{internal, openai::Client},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ListModelEntry {
    id: String,
    created: u64,
    owned_by: String,
}

impl From<ListModelEntry> for Model {
    fn from(value: ListModelEntry) -> Self {
        let mut model = Model::from_id(value.id);
        model.created_at = Some(value.created);
        model.owned_by = Some(value.owned_by);
        model
    }
}

/// [`ModelLister`] implementation for the OpenAI API (`GET /models`).
#[derive(Clone)]
pub struct OpenAIModelLister<H = reqwest::Client> {
    client: Client<H>,
}

impl<H> ModelLister<H> for OpenAIModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    type Client = Client<H>;

    fn new(client: Self::Client) -> Self {
        Self { client }
    }

    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        internal::model_listing::list_models::<ListModelEntry, _, _>(
            &self.client,
            "OpenAI",
            "/models",
        )
        .await
    }
}
