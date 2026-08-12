use crate::{
    client::ModelLister,
    http_client::HttpClientExt,
    model::{ModelList, ModelListingError},
    providers::{internal, internal::model_listing::ListModelEntry, openai::Client},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

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
