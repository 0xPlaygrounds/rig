use crate::{
    client::{self, ModelLister, Provider},
    http_client::{self, HttpClientExt},
    model::{Model, ModelList, ModelListingError},
    providers::gemini::{Client, InteractionsClient},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use serde::Deserialize;
use std::convert::TryFrom;

const MAX_PAGE_SIZE: usize = 1000;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListModelsResponse {
    models: Vec<ListModelEntry>,
    next_page_token: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListModelEntry {
    base_model_id: String,
    display_name: Option<String>,
    description: Option<String>,
    input_token_limit: Option<u64>,
}

impl From<ListModelEntry> for Model {
    fn from(value: ListModelEntry) -> Self {
        let mut model = Model::from_id(value.base_model_id);
        model.name = value.display_name;
        model.description = value.description;
        model.context_length = value
            .input_token_limit
            .and_then(|limit| u32::try_from(limit).ok());
        model
    }
}

fn list_models_path(page_token: Option<&str>) -> String {
    let mut serializer = url::form_urlencoded::Serializer::new(String::new());
    serializer.append_pair("pageSize", &MAX_PAGE_SIZE.to_string());

    if let Some(page_token) = page_token {
        serializer.append_pair("pageToken", page_token);
    }

    format!("/v1beta/models?{}", serializer.finish())
}

async fn list_all_models<Ext, H>(
    client: &client::Client<Ext, H>,
) -> Result<ModelList, ModelListingError>
where
    Ext: Provider + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    let mut all_models = Vec::new();
    let mut next_page_token: Option<String> = None;

    loop {
        let path = list_models_path(next_page_token.as_deref());
        let req = client.get(&path)?.body(http_client::NoBody)?;
        let response = client.send(req).await?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let text = http_client::text(response).await?;
            return Err(ModelListingError::api_error(status_code, text));
        }

        let body = response.into_body().await?;
        let page: ListModelsResponse = serde_json::from_slice(&body)?;

        all_models.extend(page.models.into_iter().map(Model::from));

        if page.next_page_token.is_none() {
            break;
        }

        next_page_token = page.next_page_token;
    }

    Ok(ModelList::new(all_models))
}

/// [`ModelLister`] implementation for Gemini GenerateContent clients.
#[derive(Clone)]
pub struct GeminiModelLister<H = reqwest::Client> {
    client: Client<H>,
}

impl<H> ModelLister<H> for GeminiModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    type Client = Client<H>;

    fn new(client: Self::Client) -> Self {
        Self { client }
    }

    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        list_all_models(&self.client).await
    }
}

/// [`ModelLister`] implementation for Gemini Interactions API clients.
#[derive(Clone)]
pub struct GeminiInteractionsModelLister<H = reqwest::Client> {
    client: InteractionsClient<H>,
}

impl<H> ModelLister<H> for GeminiInteractionsModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    type Client = InteractionsClient<H>;

    fn new(client: Self::Client) -> Self {
        Self { client }
    }

    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        list_all_models(&self.client).await
    }
}
