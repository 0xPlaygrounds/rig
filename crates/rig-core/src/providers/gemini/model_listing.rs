use crate::{
    client::{self, ModelLister, Provider},
    http_client::HttpClientExt,
    model::{Model, ModelList, ModelListingError},
    providers::{
        gemini::{Client, InteractionsClient},
        internal,
    },
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use serde::Deserialize;
use std::{convert::TryFrom, fmt};

const MAX_PAGE_SIZE: usize = 1000;

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListModelsResponse {
    #[serde(default)]
    models: Vec<ListModelEntry>,
    next_page_token: Option<String>,
}

#[derive(Debug, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ListModelEntry {
    #[serde(default)]
    name: String,
    base_model_id: Option<String>,
    display_name: Option<String>,
    description: Option<String>,
    input_token_limit: Option<u64>,
    /// The model's output ceiling. Gemini reports this for every model
    /// (`gemini-2.5-flash`: 65536) and rig used to drop it on the floor, which
    /// is why a hardcoded 4096 default went unnoticed for so long — nothing in
    /// the library ever knew the real limit was ~16x larger (rig#2322).
    output_token_limit: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct MissingModelIdError;

impl fmt::Display for MissingModelIdError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "parse_error=model entry missing usable `baseModelId` and `name` values"
        )
    }
}

impl std::error::Error for MissingModelIdError {}

fn normalize_gemini_model_id(name: &str) -> Option<String> {
    let trimmed = name.trim();
    let trimmed = trimmed.strip_prefix("models/").unwrap_or(trimmed);

    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

impl TryFrom<ListModelEntry> for Model {
    type Error = MissingModelIdError;

    fn try_from(value: ListModelEntry) -> Result<Self, Self::Error> {
        let id = value
            .base_model_id
            .as_deref()
            .map(str::trim)
            .filter(|id| !id.is_empty())
            .map(str::to_owned)
            .or_else(|| normalize_gemini_model_id(&value.name))
            .ok_or(MissingModelIdError)?;

        let mut model = Model::from_id(id);
        model.name = value.display_name;
        model.description = value.description;
        model.context_length = value
            .input_token_limit
            .and_then(|limit| u32::try_from(limit).ok());
        model.max_output_tokens = value
            .output_token_limit
            .and_then(|limit| u32::try_from(limit).ok());
        Ok(model)
    }
}

fn list_models_path(page_token: Option<&str>) -> String {
    let page_size = MAX_PAGE_SIZE.to_string();
    let mut pairs = vec![("pageSize", page_size.as_str())];
    if let Some(page_token) = page_token {
        pairs.push(("pageToken", page_token));
    }
    internal::model_listing::with_query_pairs("/v1beta/models", &pairs)
}

fn parse_models_page(
    body: &[u8],
    path: &str,
) -> Result<internal::model_listing::ListingPage, ModelListingError> {
    let page: ListModelsResponse = serde_json::from_slice(body).map_err(|error| {
        ModelListingError::parse_error_with_context("Gemini", path, &error, body)
    })?;

    let models = page
        .models
        .into_iter()
        .map(|entry| {
            Model::try_from(entry).map_err(|error| {
                ModelListingError::parse_error_with_details("Gemini", path, error, body)
            })
        })
        .collect::<Result<Vec<_>, _>>()?;

    // An empty cursor counts as absent, matching how every other
    // provider-reported identifier in rig is read. Reporting `Some("")` would
    // tell the shared loop there is a next page, and re-sending an empty
    // `pageToken` returns the same page forever.
    Ok(internal::model_listing::ListingPage {
        models,
        next_cursor: page.next_page_token.filter(|token| !token.is_empty()),
    })
}

async fn list_all_models<Ext, H>(
    client: &client::Client<Ext, H>,
) -> Result<ModelList, ModelListingError>
where
    Ext: Provider + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    internal::model_listing::paginate_models(client, "Gemini", list_models_path, parse_models_page)
        .await
}

/// [`ModelLister`] implementation for Gemini GenerateContent clients.
#[derive(Clone)]
pub struct GeminiModelLister<H> {
    client: Client<H>,
}

impl<H> ModelLister<H> for GeminiModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        list_all_models(&self.client).await
    }
}

impl<H> crate::client::ConstructModelLister<Client<H>> for GeminiModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static + Clone,
{
    fn construct(client: &Client<H>) -> Self {
        let client = client.clone();
        Self { client }
    }
}

#[cfg(test)]
mod tests;

/// [`ModelLister`] implementation for Gemini Interactions API clients.
#[derive(Clone)]
pub struct GeminiInteractionsModelLister<H> {
    client: InteractionsClient<H>,
}

impl<H> ModelLister<H> for GeminiInteractionsModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        list_all_models(&self.client).await
    }
}

impl<H> crate::client::ConstructModelLister<InteractionsClient<H>>
    for GeminiInteractionsModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static + Clone,
{
    fn construct(client: &InteractionsClient<H>) -> Self {
        let client = client.clone();
        Self { client }
    }
}
