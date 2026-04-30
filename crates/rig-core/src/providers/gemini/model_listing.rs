use crate::{
    client::{self, ModelLister, Provider},
    http_client::{self, HttpClientExt},
    model::{Model, ModelList, ModelListingError},
    providers::gemini::{Client, InteractionsClient},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use serde::Deserialize;
use std::{convert::TryFrom, fmt};

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
    name: String,
    base_model_id: Option<String>,
    display_name: Option<String>,
    description: Option<String>,
    input_token_limit: Option<u64>,
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
        Ok(model)
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

fn parse_models_page(
    body: &[u8],
    path: &str,
) -> Result<(Vec<Model>, Option<String>), ModelListingError> {
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

    Ok((models, page.next_page_token))
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
        let response = client.send::<_, Vec<u8>>(req).await?;

        if !response.status().is_success() {
            let status_code = response.status().as_u16();
            let body = response.into_body().await?;
            return Err(ModelListingError::api_error_with_context(
                "Gemini",
                &path,
                status_code,
                &body,
            ));
        }

        let body = response.into_body().await?;
        let (models, next_page_token_for_page) = parse_models_page(&body, &path)?;
        all_models.extend(models);

        if next_page_token_for_page.is_none() {
            break;
        }

        next_page_token = next_page_token_for_page;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parse_models_page_falls_back_to_name_when_base_model_id_is_missing() {
        let body = br#"{
            "models": [
                {
                    "name": "models/gemini-2.0-flash-001",
                    "displayName": "Gemini 2.0 Flash 001",
                    "description": "Stable Gemini 2.0 Flash",
                    "inputTokenLimit": 1048576
                }
            ]
        }"#;

        let (models, next_page_token) =
            parse_models_page(body, "/v1beta/models?pageSize=1000").expect("page should parse");

        assert_eq!(next_page_token, None);
        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "gemini-2.0-flash-001");
        assert_eq!(models[0].name.as_deref(), Some("Gemini 2.0 Flash 001"));
        assert_eq!(
            models[0].description.as_deref(),
            Some("Stable Gemini 2.0 Flash")
        );
        assert_eq!(models[0].context_length, Some(1_048_576));
    }

    #[test]
    fn parse_models_page_prefers_base_model_id_when_present() {
        let body = br#"{
            "models": [
                {
                    "name": "models/gemini-2.0-flash-001",
                    "baseModelId": "gemini-2.0-flash",
                    "displayName": "Gemini 2.0 Flash 001"
                }
            ]
        }"#;

        let (models, _) =
            parse_models_page(body, "/v1beta/models?pageSize=1000").expect("page should parse");

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "gemini-2.0-flash");
    }

    #[test]
    fn parse_models_page_returns_parse_error_when_entry_has_no_usable_id() {
        let body = br#"{
            "models": [
                {
                    "name": "models/",
                    "baseModelId": "   ",
                    "displayName": "Broken Gemini"
                }
            ]
        }"#;

        let error = parse_models_page(body, "/v1beta/models?pageSize=1000")
            .expect_err("page should fail when no usable ID is available");

        match error {
            ModelListingError::ParseError { message } => {
                assert!(message.contains("provider=Gemini"));
                assert!(message.contains("path=/v1beta/models?pageSize=1000"));
                assert!(message.contains(
                    "parse_error=model entry missing usable `baseModelId` and `name` values"
                ));
                assert!(message.contains(r#""name": "models/""#));
            }
            _ => panic!("expected parse error"),
        }
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
