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
    fn parse_models_page_accepts_omitted_empty_models_list() {
        let page =
            parse_models_page(br#"{}"#, "/v1beta/models?pageSize=1000").expect("page should parse");
        let (models, next_page_token) = (page.models, page.next_cursor);

        assert!(models.is_empty());
        assert_eq!(next_page_token, None);
    }

    /// The request path is what the recorded cassette matches on, so its exact
    /// shape is load-bearing. Page 1 is covered by that replay; this pins the
    /// cursored form, which no fixture exercises because Gemini's catalog fits
    /// in one page.
    #[test]
    fn list_models_path_puts_page_size_first_and_encodes_the_cursor() {
        assert_eq!(list_models_path(None), "/v1beta/models?pageSize=1000");
        assert_eq!(
            list_models_path(Some("abc123")),
            "/v1beta/models?pageSize=1000&pageToken=abc123",
        );
        assert_eq!(
            list_models_path(Some("weird token&x=1")),
            "/v1beta/models?pageSize=1000&pageToken=weird+token%26x%3D1",
        );
    }

    /// An empty `nextPageToken` must read as "no more pages", not as a cursor.
    ///
    /// Treated as a cursor it re-sends an empty `pageToken`, gets the same
    /// page back, and loops forever — the listing never returns rather than
    /// returning a short list, so the only observable symptom is a hang.
    #[test]
    fn parse_models_page_treats_an_empty_next_page_token_as_absent() {
        let next_page_token = parse_models_page(
            br#"{"models": [], "nextPageToken": ""}"#,
            "/v1beta/models?pageSize=1000",
        )
        .expect("page should parse")
        .next_cursor;

        assert_eq!(next_page_token, None);
    }

    /// A real cursor still advances the loop.
    #[test]
    fn parse_models_page_keeps_a_non_empty_next_page_token() {
        let next_page_token = parse_models_page(
            br#"{"models": [], "nextPageToken": "abc123"}"#,
            "/v1beta/models?pageSize=1000",
        )
        .expect("page should parse")
        .next_cursor;

        assert_eq!(next_page_token.as_deref(), Some("abc123"));
    }

    /// Loop-level: a server that keeps echoing the same cursor cannot advance
    /// the listing, so the loop must stop rather than fetch the same page
    /// forever. The parser-level guard above only covers the *empty* cursor;
    /// this covers the other way a cursor fails to move.
    #[tokio::test]
    async fn list_all_stops_on_a_cursor_that_does_not_advance() {
        use crate::client::ModelLister as _;
        use crate::test_utils::{MockHttpResponse, SequencedHttpClient};

        let page = |id: &str, token: &str| {
            MockHttpResponse::success(
                serde_json::json!({
                    "models": [{
                        "name": format!("models/{id}"),
                        "displayName": id,
                        "inputTokenLimit": 1024
                    }],
                    "nextPageToken": token
                })
                .to_string(),
            )
        };
        let http_client = SequencedHttpClient::new(vec![
            page("a", "stuck"),
            page("b", "stuck"),
            page("c", "stuck"),
        ]);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("client should build");

        let models = GeminiModelLister::new(client)
            .list_all()
            .await
            .expect("listing should terminate");

        assert_eq!(
            models.data.len(),
            2,
            "the repeat is only detectable on the second page, so both are kept",
        );
        assert_eq!(http_client.remaining_responses(), 1);
    }

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

        let page =
            parse_models_page(body, "/v1beta/models?pageSize=1000").expect("page should parse");
        let (models, next_page_token) = (page.models, page.next_cursor);

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

        let models = parse_models_page(body, "/v1beta/models?pageSize=1000")
            .expect("page should parse")
            .models;

        assert_eq!(models.len(), 1);
        assert_eq!(models[0].id, "gemini-2.0-flash");
    }

    #[test]
    fn parse_models_page_reports_missing_model_id_when_name_is_omitted() {
        let error = parse_models_page(br#"{"models":[{}]}"#, "/v1beta/models?pageSize=1000")
            .expect_err("entry without name/baseModelId should fail with contextual error");

        match error {
            ModelListingError::ParseError { message } => {
                assert!(message.contains("provider=Gemini"));
                assert!(message.contains("path=/v1beta/models?pageSize=1000"));
                assert!(message.contains(
                    "parse_error=model entry missing usable `baseModelId` and `name` values"
                ));
            }
            _ => panic!("expected parse error"),
        }
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
