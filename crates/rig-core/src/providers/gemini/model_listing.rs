use crate::{
    client::VerifyError,
    model::{Model, ModelList, ModelListingError},
    operation::{ModelListing, Verify},
    providers::internal::{wire::classify_marker_keyed_frame, with_query_pairs},
    wire::{Body, Decoder, Encoded, Framing, Mode, Output, Sink, Wire, WireEvent, WireFrame},
};
use serde::{Deserialize, Serialize};
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
    with_query_pairs("/v1beta/models", &pairs)
}

/// One decoded page of `GET /v1beta/models`: the models it carried, and the
/// cursor it named when it named a usable one.
#[derive(Debug)]
struct ListingPage {
    models: Vec<Model>,
    next_cursor: Option<String>,
}

fn parse_models_page(body: &[u8], path: &str) -> Result<ListingPage, ModelListingError> {
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
    Ok(ListingPage {
        models,
        next_cursor: page.next_page_token.filter(|token| !token.is_empty()),
    })
}

#[cfg(test)]
mod tests;

/// The top-level keys a genuine `GET /v1beta/models` reply carries.
///
/// A frame naming either must decode fully or it is corrupt; JSON naming
/// neither is some other endpoint's reply rather than a page of this one.
const MODEL_PAGE_MARKERS: &[&str] = &["models", "nextPageToken"];

/// How a listing request carries the API key.
///
/// Both Gemini providers list from the same endpoint and differ in nothing
/// else, which is why one endpoint has two wires: GenerateContent appends
/// the credential as the last `key=` query pair (`Gemini::uri`) while the
/// Interactions API sends `x-goog-api-key` and names no query credential at
/// all (`Gemini::interactions_uri`).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Auth {
    Query,
    Header,
}

/// One page's request, after `page_token` when a page named one.
fn list_models_request(
    provider: &super::Gemini,
    auth: Auth,
    page_token: Option<&str>,
) -> Result<http::Request<Body>, ModelListingError> {
    let path = list_models_path(page_token);
    let trimmed = path.trim_start_matches('/');
    let base_url = &provider.base_url;
    // `list_models_path` always names `pageSize`, so a query credential joins
    // an existing query string and lands last, as `Gemini::uri` writes it.
    let request = match auth {
        Auth::Query => http::Request::get(format!(
            "{base_url}/{trimmed}&key={}",
            provider.api_key.expose()
        )),
        Auth::Header => http::Request::get(format!("{base_url}/{trimmed}"))
            .header("x-goog-api-key", provider.api_key.expose()),
    };
    request
        .body(Body::empty())
        .map_err(|error| ModelListingError::request_error(error.to_string()))
}

/// The GenerateContent model-listing wire: `GET /v1beta/models`, paged on
/// `nextPageToken`, credential in the query.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Models {
    /// The provider this wire speaks to.
    pub provider: super::Gemini,
}

impl Models {
    /// Build the wire over `provider`.
    pub fn new(provider: super::Gemini) -> Self {
        Self { provider }
    }
}

impl Wire for Models {
    type Op = ModelListing;
    type Decoder = ModelsDecoder;

    fn name(&self) -> &str {
        super::PROVIDER_NAME
    }

    /// A listing never streams, so both modes send the one request.
    fn encode(&self, _request: (), _mode: Mode) -> Result<Encoded, ModelListingError> {
        Ok(Encoded::new(
            list_models_request(&self.provider, Auth::Query, None)?,
            Framing::Whole,
        ))
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        ModelsDecoder::new(self.provider.clone(), Auth::Query)
    }
}

/// The Interactions API model-listing wire: the same `GET /v1beta/models`,
/// authenticated with `x-goog-api-key`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InteractionsModels {
    /// The provider this wire speaks to.
    pub provider: super::Gemini,
}

impl InteractionsModels {
    /// Build the wire over `provider`.
    pub fn new(provider: super::Gemini) -> Self {
        Self { provider }
    }
}

impl Wire for InteractionsModels {
    type Op = ModelListing;
    type Decoder = ModelsDecoder;

    fn name(&self) -> &str {
        super::PROVIDER_NAME
    }

    /// A listing never streams, so both modes send the one request.
    fn encode(&self, _request: (), _mode: Mode) -> Result<Encoded, ModelListingError> {
        Ok(Encoded::new(
            list_models_request(&self.provider, Auth::Header, None)?,
            Framing::Whole,
        ))
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        ModelsDecoder::new(self.provider.clone(), Auth::Header)
    }
}

/// Decodes one `GET /v1beta/models` page and follows the cursor it names.
///
/// The authentication is the only thing that differs between the two
/// listing wires' requests, so `auth` keeps the paging rules in one state
/// machine instead of two copies of them.
pub struct ModelsDecoder {
    provider: super::Gemini,
    auth: Auth,
    /// The cursor the page just interpreted named, when it named a usable one.
    next: Option<String>,
}

impl ModelsDecoder {
    fn new(provider: super::Gemini, auth: Auth) -> Self {
        Self {
            provider,
            auth,
            next: None,
        }
    }
}

impl Decoder<ModelListing> for ModelsDecoder {
    /// The page's raw bytes. `parse_models_page` owns the entry mapping and
    /// reads the body for its error context, so classification validates the
    /// page's shape and hands the payload on rather than mapping it twice.
    type Event = String;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        let payload = frame.as_str().into_owned();
        let classified =
            classify_marker_keyed_frame::<ListModelsResponse>(&payload, MODEL_PAGE_MARKERS);
        classified.map(|_| payload)
    }

    fn interpret(&mut self, payload: Self::Event, out: &mut Output<ModelListing>) {
        // Every page gets a fresh decoder, so the cursor that fetched this
        // one is not in hand here; the endpoint it names is the same either
        // way, and that is all the diagnostic path reports.
        let path = list_models_path(None);
        match parse_models_page(payload.as_bytes(), &path) {
            Ok(page) => {
                self.next = page.next_cursor;
                out.push(Ok(ModelList::new(page.models)));
            }
            Err(error) => out.push(Err(error)),
        }
    }

    fn continuation(&self) -> Option<http::Request<Body>> {
        let cursor = self.next.as_deref()?;
        list_models_request(&self.provider, self.auth, Some(cursor)).ok()
    }
}

/// The credential-check wire: `GET /v1beta/models`, status only.
///
/// The listing endpoint is `Provider::VERIFY_PATH` for both Gemini
/// providers, and the credential goes in the query as every other
/// GenerateContent-family request puts it.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerifyKey {
    /// The provider this wire speaks to.
    pub provider: super::Gemini,
}

impl VerifyKey {
    /// Build the wire over `provider`.
    pub fn new(provider: super::Gemini) -> Self {
        Self { provider }
    }
}

impl Wire for VerifyKey {
    type Op = Verify;
    type Decoder = VerifyKeyDecoder;

    fn name(&self) -> &str {
        super::PROVIDER_NAME
    }

    fn encode(&self, _request: (), _mode: Mode) -> Result<Encoded, VerifyError> {
        let request = http::Request::get(format!(
            "{}/v1beta/models?key={}",
            self.provider.base_url,
            self.provider.api_key.expose()
        ))
        .body(Body::empty())
        .map_err(|error| VerifyError::HttpError(crate::http_client::Error::Protocol(error)))?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder(&self, _mode: Mode) -> Self::Decoder {
        VerifyKeyDecoder::default()
    }
}

/// A success status is the whole answer; the page is not read for meaning.
#[derive(Default)]
pub struct VerifyKeyDecoder {
    answered: bool,
}

impl Decoder<Verify> for VerifyKeyDecoder {
    type Event = ();

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_marker_keyed_frame::<ListModelsResponse>(&frame.as_str(), MODEL_PAGE_MARKERS)
            .map(|_| ())
    }

    fn interpret(&mut self, _event: Self::Event, out: &mut Output<Verify>) {
        self.answered = true;
        out.push(Ok(()));
    }

    /// A 2xx whose body carried nothing this decoder recognizes still
    /// verifies: the status already answered, and the fold needs one event.
    fn finish(&mut self, out: &mut Output<Verify>) {
        if !self.answered {
            out.push(Ok(()));
        }
    }
}
