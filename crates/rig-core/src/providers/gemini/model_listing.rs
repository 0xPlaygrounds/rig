use crate::error::EncodeError;
use crate::error::ProviderError;
use crate::json_utils::Lenient;
use crate::wire::Flow;
use crate::{
    model::{ModelInfo, ModelList, listing},
    operation::{ModelListing, ModelPage, Verify},
    providers::internal::{wire::classify_marker_keyed_frame, with_query_pairs},
    wire::{Body, Decoder, Descriptor, Encoded, Framing, Mode, Out, Wire, WireEvent, WireFrame},
};
use serde::{Deserialize, Serialize};
use serde_json::Value;

const MAX_PAGE_SIZE: usize = 1000;

/// The model a listing entry names: its `baseModelId`, else its `name`
/// without the `models/` prefix, with the limits it reports.
fn model_of(entry: &Value) -> Option<ModelInfo> {
    let id = entry
        .str("baseModelId")
        .map(str::trim)
        .filter(|id| !id.is_empty());
    let name = entry.str("name").map(str::trim).unwrap_or_default();
    let name = name.strip_prefix("models/").unwrap_or(name);
    let id = id.or((!name.is_empty()).then_some(name))?;
    let limit = |key: &str| entry.u64(key).and_then(|limit| u32::try_from(limit).ok());
    let mut model = ModelInfo::from_id(id.to_owned());
    model.name = entry.str("displayName").map(str::to_owned);
    model.description = entry.str("description").map(str::to_owned);
    model.context_length = limit("inputTokenLimit");
    model.max_output_tokens = limit("outputTokenLimit");
    Some(model)
}

fn list_models_path(page_token: Option<&str>) -> String {
    let page_size = MAX_PAGE_SIZE.to_string();
    let mut pairs = vec![("pageSize", page_size.as_str())];
    if let Some(page_token) = page_token {
        pairs.push(("pageToken", page_token));
    }
    with_query_pairs("/v1beta/models", &pairs)
}

/// A decoded model page with an optional nonempty continuation cursor.
#[derive(Debug)]
struct ListingPage {
    models: Vec<ModelInfo>,
    next_cursor: Option<String>,
}

fn parse_models_page(body: &[u8], path: &str) -> Result<ListingPage, ProviderError> {
    let error =
        |details: &dyn std::fmt::Display| listing::parse_error("Gemini", path, details, body);
    let page: Value = serde_json::from_slice(body)
        .map_err(|parse| error(&format_args!("parse_error={parse}")))?;
    if page.get("models").is_some_and(|models| !models.is_array()) {
        return Err(error(&"parse_error=`models` is not a list"));
    }
    let models = page
        .arr("models")
        .iter()
        .map(|entry| {
            model_of(entry).ok_or_else(|| {
                error(&"parse_error=model entry missing usable `baseModelId` and `name` values")
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    // Sending an empty cursor would repeatedly fetch the same page.
    Ok(ListingPage {
        models,
        next_cursor: page
            .str("nextPageToken")
            .filter(|token| !token.is_empty())
            .map(str::to_owned),
    })
}

impl super::GeminiConfig {
    /// The GenerateContent model-listing wire.
    pub(crate) fn models(&self) -> Models {
        Models::new(self.clone())
    }

    /// The credential-check wire.
    pub(crate) fn verify(&self) -> VerifyKey {
        VerifyKey::new(self.clone())
    }
}

#[cfg(test)]
mod tests;

/// The top-level keys a genuine `GET /v1beta/models` reply carries.
///
/// A frame naming either must decode fully or it is corrupt; JSON naming
/// neither is some other endpoint's reply rather than a page of this one.
const MODEL_PAGE_MARKERS: &[&str] = &["models", "nextPageToken"];

/// API-key placement for a model-listing request.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Auth {
    Query,
    Header,
}

/// One page's request, after `page_token` when a page named one.
fn list_models_request(
    provider: &super::GeminiConfig,
    auth: Auth,
    page_token: Option<&str>,
) -> Result<http::Request<Body>, EncodeError> {
    let path = list_models_path(page_token);
    let trimmed = path.trim_start_matches('/');
    let base_url = &provider.base_url;
    // The page-size parameter guarantees an existing query string.
    let request = match auth {
        Auth::Query => http::Request::get(format!(
            "{base_url}/{trimmed}&key={}",
            provider.api_key.expose()
        )),
        Auth::Header => http::Request::get(format!("{base_url}/{trimmed}"))
            .header("x-goog-api-key", provider.api_key.expose()),
    };
    request.body(Body::empty()).map_err(EncodeError::from)
}

/// The GenerateContent model-listing wire: `GET /v1beta/models`, paged on
/// `nextPageToken`, credential in the query.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct Models {
    /// The provider this wire speaks to.
    pub provider: super::GeminiConfig,
}

impl Models {
    /// Build the wire over `provider`.
    pub fn new(provider: super::GeminiConfig) -> Self {
        Self { provider }
    }
}

impl Wire for Models {
    type Op = ModelListing;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ModelsDecoder;
    type Reassembler = crate::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME)
    }

    /// A listing never streams, so both modes send the one request.
    fn encode(&self, cursor: Option<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        Ok(Encoded::new(
            list_models_request(&self.provider, Auth::Query, cursor.as_deref())?,
            Framing::Whole,
        ))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ModelsDecoder
    }
}

/// The Interactions API model-listing wire: the same `GET /v1beta/models`,
/// authenticated with `x-goog-api-key`.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct InteractionsModels {
    /// The provider this wire speaks to.
    pub provider: super::GeminiConfig,
}

impl InteractionsModels {
    /// Build the wire over `provider`.
    pub fn new(provider: super::GeminiConfig) -> Self {
        Self { provider }
    }
}

impl Wire for InteractionsModels {
    type Op = ModelListing;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = ModelsDecoder;
    type Reassembler = crate::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME)
    }

    /// A listing never streams, so both modes send the one request.
    fn encode(&self, cursor: Option<String>, _mode: Mode) -> Result<Encoded, EncodeError> {
        Ok(Encoded::new(
            list_models_request(&self.provider, Auth::Header, cursor.as_deref())?,
            Framing::Whole,
        ))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        ModelsDecoder
    }
}

/// Decode model pages and the cursor each names.
#[derive(Default)]
pub struct ModelsDecoder;

impl<'id> Decoder<'id, ModelListing> for ModelsDecoder {
    /// The page JSON, retained for model-entry error context.
    type Event = String;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        let payload = frame.as_str().into_owned();
        let classified = classify_marker_keyed_frame::<Value>(&payload, MODEL_PAGE_MARKERS);
        classified.map(|_| payload)
    }

    fn decode(
        &mut self,
        payload: Self::Event,
        out: Out<'id, ModelListing>,
    ) -> Result<Flow, ProviderError> {
        // Diagnostics identify the endpoint without retaining the previous page's cursor.
        let path = list_models_path(None);
        match parse_models_page(payload.as_bytes(), &path) {
            Ok(page) => Ok(out.end(ModelPage {
                models: ModelList::new(page.models),
                next: page.next_cursor,
            })),
            Err(error) => Err(error),
        }
    }
}

/// Check credentials with `GET /v1beta/models`, authenticating through the query.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct VerifyKey {
    /// The provider this wire speaks to.
    pub provider: super::GeminiConfig,
}

impl VerifyKey {
    /// Build the wire over `provider`.
    pub fn new(provider: super::GeminiConfig) -> Self {
        Self { provider }
    }
}

impl Wire for VerifyKey {
    type Op = Verify;
    type Payload = crate::wire::Encoded;
    type Frame = crate::wire::WireFrame;
    type Decoder<'id> = VerifyKeyDecoder;
    type Reassembler = crate::wire::document::Unreassembled;

    fn describe(&self) -> Descriptor<'_> {
        Descriptor::new(super::PROVIDER_NAME)
    }

    fn encode(&self, _request: (), _mode: Mode) -> Result<Encoded, EncodeError> {
        let request = http::Request::get(format!(
            "{}/v1beta/models?key={}",
            self.provider.base_url,
            self.provider.api_key.expose()
        ))
        .body(Body::empty())?;
        Ok(Encoded::new(request, Framing::Whole))
    }

    fn decoder<'id>(&self) -> Self::Decoder<'id> {
        VerifyKeyDecoder
    }
}

/// Verify a successful response without normalizing model entries.
#[derive(Default)]
pub struct VerifyKeyDecoder;

impl<'id> Decoder<'id, Verify> for VerifyKeyDecoder {
    type Event = ();

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        classify_marker_keyed_frame::<Value>(&frame.as_str(), MODEL_PAGE_MARKERS).map(|_| ())
    }

    fn decode(
        &mut self,
        _event: Self::Event,
        out: Out<'id, Verify>,
    ) -> Result<Flow, ProviderError> {
        Ok(out.end(()))
    }

    /// A success status verifies even when no recognized page arrived.
    fn eof(&mut self, out: Out<'id, Verify>) -> Result<Flow, ProviderError> {
        Ok(out.end(()))
    }
}
