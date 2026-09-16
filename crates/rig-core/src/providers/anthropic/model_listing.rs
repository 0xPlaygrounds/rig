use super::client::Anthropic;
use crate::operation::ModelListing;
use crate::wire::{Body, Decoder, Encoded, Framing, Output, Wire, WireEvent, WireFrame};
use serde::Serialize;
use std::sync::{Arc, Mutex};

/// The Models wire for Anthropic model listing (GET /v1/models).
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Models {
    pub provider: Anthropic,
}

impl Models {
    pub fn new(provider: Anthropic) -> Self {
        Self { provider }
    }
}

impl Wire for Models {
    type Op = ModelListing;
    type Decoder = AnthropicModelsDecoder;

    fn name(&self) -> &str {
        self.provider.dialect.name
    }

    fn encode(&self, _request: ()) -> Result<Encoded, ModelListingError> {
        let url = format!("{}/v1/models", self.provider.base_url.trim_end_matches('/'));
        let mut req = http::Request::builder()
            .method(http::Method::GET)
            .uri(&url)
            .body(Body::Bytes(Vec::new()))
            .map_err(|e| ModelListingError::request_error(e.to_string()))?;
        req.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        req.headers_mut()
            .insert(http::header::ACCEPT, http::HeaderValue::from_static("*/*"));
        self.provider.apply_headers(req.headers_mut());

        let mut encoded = Encoded::new(req, Framing::Whole, "/v1/models");
        if let Some(header) = self.provider.dialect.request_id_header {
            encoded = encoded.with_request_id_header(header);
        }
        Ok(encoded)
    }

    fn decoder(&self) -> Self::Decoder {
        AnthropicModelsDecoder {
            provider: self.provider.clone(),
            state: Arc::new(Mutex::new(ListingState {
                page_count: 1,
                ..Default::default()
            })),
        }
    }

    fn capabilities(&self) {}
}

#[derive(Default)]
struct ListingState {
    next_cursor: Option<String>,
    prev_cursor: Option<String>,
    page_count: usize,
}

pub struct AnthropicModelsDecoder {
    provider: Anthropic,
    state: Arc<Mutex<ListingState>>,
}

impl Decoder<ModelListing> for AnthropicModelsDecoder {
    type Event = Vec<Model>;

    fn classify(&self, frame: WireFrame) -> WireEvent<Self::Event> {
        let body = frame.as_str();
        crate::providers::internal::wire::classify_unary_frame::<ListModelsResponse>(&body).map(
            |page| {
                let next_cursor = page.last_id.filter(|cursor| !cursor.is_empty());
                let next = page.has_more.then_some(next_cursor).flatten();
                if let Ok(mut guard) = self.state.lock() {
                    guard.next_cursor = next;
                }
                page.data.into_iter().map(Model::from).collect()
            },
        )
    }

    fn interpret(&mut self, event: Self::Event, out: &mut Output<ModelListing>) {
        out.emit(event);
    }

    fn finish(&mut self, _out: &mut Output<ModelListing>) {}

    fn flush_before_terminal_error(&mut self, _out: &mut Output<ModelListing>) {}

    fn project(&self, _payload: &[u8], _sink: &mut dyn crate::wire::ObservationSink) {}

    fn continuation(&self) -> Option<http::Request<Body>> {
        let mut guard = self.state.lock().ok()?;
        if guard.page_count >= crate::providers::internal::model_listing::MAX_LISTING_PAGES {
            tracing::warn!(
                provider = "Anthropic",
                pages = crate::providers::internal::model_listing::MAX_LISTING_PAGES,
                "model listing hit its page ceiling with a cursor still advancing; returning the pages fetched so far"
            );
            return None;
        }
        let cursor = guard.next_cursor.take()?;
        if guard.prev_cursor.as_ref() == Some(&cursor) {
            tracing::warn!(
                provider = "Anthropic",
                "model listing repeated its pagination cursor; returning the pages fetched so far"
            );
            return None;
        }
        guard.prev_cursor = Some(cursor.clone());
        guard.page_count += 1;

        let encoded_cursor: String =
            url::form_urlencoded::byte_serialize(cursor.as_bytes()).collect();
        let url = format!(
            "{}/v1/models?after_id={}",
            self.provider.base_url.trim_end_matches('/'),
            encoded_cursor
        );
        let mut req = http::Request::builder()
            .method(http::Method::GET)
            .uri(&url)
            .body(Body::Bytes(Vec::new()))
            .ok()?;
        req.headers_mut().insert(
            http::header::CONTENT_TYPE,
            http::HeaderValue::from_static("application/json"),
        );
        req.headers_mut()
            .insert(http::header::ACCEPT, http::HeaderValue::from_static("*/*"));
        self.provider.apply_headers(req.headers_mut());
        Some(req)
    }
}
use crate::{
    client::ModelLister,
    http_client::HttpClientExt,
    model::{Model, ModelList, ModelListingError},
    providers::anthropic::Client,
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct ListModelsResponse {
    data: Vec<ListModelEntry>,
    has_more: bool,
    last_id: Option<String>,
}

#[derive(Debug, Deserialize)]
struct ListModelEntry {
    id: String,
    display_name: String,
}

impl From<ListModelEntry> for Model {
    fn from(value: ListModelEntry) -> Self {
        Model::new(value.id, value.display_name)
    }
}

/// [`ModelLister`] implementation for the Anthropic API (`GET /v1/models`).
///
/// Automatically paginates through all pages using cursor-based pagination.
#[derive(Clone)]
pub struct AnthropicModelLister<H = crate::http_client::BoxedHttpClient> {
    client: Client<H>,
}

impl<H> ModelLister<H> for AnthropicModelLister<H>
where
    H: HttpClientExt + Clone + WasmCompatSend + WasmCompatSync + 'static,
{
    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        self.client.models().list_all().await
    }
}

impl<H> AnthropicModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static + Clone,
{
    /// Build the lister over `client`.
    pub fn new(client: Client<H>) -> Self {
        Self { client }
    }
}

/// Edge matrix for the pagination loop's termination.
///
/// The loop's input space is the cross-product of two response fields, and it
/// is small enough to enumerate completely:
///
/// | # | Cell | `has_more` | `last_id` | behavior |
/// |---|------|-----------|-----------|----------|
/// | 1 | `single_page_listing_is_unchanged` | `false` | `Some` | stop |
/// | 2 | `stops_when_the_last_page_names_no_cursor` | `false` | `None` | stop |
/// | 3 | `pagination_follows_the_cursor_across_pages` | `true` | `Some(id)` | continue |
/// | 4 | `pagination_stops_when_a_page_claims_more_but_names_no_cursor` | `true` | `None` | stop |
/// | 5 | `pagination_stops_on_an_empty_cursor` | `true` | `Some("")` | stop |
/// | 6 | `pagination_stops_midway_and_keeps_earlier_pages` | mixed | mixed | partial |
/// | 7 | `pagination_stops_on_an_empty_page_claiming_more` | `true` | `None` | stop, empty |
/// | 8 | `pagination_stops_on_a_cursor_that_does_not_advance` | `true` | repeated | stop |
/// | 9 | `pagination_stops_at_the_page_ceiling_on_an_alternating_cursor` | `true` | alternating | stop at cap |
/// | 10 | `pagination_percent_encodes_the_cursor` | `true` | `Some("weird id&x=1")` | encoded |
///
/// Rows 1–5 are every combination of the two fields. Rows 6–9 are the ways a
/// cursor can fail to advance that only show up across multiple pages: one
/// arriving *after* a good page, a page with no models, a server that echoes
/// the same cursor forever, and one that alternates so no repeat is ever
/// observed. Row 10 covers how the cursor is serialized.
///
/// No cell is recorded. Anthropic pairs `has_more` with `last_id`, so rows 2
/// and 4–8 describe responses no live request can produce, and row 3 needs a
/// catalog larger than one page — Anthropic's fits in one, which is why the
/// recorded `models` cassette answers `has_more: false` and the loop body
/// never ran before this suite existed.
#[cfg(test)]
mod tests;
