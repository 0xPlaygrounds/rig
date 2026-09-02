use crate::{
    client::ModelLister,
    http_client::HttpClientExt,
    model::{Model, ModelList, ModelListingError},
    providers::{anthropic::Client, internal},
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
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        internal::model_listing::paginate_models(
            &self.client,
            "Anthropic",
            |cursor| match cursor {
                Some(cursor) => {
                    internal::model_listing::with_query_pairs("/v1/models", &[("after_id", cursor)])
                }
                None => "/v1/models".to_string(),
            },
            parse_page,
        )
        .await
    }
}

impl<H> crate::client::ConstructModelLister<Client<H>> for AnthropicModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static + Clone,
{
    fn construct(client: &Client<H>) -> Self {
        let client = client.clone();
        Self { client }
    }
}

/// Anthropic pages with a `has_more` flag beside the `last_id` cursor, so the
/// "more pages, no cursor" shape is expressible on this wire and worth
/// reporting. The shared loop only needs to know whether there is a cursor.
fn parse_page(
    body: &[u8],
    path: &str,
) -> Result<internal::model_listing::ListingPage, ModelListingError> {
    let page: ListModelsResponse = serde_json::from_slice(body).map_err(|error| {
        ModelListingError::parse_error_with_context("Anthropic", path, &error, body)
    })?;

    // An empty cursor counts as absent, matching how every other
    // provider-reported identifier in rig is read.
    let next_cursor = page.last_id.filter(|cursor| !cursor.is_empty());
    if page.has_more && next_cursor.is_none() {
        // Anthropic pairs the two, so this is unreachable against the real
        // API; it is reachable because a caller can point this client at an
        // Anthropic-compatible gateway base URL. There is no next page to ask
        // for without a cursor either way.
        tracing::warn!(
            "Anthropic model listing reported more pages but no usable `last_id` cursor; \
             returning the pages fetched so far"
        );
    }

    Ok(internal::model_listing::ListingPage {
        models: page.data.into_iter().map(Model::from).collect(),
        // `has_more: false` ends the listing even if a cursor is present:
        // the flag is authoritative for *stopping*, the cursor only for
        // *continuing*.
        next_cursor: page.has_more.then_some(next_cursor).flatten(),
    })
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
