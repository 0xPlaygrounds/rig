//! Shared plumbing for provider `GET /models` listings.
//!
//! Every provider's model listing is the same HTTP conversation — GET a
//! path, triage the status, decode a `{ "data": [...] }` envelope, convert
//! entries into [`Model`]s — differing only in the path, the provider label
//! used in error context, and the entry DTO. The DTOs and their
//! `From<Entry> for Model` impls stay in each provider module (that mapping
//! is genuinely provider-specific); the conversation lives here once.

use crate::{
    client::{Client, Provider},
    http_client::{self, HttpClientExt},
    model::{Model, ModelList, ModelListingError},
    wasm_compat::{WasmCompatSend, WasmCompatSync},
};

/// The standard `{ "data": [...] }` list envelope shared by OpenAI-style
/// listing endpoints.
#[derive(Debug, serde::Deserialize)]
pub(crate) struct DataEnvelope<Entry> {
    pub(crate) data: Vec<Entry>,
}

/// The standard OpenAI-style listing entry (OpenAI, Mistral, DeepSeek,
/// Xiaomi MiMo). `id` is the one field every listing carries; the rest are
/// optional so providers that omit them still decode. Providers whose
/// entries genuinely diverge keep their own DTO.
#[derive(Debug, serde::Deserialize)]
pub(crate) struct ListModelEntry {
    pub(crate) id: String,
    pub(crate) name: Option<String>,
    pub(crate) created: Option<u64>,
    pub(crate) owned_by: Option<String>,
}

impl From<ListModelEntry> for Model {
    fn from(value: ListModelEntry) -> Self {
        let mut model = Model::from_id(value.id);
        model.name = value.name;
        model.created_at = value.created;
        model.owned_by = value.owned_by;
        model
    }
}

/// Define a [`ModelLister`](crate::client::ModelLister) that lists models
/// from an OpenAI-style `{ "data": [...] }` endpoint via
/// [`list_models`]. Providers whose listing needs pagination or a bespoke
/// envelope (Gemini, Anthropic, Ollama, Copilot) keep hand-written listers.
macro_rules! impl_model_lister {
    ($(#[$meta:meta])* $name:ident, $client:ty, $entry:ty, $label:literal, $path:literal) => {
        $(#[$meta])*
        #[derive(Clone)]
        pub struct $name<H = reqwest::Client> {
            client: $client,
        }

        impl<H> $crate::client::ModelLister<H> for $name<H>
        where
            H: $crate::http_client::HttpClientExt
                + $crate::wasm_compat::WasmCompatSend
                + $crate::wasm_compat::WasmCompatSync
                + 'static,
        {
            type Client = $client;

            fn new(client: Self::Client) -> Self {
                Self { client }
            }

            async fn list_all(
                &self,
            ) -> Result<$crate::model::ModelList, $crate::model::ModelListingError> {
                $crate::providers::internal::model_listing::list_models::<$entry, _, _>(
                    &self.client,
                    $label,
                    $path,
                )
                .await
            }
        }
    };
}
pub(crate) use impl_model_lister;

/// Map a transport-level send error into listing-flavored context: an
/// [`http_client::Error::InvalidStatusCodeWithMessage`] (backends that reject
/// non-2xx before handing back a response) keeps the provider label, path,
/// status, and body preview, exactly like a non-2xx status on a returned
/// response. Shared with listings that build their own request (copilot's
/// auth-derived base URL cannot go through [`get_json`]).
pub(crate) fn map_transport_error(
    provider_name: &str,
    path: &str,
    error: http_client::Error,
) -> ModelListingError {
    match error {
        http_client::Error::InvalidStatusCodeWithMessage(status, message) => {
            ModelListingError::api_error_with_context(
                provider_name,
                path,
                status.as_u16(),
                message.as_bytes(),
            )
        }
        // The reqwest transport reports non-success with preserved headers
        // (rig#2314); listings have no request-id contract, so only the
        // status and body matter here.
        http_client::Error::InvalidStatusCodeWithDetails { status, body, .. } => {
            ModelListingError::api_error_with_context(
                provider_name,
                path,
                status.as_u16(),
                body.as_bytes(),
            )
        }
        other => ModelListingError::from(other),
    }
}

/// GET `path` and decode the response body as `T`, with listing-flavored
/// error context.
///
/// Error triage is standardized on the most informative behavior: an
/// [`http_client::Error::InvalidStatusCodeWithMessage`] surfaced by the
/// transport (backends that reject non-2xx before handing back a response)
/// is mapped into [`ModelListingError::api_error_with_context`] so the
/// provider label, path, status, and body preview survive, exactly like a
/// non-2xx status on a returned response.
pub(crate) async fn get_json<T, Ext, H>(
    client: &Client<Ext, H>,
    provider_name: &str,
    path: &str,
) -> Result<T, ModelListingError>
where
    T: serde::de::DeserializeOwned,
    Ext: Provider + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    let body = get_bytes(client, provider_name, path).await?;
    serde_json::from_slice(&body).map_err(|error| {
        ModelListingError::parse_error_with_context(provider_name, path, &error, &body)
    })
}

/// Triage a listing response's status and decode its JSON body, keeping the
/// provider label, path, status, and body preview in every error. Shared with
/// listings that build their own request (copilot's auth-derived base URL
/// cannot go through [`get_json`]).
pub(crate) async fn decode_json_response<T>(
    response: http::Response<http_client::LazyBody<Vec<u8>>>,
    provider_name: &str,
    path: &str,
) -> Result<T, ModelListingError>
where
    T: serde::de::DeserializeOwned,
{
    if !response.status().is_success() {
        let status_code = response.status().as_u16();
        let body = response.into_body().await?;
        return Err(ModelListingError::api_error_with_context(
            provider_name,
            path,
            status_code,
            &body,
        ));
    }

    let body = response.into_body().await?;
    serde_json::from_slice(&body).map_err(|error| {
        ModelListingError::parse_error_with_context(provider_name, path, &error, &body)
    })
}

/// Ceiling on pages a single listing will fetch.
///
/// Generous by orders of magnitude: the largest provider catalog is a few
/// hundred models and pages hold up to 1000, so a real listing finishes in one
/// or two requests. This exists only so a cursor that changes without
/// advancing terminates.
pub(crate) const MAX_LISTING_PAGES: usize = 1000;

/// One page of a cursor-paginated listing.
///
/// `next_cursor` is already normalized by the provider's page parser: `None`
/// means "no next page to ask for", whether the wire said so with a flag, an
/// absent cursor, or an empty one.
#[derive(Debug)]
pub(crate) struct ListingPage {
    pub(crate) models: Vec<Model>,
    pub(crate) next_cursor: Option<String>,
}

/// Drive a cursor-paginated listing to exhaustion.
///
/// The providers differ in how they spell a cursor (`after_id`, `pageToken`),
/// where they put it, and how they signal the end (Anthropic's `has_more`
/// flag, Gemini's bare `nextPageToken`) — so `path_for` and `parse_page` stay
/// provider-specific. What must *not* differ is when the loop stops, and that
/// lives here:
///
/// - **No next cursor ends the listing.** Termination follows the cursor, not
///   any flag beside it. A provider that claims more pages while naming no
///   cursor has nothing to ask for, so its parser reports `None` and the
///   listing ends with what it has.
/// - **A repeated cursor ends the listing.** The next request would be
///   byte-identical to the one just answered, so the page would repeat
///   forever.
///
/// - **A page ceiling ends the listing.** The cursor checks above catch a
///   cursor that stops moving, but not one that keeps changing without making
///   progress — a gateway alternating `c1, c2, c1, …`, or minting a fresh
///   cursor per request. Only a hard bound catches that, and a model catalog
///   that needs more than [`MAX_LISTING_PAGES`] pages does not exist.
///
/// All three rules exist because breaking any of them is an *unbounded loop* —
/// the listing never returns and `models` grows without limit — rather than a
/// short list. Anthropic and Gemini each hand-rolled this loop and each
/// shipped the same class of hang (rig#2334); one implementation is the point,
/// and one place to add the bound neither of them had.
pub(crate) async fn paginate_models<Ext, H, P, Q>(
    client: &Client<Ext, H>,
    provider_name: &str,
    mut path_for: P,
    mut parse_page: Q,
) -> Result<ModelList, ModelListingError>
where
    Ext: Provider + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
    P: FnMut(Option<&str>) -> String,
    Q: FnMut(&[u8], &str) -> Result<ListingPage, ModelListingError>,
{
    let mut all_models = Vec::new();
    let mut cursor: Option<String> = None;
    // Only the loop running out of iterations is a ceiling. Every `break`
    // below is the provider ending the listing, which is the normal path and
    // must stay silent — inferring the ceiling from `cursor` instead would
    // report one on any listing that fetched more than a single page, since
    // the cursor of the *previous* page is still held when the loop breaks.
    let mut exhausted_page_budget = true;

    for _ in 0..MAX_LISTING_PAGES {
        let path = path_for(cursor.as_deref());
        let body = get_bytes(client, provider_name, &path).await?;
        let page = parse_page(&body, &path)?;
        all_models.extend(page.models);

        let Some(next) = page.next_cursor else {
            exhausted_page_budget = false;
            break;
        };
        if cursor.as_deref() == Some(next.as_str()) {
            tracing::warn!(
                provider = provider_name,
                models = all_models.len(),
                "model listing repeated its pagination cursor; returning the pages fetched \
                 so far"
            );
            exhausted_page_budget = false;
            break;
        }
        cursor = Some(next);
        continue;
    }

    if exhausted_page_budget {
        tracing::warn!(
            provider = provider_name,
            models = all_models.len(),
            pages = MAX_LISTING_PAGES,
            "model listing hit its page ceiling with a cursor still advancing; returning \
             the pages fetched so far"
        );
    }

    Ok(ModelList::new(all_models))
}

/// Append `pairs` to `path` as a query string, percent-encoding each value in
/// the order given. Cursors are provider-supplied strings landing in a URL, so
/// they are encoded rather than interpolated.
///
/// Callers pass at least one pair; an empty slice would yield a dangling `?`.
pub(crate) fn with_query_pairs(path: &str, pairs: &[(&str, &str)]) -> String {
    let mut serializer = url::form_urlencoded::Serializer::new(String::new());
    for (name, value) in pairs {
        serializer.append_pair(name, value);
    }
    format!("{path}?{}", serializer.finish())
}

/// GET `path` and return the raw body, with listing-flavored status triage.
///
/// The paginated listers need the bytes rather than a decoded value: their
/// page parsers convert entries fallibly and want the raw body for error
/// context.
pub(crate) async fn get_bytes<Ext, H>(
    client: &Client<Ext, H>,
    provider_name: &str,
    path: &str,
) -> Result<Vec<u8>, ModelListingError>
where
    Ext: Provider + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    let req = client.get(path)?.body(http_client::NoBody)?;
    let response = client
        .send::<_, Vec<u8>>(req)
        .await
        .map_err(|error| map_transport_error(provider_name, path, error))?;

    if !response.status().is_success() {
        let status_code = response.status().as_u16();
        let body = response.into_body().await?;
        return Err(ModelListingError::api_error_with_context(
            provider_name,
            path,
            status_code,
            &body,
        ));
    }

    Ok(response.into_body().await?)
}

/// List models from an OpenAI-style `{ "data": [Entry, ...] }` endpoint,
/// converting each entry via `Entry: Into<Model>`.
pub(crate) async fn list_models<Entry, Ext, H>(
    client: &Client<Ext, H>,
    provider_name: &str,
    path: &str,
) -> Result<ModelList, ModelListingError>
where
    Entry: serde::de::DeserializeOwned + Into<Model>,
    Ext: Provider + WasmCompatSend + WasmCompatSync + 'static,
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    let envelope: DataEnvelope<Entry> = get_json(client, provider_name, path).await?;
    let models = envelope.data.into_iter().map(Into::into).collect();
    Ok(ModelList::new(models))
}

#[cfg(test)]
mod tests {
    use super::ListModelEntry;
    use crate::model::Model;

    /// `id` is the one required field: an entry that carries nothing else
    /// (some OpenAI-compatible gateways omit `created`/`owned_by`) still
    /// decodes, mapping the absent fields to `None` on the `Model`.
    #[test]
    fn minimal_entry_decodes_with_id_alone() {
        let entry: ListModelEntry =
            serde_json::from_str(r#"{"id":"gpt-test"}"#).expect("minimal entry should decode");
        let model = Model::from(entry);
        assert_eq!(model.id, "gpt-test");
        assert_eq!(model.name, None);
        assert_eq!(model.created_at, None);
        assert_eq!(model.owned_by, None);
    }
}

#[cfg(test)]
mod transport_error_tests {
    use super::*;

    /// Regression (rig#2314 review): the header-preserving transport variant
    /// must classify as an ApiError with provider/path context exactly like
    /// the header-less one — the reqwest transport now emits it for every
    /// non-2xx.
    #[test]
    fn details_variant_maps_to_api_error_with_context() {
        let error = map_transport_error(
            "test-provider",
            "/models",
            http_client::Error::InvalidStatusCodeWithDetails {
                status: http::StatusCode::UNAUTHORIZED,
                body: r#"{"error":"no"}"#.to_string(),
                headers: Box::new(http::HeaderMap::new()),
            },
        );
        let with_message = map_transport_error(
            "test-provider",
            "/models",
            http_client::Error::InvalidStatusCodeWithMessage(
                http::StatusCode::UNAUTHORIZED,
                r#"{"error":"no"}"#.to_string(),
            ),
        );
        assert_eq!(format!("{error}"), format!("{with_message}"));
        assert!(
            matches!(error, ModelListingError::ApiError { .. }),
            "got {error:?}"
        );
    }
}
