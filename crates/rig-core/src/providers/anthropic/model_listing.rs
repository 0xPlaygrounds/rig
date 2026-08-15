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
pub struct AnthropicModelLister<H = reqwest::Client> {
    client: Client<H>,
}

impl<H> ModelLister<H> for AnthropicModelLister<H>
where
    H: HttpClientExt + WasmCompatSend + WasmCompatSync + 'static,
{
    type Client = Client<H>;

    fn new(client: Self::Client) -> Self {
        Self { client }
    }

    async fn list_all(&self) -> Result<ModelList, ModelListingError> {
        let mut all_models = Vec::new();
        let mut after_id: Option<String> = None;

        loop {
            let path = match &after_id {
                Some(cursor) => format!("/v1/models?after_id={cursor}"),
                None => "/v1/models".to_string(),
            };

            let page: ListModelsResponse =
                internal::model_listing::get_json(&self.client, "Anthropic", &path).await?;

            all_models.extend(page.data.into_iter().map(Model::from));

            if !page.has_more {
                break;
            }

            // Termination follows the *cursor*, not the flag. `has_more` with
            // no `last_id` would otherwise re-request the uncursored first
            // page forever, appending the same models on every pass — an
            // unbounded loop, not a truncated list. Anthropic pairs the two,
            // but the Anthropic-compatible gateways sharing this client are
            // exactly the sources that report a flag without its companion
            // field, and there is no next page to ask for without a cursor.
            let Some(cursor) = page.last_id else {
                tracing::warn!(
                    models = all_models.len(),
                    "Anthropic model listing reported more pages but no `last_id` cursor; \
                     returning the pages fetched so far"
                );
                break;
            };
            after_id = Some(cursor);
        }

        Ok(ModelList::new(all_models))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_utils::{MockHttpResponse, SequencedHttpClient};

    fn page(models: &[&str], has_more: bool, last_id: Option<&str>) -> MockHttpResponse {
        let data: Vec<_> = models
            .iter()
            .map(|id| serde_json::json!({"id": id, "display_name": id, "type": "model"}))
            .collect();
        MockHttpResponse::success(
            serde_json::json!({
                "data": data,
                "has_more": has_more,
                "last_id": last_id,
            })
            .to_string(),
        )
    }

    fn lister(
        pages: Vec<MockHttpResponse>,
    ) -> (
        AnthropicModelLister<SequencedHttpClient>,
        SequencedHttpClient,
    ) {
        let http_client = SequencedHttpClient::new(pages);
        let client = Client::builder()
            .api_key("test-key")
            .http_client(http_client.clone())
            .build()
            .expect("client should build");
        (AnthropicModelLister::new(client), http_client)
    }

    /// The pagination loop follows the cursor across pages, sending `after_id`
    /// on every request after the first. No fixture can cover this: Anthropic's
    /// own catalog fits in one page, so the recorded `models` cassette always
    /// answers `has_more: false` and the loop body never runs.
    #[tokio::test]
    async fn pagination_follows_the_cursor_across_pages() {
        let (lister, http_client) = lister(vec![
            page(&["claude-a"], true, Some("claude-a")),
            page(&["claude-b"], true, Some("claude-b")),
            page(&["claude-c"], false, Some("claude-c")),
        ]);

        let models = lister.list_all().await.expect("listing should succeed");

        let ids: Vec<_> = models.data.iter().map(|model| model.id.as_str()).collect();
        assert_eq!(ids, ["claude-a", "claude-b", "claude-c"]);

        let uris: Vec<_> = http_client
            .requests()
            .into_iter()
            .map(|request| request.uri)
            .collect();
        assert!(
            uris[0].ends_with("/v1/models"),
            "first page is uncursored: {uris:?}"
        );
        assert!(
            uris[1].ends_with("/v1/models?after_id=claude-a"),
            "second page must carry the first page's cursor: {uris:?}",
        );
        assert!(
            uris[2].ends_with("/v1/models?after_id=claude-b"),
            "third page must carry the second page's cursor: {uris:?}",
        );
    }

    /// A page claiming more pages while naming no cursor must end the loop.
    ///
    /// Termination follows the cursor, not the flag: re-requesting the
    /// uncursored first page cannot make progress, so the pre-fix loop
    /// refetched page 1 forever and appended its models on every pass. The
    /// scripted pages are finite, so on the unfixed code this test fails
    /// (the loop drains them and errors) rather than hanging the suite.
    ///
    /// Unit-tested rather than recorded because Anthropic pairs `has_more`
    /// with `last_id`; no live request can produce the malformed page.
    #[tokio::test]
    async fn pagination_stops_when_a_page_claims_more_but_names_no_cursor() {
        let (lister, http_client) = lister(vec![
            page(&["claude-a"], true, None),
            page(&["claude-a"], true, None),
            page(&["claude-a"], true, None),
        ]);

        let models = lister
            .list_all()
            .await
            .expect("a cursor-less page ends the listing instead of looping");

        let ids: Vec<_> = models.data.iter().map(|model| model.id.as_str()).collect();
        assert_eq!(
            ids,
            ["claude-a"],
            "the page's models are returned exactly once"
        );
        assert_eq!(
            http_client.remaining_responses(),
            2,
            "the loop must stop after the first page rather than re-requesting it",
        );
    }

    /// An empty page that claims more is the same shape with nothing to return,
    /// and must also terminate rather than spin.
    #[tokio::test]
    async fn pagination_stops_on_an_empty_page_claiming_more() {
        let (lister, http_client) = lister(vec![
            page(&[], true, None),
            page(&["claude-a"], false, None),
        ]);

        let models = lister.list_all().await.expect("listing should terminate");

        assert!(models.data.is_empty());
        assert_eq!(http_client.remaining_responses(), 1);
    }

    /// The ordinary single-page catalog is unchanged by the guard.
    #[tokio::test]
    async fn single_page_listing_is_unchanged() {
        let (lister, http_client) = lister(vec![page(
            &["claude-a", "claude-b"],
            false,
            Some("claude-b"),
        )]);

        let models = lister.list_all().await.expect("listing should succeed");

        let ids: Vec<_> = models.data.iter().map(|model| model.id.as_str()).collect();
        assert_eq!(ids, ["claude-a", "claude-b"]);
        assert_eq!(http_client.remaining_responses(), 0);
    }
}
