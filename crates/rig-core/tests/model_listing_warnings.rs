//! What a paginated model listing *reports*, as distinct from what it returns
//! (rig#2339 review).
//!
//! The page ceiling is a bound on rig's own loop, so warning about it tells an
//! operator their catalog was truncated. Deciding that from the pagination
//! cursor rather than from loop exhaustion made every multi-page listing say
//! so: the loop breaks while still holding the *previous* page's cursor, so
//! `cursor.is_some()` is true for any listing past its first page. Gemini's
//! catalog paginates in production, so that fired on every real listing.
//!
//! These live in their own test binary on purpose. `tracing` caches callsite
//! interest **globally**, computed by whichever thread reaches the callsite
//! first, while `set_default` installs a subscriber only for the current
//! thread. A sibling test emitting these same warnings without a subscriber
//! therefore caches `Interest::never` and silences the capture mid-test — which
//! is exactly what happened when these assertions lived beside the other
//! listing tests. Alone in a binary, and serialized by
//! `scoped_tracing_subscriber_guard`, nothing else can reach the callsites.

#![allow(clippy::expect_used)]

use std::io::Write;
use std::sync::{Arc, Mutex};

use rig_core::client::ModelListingClient;
use rig_core::providers::anthropic;
use rig_core::test_utils::{MockHttpResponse, SequencedHttpClient};

/// One page of Anthropic's `/v1/models` envelope.
fn page(models: &[&str], has_more: bool, last_id: Option<&str>) -> MockHttpResponse {
    let data: Vec<_> = models
        .iter()
        .map(|id| serde_json::json!({"id": id, "display_name": id, "type": "model"}))
        .collect();
    MockHttpResponse::success(
        serde_json::json!({"data": data, "has_more": has_more, "last_id": last_id}).to_string(),
    )
}

#[derive(Clone)]
struct SharedWriter(Arc<Mutex<Vec<u8>>>);

impl Write for SharedWriter {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().expect("log buffer").extend_from_slice(buf);
        Ok(buf.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

/// List `pages` through a mock transport and return everything it logged.
async fn logs_from_listing(pages: Vec<MockHttpResponse>) -> String {
    // Scoped-subscriber tests must not run concurrently; see
    // `test_utils::scoped_tracing_subscriber_guard`.
    let _isolation = rig_core::test_utils::scoped_tracing_subscriber_guard().await;
    let captured = Arc::new(Mutex::new(Vec::new()));
    let subscriber = tracing_subscriber::fmt()
        .with_max_level(tracing::Level::WARN)
        .with_ansi(false)
        .without_time()
        .with_writer({
            let captured = captured.clone();
            move || SharedWriter(captured.clone())
        })
        .finish();

    {
        let _guard = tracing::subscriber::set_default(subscriber);
        // Interest is cached per callsite, globally; rebuild so it is computed
        // against the subscriber installed just above.
        tracing::callsite::rebuild_interest_cache();

        let client = anthropic::Client::builder()
            .api_key("test-key")
            .http_client(SequencedHttpClient::new(pages))
            .build()
            .expect("client should build");
        client.list_models().await.expect("listing should succeed");
    }

    let bytes = captured.lock().expect("log buffer").clone();
    String::from_utf8(bytes).expect("log output is utf-8")
}

const CEILING_WARNING: &str = "hit its page ceiling";

/// A listing that ends because the provider said so is not a ceiling, however
/// many pages it took.
#[tokio::test]
async fn a_completed_multi_page_listing_reports_no_page_ceiling() {
    let logs = logs_from_listing(vec![
        page(&["claude-a"], true, Some("claude-a")),
        page(&["claude-b"], true, Some("claude-b")),
        page(&["claude-c"], false, Some("claude-c")),
    ])
    .await;

    assert!(
        !logs.contains(CEILING_WARNING),
        "a completed three-page listing must not report a page ceiling; logged:\n{logs}"
    );
}

/// The counterpart, so the fix cannot be "stop warning at all": a listing that
/// genuinely runs out of its page budget still reports the ceiling.
#[tokio::test]
async fn a_listing_that_exhausts_the_page_budget_still_reports_the_ceiling() {
    // Two cursors that alternate forever: every request differs from the one
    // before, so no repeat is ever observed and only the bound stops it.
    let pages: Vec<_> = (0..1_010)
        .map(|i| {
            page(
                &["claude-a"],
                true,
                Some(if i % 2 == 0 { "ping" } else { "pong" }),
            )
        })
        .collect();

    let logs = logs_from_listing(pages).await;

    assert!(
        logs.contains(CEILING_WARNING),
        "exhausting the page budget must still be reported; logged:\n{logs}"
    );
}

/// A repeated cursor is its own diagnosis and ends the listing, so it reports
/// that alone — not that *and* a page ceiling for one event.
#[tokio::test]
async fn a_repeated_cursor_reports_only_its_own_warning() {
    let logs = logs_from_listing(vec![
        page(&["claude-a"], true, Some("stuck")),
        page(&["claude-b"], true, Some("stuck")),
    ])
    .await;

    assert!(
        logs.contains("repeated its pagination cursor"),
        "the repeated cursor is what ended the listing; logged:\n{logs}"
    );
    assert!(
        !logs.contains(CEILING_WARNING),
        "one event must not also be reported as a page ceiling; logged:\n{logs}"
    );
}
