use futures::FutureExt;
use rig::providers::gemini;
use serde::Deserialize;
use std::future::Future;
use std::panic::{AssertUnwindSafe, resume_unwind};

use crate::cassettes::{CassetteSpec, ProviderCassette};

/// Every `generationConfig` object in a recorded scenario's **request** bodies,
/// one per recorded turn. A turn that sent no `generationConfig` at all yields
/// [`serde_json::Value::Null`].
///
/// Per `tests/README.md`'s "assert on the request boundary too": a frozen
/// cassette replays the provider's responses and cannot by itself catch
/// outbound drift. Reading the recorded request back lets a test state its
/// guarantee explicitly instead of leaving it implied by mock body matching,
/// which a future harness change could relax.
///
/// Parses the YAML and indexes into the decoded JSON rather than substring
/// matching the raw file, so an incidental occurrence of a field name elsewhere
/// in the cassette (a response body, a JSON schema property) cannot make an
/// absence assertion silently pass or fail.
pub(super) fn recorded_request_generation_configs(scenario: &str) -> Vec<serde_json::Value> {
    recorded_request_bodies(scenario)
        .into_iter()
        .map(|body| {
            body.get("generationConfig")
                .cloned()
                .unwrap_or(serde_json::Value::Null)
        })
        .collect()
}

/// Every recorded **request** body of a scenario, parsed, in recorded order.
///
/// The general form of [`recorded_request_generation_configs`]. A cell whose
/// claim is about what rig *sent* — a handle present, a field absent — can only
/// prove it here: the response says nothing about which of them the request
/// carried, and a cassette replays regardless.
///
/// Bodyless interactions are skipped rather than treated as an error, because a
/// `cachedContents` scenario records its own `GET` and `DELETE` alongside the
/// completion, and those carry no body by construction. The non-empty assert
/// still catches a scenario that recorded nothing to read.
pub(super) fn recorded_request_bodies(scenario: &str) -> Vec<serde_json::Value> {
    let path = crate::cassettes::cassette_path("gemini", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cassette {} should be readable: {error}", path.display()));

    let bodies: Vec<serde_json::Value> =
        serde_yaml::Deserializer::from_str(&contents)
            .filter_map(|document| {
                let document = serde_yaml::Value::deserialize(document)
                    .unwrap_or_else(|error| panic!("cassette document should parse: {error}"));
                let body = document
                    .get("when")
                    .and_then(|when| when.get("body"))
                    .and_then(serde_yaml::Value::as_str)
                    .filter(|body| !body.is_empty())?
                    .to_owned();
                Some(serde_json::from_str(&body).unwrap_or_else(|error| {
                    panic!("recorded request body should be JSON: {error}")
                }))
            })
            .collect();

    assert!(
        !bodies.is_empty(),
        "scenario {scenario} recorded no request bodies, so it asserts nothing"
    );
    bodies
}

/// Assert that every recorded `generateContent` request in `scenario` carried
/// `cachedContent` and none of the three fields a cached content owns.
///
/// The cassette's own body matching would fail a request that changed, but only
/// as an opaque "no recorded interaction matched" — this states the guarantee,
/// so a cell about reading from a cache cannot quietly become a cell about
/// something else.
///
/// `cachedContents` lifecycle calls are skipped, and a top-level `model` is what
/// identifies them rather than the absence of `contents`: a cache created with
/// `NewCachedContent::content(..)` carries `contents` too, and would otherwise
/// be held to a rule written for completions — failing for want of a handle it
/// is in the middle of minting. `generateContent` never carries `model` in the
/// body; the model is in the path.
pub(super) fn assert_recorded_requests_read_from_a_cache(scenario: &str) {
    let mut generate_requests = 0;
    for (turn, body) in recorded_request_bodies(scenario).iter().enumerate() {
        let is_completion = body.get("contents").is_some_and(|c| !c.is_null())
            && body.get("model").is_none_or(serde_json::Value::is_null);
        if !is_completion {
            continue;
        }
        generate_requests += 1;
        assert!(
            body.get("cachedContent")
                .and_then(serde_json::Value::as_str)
                .is_some_and(|handle| handle.starts_with("cachedContents/")),
            "{scenario} turn {turn}: this cell is about reading from a cache, so every \
             generateContent request has to carry the handle: {body}"
        );
        for field in ["systemInstruction", "tools", "toolConfig"] {
            assert!(
                body.get(field).is_none_or(serde_json::Value::is_null),
                "{scenario} turn {turn}: the cache owns {field}, so the request must not carry \
                 its own — and the provider would reject it: {body}"
            );
        }
    }
    assert!(
        generate_requests > 0,
        "{scenario} recorded no generateContent request, so it proves nothing about caching"
    );
}

/// Assert that no recorded request for `scenario` carried a `generationConfig`
/// field the caller never set.
///
/// `expected` lists the fields the test *did* ask for, as
/// `(field, Some(json_value))`; every other sampling field must be absent from
/// the wire so Gemini applies the model's own documented default (rig#2322).
pub(super) fn assert_recorded_sampling_fields(
    scenario: &str,
    expected: &[(&str, serde_json::Value)],
) {
    // The fields rig#2322's hardcoded `Default` used to inject.
    const SAMPLING_FIELDS: &[&str] = &["maxOutputTokens", "temperature"];

    for (turn, config) in recorded_request_generation_configs(scenario)
        .iter()
        .enumerate()
    {
        for field in SAMPLING_FIELDS {
            let recorded = config.get(*field);
            match expected.iter().find(|(name, _)| name == field) {
                Some((_, want)) => assert_eq!(
                    recorded,
                    Some(want),
                    "{scenario} turn {turn}: generationConfig.{field} must reach Gemini as the \
                     caller set it; a dropped value silently hands the turn back to the model's \
                     own default"
                ),
                None => assert_eq!(
                    recorded, None,
                    "{scenario} turn {turn}: generationConfig.{field} must stay off the wire when \
                     the caller never set it — rig#2322 injected a hardcoded \
                     maxOutputTokens=4096/temperature=1.0 here, capping the turn far below the \
                     caller's budget"
                ),
            }
        }
    }
}

/// Every recorded **response** body of a scenario, in recorded order.
///
/// The response side of [`recorded_request_generation_configs`]. Returned raw
/// rather than parsed because the two Gemini transports disagree on the
/// envelope — `generateContent` answers with one JSON object, and
/// `streamGenerateContent` with an SSE stream of them — while the question a
/// matrix cell asks of the bytes ("did the provider actually send the part
/// kind this cell is about?") is the same either way.
pub(super) fn recorded_response_bodies(scenario: &str) -> Vec<String> {
    let path = crate::cassettes::cassette_path("gemini", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cassette {} should be readable: {error}", path.display()));

    let bodies: Vec<String> = serde_yaml::Deserializer::from_str(&contents)
        .map(|document| {
            let document = serde_yaml::Value::deserialize(document)
                .unwrap_or_else(|error| panic!("cassette document should parse: {error}"));
            document
                .get("then")
                .and_then(|then| then.get("body"))
                .and_then(serde_yaml::Value::as_str)
                .unwrap_or_default()
                .to_owned()
        })
        .collect();

    assert!(
        !bodies.is_empty(),
        "scenario {scenario} recorded no interactions, so it asserts nothing"
    );
    bodies
}

/// Assert that `scenario` actually recorded the Gemini part kinds it is about.
///
/// A matrix cell whose provider turn silently stopped producing the shape
/// under test would keep passing while covering nothing; asserting against the
/// recorded bytes makes the cell's premise part of the test rather than an
/// assumption about what the model chose to do on recording day.
pub(super) fn assert_recorded_response_contains(scenario: &str, needles: &[&str]) {
    let bodies = recorded_response_bodies(scenario);
    for needle in needles {
        assert!(
            bodies.iter().any(|body| body.contains(needle)),
            "{scenario}: no recorded response body carries {needle:?}, so this cell does not \
             exercise the shape it claims to cover"
        );
    }
}

/// The mirror of [`assert_recorded_response_contains`]: a cell whose premise
/// is the *absence* of a wire shape has to prove that too, or it silently
/// becomes a duplicate of the cell next to it.
pub(super) fn assert_recorded_response_excludes(scenario: &str, needles: &[&str]) {
    let bodies = recorded_response_bodies(scenario);
    for needle in needles {
        assert!(
            bodies.iter().all(|body| !body.contains(needle)),
            "{scenario}: a recorded response body carries {needle:?}, which this cell exists to \
             show is absent"
        );
    }
}

/// The `usageMetadata.totalTokenCount` on the **last** recorded SSE frame of
/// `scenario` that carries one.
///
/// Gemini's streaming usage is cumulative per chunk, so this is the turn's
/// real total — and the number a terminal built at an *intermediate*
/// `finishReason` gets wrong. Read from the fixture rather than hardcoded, so
/// a re-record cannot leave the expectation behind.
pub(super) fn last_frame_total_tokens(scenario: &str) -> u64 {
    recorded_response_bodies(scenario)
        .iter()
        .flat_map(|body| {
            body.lines()
                .filter_map(|line| line.strip_prefix("data: "))
                .filter_map(|frame| serde_json::from_str::<serde_json::Value>(frame).ok())
                .filter_map(|frame| frame.get("usageMetadata")?.get("totalTokenCount")?.as_u64())
                .collect::<Vec<_>>()
        })
        .next_back()
        .unwrap_or_else(|| panic!("{scenario}: no recorded frame carries a totalTokenCount"))
}

/// Whether any recorded `streamGenerateContent` body in `scenario` carries a
/// `finishReason` frame that is **not** its last frame.
///
/// That is the wire shape the stream-terminal matrix exists for: Gemini emits
/// an intermediate `finishReason` when a built-in tool runs a round and then
/// keeps streaming.
fn recorded_stream_finishes_early(scenario: &str) -> bool {
    recorded_response_bodies(scenario).iter().any(|body| {
        let frames: Vec<serde_json::Value> = body
            .lines()
            .filter_map(|line| line.strip_prefix("data: "))
            .filter_map(|frame| serde_json::from_str(frame).ok())
            .collect();
        let first_finish = frames.iter().position(|frame| {
            frame
                .get("candidates")
                .and_then(serde_json::Value::as_array)
                .is_some_and(|candidates| {
                    candidates
                        .iter()
                        .any(|candidate| candidate.get("finishReason").is_some())
                })
        });

        first_finish.is_some_and(|first| first + 1 < frames.len())
    })
}

/// Assert whether `scenario`'s recorded stream carries the intermediate
/// `finishReason` shape.
///
/// Asserted in both directions: the model only sometimes takes two tool
/// rounds, so a repro cell whose fixture stopped carrying the shape would
/// otherwise keep passing while covering nothing — and a control cell that
/// started carrying it is no longer a control.
pub(super) fn assert_recorded_stream_finishes_early(scenario: &str, expected: bool) {
    assert_eq!(
        recorded_stream_finishes_early(scenario),
        expected,
        "{scenario}: recorded stream should {}carry a finishReason before its last frame; \
         re-record the cell",
        if expected { "" } else { "not " }
    );
}

async fn gemini_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, gemini::Client) {
    let cassette =
        ProviderCassette::start("gemini", spec, "https://generativelanguage.googleapis.com").await;
    let client = gemini::Client::builder()
        .api_key(cassette.api_key("GEMINI_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");

    (cassette, client)
}

async fn gemini_interactions_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, gemini::InteractionsClient) {
    let (cassette, client) = gemini_cassette(spec).await;
    (cassette, client.interactions_api())
}

/// Cassette wrapper for the run-lifecycle matrix (PR #2407): the client sends
/// through a [`rig::http_client::BoxedHttpClient`] carrying the supplied
/// [`rig::http_client::HttpMiddleware`], so the same recorded exchange
/// exercises the transport middleware seam and the run lifecycle hooks
/// together (see `tests/cassettes/gemini/lifecycle_matrix/`).
pub(super) async fn with_gemini_lifecycle_cassette<M, F, Fut>(
    spec: impl Into<CassetteSpec>,
    middleware: M,
    test_body: F,
) where
    M: rig::http_client::HttpMiddleware + 'static,
    F: FnOnce(gemini::Client<rig::http_client::BoxedHttpClient>) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette =
        ProviderCassette::start("gemini", spec, "https://generativelanguage.googleapis.com").await;
    let client = gemini::Client::builder()
        .api_key(cassette.api_key("GEMINI_API_KEY"))
        .base_url(cassette.base_url())
        .http_client(
            rig::http_client::BoxedHttpClient::new(rig::http_client::reqwest::Client::default())
                .with_middleware(middleware),
        )
        .build()
        .expect("client should build");
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_gemini_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(gemini::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = gemini_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Per-bug wrapper for the model-turn termination-metadata matrix
/// (`tests/cassettes/gemini/turn_termination_matrix/`), rig#2184.
pub(super) async fn with_gemini_turn_metadata_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(gemini::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_gemini_cassette(spec, test_body).await;
}

pub(super) async fn with_gemini_interactions_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(gemini::InteractionsClient) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = gemini_interactions_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Code-execution edge matrix (`tests/cassettes/gemini/code_execution_matrix/`).
///
/// A separate registered wrapper so one bug's matrix stays auditable as a
/// unit: `cassette_files_match_registered_scenarios` pairs the fixtures under
/// that directory with the calls made through this name and nothing else.
pub(super) async fn with_gemini_code_execution_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(gemini::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_gemini_cassette(spec, test_body).await;
}

/// Stream-terminal edge matrix (`tests/cassettes/gemini/stream_terminal_matrix/`).
pub(super) async fn with_gemini_stream_terminal_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(gemini::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_gemini_cassette(spec, test_body).await;
}

/// Thought-text edge matrix (`tests/cassettes/gemini/thought_text_matrix/`).
///
/// Separate from [`with_gemini_code_execution_cassette`] for the same reason:
/// one registered wrapper per bug keeps each matrix's fixture set closed.
pub(super) async fn with_gemini_thought_text_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(gemini::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_gemini_cassette(spec, test_body).await;
}

/// Bogus-key variant for recording real 401/403s (rig#2314 error matrix).
pub(super) async fn with_gemini_cassette_bogus_key<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(gemini::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette =
        ProviderCassette::start("gemini", spec, "https://generativelanguage.googleapis.com").await;
    let client = gemini::Client::builder()
        .api_key(cassette.bogus_api_key())
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Cassette wrapper for the gemini prompt-caching matrix
/// (`tests/cassettes/gemini/prompt_caching/`).
///
/// Delegates to [`with_gemini_cassette`] — the behavior is identical, and deliberately
/// shared so the two cannot drift apart when the base wrapper gains policy. What
/// the separate name buys is a per-suite entry in the cassette-safety registry,
/// so the cache fixtures are auditable as one concern's evidence.
pub(super) async fn with_gemini_prompt_caching_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(gemini::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_gemini_cassette(spec, test_body).await;
}

/// Run `body`, then delete `handles` — including when `body` panics.
///
/// Gemini bills `cachedContents` storage per token-hour from the moment
/// `create` returns until the delete lands, so an assertion that takes a cell
/// down between the two leaks a billed cache until its TTL lapses. That only
/// costs money under `RIG_PROVIDER_TEST_MODE=record` — which is exactly when a
/// newly written assertion is most likely to be the one that fails.
///
/// The failure is *caught* rather than threaded back as a `Result` because the
/// checks these cells run are not all this suite's to reshape: `run_cache_probe`
/// and `assert_cache_conformance` (`tests/common/cache_conformance.rs`) are
/// shared by every provider's cache suite and panic from the inside, as does
/// every `.expect()` on an intermediate `get`/`list`/`update_expiry`.
///
/// Delete failures are treated differently on the two paths, and the asymmetry
/// is the point:
///
/// * the body panicked — the body's panic wins, because a cleanup that also
///   failed must never overwrite the assertion actually under test;
/// * the body passed — a delete that failed *is* the leak this exists to
///   prevent, so it fails the cell.
///
/// Every handle is attempted regardless of the ones before it failing, so a
/// cell holding three caches cannot leak two because the first delete 500'd.
pub(super) async fn always_deleting_cached_contents<Fut>(
    client: &gemini::Client,
    handles: &[String],
    body: Fut,
) where
    Fut: Future<Output = ()>,
{
    always_deleting_cached_contents_reporting(client, handles, body, |warning| {
        // libtest prints captured stderr under the failing test, which is
        // exactly when someone needs this.
        eprintln!("{warning}");
    })
    .await;
}

/// [`always_deleting_cached_contents`] with the leak warning routed somewhere a
/// test can read it.
///
/// The warning only fires when the body panicked *and* the cleanup failed, and
/// on that path the body's panic is re-raised — so from the outside the guard is
/// indistinguishable from one that never warned at all. Left on `eprintln!`, the
/// whole block could be deleted with all four cells still green, and the one
/// path that names a still-billing handle would be gone silently.
async fn always_deleting_cached_contents_reporting<Fut, R>(
    client: &gemini::Client,
    handles: &[String],
    body: Fut,
    report: R,
) where
    Fut: Future<Output = ()>,
    R: FnOnce(String),
{
    let outcome = AssertUnwindSafe(body).catch_unwind().await;

    let caches = client.cached_contents();
    let mut failures = Vec::new();
    for handle in handles {
        if let Err(error) = caches.delete(handle).await {
            failures.push(format!("{handle}: {error}"));
        }
    }

    if let Err(payload) = outcome {
        if !failures.is_empty() {
            // The body's panic wins the cell, but a cleanup that also failed
            // leaves a billed cache behind, and this is the only place its
            // handle is ever named.
            report(format!(
                "WARNING: cached contents left billing after a failed cell — delete by hand: {}",
                failures.join("; ")
            ));
        }
        resume_unwind(payload);
    }
    assert!(
        failures.is_empty(),
        "every cached content a cell creates must be deleted — storage bills until it lands: {}",
        failures.join("; ")
    );
}

#[cfg(test)]
mod always_deleting_cached_contents_tests {
    use super::*;
    use axum::http::StatusCode;
    use std::sync::{Arc, Mutex};

    /// A `cachedContents` endpoint that logs every request it sees and answers
    /// `status` with `{}` — enough for `delete`, which discards its body.
    ///
    /// Local rather than a cassette because the behaviour under test is the
    /// *failure* path, and a fixture only ever records a passing run.
    async fn stub_cached_contents(
        seen: Arc<Mutex<Vec<String>>>,
        status: StatusCode,
    ) -> (gemini::Client, tokio::task::JoinHandle<()>) {
        let app = axum::Router::new().fallback(axum::routing::any(
            move |method: axum::http::Method, uri: axum::http::Uri| {
                let seen = Arc::clone(&seen);
                async move {
                    seen.lock()
                        .expect("stub log should not be poisoned")
                        .push(format!("{method} {}", uri.path()));
                    (status, "{}")
                }
            },
        ));
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("stub should bind");
        let addr = listener.local_addr().expect("stub should have an address");
        let server = tokio::spawn(async move {
            axum::serve(listener, app).await.expect("stub should serve");
        });
        let client = gemini::Client::builder()
            .api_key("stub-key")
            .base_url(format!("http://{addr}"))
            .build()
            .expect("client should build");

        (client, server)
    }

    /// The guard must issue the delete when the body panics, and must let the
    /// *body's* panic be what the runner reports.
    ///
    /// This is the whole reason the guard exists and the one path no fixture
    /// covers: a regression that dropped the cleanup would keep every recorded
    /// cell green while leaking a billed cache on every failed recording run,
    /// and one that reported the cleanup's failure instead would hide the
    /// assertion the cell was written to make.
    #[tokio::test]
    async fn a_panicking_body_still_deletes_and_still_reports_its_own_failure() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (client, _stub) = stub_cached_contents(Arc::clone(&seen), StatusCode::OK).await;

        let panicked = AssertUnwindSafe(always_deleting_cached_contents(
            &client,
            &["cachedContents/leaky".to_owned()],
            async { panic!("the assertion under test failed") },
        ))
        .catch_unwind()
        .await
        .expect_err("the body's panic should propagate");

        assert_eq!(
            panicked.downcast_ref::<&str>().copied(),
            Some("the assertion under test failed"),
            "the guard must surface the body's failure, not the cleanup's"
        );
        assert_eq!(
            seen.lock()
                .expect("stub log should not be poisoned")
                .as_slice(),
            ["DELETE /v1beta/cachedContents/leaky"],
            "the cache must be deleted even though the body panicked"
        );
    }

    /// A cleanup delete that fails after a *passing* body is itself the leak,
    /// so it has to fail the cell rather than pass quietly — and the failure has
    /// to name the handle, because that is the string someone needs to delete
    /// the cache by hand.
    #[tokio::test]
    async fn a_cleanup_delete_that_fails_after_a_passing_body_fails_the_cell() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (client, _stub) =
            stub_cached_contents(Arc::clone(&seen), StatusCode::INTERNAL_SERVER_ERROR).await;

        let panicked = AssertUnwindSafe(always_deleting_cached_contents(
            &client,
            &["cachedContents/undeletable".to_owned()],
            async {},
        ))
        .catch_unwind()
        .await
        .expect_err("a cleanup delete that fails should fail the cell");

        let message = panicked
            .downcast_ref::<String>()
            .map(String::as_str)
            .unwrap_or_default();
        assert!(
            message.contains("cachedContents/undeletable"),
            "the failure should name the handle that is still being billed: {message}"
        );
    }

    /// One handle failing to delete must not strand the handles after it.
    ///
    /// The pagination cell creates three caches; an early 500 that aborted the
    /// loop would leak two of them, which is the same bug as no cleanup at all
    /// for two thirds of the cost.
    #[tokio::test]
    async fn every_handle_is_deleted_even_when_an_earlier_delete_fails() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (client, _stub) =
            stub_cached_contents(Arc::clone(&seen), StatusCode::INTERNAL_SERVER_ERROR).await;

        let _ = AssertUnwindSafe(always_deleting_cached_contents(
            &client,
            &[
                "cachedContents/one".to_owned(),
                "cachedContents/two".to_owned(),
                "cachedContents/three".to_owned(),
            ],
            async {},
        ))
        .catch_unwind()
        .await;

        assert_eq!(
            seen.lock()
                .expect("stub log should not be poisoned")
                .as_slice(),
            [
                "DELETE /v1beta/cachedContents/one",
                "DELETE /v1beta/cachedContents/two",
                "DELETE /v1beta/cachedContents/three",
            ],
            "a failed delete must not abandon the handles behind it"
        );
    }

    /// The cell that decides the asymmetry: a body that panicked *and* a
    /// cleanup that failed.
    ///
    /// Neither test above can see the ordering — the panicking-body case stubs
    /// `OK`, so there is no cleanup failure to lose to, and the two 500 cases
    /// pass a body that never panics. Swap the `resume_unwind` and the
    /// `failures` assert and all three stay green while the assertion the cell
    /// was written to make is replaced by the delete's 500. That is exactly
    /// what a failed recording run looks like: the new assertion fails, and the
    /// delete that follows it answers 403 because the cache is already gone.
    #[tokio::test]
    async fn a_failing_cleanup_never_overwrites_a_panicking_body() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (client, _stub) =
            stub_cached_contents(Arc::clone(&seen), StatusCode::INTERNAL_SERVER_ERROR).await;

        let warnings = Arc::new(Mutex::new(Vec::new()));
        let collected = Arc::clone(&warnings);
        let panicked = AssertUnwindSafe(always_deleting_cached_contents_reporting(
            &client,
            &["cachedContents/leaky".to_owned()],
            async { panic!("the assertion under test failed") },
            move |warning| {
                collected
                    .lock()
                    .expect("warning sink should not be poisoned")
                    .push(warning);
            },
        ))
        .catch_unwind()
        .await
        .expect_err("the body's panic should propagate");

        assert_eq!(
            panicked.downcast_ref::<&str>().copied(),
            Some("the assertion under test failed"),
            "a cleanup that also failed must not overwrite the assertion under test"
        );
        assert_eq!(
            seen.lock()
                .expect("stub log should not be poisoned")
                .as_slice(),
            ["DELETE /v1beta/cachedContents/leaky"],
            "the delete must still be attempted, even though its failure is discarded"
        );

        // The panic is re-raised either way, so this warning is the *only*
        // trace the still-billing cache leaves. Without this assertion the
        // whole block can be deleted with every cell still green.
        let warnings = warnings
            .lock()
            .expect("warning sink should not be poisoned");
        let [warning] = warnings.as_slice() else {
            panic!("a discarded cleanup failure must still be reported: {warnings:?}");
        };
        assert!(
            warning.contains("cachedContents/leaky"),
            "the warning has to name the handle — it is what someone types to delete it by hand: \
             {warning}"
        );
    }

    /// The converse: a cleanup that *succeeded* after a panicking body has
    /// nothing to report, so a reader is not sent hunting for a cache that is
    /// already gone.
    #[tokio::test]
    async fn a_clean_cleanup_after_a_panicking_body_reports_no_leak() {
        let seen = Arc::new(Mutex::new(Vec::new()));
        let (client, _stub) = stub_cached_contents(Arc::clone(&seen), StatusCode::OK).await;

        let warnings = Arc::new(Mutex::new(Vec::new()));
        let collected = Arc::clone(&warnings);
        let _ = AssertUnwindSafe(always_deleting_cached_contents_reporting(
            &client,
            &["cachedContents/tidy".to_owned()],
            async { panic!("the assertion under test failed") },
            move |warning| {
                collected
                    .lock()
                    .expect("warning sink should not be poisoned")
                    .push(warning);
            },
        ))
        .catch_unwind()
        .await;

        assert!(
            warnings
                .lock()
                .expect("warning sink should not be poisoned")
                .is_empty(),
            "the cache was deleted; there is no leak to warn about"
        );
    }
}
