use futures::FutureExt;
use rig::providers::gemini;
use serde::Deserialize;
use std::future::Future;
use std::panic::AssertUnwindSafe;

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
    let path = crate::cassettes::cassette_path("gemini", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cassette {} should be readable: {error}", path.display()));

    let configs: Vec<serde_json::Value> = serde_yaml::Deserializer::from_str(&contents)
        .map(|document| {
            let document = serde_yaml::Value::deserialize(document)
                .unwrap_or_else(|error| panic!("cassette document should parse: {error}"));
            let body = document
                .get("when")
                .and_then(|when| when.get("body"))
                .and_then(serde_yaml::Value::as_str)
                .expect("each recorded interaction should carry a request body")
                .to_owned();
            let body: serde_json::Value = serde_json::from_str(&body)
                .unwrap_or_else(|error| panic!("recorded request body should be JSON: {error}"));
            body.get("generationConfig")
                .cloned()
                .unwrap_or(serde_json::Value::Null)
        })
        .collect();

    assert!(
        !configs.is_empty(),
        "scenario {scenario} recorded no interactions, so it asserts nothing"
    );
    configs
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

pub(super) async fn with_gemini_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(gemini::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = gemini_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
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
    with_gemini_cassette(spec, test_body).await
}

/// Stream-terminal edge matrix (`tests/cassettes/gemini/stream_terminal_matrix/`).
pub(super) async fn with_gemini_stream_terminal_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(gemini::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_gemini_cassette(spec, test_body).await
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
    with_gemini_cassette(spec, test_body).await
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
