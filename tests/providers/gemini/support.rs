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
