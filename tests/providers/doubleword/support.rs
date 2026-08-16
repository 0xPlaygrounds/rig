use futures::FutureExt;
use rig::providers::doubleword;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};

const DOUBLEWORD_BASE_URL: &str = "https://api.doubleword.ai/v1";

async fn doubleword_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, doubleword::Client) {
    let cassette = ProviderCassette::start("doubleword", spec, DOUBLEWORD_BASE_URL).await;
    let client = doubleword::Client::builder()
        .api_key(cassette.api_key("DOUBLEWORD_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");

    (cassette, client)
}

pub(super) async fn with_doubleword_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(doubleword::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = doubleword_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_doubleword_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(doubleword::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = doubleword_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// One recorded embeddings round trip, read back out of the cassette file.
///
/// The matrix below is entirely about the relationship between two numbers —
/// the width a request asked for and the width the response came back with —
/// and both live in the fixture, not in the assertion. Reading them back is
/// what stops a cell from passing while covering nothing.
pub(super) struct RecordedEmbeddingCall {
    /// The `dimensions` field the request carried, or `None` when it carried
    /// the field not at all.
    pub(super) requested_dimensions: Option<usize>,
    /// The width of every vector the response returned, in order.
    pub(super) returned_widths: Vec<usize>,
}

/// Runs an embeddings cell and hands back what the cassette recorded.
///
/// Identical to [`with_doubleword_cassette`] up to the point the fixture is
/// written; then it re-opens that fixture and parses each interaction, so
/// every cell in the dimensions matrix can assert its premise against the
/// recorded bytes rather than against its own expectations.
pub(super) async fn with_doubleword_embedding_cassette<F, Fut>(
    scenario: &'static str,
    test_body: F,
) -> Vec<RecordedEmbeddingCall>
where
    F: FnOnce(doubleword::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_doubleword_cassette(scenario, test_body).await;
    recorded_embedding_calls(scenario)
}

fn recorded_embedding_calls(scenario: &str) -> Vec<RecordedEmbeddingCall> {
    use serde::Deserialize as _;

    let path = crate::cassettes::cassette_path("doubleword", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cassette {} should be readable: {error}", path.display()));

    serde_yaml::Deserializer::from_str(&contents)
        .map(|document| {
            let interaction = serde_yaml::Value::deserialize(document).unwrap_or_else(|error| {
                panic!("cassette {} should deserialize: {error}", path.display())
            });
            let body = |side: &str| -> serde_json::Value {
                let raw = interaction[side]["body"].as_str().unwrap_or_else(|| {
                    panic!("cassette {} {side} should carry a body", path.display())
                });
                serde_json::from_str(raw).unwrap_or_else(|error| {
                    panic!(
                        "cassette {} {side} body should be JSON: {error}",
                        path.display()
                    )
                })
            };
            let (request, response) = (body("when"), body("then"));

            RecordedEmbeddingCall {
                requested_dimensions: request.get("dimensions").map(|value| {
                    value.as_u64().unwrap_or_else(|| {
                        panic!("{}: dimensions should be a number", path.display())
                    }) as usize
                }),
                // An error turn carries no `data`; the cell that records one
                // is asserting on the request it sent, so an empty width list
                // is the honest answer rather than a panic.
                returned_widths: response
                    .get("data")
                    .and_then(serde_json::Value::as_array)
                    .map(|data| {
                        data.iter()
                            .map(|datum| {
                                datum["embedding"]
                                    .as_array()
                                    .unwrap_or_else(|| {
                                        panic!(
                                            "{}: datum should carry an embedding",
                                            path.display()
                                        )
                                    })
                                    .len()
                            })
                            .collect()
                    })
                    .unwrap_or_default(),
            }
        })
        .collect()
}
