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

pub(super) async fn with_doubleword_bogus_key_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(doubleword::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start("doubleword", spec, DOUBLEWORD_BASE_URL).await;
    let client = doubleword::Client::builder()
        .api_key("rig-deliberately-invalid-doubleword-key")
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");
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

/// One recorded `/chat/completions` interaction, decoded from its cassette.
///
/// Matrices use this to prove their premise from the bytes that crossed the
/// provider boundary. Successful blocking responses populate `response_json`;
/// streaming responses populate `stream_chunks`; provider errors keep their
/// status and JSON envelope in `response_json`.
pub(super) struct RecordedChatCall {
    pub(super) request: serde_json::Value,
    pub(super) status: u16,
    pub(super) response_json: Option<serde_json::Value>,
    pub(super) stream_chunks: Vec<serde_json::Value>,
}

pub(super) fn recorded_chat_calls(scenario: &str) -> Vec<RecordedChatCall> {
    use serde::Deserialize as _;

    let path = crate::cassettes::cassette_path("doubleword", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cassette {} should be readable: {error}", path.display()));

    serde_yaml::Deserializer::from_str(&contents)
        .map(|document| {
            let interaction = serde_yaml::Value::deserialize(document).unwrap_or_else(|error| {
                panic!("cassette {} should deserialize: {error}", path.display())
            });
            assert_eq!(
                interaction["when"]["path"].as_str(),
                Some("/v1/chat/completions"),
                "cassette {} should record only chat-completion turns",
                path.display()
            );
            assert!(
                interaction["then"]["body_encoding"].is_null()
                    && interaction["when"]["body_encoding"].is_null(),
                "cassette {} records an encoded body this reader cannot decode",
                path.display()
            );

            let request_raw = interaction["when"]["body"].as_str().unwrap_or_else(|| {
                panic!("cassette {} request should carry a body", path.display())
            });
            let request = serde_json::from_str(request_raw).unwrap_or_else(|error| {
                panic!(
                    "cassette {} request body should be JSON: {error}",
                    path.display()
                )
            });
            let status = interaction["then"]["status"]
                .as_u64()
                .unwrap_or_else(|| panic!("cassette {} should carry a status", path.display()))
                as u16;
            let response_raw = interaction["then"]["body"].as_str().unwrap_or_else(|| {
                panic!("cassette {} response should carry a body", path.display())
            });
            let response_json = serde_json::from_str(response_raw).ok();
            let stream_chunks = response_raw
                .lines()
                .filter_map(|line| line.trim().strip_prefix("data: "))
                .filter(|payload| *payload != "[DONE]")
                .map(|payload| {
                    serde_json::from_str(payload).unwrap_or_else(|error| {
                        panic!(
                            "cassette {} SSE payload should be JSON: {error}",
                            path.display()
                        )
                    })
                })
                .collect();

            RecordedChatCall {
                request,
                status,
                response_json,
                stream_chunks,
            }
        })
        .collect()
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
            // Asserted rather than assumed: this reader treats every
            // interaction as a JSON embeddings turn, so a cassette that ever
            // mixed in a completion turn — or a base64-encoded body — would
            // otherwise be read as an embeddings call that returned nothing.
            assert_eq!(
                interaction["when"]["path"].as_str(),
                Some("/v1/embeddings"),
                "cassette {} should record only embeddings turns",
                path.display()
            );
            assert!(
                interaction["then"]["body_encoding"].is_null()
                    && interaction["when"]["body_encoding"].is_null(),
                "cassette {} records an encoded body this reader cannot decode",
                path.display()
            );
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
