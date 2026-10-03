use super::*;
use crate::test_utils::json_body;
use crate::test_utils::{MockHttpResponse, RecordingHttpClient, SequencedHttpClient};
use crate::wire::secret::tests::a_config_reloads_without_its_credential;

fn cohere() -> CohereConfig {
    CohereConfig::new("cohere-test-key")
}

/// Chat goes to the Compatibility API under the configured root, with the
/// key as a bearer token.
#[test]
fn chat_addresses_the_compatibility_api() {
    let wire = cohere()
        .with_base_url("http://127.0.0.1:9/")
        .completion("command-a-03-2025");
    let encoded = wire
        .encode(crate::completion::CompletionRequest::new("hi"), Mode::Unary)
        .expect("the request encodes");
    assert_eq!(
        encoded.request.uri().to_string(),
        "http://127.0.0.1:9/compatibility/v1/chat/completions"
    );
    assert_eq!(
        encoded.request.headers()[http::header::AUTHORIZATION],
        "Bearer cohere-test-key"
    );
    assert_eq!(json_body(&encoded.request)["model"], "command-a-03-2025");
}

#[test]
fn a_serialized_config_carries_no_key_material() {
    a_config_reloads_without_its_credential(&cohere(), "cohere-test-key", |cohere| &cohere.api_key);

    let wire = cohere().completion("command-a-03-2025");
    let serialized = serde_json::to_string(&wire).expect("the wire serializes");
    assert!(
        !serialized.contains("cohere-test-key"),
        "a wire a host may persist must not carry the credential: {serialized}"
    );
}

/// A `/v1/embed` reply, shaped as `crates/rig-cassette/fixtures/cassettes/cohere/embeddings/
/// embed_texts_smoke.yaml` records it (`id`, `embeddings`, `texts`, and
/// `meta.billed_units`), with two-element vectors in place of the recorded
/// 1024-element ones.
const EMBED_BODY: &str = r#"{"id":"b2e4b0f7-0000-0000-0000-000000000000","texts":["first","second"],"embeddings":[[0.5,-0.25],[0.125,0.0]],"meta":{"api_version":{"version":"1"},"billed_units":{"input_tokens":7,"search_units":0,"classifications":0,"images":0}}}"#;

#[tokio::test]
async fn an_embedding_reply_pairs_its_vectors_with_the_texts_that_were_sent() {
    let response = crate::driver::Model::new(
        cohere().embedding("embed-v4.0", None),
        RecordingHttpClient::new(EMBED_BODY),
    )
    .call(vec!["first".to_owned(), "second".to_owned()])
    .await
    .expect("the reply decodes");

    assert_eq!(
        response
            .embeddings
            .iter()
            .map(|embedding| (embedding.document.as_str(), embedding.vec.as_slice()))
            .collect::<Vec<_>>(),
        vec![
            ("first", [0.5, -0.25].as_slice()),
            ("second", [0.125, 0.0].as_slice()),
        ]
    );
    assert_eq!(response.usage.input_tokens, Some(7));
    assert_eq!(response.usage.total_tokens, Some(7));
    assert_eq!(
        response.response_id.as_deref(),
        Some("b2e4b0f7-0000-0000-0000-000000000000")
    );
}

#[test]
fn an_embedding_wire_reports_the_models_published_width() {
    let wire = cohere().embedding("embed-english-light-v3.0", None);
    assert_eq!(
        wire.describe().capabilities,
        Capabilities::embedding(96, 384)
    );
    assert_eq!(
        cohere()
            .embedding("embed-english-light-v3.0", Some(64))
            .ndims,
        64,
        "a width the caller named wins over the model's table"
    );
}

/// A one-pixel PNG-headed byte string: enough for the media-type sniff the
/// wire runs before it builds a request.
fn png(tail: &[u8]) -> Vec<u8> {
    let mut bytes = b"\x89PNG\r\n\x1a\n".to_vec();
    bytes.extend_from_slice(tail);
    bytes
}

/// Cohere embeds ONE image per call: a larger batch is the caller's to
/// split, and the one image travels as a data URL.
#[test]
fn an_image_request_carries_exactly_one_image() {
    let wire = cohere().image_embedding();
    let refused = wire.encode(vec![png(b"first"), png(b"second")], Mode::Unary);
    assert!(refused.is_err(), "two images are two calls");

    let encoded = wire
        .encode(vec![png(b"first")], Mode::Unary)
        .expect("the image is a PNG");
    let body = json_body(&encoded.request);
    assert!(
        body["images"][0]
            .as_str()
            .is_some_and(|url| url.starts_with("data:image/png;base64,"))
    );
}

#[test]
fn an_image_the_provider_will_not_accept_never_reaches_the_wire() {
    let rejected = cohere()
        .image_embedding()
        .encode(vec![b"not an image".to_vec()], Mode::Unary);
    let Err(error) = rejected else {
        panic!("an unsniffable format must be rejected before the request is built");
    };
    assert!(matches!(
        ProviderError::from(error),
        ProviderError::Request(_)
    ));
}

/// One `/v1/embed` image reply, shaped as the recorded image cells are:
/// `embeddings.float` holds exactly one vector.
fn image_reply(first: f64) -> MockHttpResponse {
    MockHttpResponse::success(format!(
        r#"{{"id":"img-{first}","embeddings":{{"float":[[{first},1.0]]}},"meta":{{"api_version":{{"version":"1"}},"billed_units":{{"search_units":0,"classifications":0,"images":1}}}}}}"#
    ))
}

#[tokio::test]
async fn a_single_image_embed_captures_the_bare_document() {
    let http = SequencedHttpClient::new([image_reply(0.5)]);
    let response = crate::driver::Model::new(cohere().image_embedding(), http.clone())
        .call(vec![png(b"only")])
        .await
        .expect("the reply decodes");

    assert_eq!(http.requests().len(), 1);
    // One request, one page: `raw` is that document itself, not a one-element
    // array, so a single-image embed reads the same as any non-batched wire.
    assert!(
        response.raw.is_object(),
        "one page is captured bare: {}",
        response.raw
    );
    let page: super::super::embeddings::ImageEmbeddingResponse =
        serde_json::from_value(response.raw.clone()).expect("raw is Cohere's own answer");
    assert_eq!(page.id.as_deref(), Some("img-0.5"));
}

/// `raw` is the whole reply body, so a field the reply type does not model
/// survives in it.
#[tokio::test]
async fn an_embedding_reply_keeps_its_whole_body_as_raw() {
    let body = r#"{"id":"b2e4b0f7-0000-0000-0000-000000000000","texts":["first","second"],"embeddings":[[0.5,-0.25],[0.125,0.0]],"meta":{"api_version":{"version":"1"},"billed_units":{"input_tokens":7,"search_units":0,"classifications":0,"images":0}},"unmodeled":"kept"}"#;
    let response = crate::driver::Model::new(
        cohere().embedding("embed-v4.0", None),
        RecordingHttpClient::new(body),
    )
    .call(vec!["first".to_owned(), "second".to_owned()])
    .await
    .expect("the reply decodes");

    assert_eq!(
        response.raw,
        serde_json::from_str::<serde_json::Value>(body).expect("the body is JSON")
    );
    assert_eq!(response.raw["unmodeled"], "kept");
}
