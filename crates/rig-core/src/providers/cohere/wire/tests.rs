use super::*;
use crate::completion::CompletionModel as _;
use crate::driver::Bound;
use crate::embeddings::{EmbeddingModel as _, ImageEmbeddingModel as _};
use crate::message::AssistantContent;
use crate::test_utils::{
    MockHttpResponse, MockStreamingClient, RecordingHttpClient, SequencedHttpClient,
};
use crate::wire::secret::tests::a_config_reloads_without_its_credential;
use futures::StreamExt;

/// The recorded request of `crates/rig-cassette/fixtures/cassettes/cohere/agent/
/// max_tokens_sets_max_tokens_finish_reason.yaml` (`POST /v2/chat`).
const RECORDED_REQUEST: &str = r#"{"documents":[],"max_tokens":4,"messages":[{"content":[{"text":"Write a detailed fifty-word description of the ocean.","type":"text"}],"role":"user"}],"model":"command-a-03-2025"}"#;

/// That cassette's recorded reply, byte for byte.
const UNARY_BODY: &str = r#"{"finish_reason":"MAX_TOKENS","id":"20ae3cc4-46d2-4e78-8566-83649fbfc218","message":{"content":[{"text":"The ocean,","type":"text"}],"role":"assistant"},"usage":{"billed_units":{"input_tokens":11,"output_tokens":3},"cached_tokens":448,"tokens":{"input_tokens":506,"output_tokens":4}}}"#;

/// The same turn as a stream: the frame shapes are verbatim from
/// `crates/rig-cassette/fixtures/cassettes/cohere/streaming/streaming_smoke.yaml` (`message-start`
/// carrying the id, an empty `content-start`, `content-delta` text
/// fragments, `content-end`, then `message-end` with usage and finish
/// reason), carrying the unary reply's id, text, usage and finish reason.
/// Cohere records the two modes from separate calls, so a byte-identical
/// turn only exists when it is built this way — and the property under test
/// is exactly that the two shapes fold alike.
const STREAM_BODY: &str = concat!(
    "event: message-start\n",
    r#"data: {"delta":{"message":{"citations":[],"content":[],"role":"assistant","tool_calls":[],"tool_plan":""}},"id":"20ae3cc4-46d2-4e78-8566-83649fbfc218","type":"message-start"}"#,
    "\n\n",
    "event: content-start\n",
    r#"data: {"delta":{"message":{"content":{"text":"","type":"text"}}},"index":0,"type":"content-start"}"#,
    "\n\n",
    "event: content-delta\n",
    r#"data: {"delta":{"message":{"content":{"text":"The ocean"}}},"index":0,"type":"content-delta"}"#,
    "\n\n",
    "event: content-delta\n",
    r#"data: {"delta":{"message":{"content":{"text":","}}},"index":0,"type":"content-delta"}"#,
    "\n\n",
    "event: content-end\n",
    r#"data: {"index":0,"type":"content-end"}"#,
    "\n\n",
    "event: message-end\n",
    r#"data: {"delta":{"finish_reason":"MAX_TOKENS","usage":{"billed_units":{"input_tokens":11,"output_tokens":3},"cached_tokens":448,"tokens":{"input_tokens":506,"output_tokens":4}}},"type":"message-end"}"#,
    "\n\n",
    "data: [DONE]\n\n",
);

fn cohere() -> Cohere {
    Cohere::new("cohere-test-key")
}

/// The request the recorded cell sent.
fn recorded_request() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![crate::message::Message::user(
            "Write a detailed fifty-word description of the ocean.",
        )],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: Some(4),
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

fn body_of(encoded: &Encoded) -> serde_json::Value {
    let [request] = encoded.requests.as_slice() else {
        panic!(
            "expected exactly one request, got {}",
            encoded.requests.len()
        );
    };
    match request.body() {
        Body::Bytes(bytes) => serde_json::from_slice(bytes).expect("the body is JSON"),
        Body::Multipart(_) => panic!("the chat wire sends no multipart body"),
    }
}

fn text_of(choice: &[AssistantContent]) -> String {
    choice
        .iter()
        .filter_map(|block| match block {
            AssistantContent::Text(text) => Some(text.text.clone()),
            _ => None,
        })
        .collect()
}

#[tokio::test]
async fn a_unary_reply_and_a_streamed_reply_fold_to_the_same_turn() {
    let buffered = Bound::new(
        cohere().chat("command-a-03-2025"),
        RecordingHttpClient::new(UNARY_BODY),
    )
    .completion(recorded_request())
    .await
    .expect("the recorded reply decodes");

    let streaming = Bound::new(
        cohere().chat("command-a-03-2025"),
        MockStreamingClient {
            sse_bytes: bytes::Bytes::from_static(STREAM_BODY.as_bytes()),
        },
    );
    let mut response = streaming
        .stream(recorded_request())
        .await
        .expect("the stream opens");
    while response.next().await.is_some() {}
    let streamed = response
        .finish()
        .expect("the stream produced a terminal record");

    assert_eq!(text_of(&buffered.choice), "The ocean,");
    assert_eq!(buffered.choice, streamed.choice);
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    // Cohere's `/v2/chat` names no model in either mode.
    assert_eq!(buffered.model, streamed.model);
    assert_eq!(buffered.response_id, streamed.response_id);
    assert_eq!(
        buffered.usage.output_tokens,
        Some(4),
        "the total-usage counter is the one that is read, not `billed_units`"
    );
    assert_eq!(
        buffered.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
}

/// The two modes differ by `stream` and by nothing else — both spellings are
/// recorded (`agent/max_tokens_sets_max_tokens_finish_reason.yaml` without,
/// `streaming/streaming_smoke.yaml` with).
#[test]
fn the_mode_is_the_only_difference_between_the_two_requests() {
    let wire = cohere().chat("command-a-03-2025");
    let unary = wire
        .encode(recorded_request(), Mode::Unary)
        .expect("the request encodes");
    let streamed = wire
        .encode(recorded_request(), Mode::Streaming)
        .expect("the request encodes");

    assert_eq!(
        body_of(&unary),
        serde_json::from_str::<serde_json::Value>(RECORDED_REQUEST).expect("the fixture is JSON")
    );
    assert_eq!(unary.framing, Framing::Whole);

    let mut expected =
        serde_json::from_str::<serde_json::Value>(RECORDED_REQUEST).expect("the fixture is JSON");
    expected["stream"] = serde_json::Value::Bool(true);
    assert_eq!(body_of(&streamed), expected);
    assert_eq!(streamed.framing, Framing::Sse);
}

#[test]
fn a_serialized_config_carries_no_key_material() {
    a_config_reloads_without_its_credential(&cohere(), "cohere-test-key", |cohere| &cohere.api_key);

    let wire = cohere().chat("command-a-03-2025");
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
    let response = Bound::new(
        cohere().embeddings("embed-v4.0", None),
        RecordingHttpClient::new(EMBED_BODY),
    )
    .embed_texts_response(vec!["first".to_owned(), "second".to_owned()])
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
    let wire = cohere().embeddings("embed-english-light-v3.0", None);
    assert_eq!(wire.capabilities(), EmbeddingCapabilities::new(96, 384));
    assert_eq!(
        cohere()
            .embeddings("embed-english-light-v3.0", Some(64))
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

/// Cohere embeds ONE image per call, so a batch is a batch of requests.
#[test]
fn an_image_batch_encodes_one_request_per_image_in_input_order() {
    let encoded = cohere()
        .image_embeddings()
        .encode(vec![png(b"first"), png(b"second")], Mode::Unary)
        .expect("both images are PNGs");

    assert_eq!(encoded.requests.len(), 2);
    let images = encoded
        .requests
        .iter()
        .map(|request| match request.body() {
            Body::Bytes(bytes) => {
                let body: serde_json::Value =
                    serde_json::from_slice(bytes).expect("the body is JSON");
                body["images"][0].as_str().unwrap_or_default().to_owned()
            }
            Body::Multipart(_) => panic!("the image wire sends no multipart body"),
        })
        .collect::<Vec<_>>();
    assert_eq!(images.len(), 2);
    assert!(
        images
            .iter()
            .all(|url| url.starts_with("data:image/png;base64,"))
    );
    assert_ne!(images[0], images[1], "each request carries its own image");
}

#[test]
fn an_image_the_provider_will_not_accept_never_reaches_the_wire() {
    let rejected = cohere()
        .image_embeddings()
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
async fn an_image_batch_folds_its_replies_in_input_order() {
    let http = SequencedHttpClient::new([image_reply(0.5), image_reply(0.75)]);
    let response = Bound::new(cohere().image_embeddings(), http.clone())
        .embed_images_response(vec![png(b"first"), png(b"second")])
        .await
        .expect("both replies decode");

    assert_eq!(http.requests().len(), 2);
    assert_eq!(
        response
            .embeddings
            .iter()
            .map(|embedding| embedding.vec.as_slice())
            .collect::<Vec<_>>(),
        vec![[0.5, 1.0].as_slice(), [0.75, 1.0].as_slice()]
    );
    // An image has no text to name it: the identity is a digest of its
    // bytes, and the bytes themselves never travel back.
    assert_eq!(
        response.embeddings[0].document,
        crate::embeddings::image_document(&png(b"first"))
    );
    assert_ne!(
        response.embeddings[0].document,
        response.embeddings[1].document
    );
    // Both replies billed one image each.
    assert_eq!(response.usage.input_tokens, None);
    // The per-image sequence, in input order: Cohere bills an image embed in
    // images, not tokens, so `meta.billed_units.images` on each page is the
    // only route to an image count and every page has to be reachable.
    let pages: Vec<super::super::embeddings::ImageEmbeddingResponse> =
        serde_json::from_value(response.raw.clone()).expect("raw is the per-image array");
    assert_eq!(pages.len(), 2);
    assert_eq!(
        pages
            .iter()
            .map(|page| page.id.as_deref())
            .collect::<Vec<_>>(),
        vec![Some("img-0.5"), Some("img-0.75")]
    );
    assert!(
        pages
            .iter()
            .all(|page| page.meta.as_ref().map(|meta| meta.billed_units.images) == Some(1))
    );
}

#[tokio::test]
async fn a_single_image_embed_captures_the_bare_document() {
    let http = SequencedHttpClient::new([image_reply(0.5)]);
    let response = Bound::new(cohere().image_embeddings(), http.clone())
        .embed_images_response(vec![png(b"only")])
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
