use super::*;
use crate::completion::CompletionModel as _;
use crate::driver::Bound;
use crate::embeddings::EmbeddingModel as _;
use crate::message::AssistantContent;
use crate::model::ModelLister as _;
use crate::test_utils::{MockStreamingClient, RecordingHttpClient};
use crate::wire::secret::tests::a_config_reloads_without_its_credential;
use futures::StreamExt;

/// The recorded request of `crates/rig-cassette/fixtures/cassettes/ollama/agent/max_tokens.yaml`
/// (`POST /api/chat`): `max_tokens` rides in `options.num_predict`, and
/// `stream` is spelled out.
const RECORDED_REQUEST: &str = r#"{"messages":[{"content":"You are a concise assistant. Answer directly.","role":"system"},{"content":"In one or two sentences, explain what Rust programming language is and why memory safety matters.","role":"user"}],"model":"qwen3:4b","options":{"num_predict":24},"stream":false,"think":false}"#;

/// That cassette's recorded reply, byte for byte.
const UNARY_BODY: &str = r#"{"created_at":"1970-01-01T00:00:00Z","done":true,"done_reason":"length","eval_count":24,"eval_duration":318697665,"load_duration":4220408500,"message":{"content":"Hmm, the user wants a concise explanation of Rust and why memory safety matters. They specifically asked for one or two sentences","role":"assistant"},"model":"qwen3:4b","prompt_eval_count":42,"prompt_eval_duration":2035633750,"total_duration":6668117083}"#;

/// The same turn as a stream. The record shapes are verbatim from
/// `crates/rig-cassette/fixtures/cassettes/ollama/streaming/streaming_smoke.yaml` — content
/// records with `"done":false`, then an empty-content `"done":true` record
/// carrying the counters — and they carry the unary reply's text, model,
/// counters and `done_reason`. Ollama records the two modes from separate
/// calls, so a byte-identical turn only exists when it is built this way,
/// and the property under test is that the two fold alike.
const STREAM_BODY: &str = concat!(
    r#"{"model":"qwen3:4b","created_at":"1970-01-01T00:00:00Z","message":{"role":"assistant","content":"Hmm, the user wants a concise explanation of Rust and why memory safety matters."},"done":false}"#,
    "\n",
    r#"{"model":"qwen3:4b","created_at":"1970-01-01T00:00:00Z","message":{"role":"assistant","content":" They specifically asked for one or two sentences"},"done":false}"#,
    "\n",
    r#"{"model":"qwen3:4b","created_at":"1970-01-01T00:00:00Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"length","total_duration":6668117083,"load_duration":4220408500,"prompt_eval_count":42,"prompt_eval_duration":2035633750,"eval_count":24,"eval_duration":318697665}"#,
    "\n",
);

/// The request the recorded cell sent: a preamble, a prompt, 24 tokens, and
/// `think: false` through the provider escape hatch.
fn recorded_request() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![
            crate::message::Message::system("You are a concise assistant. Answer directly."),
            crate::message::Message::user(
                "In one or two sentences, explain what Rust programming language is and why \
                 memory safety matters.",
            ),
        ],
        documents: Vec::new(),
        tools: Vec::new(),
        temperature: None,
        max_tokens: Some(24),
        tool_choice: None,
        additional_params: Some(serde_json::json!({ "think": false })),
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
        Ollama::new().chat("qwen3:4b"),
        RecordingHttpClient::new(UNARY_BODY),
    )
    .completion(recorded_request())
    .await
    .expect("the recorded reply decodes");

    let streaming = Bound::new(
        Ollama::new().chat("qwen3:4b"),
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

    assert_eq!(buffered.choice, streamed.choice);
    assert_eq!(buffered.usage, streamed.usage);
    assert_eq!(buffered.finish_reason(), streamed.finish_reason());
    assert_eq!(buffered.model, streamed.model);
    assert_eq!(
        text_of(&buffered.choice),
        "Hmm, the user wants a concise explanation of Rust and why memory safety matters. They \
         specifically asked for one or two sentences"
    );
    assert_eq!(buffered.usage.input_tokens, Some(42));
    assert_eq!(buffered.usage.output_tokens, Some(24));
    assert_eq!(buffered.usage.total_tokens, Some(66));
    assert_eq!(
        buffered.finish_reason(),
        Some(crate::completion::FinishReason::Length)
    );
    assert_eq!(buffered.model.as_deref(), Some("qwen3:4b"));
}

/// The mode picks `stream` and the framer, and changes nothing else. Both
/// spellings are recorded (`agent/max_tokens.yaml` false,
/// `streaming/streaming_smoke.yaml` true).
#[test]
fn the_mode_is_the_only_difference_between_the_two_requests() {
    let wire = Ollama::new().chat("qwen3:4b");
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
    // A streamed reply is newline-delimited JSON, never SSE.
    assert_eq!(streamed.framing, Framing::Ndjson);
}

/// A reasoning model that puts its reasoning in `content` is split on the
/// whole reply only: the shape is the one recorded in
/// `crates/rig-cassette/fixtures/cassettes/ollama/structured_output/raw_with_thinking.yaml`.
#[tokio::test]
async fn a_buffered_reply_splits_legacy_reasoning_out_of_its_content() {
    let body = r#"{"model":"deepseek-r1","created_at":"1970-01-01T00:00:00Z","message":{"role":"assistant","content":"<think>weighing it up</think>the answer"},"done":true,"done_reason":"stop","prompt_eval_count":3,"eval_count":5}"#;
    let response = Bound::new(
        Ollama::new().chat("deepseek-r1"),
        RecordingHttpClient::new(body),
    )
    .completion(recorded_request())
    .await
    .expect("the reply decodes");

    assert_eq!(text_of(&response.choice), "the answer");
    assert!(
        response.choice.iter().any(|block| matches!(
            block,
            AssistantContent::Reasoning(reasoning)
                if reasoning.display_text() == "weighing it up"
        )),
        "the reasoning must survive into history: {:?}",
        response.choice
    );
}

/// A streamed fragment that merely opens a `<think>` marker is not a whole
/// content: splitting it would strip text the turn never finished writing.
#[tokio::test]
async fn a_streamed_fragment_is_never_split_as_legacy_reasoning() {
    let stream = concat!(
        r#"{"model":"deepseek-r1","created_at":"1970-01-01T00:00:00Z","message":{"role":"assistant","content":"<think>weighing"},"done":false}"#,
        "\n",
        r#"{"model":"deepseek-r1","created_at":"1970-01-01T00:00:00Z","message":{"role":"assistant","content":" it up</think>the answer"},"done":false}"#,
        "\n",
        r#"{"model":"deepseek-r1","created_at":"1970-01-01T00:00:00Z","message":{"role":"assistant","content":""},"done":true,"done_reason":"stop","prompt_eval_count":3,"eval_count":5}"#,
        "\n",
    );
    let bound = Bound::new(
        Ollama::new().chat("deepseek-r1"),
        MockStreamingClient {
            sse_bytes: bytes::Bytes::from(stream),
        },
    );
    let mut response = bound
        .stream(recorded_request())
        .await
        .expect("the stream opens");
    while response.next().await.is_some() {}
    let streamed = response
        .finish()
        .expect("the stream produced a terminal record");

    assert_eq!(
        text_of(&streamed.choice),
        "<think>weighing it up</think>the answer"
    );
}

/// A config is data a host persists, so what survives the round trip is the
/// part that is not a credential: a reloaded wire addresses the same daemon
/// with the same model and gets its credential from the environment again,
/// never from the file.
#[test]
fn a_serialized_config_round_trips_everything_but_the_credential() {
    let wire = Ollama::new()
        .with_base_url("http://ollama.internal:11434")
        .chat("qwen3:4b");
    let serialized = serde_json::to_string(&wire).expect("the wire serializes");
    let restored: Chat = serde_json::from_str(&serialized).expect("the wire deserializes");

    assert_eq!(restored.model, wire.model);
    assert_eq!(restored.provider.base_url, "http://ollama.internal:11434");

    // A proxied daemon does take a credential, and that one never travels.
    a_config_reloads_without_its_credential(
        &Ollama::new().with_api_key("ollama-proxy-key"),
        "ollama-proxy-key",
        |ollama| &ollama.api_key,
    );
}

#[test]
fn a_local_daemon_sends_no_authorization_header() {
    let encoded = Ollama::new()
        .chat("qwen3:4b")
        .encode(recorded_request(), Mode::Unary)
        .expect("the request encodes");
    let [request] = encoded.requests.as_slice() else {
        panic!("one request");
    };
    assert_eq!(request.uri(), "http://localhost:11434/api/chat");
    assert!(!request.headers().contains_key(http::header::AUTHORIZATION));

    let encoded = Ollama::new()
        .with_api_key("ollama-proxy-key")
        .chat("qwen3:4b")
        .encode(recorded_request(), Mode::Unary)
        .expect("the request encodes");
    let [request] = encoded.requests.as_slice() else {
        panic!("one request");
    };
    assert_eq!(
        request
            .headers()
            .get(http::header::AUTHORIZATION)
            .and_then(|value| value.to_str().ok()),
        Some("Bearer ollama-proxy-key")
    );
}

/// The reply shape of `crates/rig-cassette/fixtures/cassettes/ollama/models/list_models_smoke.yaml`
/// (`GET /api/tags`), with two of its entries.
const MODELS_BODY: &str = r#"{"models":[{"name":"all-minilm:latest","model":"all-minilm:latest","modified_at":"2026-06-19T17:15:40.188240254-07:00","size":45960996},{"name":"qwen3:4b","model":"qwen3:4b","modified_at":"2026-06-19T16:26:52.429441648-07:00","size":2497293931}]}"#;

#[tokio::test]
async fn the_model_listing_reads_every_installed_model() {
    let models = Bound::new(
        Ollama::new().models(),
        RecordingHttpClient::new(MODELS_BODY),
    )
    .list_all()
    .await
    .expect("the recorded reply decodes");

    assert_eq!(
        models
            .data
            .iter()
            .map(|model| model.id.as_str())
            .collect::<Vec<_>>(),
        vec!["all-minilm:latest", "qwen3:4b"]
    );
}

/// `POST /api/embed`'s reply shape, with two-element vectors in place of the
/// recorded 384-element ones.
const EMBED_BODY: &str = r#"{"model":"all-minilm","embeddings":[[0.5,-0.25],[0.125,0.0]],"total_duration":1000,"load_duration":10,"prompt_eval_count":6}"#;

#[tokio::test]
async fn an_embedding_reply_pairs_its_vectors_with_the_texts_that_were_sent() {
    let response = Bound::new(
        Ollama::new().embeddings("all-minilm", None),
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
    assert_eq!(response.model.as_deref(), Some("all-minilm"));
    // Every token of an embedding is input; Ollama reports one counter.
    assert_eq!(response.usage.input_tokens, Some(6));
    assert_eq!(response.usage.total_tokens, Some(6));
    assert_eq!(response.usage.output_tokens, None);
}

#[test]
fn an_embedding_wire_reports_the_models_published_width() {
    assert_eq!(
        Ollama::new().embeddings("all-minilm", None).capabilities(),
        EmbeddingCapabilities::new(1024, 384)
    );
    assert_eq!(
        Ollama::new()
            .embeddings("qwen3-embedding", Some(2048))
            .capabilities(),
        EmbeddingCapabilities::new(1024, 2048),
        "a family whose width varies by size takes the caller's"
    );
}
