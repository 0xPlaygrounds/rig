use super::*;
use crate::client::{EmbeddingsClient, RerankingClient};
use crate::embeddings::EmbeddingModel as _;
use crate::providers::openai::embedding::EncodingFormat;
use crate::test_utils::RecordingHttpClient;

#[test]
fn client_initialization() {
    let _from_new = Client::new_with(
        LlamacppApiKey::default(),
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::new() failed");
    let _from_builder = Client::builder()
        .api_key(LlamacppApiKey::default())
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("Client::builder() failed");
    let _from_url = Client::from_url_with(
        "http://localhost:8080",
        crate::test_utils::RecordingHttpClient::new(""),
    )
    .expect("Client::from_url_with() failed");
    // A bare `&str` key is accepted by the builder, which is the
    // `--api-key` path.
    let _keyed = Client::builder()
        .api_key("hunter2")
        .http_client(crate::test_utils::RecordingHttpClient::new(""))
        .build()
        .expect("keyed Client::builder() failed");
}

/// `/v1` is added when the base URL lacks it and *not* added when it has
/// it. The predecessor provider appended unconditionally, so the second
/// case produced `/v1/v1/chat/completions`.
#[test]
fn build_uri_adds_v1_once_and_only_once() {
    let ext = Llamacpp;

    for base in ["http://localhost:8080", "http://localhost:8080/"] {
        assert_eq!(
            ext.build_uri(base, "/chat/completions"),
            "http://localhost:8080/v1/chat/completions",
            "bare host base URL should gain /v1"
        );
    }

    for base in ["http://localhost:8080/v1", "http://localhost:8080/v1/"] {
        assert_eq!(
            ext.build_uri(base, "/chat/completions"),
            "http://localhost:8080/v1/chat/completions",
            "a base URL that already ends in /v1 must not double it"
        );
    }

    // Every path this provider uses composes the same way.
    assert_eq!(
        ext.build_uri("http://localhost:8080", "/embeddings"),
        "http://localhost:8080/v1/embeddings"
    );
    assert_eq!(
        ext.build_uri("http://localhost:8080", "/rerank"),
        "http://localhost:8080/v1/rerank"
    );
    assert_eq!(
        ext.build_uri("http://localhost:8080", "/models"),
        "http://localhost:8080/v1/models"
    );
}

/// llama.cpp's operational routes are relative to the server root, not to
/// `/v1` — `GET /v1/props` is a 404 — and that has to hold whichever of
/// the two accepted base-URL spellings the caller used.
#[test]
fn build_uri_keeps_the_unversioned_routes_off_the_v1_namespace() {
    let ext = Llamacpp;
    for base in [
        "http://localhost:8080",
        "http://localhost:8080/",
        "http://localhost:8080/v1",
        "http://localhost:8080/v1/",
    ] {
        assert_eq!(
            ext.build_uri(base, Llamacpp::VERIFY_PATH),
            "http://localhost:8080/props",
            "the verify path must reach the server root from base URL `{base}`"
        );
    }
    assert_eq!(
        ext.build_uri("http://localhost:8080/v1", "/health"),
        "http://localhost:8080/health"
    );
    // Only the routes llama.cpp actually serves unversioned are exempt; an
    // OpenAI route that merely looks operational is not.
    assert_eq!(
        ext.build_uri("http://localhost:8080", "/models"),
        "http://localhost:8080/v1/models"
    );
}

/// Only a *trailing* `/v1` suppresses the prefix. A reverse proxy mounted
/// under a path that merely contains the segment still needs it, because
/// the OpenAI routes hang off the mount point.
#[test]
fn build_uri_only_treats_a_trailing_v1_as_the_prefix() {
    let ext = Llamacpp;
    assert_eq!(
        ext.build_uri("https://gw.example/v1/llama", "/chat/completions"),
        "https://gw.example/v1/llama/v1/chat/completions"
    );
    assert_eq!(
        ext.build_uri("https://gw.example/v10", "/chat/completions"),
        "https://gw.example/v10/v1/chat/completions"
    );
}

/// The header is present when a key is set and **absent** when it is not.
///
/// Both halves matter: a local server started without `--api-key` must
/// keep working, and a server started with one is unreachable unless the
/// header is really sent. The predecessor provider could only ever do the
/// first, because its key type was `Nothing`.
#[tokio::test]
async fn authorization_header_is_sent_only_when_a_key_is_set() {
    let recorder = RecordingHttpClient::new("{}");
    let keyed = Client::builder()
        .api_key("hunter2")
        .http_client(recorder.clone())
        .build()
        .expect("client should build");
    let _ = keyed
        .embedding_model("m")
        .embed_texts(["hello".to_string()])
        .await;
    let sent = &recorder.requests()[0];
    assert_eq!(
        sent.headers
            .get("authorization")
            .map(|v| v.to_str().unwrap_or_default()),
        Some("Bearer hunter2"),
        "a set key must reach the wire as a bearer token"
    );

    let recorder = RecordingHttpClient::new("{}");
    let unkeyed = Client::builder()
        .api_key(LlamacppApiKey::default())
        .http_client(recorder.clone())
        .build()
        .expect("client should build");
    let _ = unkeyed
        .embedding_model("m")
        .embed_texts(["hello".to_string()])
        .await;
    let sent = &recorder.requests()[0];
    assert!(
        sent.headers.get("authorization").is_none(),
        "no key means no Authorization header at all, not an empty one"
    );

    // An empty string is treated as "no key" rather than as the literal
    // credential `Bearer `, which every server rejects.
    let recorder = RecordingHttpClient::new("{}");
    let empty = Client::builder()
        .api_key("")
        .http_client(recorder.clone())
        .build()
        .expect("client should build");
    let _ = empty
        .embedding_model("m")
        .embed_texts(["hello".to_string()])
        .await;
    assert!(
        recorder.requests()[0]
            .headers
            .get("authorization")
            .is_none(),
        "an empty key is absence, not a blank credential"
    );
}

#[tokio::test]
async fn embedding_model_preserves_v1_path_and_usage() {
    let response = r#"{
            "object": "list",
            "model": "LLaMA_CPP",
            "usage": { "prompt_tokens": 2, "total_tokens": 2 },
            "data": [{ "object": "embedding", "index": 0, "embedding": [0.1, 0.2] }]
        }"#;
    let http_client = RecordingHttpClient::new(response);
    let client = Client::builder()
        .api_key(LlamacppApiKey::default())
        .http_client(http_client.clone())
        .build()
        .expect("client should build");
    let model = client.embedding_model(super::super::LLAMA_CPP);

    let response = model
        .embed_texts_response(["hello".to_string()])
        .await
        .expect("embedding request should succeed");

    assert_eq!(response.usage.total_tokens, 2);
    assert_eq!(
        http_client.requests()[0].uri,
        "http://localhost:8080/v1/embeddings"
    );
}

#[tokio::test]
async fn embedding_model_rejects_base64_before_sending() {
    let http_client = RecordingHttpClient::new("{}");
    let client = Client::builder()
        .api_key(LlamacppApiKey::default())
        .http_client(http_client.clone())
        .build()
        .expect("client should build");
    let model = client
        .embedding_model(super::super::LLAMA_CPP)
        .encoding_format(EncodingFormat::Base64);

    let error = model
        .embed_texts(["hello".to_string()])
        .await
        .expect_err("numeric response parser should reject base64");

    assert!(matches!(
        error,
        crate::embeddings::EmbeddingError::UnsupportedResponseEncoding {
            provider: "llamacpp",
            encoding_format: "base64"
        }
    ));
    assert!(http_client.requests().is_empty());
}

/// The rerank request rig sends is the Jina-shaped body llama.cpp parses,
/// on the `/v1`-prefixed path, and `top_n` is omitted unless asked for.
#[tokio::test]
async fn rerank_request_shape_and_path() {
    use crate::rerank::RerankModel as _;

    let response = r#"{
            "model": "reranker",
            "object": "list",
            "usage": { "prompt_tokens": 42, "total_tokens": 42 },
            "results": [
                { "index": 1, "relevance_score": 0.9 },
                { "index": 0, "relevance_score": 0.1 }
            ]
        }"#;
    let http_client = RecordingHttpClient::new(response);
    let client = Client::builder()
        .api_key(LlamacppApiKey::default())
        .http_client(http_client.clone())
        .build()
        .expect("client should build");

    let reranked = client
        .rerank_model("reranker")
        .rerank("what is a panda?", vec!["hi".into(), "it is a bear".into()])
        .await
        .expect("rerank should succeed");

    let sent = &http_client.requests()[0];
    assert_eq!(sent.uri, "http://localhost:8080/v1/rerank");
    let body: serde_json::Value =
        serde_json::from_slice(&sent.body).expect("request body should be JSON");
    assert_eq!(
        body,
        serde_json::json!({
            "model": "reranker",
            "query": "what is a panda?",
            "documents": ["hi", "it is a bear"],
        }),
        "no top_n unless the caller set one"
    );

    assert_eq!(reranked.model.as_deref(), Some("reranker"));
    assert_eq!(reranked.usage.input_tokens, 42);
    assert_eq!(reranked.usage.total_tokens, 42);
    assert_eq!(
        reranked.results.iter().map(|r| r.index).collect::<Vec<_>>(),
        vec![1, 0],
        "results keep the server's ranking order"
    );
    assert!(
        reranked.results.iter().all(|r| r.document.is_none()),
        "llama.cpp does not echo documents back on this path"
    );
}

/// A 200 whose body is not a rerank payload is a *named* error, not a bare
/// serde failure.
///
/// The one path in the shared driver a live server cannot produce:
/// llama.cpp always answers the Jina shape on this route, so the case only
/// arises behind a proxy or gateway that returns something else with a 200.
/// Without the provider label the caller gets a `serde_json` message with
/// no indication of which server produced it, and this driver is shared.
#[tokio::test]
async fn rerank_names_the_provider_when_the_body_is_not_a_ranking() {
    use crate::rerank::RerankModel as _;

    let http_client = RecordingHttpClient::new(r#"{"detail":"upstream unavailable"}"#);
    let client = Client::builder()
        .api_key(LlamacppApiKey::default())
        .http_client(http_client)
        .build()
        .expect("client should build");

    let error = client
        .rerank_model("reranker")
        .rerank("q", vec!["a".into()])
        .await
        .expect_err("a 200 that is not a ranking must fail");

    match error {
        crate::rerank::RerankError::ResponseError(message) => {
            assert!(
                message.starts_with("llamacpp:"),
                "the shared driver must name which provider produced it: {message}"
            );
            assert!(
                message.contains("Jina-shaped"),
                "and say what it expected: {message}"
            );
        }
        other => panic!("expected a named ResponseError, got {other:?}"),
    }
}

#[tokio::test]
async fn rerank_sends_top_n_when_set() {
    use crate::rerank::RerankModel as _;

    let http_client = RecordingHttpClient::new(
        r#"{"model":"m","object":"list","usage":{"prompt_tokens":1,"total_tokens":1},"results":[]}"#,
    );
    let client = Client::builder()
        .api_key(LlamacppApiKey::default())
        .http_client(http_client.clone())
        .build()
        .expect("client should build");

    let _ = client
        .rerank_model("m")
        .top_n(1)
        .rerank("q", vec!["a".into(), "b".into()])
        .await
        .expect("rerank should succeed");

    let body: serde_json::Value = serde_json::from_slice(&http_client.requests()[0].body)
        .expect("request body should be JSON");
    assert_eq!(body["top_n"], serde_json::json!(1));
}
