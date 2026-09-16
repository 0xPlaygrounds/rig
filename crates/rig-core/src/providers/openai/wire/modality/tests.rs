//! The non-chat wires, driven from recorded bytes.

use super::super::tests::{recorded, recorded_json};
use super::*;
use crate::driver::Bound;
use crate::embeddings::EmbeddingModel as _;
use crate::model::ModelLister as _;
use crate::providers::openai::embedding::TEXT_EMBEDDING_ADA_002;
use crate::providers::openai::wire::{
    AZURE, Dialect, GROQ, LLAMACPP, MISTRAL, OPENAI, OpenAI, TOGETHER,
};
use crate::test_utils::RecordingHttpClient;

/// The batch the embedding cassettes were recorded against.
fn documents() -> Vec<String> {
    vec![
        "Rust values memory safety and predictable performance.".to_owned(),
        "Streaming responses arrive incrementally instead of all at once.".to_owned(),
        "Embeddings turn text into numeric vectors for similarity search.".to_owned(),
    ]
}

/// The vectors come back in request order, joined onto the inputs the
/// request carried — which are not on the wire, so only the operation's fold
/// can supply them.
#[tokio::test]
async fn a_recorded_embedding_reply_zips_onto_the_requests_inputs() {
    let reply = recorded("then", "embedding_matrix/normalized_response_is_complete.yaml");
    let wire = OpenAI::new("sk-test").embeddings("text-embedding-3-small", None);
    let bound = Bound::new(wire, RecordingHttpClient::new(reply));

    let response = bound
        .embed_texts_response(documents())
        .await
        .expect("the recorded reply decodes");

    let inputs: Vec<&str> = response
        .embeddings
        .iter()
        .map(|embedding| embedding.document.as_str())
        .collect();
    assert_eq!(
        inputs,
        documents().iter().map(String::as_str).collect::<Vec<_>>(),
        "vectors must stay paired with the input they belong to, in order"
    );
    assert!(
        response
            .embeddings
            .iter()
            .all(|embedding| embedding.vec.len() == 1536),
        "widths: {:?}",
        response
            .embeddings
            .iter()
            .map(|embedding| embedding.vec.len())
            .collect::<Vec<_>>()
    );
    assert_eq!(response.provider, "openai");
    assert_eq!(response.model.as_deref(), Some("text-embedding-3-small"));
    assert!(response.usage.input_tokens.is_some());
}

/// A batch whose reply carries the wrong number of vectors is a provider
/// defect, and the check lives in the operation's fold rather than in this
/// wire.
#[tokio::test]
async fn a_short_embedding_reply_fails_the_call() {
    let reply = r#"{"object":"list","model":"m","data":[{"object":"embedding","index":0,"embedding":[0.5]}],"usage":{"prompt_tokens":1,"total_tokens":1}}"#;
    let bound = Bound::new(
        OpenAI::new("sk-test").embeddings("text-embedding-3-small", None),
        RecordingHttpClient::new(reply),
    );
    let error = bound
        .embed_texts_response(documents())
        .await
        .expect_err("three inputs and one vector cannot pair up");
    assert!(
        error.to_string().contains('1') && error.to_string().contains('3'),
        "the error names both counts: {error}"
    );
}

/// A requested width goes on the wire in the field the dialect spells it
/// with — and the recorded request is what that looks like.
#[test]
fn a_requested_width_matches_the_recorded_request() {
    let encoded = OpenAI::new("sk-test")
        .embeddings("text-embedding-3-small", Some(512))
        .encode(documents(), Mode::Unary)
        .expect("the request encodes");
    let [request] = encoded.requests.as_slice() else {
        panic!("an embedding batch on this wire is one request");
    };
    let Body::Bytes(bytes) = request.body() else {
        panic!("an embedding request body is bytes");
    };
    let body: serde_json::Value = serde_json::from_slice(bytes).expect("the body is JSON");
    assert_eq!(
        body,
        recorded_json("when", "embedding_matrix/dimensions_request.yaml")
    );
}

/// Mistral takes a width in `output_dimension`, and `llama-server` reads no
/// width field at all — so neither can be expressed by sending `dimensions`
/// unconditionally.
#[test]
fn the_dialect_decides_the_width_field() {
    fn width_field(dialect: &Dialect, model: &str) -> Option<String> {
        let encoded = OpenAI::with_key(dialect, "k")
            .embeddings(model, Some(256))
            .encode(documents(), Mode::Unary)
            .expect("the request encodes");
        let [request] = encoded.requests.as_slice() else {
            panic!("one request")
        };
        let Body::Bytes(bytes) = request.body() else {
            panic!("bytes")
        };
        let body: serde_json::Value = serde_json::from_slice(bytes).expect("JSON");
        ["dimensions", "output_dimension"]
            .into_iter()
            .find(|field| body.get(*field).is_some())
            .map(str::to_owned)
    }

    assert_eq!(
        width_field(&OPENAI, "text-embedding-3-small"),
        Some("dimensions".to_owned())
    );
    assert_eq!(
        width_field(&MISTRAL, "codestral-embed"),
        Some("output_dimension".to_owned())
    );
    assert_eq!(
        width_field(&LLAMACPP, "nomic-embed"),
        None,
        "`llama-server` ignores a width field, so sending one would leave \
         `ndims()` describing vectors it never returned"
    );
    // OpenAI's legacy Ada model rejects the field outright.
    assert_eq!(
        width_field(
            &OPENAI,
            TEXT_EMBEDDING_ADA_002
        ),
        None
    );
}

/// Azure addresses a deployment in the URL and therefore sends no `model`.
#[test]
fn azure_sends_no_model_field() {
    let encoded = OpenAI::with_key(&AZURE, "k")
        .with_base_url("https://example.openai.azure.com")
        .with_api_version("2024-10-21")
        .embeddings("my-deployment", None)
        .encode(documents(), Mode::Unary)
        .expect("the request encodes");
    let [request] = encoded.requests.as_slice() else {
        panic!("one request")
    };
    assert_eq!(
        request.uri().to_string(),
        "https://example.openai.azure.com/openai/deployments/my-deployment/embeddings?api-version=2024-10-21"
    );
    let Body::Bytes(bytes) = request.body() else {
        panic!("bytes")
    };
    let body: serde_json::Value = serde_json::from_slice(bytes).expect("JSON");
    assert!(body.get("model").is_none(), "{body}");
}

/// A dialect that must report usage and does not is a defect, not an
/// embedding with no accounting.
#[tokio::test]
async fn a_usage_less_reply_fails_a_dialect_that_requires_usage() {
    let reply = r#"{"object":"list","model":"m","data":[{"object":"embedding","index":0,"embedding":[0.5]}]}"#;
    let error = Bound::new(
        OpenAI::new("sk-test").embeddings("text-embedding-3-small", None),
        RecordingHttpClient::new(reply),
    )
    .embed_texts_response(vec!["one".to_owned()])
    .await
    .expect_err("OpenAI always reports usage");
    assert!(error.to_string().contains("usage"), "{error}");

    // Together does not guarantee it, so the same reply succeeds there.
    let response = Bound::new(
        OpenAI::with_key(&TOGETHER, "k")
            .embeddings("togethercomputer/m2-bert-80M-8k-retrieval", None),
        RecordingHttpClient::new(reply),
    )
    .embed_texts_response(vec!["one".to_owned()])
    .await
    .expect("Together may omit usage");
    assert_eq!(response.embeddings.len(), 1);
}

/// The catalogue decodes, and Groq's context/output limits land on the
/// normalized model rather than being dropped.
#[tokio::test]
async fn a_recorded_model_listing_decodes() {
    let reply = recorded("then", "models/list_models_smoke.yaml");
    let models = Bound::new(OpenAI::new("sk-test").models(), RecordingHttpClient::new(reply))
        .list_all()
        .await
        .expect("the recorded catalogue decodes");
    assert!(!models.is_empty(), "the catalogue is not empty");
    assert!(
        models.iter().all(|model| !model.id.is_empty()),
        "every entry names a model"
    );
}

#[tokio::test]
async fn a_listing_entry_keeps_the_limits_a_dialect_reports() {
    let reply = r#"{"object":"list","data":[{"id":"llama-3.3-70b","object":"model","created":1,"owned_by":"Meta","context_window":131072,"max_completion_tokens":32768}]}"#;
    let models = Bound::new(
        OpenAI::with_key(&GROQ, "k").models(),
        RecordingHttpClient::new(reply),
    )
    .list_all()
    .await
    .expect("the catalogue decodes");
    let model = models.iter().next().expect("one entry");
    assert_eq!(model.id, "llama-3.3-70b");
    assert_eq!(model.owned_by.as_deref(), Some("Meta"));
    assert_eq!(model.context_length, Some(131_072));
    assert_eq!(model.max_output_tokens, Some(32_768));
}

/// The transcription body is multipart, with the model as a form field
/// everywhere but Azure.
#[test]
fn a_transcription_request_is_multipart() {
    let request = crate::transcription::TranscriptionRequest {
        data: b"RIFF".to_vec(),
        filename: "clip.wav".to_owned(),
        language: Some("en".to_owned()),
        prompt: None,
        temperature: None,
        additional_params: Some(serde_json::json!({"response_format": "verbose_json"})),
    };
    let encoded = OpenAI::new("sk-test")
        .transcriptions("whisper-1")
        .encode(request, Mode::Unary)
        .expect("the request encodes");
    let [http_request] = encoded.requests.as_slice() else {
        panic!("one request")
    };
    let Body::Multipart(form) = http_request.body() else {
        panic!("a transcription body is multipart");
    };
    let names: Vec<&str> = form.parts().iter().map(|part| part.name()).collect();
    assert_eq!(
        names,
        vec!["model", "file", "language", "response_format"],
        "field order is the order these endpoints were always sent in"
    );
    assert_eq!(
        http_request.uri().to_string(),
        "https://api.openai.com/v1/audio/transcriptions"
    );
}

// ── the dialect-specific modality bodies ────────────────────────────────

fn json_body(encoded: &Encoded) -> serde_json::Value {
    let [request] = encoded.requests.as_slice() else {
        panic!("one request")
    };
    let Body::Bytes(bytes) = request.body() else {
        panic!("a JSON body is bytes")
    };
    serde_json::from_slice(bytes).expect("the body is JSON")
}

/// xAI's image endpoint takes no `size` and must be asked for base64, which
/// is the only form this wire decodes.
#[cfg(feature = "image")]
#[tokio::test]
async fn the_xai_image_body_and_reply_differ_from_openais() {
    use crate::image_generation::ImageGenerationModel as _;
    use crate::providers::openai::wire::XAI;

    let request = || crate::image_generation::ImageGenerationRequest {
        prompt: "a cat".to_owned(),
        width: 1024,
        height: 1024,
        additional_params: None,
    };
    let encoded = OpenAI::with_key(&XAI, "xai-key")
        .images("grok-imagine-image-pro")
        .encode(request(), Mode::Unary)
        .expect("the request encodes");
    let [http_request] = encoded.requests.as_slice() else {
        panic!("one request")
    };
    assert_eq!(
        http_request.uri().to_string(),
        "https://api.x.ai/v1/images/generations"
    );
    let Body::Bytes(bytes) = http_request.body() else {
        panic!("bytes")
    };
    let body: serde_json::Value = serde_json::from_slice(bytes).expect("JSON");
    assert_eq!(body["response_format"], "b64_json");
    assert_eq!(body["aspect_ratio"], "1:1");
    assert!(body.get("size").is_none(), "xAI takes no `size`: {body}");

    // OpenAI's own body is the other shape.
    let openai = OpenAI::new("sk")
        .images("gpt-image-1")
        .encode(request(), Mode::Unary)
        .expect("encodes");
    let openai_body = json_body(&openai);
    assert_eq!(openai_body["size"], "1024x1024");
    assert!(openai_body.get("aspect_ratio").is_none());

    // xAI's reply carries no `created`, which the shared reply shape used to
    // require — every xAI image call would have failed to decode.
    let reply = r#"{"data":[{"b64_json":"aGk="}]}"#;
    let response = Bound::new(
        OpenAI::with_key(&XAI, "k").images("grok-imagine-image-pro"),
        RecordingHttpClient::new(reply),
    )
    .image_generation(request())
    .await
    .expect("a reply without `created` still decodes");
    assert_eq!(response.image, b"hi");
    assert_eq!(response.provider, "xai");
}

/// xAI spells its speech endpoint `/v1/tts` and takes a body of its own.
#[cfg(feature = "audio")]
#[test]
fn the_xai_speech_body_differs_from_openais() {
    use crate::providers::openai::wire::XAI;

    let request = |voice: &str| crate::audio_generation::AudioGenerationRequest {
        text: "hello".to_owned(),
        voice: voice.to_owned(),
        speed: 1.0,
        additional_params: None,
    };
    let encoded = OpenAI::with_key(&XAI, "k")
        .speech("tts-1")
        .encode(request("nova"), Mode::Unary)
        .expect("encodes");
    let [http_request] = encoded.requests.as_slice() else {
        panic!("one request")
    };
    assert_eq!(http_request.uri().to_string(), "https://api.x.ai/v1/tts");
    let body = json_body(&encoded);
    assert_eq!(body["voice_id"], "nova");
    assert_eq!(body["text"], "hello");
    assert_eq!(body["language"], "en");
    assert!(body.get("model").is_none(), "xAI's tts takes no model");

    // The voice its client defaulted to when the caller named none.
    let defaulted = OpenAI::with_key(&XAI, "k")
        .speech("tts-1")
        .encode(request(""), Mode::Unary)
        .expect("encodes");
    assert_eq!(json_body(&defaulted)["voice_id"], "eve");

    // OpenAI's own body is the other shape, at the other path.
    let openai = OpenAI::new("sk")
        .speech("tts-1")
        .encode(request("nova"), Mode::Unary)
        .expect("encodes");
    let body = json_body(&openai);
    assert_eq!(body["voice"], "nova");
    assert_eq!(body["input"], "hello");
    assert_eq!(body["model"], "tts-1");
}

/// Azure versions its speech endpoint separately from every other route, so
/// a speech request must not carry the general `api-version`.
#[cfg(feature = "audio")]
#[test]
fn azure_speech_carries_its_own_api_version() {
    let provider = OpenAI::with_key(&AZURE, "k")
        .with_base_url("https://example.openai.azure.com")
        .with_api_version("2024-10-21")
        .with_audio_api_version("2025-04-01-preview");
    let encoded = provider
        .speech("my-tts")
        .encode(
            crate::audio_generation::AudioGenerationRequest {
                text: "hi".to_owned(),
                voice: "alloy".to_owned(),
                speed: 1.0,
                additional_params: None,
            },
            Mode::Unary,
        )
        .expect("encodes");
    let [request] = encoded.requests.as_slice() else {
        panic!("one request")
    };
    assert_eq!(
        request.uri().to_string(),
        "https://example.openai.azure.com/openai/deployments/my-tts/audio/speech?api-version=2025-04-01-preview"
    );

    // Every other Azure route keeps the general version.
    let embeddings = OpenAI::with_key(&AZURE, "k")
        .with_base_url("https://example.openai.azure.com")
        .with_api_version("2024-10-21")
        .embeddings("my-embed", None)
        .encode(vec!["a".to_owned()], Mode::Unary)
        .expect("encodes");
    let [request] = embeddings.requests.as_slice() else {
        panic!("one request")
    };
    assert!(
        request.uri().to_string().ends_with("?api-version=2024-10-21"),
        "{}",
        request.uri()
    );
}

// ── reranking ───────────────────────────────────────────────────────────

/// llama.cpp's Jina-shaped reranking: the request it posts and the ranking
/// it folds.
#[tokio::test]
async fn a_recorded_rerank_reply_folds_its_ranking() {
    use crate::rerank::RerankModel as _;

    let reply = r#"{"model":"bge-reranker-v2-m3","object":"list","usage":{"prompt_tokens":37,"total_tokens":37},"results":[{"index":2,"relevance_score":0.98},{"index":0,"relevance_score":0.41},{"index":1,"relevance_score":0.02}]}"#;
    let response = Bound::new(
        OpenAI::with_key(&LLAMACPP, "").reranker("bge-reranker-v2-m3"),
        RecordingHttpClient::new(reply),
    )
    .rerank(
        "which is about cats?",
        vec![
            "dogs bark".to_owned(),
            "the sky is blue".to_owned(),
            "cats purr".to_owned(),
        ],
    )
    .await
    .expect("the reply decodes");

    // Score order is the server's, preserved as sent.
    let ranked: Vec<(usize, f64)> = response
        .results
        .iter()
        .map(|result| (result.index, result.relevance_score))
        .collect();
    assert_eq!(ranked, vec![(2, 0.98), (0, 0.41), (1, 0.02)]);
    assert_eq!(response.provider, "llamacpp");
    assert_eq!(response.model.as_deref(), Some("bge-reranker-v2-m3"));
    assert_eq!(response.usage.input_tokens, Some(37));
    assert_eq!(response.usage.total_tokens, Some(37));
    // llama.cpp never echoes the document text on this path.
    assert!(response.results.iter().all(|result| result.document.is_none()));
}

/// The text-embeddings-inference shape the same llama.cpp handler switches to
/// spells the score `score`; reading only `relevance_score` would score every
/// document zero.
#[tokio::test]
async fn a_rerank_reply_accepts_either_score_key() {
    use crate::rerank::RerankModel as _;

    let reply = r#"{"results":[{"index":0,"score":0.75}]}"#;
    let response = Bound::new(
        OpenAI::with_key(&LLAMACPP, "").reranker("r"),
        RecordingHttpClient::new(reply),
    )
    .rerank("q", vec!["a".to_owned()])
    .await
    .expect("the reply decodes");
    assert_eq!(response.results[0].relevance_score, 0.75);
    // A server that omits `model` still produced a ranking.
    assert_eq!(response.model, None);
}

#[test]
fn a_rerank_request_is_the_jina_shape() {
    let encoded = OpenAI::with_key(&LLAMACPP, "")
        .reranker("bge-reranker-v2-m3")
        .with_top_n(2)
        .encode(
            crate::operation::RerankRequest {
                query: "q".to_owned(),
                documents: vec!["a".to_owned(), "b".to_owned(), "c".to_owned()],
            },
            Mode::Unary,
        )
        .expect("encodes");
    let [request] = encoded.requests.as_slice() else {
        panic!("one request")
    };
    assert_eq!(request.uri().to_string(), "http://localhost:8080/v1/rerank");
    let body = json_body(&encoded);
    assert_eq!(
        body,
        serde_json::json!({
            "query": "q",
            "documents": ["a", "b", "c"],
            "model": "bge-reranker-v2-m3",
            "top_n": 2,
        })
    );
    // The batching hint the consumer trait asks for.
    assert_eq!(
        OpenAI::with_key(&LLAMACPP, "").reranker("r").capabilities(),
        1024
    );
}
