//! Provider-error matrix for `llama-server`.
//!
//! Nothing in rig had ever recorded a single non-2xx response from llama.cpp
//! before this suite: both pre-merge corpora were happy paths end to end. Error
//! paths are where OpenAI-compatible servers diverge most from OpenAI, and
//! llama.cpp diverges on all four axes at once — which status it picks, which
//! `type` string it uses, which extra fields it attaches, and which failures it
//! declines to treat as failures at all.
//!
//! Every cell asserts the **class and the preserved body**, never a literal id,
//! and reads the recorded bytes back to prove its own premise.
//!
//! | Cell | Server | Status | `type` | Notes |
//! | --- | --- | --- | --- | --- |
//! | [`context_overflow_preserves_the_token_counts`] | `-c 512` | 400 | `exceed_context_size_error` | carries `n_prompt_tokens` + `n_ctx` |
//! | [`streaming_context_overflow_matches_the_blocking_envelope`] | `-c 512` | 400 | `exceed_context_size_error` | the 400 lands before the SSE stream opens |
//! | [`a_missing_api_key_is_a_401_the_caller_can_read`] | `--api-key` | 401 | `authentication_error` | |
//! | [`verify_fails_without_the_key_and_succeeds_with_it`] | `--api-key` | 401 / 200 | `authentication_error` | why `verify_path` is `/props` |
//! | [`the_model_listing_requires_the_key_on_a_keyed_server`] | `--api-key` | 401 | `authentication_error` | `/v1/models` was public on b10499 |
//! | [`embeddings_without_the_flag_are_a_501`] | default | 501 | `not_supported_error` | |
//! | [`embeddings_with_pooling_none_are_a_400`] | `--pooling none` | 400 | `invalid_request_error` | not the 500 the README implies |
//! | [`an_embeddings_input_past_the_batch_size_is_a_500`] | `--embeddings` | 500 | `server_error` | the *batch* size, not the context size — a different limit with a different message |
//! | [`a_malformed_body_keeps_its_parse_error`] | default | 400 | `invalid_request_error` | mistyped field, injected through `additional_params` |
//! | [`rerank_without_a_reranker_is_a_501`] | default | 501 | `not_supported_error` | |
//! | [`rerank_with_an_empty_document_list_is_a_400`] | `--reranking` | 400 | `invalid_request_error` | |
//!
//! Two rows are the interesting ones. llama.cpp reports **`tools` without
//! `--jinja`** as `500 server_error` even though it is something the caller
//! got wrong; a client that retries 5xx and not 4xx will retry a request that
//! can never succeed until the server is restarted with a different flag. And
//! an **unknown model is not an error at all** — the field is decorative on a
//! single-model server, so a typo'd model identifier silently answers from
//! whatever is loaded.
//!
//! # Dropped, with reasons
//!
//! * **A syntactically malformed request body.** llama.cpp answers
//!   `500 server_error` carrying the nlohmann/json parse error (verified by
//!   hand against b10964-b29c606e2). rig cannot produce one: every request body
//!   it emits is serialized from typed values, and `additional_params` is a
//!   `serde_json::Value`, which is valid JSON by construction. The reachable
//!   neighbour — a field of the wrong *type* — is
//!   [`a_malformed_body_keeps_its_parse_error`], and it is a 400.
//! * **A 401 whose key is wrong rather than missing.** llama.cpp compares the
//!   key for equality and answers the same `401 authentication_error` either
//!   way; the middleware runs before routing, so even an unknown path 401s.
//!   A second cell would record identical bytes.
//! * **A streaming 401.** The API-key middleware runs before the handler, so a
//!   `stream: true` request without the key answers the same `401` body before
//!   any event stream opens — byte-identical to the blocking cell above
//!   (verified by hand against b10964-b29c606e2). The streaming-error path is
//!   already covered where it differs: `context_overflow_streaming` records a
//!   400 that must survive rig's SSE funnel rather than the unary one.

use futures::StreamExt;
use rig::error::ProviderError;
use serde_json::{Value, json};

use crate::cassettes::{recorded_json_request, recorded_statuses_and_bodies};

use super::super::cassette_support::*;
use rig::completion::CompletionRequest;
use rig::operation::RerankRequest;

/// A prompt long enough to overflow a 512-token context and short enough to
/// keep the fixture readable.
fn overflowing_prompt() -> String {
    "the quick brown fox jumps over the lazy dog. ".repeat(200)
}

/// llama.cpp's error envelope is always `{"error": {code, message, type, …}}`.
///
/// Asserting the shape rather than the text is what keeps these cells from
/// pinning a wording change, while still failing if the envelope itself is
/// flattened or swallowed.
fn assert_llamacpp_envelope(body: &str, expected_type: &str) -> Value {
    let json: Value = serde_json::from_str(body)
        .unwrap_or_else(|error| panic!("llama.cpp error body should be JSON: {error}: {body}"));
    let error = json
        .get("error")
        .and_then(Value::as_object)
        .unwrap_or_else(|| panic!("llama.cpp nests its error envelope under `error`: {json}"));
    assert_eq!(
        error.get("type").and_then(Value::as_str),
        Some(expected_type),
        "error type: {json}"
    );
    assert!(
        error
            .get("message")
            .and_then(Value::as_str)
            .is_some_and(|message| !message.trim().is_empty()),
        "an error must carry a non-empty message: {json}"
    );
    json
}

/// The status and body a *recorded* interaction carries, so a cell proves its
/// premise from the bytes rather than from what the client made of them.
fn recorded_error(scenario: &str, expected_status: u16, expected_type: &str) -> Value {
    let recorded = recorded_statuses_and_bodies("llamacpp", scenario);
    let (status, body) = recorded
        .last()
        .unwrap_or_else(|| panic!("{scenario} should have recorded an interaction"));
    assert_eq!(
        *status, expected_status,
        "{scenario}: recorded status\nbody: {body}"
    );
    assert_llamacpp_envelope(body, expected_type)
}

// ---------------------------------------------------------------------------
// Context overflow
// ---------------------------------------------------------------------------

/// A 400 whose body names both sides of the comparison.
///
/// `n_prompt_tokens` and `n_ctx` are llama.cpp's own additions to the OpenAI
/// error shape and they are the only actionable part of the failure — "try
/// increasing it" is not, by itself, a number to increase it to. The point of
/// the cell is that they survive rig's error funnel to
/// `provider_response_json()` rather than being reduced to a message string.
#[tokio::test]
async fn context_overflow_preserves_the_token_counts() {
    with_llamacpp_small_context_cassette(
        "error_matrix/context_overflow_blocking",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let error = model
                .call(CompletionRequest::new(overflowing_prompt()).max_tokens(8))
                .await
                .expect_err("a prompt past the context window must fail");

            let status = error
                .provider_response_status()
                .expect("the 400 must reach the caller");
            assert_eq!(status.as_u16(), 400, "{error}");

            let json = error
                .provider_response_json()
                .expect("the error body must be readable as JSON")
                .expect("the error body must be present");
            assert_eq!(json["error"]["type"], json!("exceed_context_size_error"));
            let n_prompt_tokens = json["error"]["n_prompt_tokens"]
                .as_u64()
                .expect("n_prompt_tokens must survive into the caller's error");
            let n_ctx = json["error"]["n_ctx"]
                .as_u64()
                .expect("n_ctx must survive into the caller's error");
            assert_eq!(n_ctx, 512, "the recording server was started with -c 512");
            assert!(
                n_prompt_tokens > n_ctx,
                "the failure is that {n_prompt_tokens} > {n_ctx}"
            );
        },
    )
    .await;

    recorded_error(
        "error_matrix/context_overflow_blocking",
        400,
        "exceed_context_size_error",
    );
}

/// The same overflow, requested as a stream.
///
/// llama.cpp validates the prompt before it opens the event stream, so this is
/// a plain 400 with a JSON body rather than an SSE frame carrying an error —
/// which means the body has to survive rig's *streaming* error path, a
/// different funnel from the blocking one. Both are asserted to produce the
/// same envelope, because a provider whose streaming errors degrade to a
/// transport string is the failure mode this pair exists to catch.
#[tokio::test]
async fn streaming_context_overflow_matches_the_blocking_envelope() {
    with_llamacpp_small_context_cassette(
        "error_matrix/context_overflow_streaming",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let request = CompletionRequest::new(overflowing_prompt()).max_tokens(8);

            // A status failure may surface either when the stream is opened or
            // as its first in-band item; both are the same contract as far as
            // this matrix is concerned, so the cell accepts either and asserts
            // on the error it gets.
            let error = match model.stream(request) {
                Err(error) => rig::ErrorReport::from(&error),
                Ok(mut stream) => match stream.next().await {
                    Some(Err(error)) => rig::ErrorReport::from(&error),
                    other => panic!("expected a preserved error, got {other:?}"),
                },
            };

            let status = error
                .provider_response_status()
                .expect("the 400 must reach the caller on the streaming path too");
            assert_eq!(status.as_u16(), 400, "{error}");
            let json = error
                .provider_response_json()
                .expect("the streaming error body must be readable as JSON")
                .expect("the streaming error body must be present");
            assert_eq!(json["error"]["type"], json!("exceed_context_size_error"));
            assert_eq!(json["error"]["n_ctx"], json!(512));
        },
    )
    .await;

    let streaming = recorded_error(
        "error_matrix/context_overflow_streaming",
        400,
        "exceed_context_size_error",
    );
    let blocking = recorded_error(
        "error_matrix/context_overflow_blocking",
        400,
        "exceed_context_size_error",
    );
    assert_eq!(
        streaming["error"]["type"], blocking["error"]["type"],
        "the streaming and blocking envelopes must agree"
    );
    assert_eq!(
        streaming["error"]["n_ctx"], blocking["error"]["n_ctx"],
        "both transports report the same context size"
    );

    // The premise: the request really did ask for a stream.
    let request = recorded_json_request("llamacpp", "error_matrix/context_overflow_streaming");
    assert_eq!(request["stream"], json!(true));
}

// ---------------------------------------------------------------------------
// The model field
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Authentication
// ---------------------------------------------------------------------------

/// `llama-server --api-key <key>`, reached without one.
///
/// This whole pair was **unreachable before this PR**: the provider being
/// replaced used `Nothing` as its `ApiKey` type, which cannot emit an
/// `Authorization` header at all, so a secured deployment could only ever
/// produce this 401 and never the 200 below.
#[tokio::test]
async fn a_missing_api_key_is_a_401_the_caller_can_read() {
    with_llamacpp_missing_api_key_cassette(
        "error_matrix/missing_api_key_is_401",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let error = model
                .call(CompletionRequest::new("hi").max_tokens(8))
                .await
                .expect_err("a server started with --api-key must reject an unkeyed request");

            assert_eq!(
                error
                    .provider_response_status()
                    .expect("the 401 must reach the caller")
                    .as_u16(),
                401,
                "{error}"
            );
            let json = error
                .provider_response_json()
                .expect("the 401 body must be readable as JSON")
                .expect("the 401 body must be present");
            assert_eq!(json["error"]["type"], json!("authentication_error"));
        },
    )
    .await;

    recorded_error(
        "error_matrix/missing_api_key_is_401",
        401,
        "authentication_error",
    );
}

/// `verify()` on a keyed server distinguishes a good credential from a bad one.
///
/// This is why the `LLAMACPP` dialect's `verify_path` quirk is `/props`
/// rather than the `/models` its predecessor used: llama.cpp b10499 served
/// `GET /v1/models` without the API-key check, so verifying against it
/// returned 200 for every key. `/props` is keyed on every build this suite
/// has recorded.
#[tokio::test]
async fn verify_fails_without_the_key_and_succeeds_with_it() {
    with_llamacpp_missing_api_key_cassette(
        "error_matrix/verify_rejects_a_missing_key",
        |client| async move {
            let error = client
                .verify()
                .await
                .expect_err("verification must fail without the key");
            assert!(
                matches!(
                    &error,
                    rig::error::ProviderError::InvalidAuthentication(response)
                        if response.status.map(|status| status.as_u16()) == Some(401)
                ),
                "a 401 from the verify path must classify as invalid authentication, got: {error}"
            );
        },
    )
    .await;

    with_llamacpp_api_key_cassette("error_matrix/verify_accepts_the_key", |client| async move {
        client
            .verify()
            .await
            .expect("verification must succeed with the key");
    })
    .await;

    for (scenario, expected) in [
        ("error_matrix/verify_rejects_a_missing_key", 401),
        ("error_matrix/verify_accepts_the_key", 200),
    ] {
        let recorded = recorded_statuses_and_bodies("llamacpp", scenario);
        assert_eq!(recorded[0].0, expected, "{scenario}");
    }
    let paths =
        crate::cassettes::recorded_request_paths("llamacpp", "error_matrix/verify_accepts_the_key");
    assert_eq!(
        paths,
        vec!["/props".to_string()],
        "verification must hit the API-key-checked route, not the public one"
    );
}

/// `GET /v1/models` checks the API key on a keyed server.
///
/// llama.cpp b10499 served the listing without the key, which is why
/// `verify_path` moved to `/props`; b10964 keys it like every other route
/// except `/health`.
#[tokio::test]
async fn the_model_listing_requires_the_key_on_a_keyed_server() {
    with_llamacpp_missing_api_key_cassette(
        "error_matrix/model_listing_is_keyed",
        |client| async move {
            let error = client
                .list_models()
                .await
                .expect_err("`/v1/models` must refuse an unkeyed client");
            assert!(
                matches!(
                    &error,
                    rig::error::ProviderError::ProviderResponse(response)
                        if response.status.map(|status| status.as_u16()) == Some(401)
                ),
                "the refusal is a readable 401: {error:?}"
            );
        },
    )
    .await;

    let recorded = recorded_statuses_and_bodies("llamacpp", "error_matrix/model_listing_is_keyed");
    assert_eq!(recorded[0].0, 401);
    assert_eq!(
        crate::cassettes::recorded_request_paths("llamacpp", "error_matrix/model_listing_is_keyed"),
        vec!["/v1/models".to_string()]
    );
}

// ---------------------------------------------------------------------------
// Embeddings
// ---------------------------------------------------------------------------

/// A server started without `--embeddings` answers 501 to the whole capability.
#[tokio::test]
async fn embeddings_without_the_flag_are_a_501() {
    with_llamacpp_cassette(
        "error_matrix/embeddings_without_the_flag",
        |client| async move {
            let error = client
                .embedding(CASSETTE_EMBEDDING_MODEL, None)
                .call(vec!["hello".to_string()])
                .await
                .map(|response| response.embeddings)
                .expect_err("a server without --embeddings must refuse");

            assert_eq!(
                error
                    .provider_response_status()
                    .expect("the 501 must reach the caller")
                    .as_u16(),
                501,
                "{error}"
            );
            let body = error
                .provider_response_body()
                .expect("the 501 body must be preserved");
            assert!(
                body.contains("--embeddings"),
                "llama.cpp names the flag to start the server with; that is the \
                 actionable half and it must survive: {body}"
            );
        },
    )
    .await;

    recorded_error(
        "error_matrix/embeddings_without_the_flag",
        501,
        "not_supported_error",
    );
}

/// `--pooling none` returns one vector per *token*, which the OpenAI
/// embeddings wire cannot express — so llama.cpp refuses with a **400**.
///
/// Recorded because the status is not the one llama.cpp's own README implies,
/// and because "the server is misconfigured" and "the request is wrong" are
/// different things for a caller deciding whether to retry.
#[tokio::test]
async fn embeddings_with_pooling_none_are_a_400() {
    with_llamacpp_pooling_none_cassette(
        "error_matrix/embeddings_with_pooling_none",
        |client| async move {
            let error = client
                .embedding(CASSETTE_EMBEDDING_MODEL, None)
                .call(vec!["hello".to_string()])
                .await
                .map(|response| response.embeddings)
                .expect_err("--pooling none is not OpenAI-compatible");

            assert_eq!(
                error
                    .provider_response_status()
                    .expect("the 400 must reach the caller")
                    .as_u16(),
                400,
                "{error}"
            );
            assert!(
                matches!(error, ProviderError::ProviderResponse(_)),
                "the provider envelope must be preserved rather than reduced: {error}"
            );
        },
    )
    .await;

    recorded_error(
        "error_matrix/embeddings_with_pooling_none",
        400,
        "invalid_request_error",
    );
}

// ---------------------------------------------------------------------------
// Request-shape failures llama.cpp reports as 5xx
// ---------------------------------------------------------------------------

/// A body llama.cpp cannot parse is a **500** carrying the parser's own
/// message.
///
/// Injected through `additional_params`, which is the only way a rig caller
/// can put arbitrary bytes on this wire — the typed request cannot produce a
/// malformed body on its own. What is under test is rig's funnel, not rig's
/// serializer: the parse error must arrive as a preserved provider envelope
/// rather than as a generic transport failure.
#[tokio::test]
async fn a_malformed_body_keeps_its_parse_error() {
    with_llamacpp_cassette(
        "error_matrix/malformed_request_field",
        |client| async move {
            let model = client.completion(CASSETTE_MODEL);
            let error = model
                .call(
                    CompletionRequest::new("hi")
                        .max_tokens(8)
                        // `temperature` is a number on this wire; a string is a
                        // type error the server reports before generating.
                        .additional_params(json!({ "temperature": "hot" })),
                )
                .await
                .expect_err("a mistyped parameter must fail");

            let status = error
                .provider_response_status()
                .expect("the status must reach the caller");
            assert_eq!(status.as_u16(), 400, "{error}");
            let body = error
                .provider_response_body()
                .expect("the body must be preserved");
            assert!(
                body.contains("temperature"),
                "the offending field must be named in the preserved body: {body}"
            );
        },
    )
    .await;

    let json = recorded_error(
        "error_matrix/malformed_request_field",
        400,
        "invalid_request_error",
    );
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("temperature")),
        "{json}"
    );
}

// ---------------------------------------------------------------------------
// Reranking
// ---------------------------------------------------------------------------

/// The rerank route exists on every server and 501s unless `--reranking` was
/// passed.
#[tokio::test]
async fn rerank_without_a_reranker_is_a_501() {
    with_llamacpp_cassette(
        "error_matrix/rerank_without_a_reranker",
        |client| async move {
            let error = client
                .rerank(CASSETTE_RERANK_MODEL)
                .call(RerankRequest {
                    query: "what is a panda?".to_owned(),
                    documents: vec!["hi".into(), "it is a bear".into()],
                })
                .await
                .expect_err("a server without --reranking must refuse");

            assert_eq!(
                error
                    .provider_response_status()
                    .expect("the 501 must reach the caller")
                    .as_u16(),
                501,
                "{error}"
            );
            let body = error
                .provider_response_body()
                .expect("the 501 body must be preserved");
            assert!(body.contains("--reranking"), "{body}");
            assert!(
                !matches!(error, ProviderError::Json(_)),
                "a 501 must not be misread as a decode failure: {error}"
            );
        },
    )
    .await;

    recorded_error(
        "error_matrix/rerank_without_a_reranker",
        501,
        "not_supported_error",
    );
}

/// An empty document list is a 400 from the server, not a client-side no-op.
///
/// rig's `RerankModel` has no minimum-length contract, so the request really is
/// sent; the cell pins that the refusal survives as an envelope rather than
/// becoming an empty successful ranking.
#[tokio::test]
async fn rerank_with_an_empty_document_list_is_a_400() {
    with_llamacpp_rerank_cassette("error_matrix/rerank_empty_documents", |client| async move {
        let error = client
            .rerank(CASSETTE_RERANK_MODEL)
            .call(RerankRequest {
                query: "what is a panda?".to_owned(),
                documents: Vec::new(),
            })
            .await
            .expect_err("an empty document list is refused by the server");

        assert_eq!(
            error
                .provider_response_status()
                .expect("the 400 must reach the caller")
                .as_u16(),
            400,
            "{error}"
        );
    })
    .await;

    let json = recorded_error(
        "error_matrix/rerank_empty_documents",
        400,
        "invalid_request_error",
    );
    assert!(
        json["error"]["message"]
            .as_str()
            .is_some_and(|message| message.contains("documents")),
        "{json}"
    );
    let request = recorded_json_request("llamacpp", "error_matrix/rerank_empty_documents");
    assert_eq!(
        request["documents"],
        json!([]),
        "the empty list really was sent rather than short-circuited"
    );
}

/// An embeddings input larger than the server's physical batch is a **500**.
///
/// A different limit from the context window, with a different message and a
/// different remedy: `-c` governs the chat context, `-b`/`--ubatch-size` the
/// embedding batch, and llama.cpp names the second one when it is the one that
/// was hit. Recorded because a caller who reads "too large to process" and
/// reaches for `-c` will not fix it — and because it is the fourth caller
/// error in this corpus that arrives as a 5xx.
#[tokio::test]
async fn an_embeddings_input_past_the_batch_size_is_a_500() {
    with_llamacpp_embeddings_cassette(
        "error_matrix/embeddings_input_past_the_batch",
        |client| async move {
            // Well past the 512-token physical batch the recording server runs
            // with, and deterministic.
            let oversized = "word ".repeat(4_000);
            let error = client
                .embedding(CASSETTE_EMBEDDING_MODEL, None)
                .call(vec![oversized])
                .await
                .map(|response| response.embeddings)
                .expect_err("an input past the physical batch must fail");

            assert_eq!(
                error
                    .provider_response_status()
                    .expect("the status must reach the caller")
                    .as_u16(),
                500,
                "{error}"
            );
            let body = error
                .provider_response_body()
                .expect("the body must be preserved");
            assert!(
                body.contains("batch size"),
                "the limit that was actually hit must survive — a caller who reads \
                 this and reaches for `-c` is fixing the wrong thing: {body}"
            );
        },
    )
    .await;

    let json = recorded_error(
        "error_matrix/embeddings_input_past_the_batch",
        500,
        "server_error",
    );
    assert!(
        json["error"]["message"].as_str().is_some_and(
            |message| message.contains("batch size") && !message.contains("context size")
        ),
        "the message names the batch, and does not confuse it with the context: {json}"
    );
}
