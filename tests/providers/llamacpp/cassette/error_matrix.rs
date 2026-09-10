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
//! | [`an_unknown_model_is_ignored_rather_than_rejected`] | default | **200** | — | llama.cpp never reads `model` for routing |
//! | [`a_missing_api_key_is_a_401_the_caller_can_read`] | `--api-key` | 401 | `authentication_error` | |
//! | [`the_api_key_the_provider_sends_is_accepted`] | `--api-key` | 200 | — | the paired positive; impossible before this PR |
//! | [`verify_fails_without_the_key_and_succeeds_with_it`] | `--api-key` | 401 / 200 | `authentication_error` | why `VERIFY_PATH` is `/props`, not the public `/models` |
//! | [`the_model_listing_is_public_even_on_a_keyed_server`] | `--api-key` | **200** | — | the measurement that justifies the row above |
//! | [`embeddings_without_the_flag_are_a_501`] | default | 501 | `not_supported_error` | |
//! | [`embeddings_with_pooling_none_are_a_400`] | `--pooling none` | 400 | `invalid_request_error` | not the 500 the README implies |
//! | [`embeddings_on_a_causal_lm_return_pooled_numbers`] | causal LM + `--pooling mean` | 200 | — | the answer to "did the old embeddings cells mean anything" |
//! | [`an_embeddings_input_past_the_batch_size_is_a_500`] | `--embeddings` | 500 | `server_error` | the *batch* size, not the context size — a different limit with a different message |
//! | [`tools_without_jinja_are_a_500`] | `--no-jinja` | 500 | `server_error` | a *request* error reported as a server error |
//! | [`a_malformed_body_keeps_its_parse_error`] | default | 400 | `invalid_request_error` | mistyped field, injected through `additional_params` |
//! | [`an_oversized_output_cap_is_clamped_not_rejected`] | default | 200 | — | truncation, not an error |
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
//!   hand against b10499-6d05498). rig cannot produce one: every request body
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
//!   (verified by hand against b10499-6d05498). The streaming-error path is
//!   already covered where it differs: `context_overflow_streaming` records a
//!   400 that must survive rig's SSE funnel rather than the unary one.

use futures::StreamExt;
use rig::client::{
    CompletionClient, EmbeddingsClient, ModelListingClient, RerankingClient, VerifyClient,
};
use rig::completion::CompletionModel;
use rig::embeddings::{EmbeddingError, EmbeddingModel};
use rig::rerank::{RerankError, RerankModel};
use serde_json::{Value, json};

use crate::cassettes::{recorded_json_request, recorded_statuses_and_bodies};

use super::super::cassette_support::*;

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
            let model = client.completion_model(CASSETTE_MODEL);
            let error = model
                .completion(
                    model
                        .completion_request(overflowing_prompt())
                        .max_tokens(8)
                        .build(),
                )
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
            let model = client.completion_model(CASSETTE_MODEL);
            let request = model
                .completion_request(overflowing_prompt())
                .max_tokens(8)
                .build();

            // A status failure may surface either when the stream is opened or
            // as its first in-band item; both are the same contract as far as
            // this matrix is concerned, so the cell accepts either and asserts
            // on the error it gets.
            let error = match model.stream(request).await {
                Err(error) => error,
                Ok(mut stream) => match stream.next().await {
                    Some(Err(error)) => error,
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

/// llama.cpp answers 200 to a model it has never heard of.
///
/// A single-model `llama-server` serves whatever GGUF it was started with and
/// treats `model` as decorative — it echoes the loaded file's path back rather
/// than the string it was asked for. There is no 404 to record, and the
/// consequence is worth pinning: a typo'd model identifier is not an error
/// here, it is a silent answer from the wrong model, which is exactly the
/// failure a user would expect a 404 to protect them from.
#[tokio::test]
async fn an_unknown_model_is_ignored_rather_than_rejected() {
    with_llamacpp_cassette(
        "error_matrix/unknown_model_is_ignored",
        |client| async move {
            let model = client.completion_model("rig/definitely-not-a-llamacpp-model");
            let response = model
                .completion(
                    model
                        .completion_request("Reply with the single word: ok")
                        .max_tokens(256)
                        .build(),
                )
                .await
                .expect("llama.cpp ignores the model field rather than rejecting it");
            assert!(!response.choice.is_empty());
        },
    )
    .await;

    let recorded =
        recorded_statuses_and_bodies("llamacpp", "error_matrix/unknown_model_is_ignored");
    let (status, body) = &recorded[0];
    assert_eq!(*status, 200, "an unknown model is not a client error");
    let response: Value = serde_json::from_str(body).expect("response should be JSON");
    let request = recorded_json_request("llamacpp", "error_matrix/unknown_model_is_ignored");
    assert_eq!(
        request["model"],
        json!("rig/definitely-not-a-llamacpp-model"),
        "the request really did name a model the server has never loaded"
    );
    assert_ne!(
        response["model"], request["model"],
        "the response echoes the loaded GGUF, not the requested identifier"
    );
}

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
            let model = client.completion_model(CASSETTE_MODEL);
            let error = model
                .completion(model.completion_request("hi").max_tokens(8).build())
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

/// The same server, reached *with* the key — the capability this PR adds.
#[tokio::test]
async fn the_api_key_the_provider_sends_is_accepted() {
    with_llamacpp_api_key_cassette("error_matrix/api_key_is_accepted", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let response = model
            .completion(
                model
                    .completion_request("Reply with the single word: ok")
                    .max_tokens(256)
                    .build(),
            )
            .await
            .expect("the bearer token the provider sends must be accepted");
        assert!(!response.choice.is_empty());
    })
    .await;

    let recorded = recorded_statuses_and_bodies("llamacpp", "error_matrix/api_key_is_accepted");
    assert_eq!(recorded[0].0, 200);
}

/// `verify()` on a keyed server distinguishes a good credential from a bad one.
///
/// This is why `Llamacpp::VERIFY_PATH` is `/props` rather than the
/// `/models` its predecessor used: `GET /v1/models` and `GET /health` are the
/// only two routes llama.cpp serves *without* the API-key check, so verifying
/// against `/models` returns 200 for every key including a wrong one, which is
/// the exact opposite of what verification means.
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
                matches!(error, rig::client::VerifyError::InvalidAuthentication),
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

/// The claim `VERIFY_PATH` rests on, recorded: `GET /v1/models` answers **200
/// without a credential** on a server that rejects everything else.
///
/// The cell above explains why `/props` replaced `/models`; this is the half
/// that makes it a measurement. Against the *same* `--api-key` server and the
/// *same* unkeyed client, the model listing succeeds while a completion 401s —
/// so a provider verifying against `/models`, as the predecessor did, reports
/// a healthy credential for a server that will reject every real request.
#[tokio::test]
async fn the_model_listing_is_public_even_on_a_keyed_server() {
    with_llamacpp_missing_api_key_cassette(
        "error_matrix/model_listing_is_public",
        |client| async move {
            let models = client
                .list_models()
                .await
                .expect("`/v1/models` is served without the API-key check");
            assert!(
                !models.data.is_empty(),
                "and it really answers, rather than returning an empty list: \
                 {models:#?}"
            );
        },
    )
    .await;

    let recorded = recorded_statuses_and_bodies("llamacpp", "error_matrix/model_listing_is_public");
    assert_eq!(
        recorded[0].0, 200,
        "no credential, and a 200 — on the same server whose completions 401"
    );
    assert_eq!(
        crate::cassettes::recorded_request_paths(
            "llamacpp",
            "error_matrix/model_listing_is_public"
        ),
        vec!["/v1/models".to_string()]
    );

    // The contrast, from the sibling scenario's own bytes: the same client
    // against the same server gets a 401 the moment it asks for anything else.
    let refused = recorded_statuses_and_bodies("llamacpp", "error_matrix/missing_api_key_is_401");
    assert_eq!(
        refused.last().expect("an interaction").0,
        401,
        "the server does reject an unkeyed request — otherwise `/models` \
         answering is unremarkable"
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
                .embedding_model(CASSETTE_EMBEDDING_MODEL)
                .embed_texts(["hello".to_string()])
                .await
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
                .embedding_model(CASSETTE_EMBEDDING_MODEL)
                .embed_texts(["hello".to_string()])
                .await
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
                matches!(
                    error,
                    EmbeddingError::ProviderResponse(_) | EmbeddingError::HttpError(_)
                ),
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

/// A **causal LM** served with `--embeddings --pooling mean` answers 200 with
/// pooled hidden states.
///
/// This is the cell that decides whether the pre-merge embeddings coverage
/// meant anything: it was recorded against a causal model, and the answer is
/// that llama.cpp does not refuse — it returns numbers of the right shape that
/// are not a trained embedding of anything. Nothing in rig can tell the
/// difference, which is precisely why the real embeddings cells in this suite
/// now run against `Qwen/Qwen3-Embedding-0.6B-GGUF` and say so.
#[tokio::test]
async fn embeddings_on_a_causal_lm_return_pooled_numbers() {
    with_llamacpp_causal_embeddings_cassette(
        "error_matrix/embeddings_on_a_causal_lm",
        |client| async move {
            let embeddings = client
                .embedding_model(CASSETTE_MODEL)
                .embed_texts(["hello".to_string()])
                .await
                .expect("llama.cpp pools a causal LM rather than refusing");

            assert_eq!(embeddings.len(), 1);
            assert!(
                !embeddings[0].vec.is_empty(),
                "a pooled causal LM still returns a vector of the right shape"
            );
        },
    )
    .await;

    let recorded =
        recorded_statuses_and_bodies("llamacpp", "error_matrix/embeddings_on_a_causal_lm");
    assert_eq!(
        recorded[0].0, 200,
        "the server accepts it; the model is the caller's problem"
    );
}

// ---------------------------------------------------------------------------
// Request-shape failures llama.cpp reports as 5xx
// ---------------------------------------------------------------------------

/// `tools` on a `--no-jinja` server is a **500**, not a 400.
///
/// Without `--jinja` llama.cpp uses its own built-in ChatML template, which
/// has no way to render a tool list, and it reports that as `server_error`.
/// The classification matters: a client that treats 5xx as retriable and 4xx
/// as fatal will retry a request that can never succeed until the server is
/// restarted with a different flag.
#[tokio::test]
async fn tools_without_jinja_are_a_500() {
    with_llamacpp_no_jinja_cassette("error_matrix/tools_without_jinja", |client| async move {
        let model = client.completion_model(CASSETTE_MODEL);
        let error = model
            .completion(
                model
                    .completion_request("Add 2 and 3 using the tool.")
                    .tool(rig::tool::tool_definition(&crate::support::Adder))
                    .max_tokens(64)
                    .build(),
            )
            .await
            .expect_err("a --no-jinja server cannot render a tool list");

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
            body.contains("--jinja"),
            "the flag to restart the server with is the actionable half: {body}"
        );
    })
    .await;

    recorded_error("error_matrix/tools_without_jinja", 500, "server_error");
}

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
            let model = client.completion_model(CASSETTE_MODEL);
            let error = model
                .completion(
                    model
                        .completion_request("hi")
                        .max_tokens(8)
                        // `temperature` is a number on this wire; a string is a
                        // type error the server reports before generating.
                        .additional_params(json!({ "temperature": "hot" }))
                        .build(),
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

/// An output cap far past the context window is **clamped**, not rejected.
#[tokio::test]
async fn an_oversized_output_cap_is_clamped_not_rejected() {
    with_llamacpp_small_context_cassette(
        "error_matrix/oversized_output_cap",
        |client| async move {
            let model = client.completion_model(CASSETTE_MODEL);
            let response = model
                .completion(
                    model
                        .completion_request("Say ok.")
                        // Two orders of magnitude past the server's -c 512.
                        .max_tokens(100_000)
                        .build(),
                )
                .await
                .expect("llama.cpp clamps an oversized cap rather than refusing");
            assert!(!response.choice.is_empty());
        },
    )
    .await;

    let request = recorded_json_request("llamacpp", "error_matrix/oversized_output_cap");
    assert_eq!(
        request["max_tokens"],
        json!(100_000),
        "the request really did ask for more than the context holds"
    );
    let recorded = recorded_statuses_and_bodies("llamacpp", "error_matrix/oversized_output_cap");
    assert_eq!(recorded[0].0, 200);
    let response: Value = serde_json::from_str(&recorded[0].1).expect("response should be JSON");
    let completion_tokens = response["usage"]["completion_tokens"]
        .as_u64()
        .expect("usage should report completion tokens");
    assert!(
        completion_tokens < 512,
        "the server generated {completion_tokens} tokens, so the cap was clamped to what \
         the context allows rather than honoured"
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
                .rerank_model(CASSETTE_RERANK_MODEL)
                .rerank("what is a panda?", vec!["hi".into(), "it is a bear".into()])
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
                !matches!(error, RerankError::JsonError(_)),
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
            .rerank_model(CASSETTE_RERANK_MODEL)
            .rerank("what is a panda?", Vec::new())
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
                .embedding_model(CASSETTE_EMBEDDING_MODEL)
                .embed_texts([oversized])
                .await
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
