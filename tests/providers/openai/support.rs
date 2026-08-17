use rig::providers::openai;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};
use futures::FutureExt;

// The direct-recording client exists for the speech-synthesis scenarios alone,
// so it is gated on the same feature that compiles them — the PR gate builds
// this target with the default feature set, where an ungated helper would be
// dead code under `-D warnings`.
#[cfg(feature = "audio")]
use crate::cassettes::DirectRecordingHttpClient;

async fn openai_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, openai::Client) {
    let cassette = ProviderCassette::start("openai", spec, "https://api.openai.com/v1").await;
    let client = openai::Client::builder()
        .api_key(cassette.api_key("OPENAI_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");

    (cassette, client)
}

async fn openai_completions_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, openai::CompletionsClient) {
    let (cassette, client) = openai_cassette(spec).await;
    (cassette, client.completions_api())
}

pub(super) async fn with_openai_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = openai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Per-bug wrapper for the model-turn termination-metadata matrix
/// (`tests/cassettes/openai/turn_termination_matrix/`), rig#2184.
pub(super) async fn with_openai_turn_metadata_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

pub(super) async fn with_openai_completions_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openai::CompletionsClient) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = openai_completions_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_openai_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = openai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

pub(super) async fn with_openai_completions_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openai::CompletionsClient) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = openai_completions_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Per-bug wrapper for the Chat Completions refusal matrix
/// (`tests/cassettes/openai/refusal_matrix/`).
///
/// Yields the Responses client; cells that drive Chat Completions call
/// [`openai::Client::completions_api`] on it, so one wrapper covers both
/// surfaces of a bug whose logic lives in the shared chat-completions types.
pub(super) async fn with_openai_refusal_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

/// Per-bug wrapper for the output-token-cap spelling matrix
/// (`tests/cassettes/openai/max_completion_tokens_matrix/`).
pub(super) async fn with_openai_max_tokens_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

/// Per-bug wrapper for the truncated-turn matrix
/// (`tests/cassettes/openai/truncated_turn_matrix/`).
pub(super) async fn with_openai_truncation_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

/// Live-recorded Chat Completions log-probability transport matrix.
pub(super) async fn with_openai_chat_stream_logprobs_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    with_openai_cassette_result(spec, test_body).await
}

/// Live-recorded Chat Completions tool-call truncation contract matrix.
pub(super) async fn with_openai_tool_truncation_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    with_openai_cassette_result(spec, test_body).await
}

/// Live-recorded Chat Completions tool-call lifecycle matrix.
pub(super) async fn with_openai_tool_lifecycle_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    with_openai_cassette_result(spec, test_body).await
}

/// Live-recorded Chat Completions terminal identity, usage, and provider
/// metadata matrix.
pub(super) async fn with_openai_terminal_metadata_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    with_openai_cassette_result(spec, test_body).await
}

/// Live-recorded Chat Completions caller-history roundtrip matrix.
pub(super) async fn with_openai_history_roundtrip_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    with_openai_cassette_result(spec, test_body).await
}

/// Per-bug wrapper for the image-generation `additional_params` matrix
/// (`tests/cassettes/openai/image_params_matrix/`).
#[cfg(feature = "image")]
pub(super) async fn with_openai_image_params_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

/// Per-bug wrapper for the transcription-usage matrix
/// (`tests/cassettes/openai/transcription_usage_matrix/`).
pub(super) async fn with_openai_transcription_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

/// Per-bug wrapper for the audio-generation `additional_params` matrix
/// (`tests/cassettes/openai/audio_params_matrix/`).
///
/// Records through the direct recorder rather than the httpmock proxy: this
/// endpoint answers with raw audio, and the proxy exports bodies as strings,
/// so a recorded speech response would come back as `body: null` and replay as
/// zero bytes. The direct path stores non-UTF-8 bodies as base64.
#[cfg(feature = "audio")]
pub(super) async fn with_openai_audio_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(openai::Client<DirectRecordingHttpClient>) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette =
        ProviderCassette::start_direct_recording("openai", spec, "https://api.openai.com/v1").await;
    let client = openai::Client::builder()
        .api_key(cassette.api_key("OPENAI_API_KEY"))
        .base_url(cassette.base_url())
        .http_client(DirectRecordingHttpClient::new(cassette.direct_recorder()))
        .build()
        .expect("client should build");

    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Per-bug wrapper for the websocket error-identity matrix
/// (`tests/cassettes/openai/websocket_error_identity_matrix/`).
///
/// Uses a deliberately invalid key in **both** modes, like
/// [`with_openai_cassette_bogus_key`]: the only websocket failure the provider
/// answers with an HTTP response is the auth rejection, so that is what these
/// cells record. The upgrade is a plain HTTP GET until the provider accepts
/// it, which is why a rejected one can be recorded and replayed at all.
#[cfg(feature = "websocket")]
pub(super) async fn with_openai_websocket_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start("openai", spec, "https://api.openai.com/v1").await;
    let client = openai::Client::builder()
        .api_key("sk-invalid-websocket-edge-matrix-key")
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Like [`with_openai_cassette`], but authenticating with a deliberately
/// invalid API key — for recording real 401s with no secret near the fixture.
pub(super) async fn with_openai_cassette_bogus_key<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(openai::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start("openai", spec, "https://api.openai.com/v1").await;
    let client = openai::Client::builder()
        .api_key("sk-invalid-edge-matrix-key")
        .base_url(cassette.base_url())
        .build()
        .expect("client should build");
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
