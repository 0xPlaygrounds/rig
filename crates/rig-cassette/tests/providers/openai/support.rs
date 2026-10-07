use rig::http_client::DynHttpClient;
use rig::providers::openai::{OpenAIConfig, Route};
use rig_test_support::cassette_models::OpenAiModels;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use crate::cassettes::{CassetteSpec, ProviderCassette};
use futures::FutureExt;

// The direct-recording client exists for the speech-synthesis scenarios alone,
// so it is gated on the same feature that compiles them — the PR gate builds
// this target with the default feature set, where an ungated helper would be
// dead code under `-D warnings`.
use crate::cassettes::DirectRecordingHttpClient;

/// The one OpenAI configuration a recorded endpoint serves, bound — twice.
///
/// `openai::Client` was `Client<OpenAIResponses, H>` — the Responses API —
/// and the client layer swapped a marker type to reach `/chat/completions`
/// on the same credential and base URL. In the wire model the endpoint is
/// configuration: [`OpenAI`] routes every completion it is asked for to
/// the dialect's flagship, `POST /responses`, unless
/// [`OpenAI::with_route`] chose `/chat/completions` once, and the modality
/// wires serve every other OpenAI REST route (embeddings, transcriptions,
/// images, speech, model listing, verification). So the same credential is
/// held here on both routes, and a cell spells the route it drives by the
/// field it reads: `rig::AgentBuilder::new(client.openai.completion(model))` beside
/// `rig::AgentBuilder::new(client.chat.completion(model))`, with `.chat(model)` / `.responses(model)`
/// still naming a typed wire when a cell reads the native reply.
pub(super) struct OpenAiCassette {
    /// The models of the configuration on its flagship route.
    pub(super) openai: OpenAiModels,
    /// The models of the same configuration routed to Chat Completions.
    pub(super) chat: OpenAiModels,
}

impl OpenAiCassette {
    /// The configuration for `api_key` at `base_url`, over `http`.
    fn new(
        api_key: impl Into<String>,
        base_url: impl Into<String>,
        http: impl rig::http_client::HttpClientExt + 'static,
    ) -> Self {
        let http = DynHttpClient::new(http);
        let openai = OpenAIConfig::new(api_key).with_base_url(base_url);
        let chat = openai.clone().with_route(Route::Chat);
        Self {
            openai: OpenAiModels::new(openai, http.clone()),
            chat: OpenAiModels::new(chat, http),
        }
    }
}

async fn openai_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, OpenAiCassette) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "openai",
        spec,
        "https://api.openai.com/v1",
    )
    .await;
    let openai = OpenAiCassette::new(
        cassette.api_key("OPENAI_API_KEY"),
        cassette.base_url(),
        rig_test_support::cassettes::local_http(),
    );

    (cassette, openai)
}

async fn openai_completions_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, OpenAiModels) {
    let (cassette, openai) = openai_cassette(spec).await;
    (cassette, openai.chat)
}

/// The effect corpus's retrieval matrix (Matrix A):
/// `crates/rig-cassette/fixtures/cassettes/openai/corpus_retrieval/`.
pub(super) async fn with_openai_corpus_retrieval_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

/// The effect corpus's provider-breadth matrix (Matrix N):
/// `crates/rig-cassette/fixtures/cassettes/openai/corpus_breadth/`.
pub(super) async fn with_openai_corpus_breadth_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

/// The effect corpus's delta-wire matrix (Matrix K):
/// `crates/rig-cassette/fixtures/cassettes/openai/corpus_delta/`.
pub(super) async fn with_openai_corpus_delta_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

/// The effect corpus's host-families matrix (Matrix I):
/// `crates/rig-cassette/fixtures/cassettes/openai/corpus_host/`.
pub(super) async fn with_openai_corpus_host_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

/// The effect corpus's output-mode matrix (Matrix H):
/// `crates/rig-cassette/fixtures/cassettes/openai/corpus_output/`.
pub(super) async fn with_openai_corpus_output_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

pub(super) async fn with_openai_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    let (cassette, openai) = openai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(openai)).catch_unwind().await;
    crate::cassettes::checkpoint_attempt(&cassette, "openai", spec.scenario()).await;
    cassette.finish_after_test(result).await;
}

/// Long-run caching recordings
/// (`crates/rig-cassette/fixtures/cassettes/openai/long_run_caching/`). The
/// body gets both routes' models and the session's clock, whose record-only
/// pause spaces a retried turn; it returns whatever the test asserts on after
/// the session is finished.
pub(super) async fn with_openai_long_run_cassette<F, Fut, R>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> R
where
    F: FnOnce(OpenAiCassette, crate::cassettes::CassetteClock) -> Fut,
    Fut: Future<Output = R>,
{
    let spec = spec.into();
    let (cassette, openai) = openai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(openai, cassette.clock()))
        .catch_unwind()
        .await;
    crate::cassettes::checkpoint_attempt(&cassette, "openai", spec.scenario()).await;
    match result {
        Ok(value) => {
            cassette.finish_after_test(Ok(())).await;
            value
        }
        Err(payload) => {
            cassette.finish_after_test(Err(payload)).await;
            unreachable!("finishing a failed session resumes its panic")
        }
    }
}

/// One model's recorded session
/// (`crates/rig-cassette/fixtures/cassettes/openai/models/<model>/`): both
/// routes' models and the session's clock.
pub(super) async fn with_openai_model_session_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> rig_test_support::model_session::Session
where
    F: FnOnce(OpenAiCassette, crate::cassettes::CassetteClock) -> Fut,
    Fut: Future<Output = rig_test_support::model_session::Session>,
{
    let spec = spec.into();
    let (cassette, openai) = openai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(openai, cassette.clock()))
        .catch_unwind()
        .await;
    crate::cassettes::checkpoint_attempt(&cassette, "openai", spec.scenario()).await;
    match result {
        Ok(session) => {
            cassette.finish_after_test(Ok(())).await;
            session
        }
        Err(payload) => {
            cassette.finish_after_test(Err(payload)).await;
            unreachable!("finishing a failed session resumes its panic")
        }
    }
}

pub(super) async fn with_openai_completions_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, chat) = openai_completions_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(chat)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_openai_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, openai) = openai_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(openai)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

pub(super) async fn with_openai_completions_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(OpenAiModels) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, chat) = openai_completions_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(chat)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Per-bug wrapper for the Chat Completions refusal matrix
/// (`crates/rig-cassette/fixtures/cassettes/openai/refusal_matrix/`).
///
/// Yields both routes; cells that drive Chat Completions take
/// [`OpenAiCassette::chat`], so one wrapper covers both surfaces of a bug
/// whose logic lives in the shared chat-completions types.
pub(super) async fn with_openai_refusal_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

/// Per-bug wrapper for the output-token-cap spelling matrix
/// (`crates/rig-cassette/fixtures/cassettes/openai/max_completion_tokens_matrix/`).
pub(super) async fn with_openai_max_tokens_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiCassette) -> Fut,
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
    F: FnOnce(OpenAiCassette) -> Fut,
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
    F: FnOnce(OpenAiCassette) -> Fut,
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
    F: FnOnce(OpenAiCassette) -> Fut,
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
    F: FnOnce(OpenAiCassette) -> Fut,
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
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    with_openai_cassette_result(spec, test_body).await
}

/// Per-bug wrapper for the image-generation `additional_params` matrix
/// (`crates/rig-cassette/fixtures/cassettes/openai/image_params_matrix/`).
pub(super) async fn with_openai_image_params_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

/// Per-bug wrapper for the transcription-usage matrix
/// (`crates/rig-cassette/fixtures/cassettes/openai/transcription_usage_matrix/`).
pub(super) async fn with_openai_transcription_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

/// Per-bug wrapper for the audio-generation `additional_params` matrix
/// (`crates/rig-cassette/fixtures/cassettes/openai/audio_params_matrix/`).
///
/// Records through the direct recorder rather than the httpmock proxy: this
/// endpoint answers with raw audio, and the proxy exports bodies as strings,
/// so a recorded speech response would come back as `body: null` and replay as
/// zero bytes. The direct path stores non-UTF-8 bodies as base64.
pub(super) async fn with_openai_audio_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start_via(
        rig_cassette::http::RecordVia::Direct,
        &crate::cassettes::cassette_root(),
        "openai",
        spec,
        "https://api.openai.com/v1",
    )
    .await;
    let openai = OpenAiCassette::new(
        cassette.api_key("OPENAI_API_KEY"),
        cassette.base_url(),
        DirectRecordingHttpClient::new(cassette.direct_recorder()),
    );

    let result = AssertUnwindSafe(test_body(openai)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Per-bug wrapper for the websocket error-identity matrix
/// (`crates/rig-cassette/fixtures/cassettes/openai/websocket_error_identity_matrix/`).
///
/// Uses a deliberately invalid key in **both** modes, like
/// [`with_openai_cassette_bogus_key`]: the only websocket failure the provider
/// answers with an HTTP response is the auth rejection, so that is what these
/// cells record. The upgrade is a plain HTTP GET until the provider accepts
/// it, which is why a rejected one can be recorded and replayed at all.
pub(super) async fn with_openai_websocket_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "openai",
        spec,
        "https://api.openai.com/v1",
    )
    .await;
    // The rejected credential is this wrapper's subject.
    cassette.expect_account_failure(crate::cassettes::AccountFailure::Auth);
    let openai = OpenAiCassette::new(
        "sk-invalid-websocket-edge-matrix-key",
        cassette.base_url(),
        rig_test_support::cassettes::local_http(),
    );
    let result = AssertUnwindSafe(test_body(openai)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Like [`with_openai_cassette`], but authenticating with a deliberately
/// invalid API key — for recording real 401s with no secret near the fixture.
pub(super) async fn with_openai_cassette_bogus_key<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "openai",
        spec,
        "https://api.openai.com/v1",
    )
    .await;
    // The rejected credential is this wrapper's subject.
    cassette.expect_account_failure(crate::cassettes::AccountFailure::Auth);
    let openai = OpenAiCassette::new(
        "sk-invalid-edge-matrix-key",
        cassette.base_url(),
        rig_test_support::cassettes::local_http(),
    );
    let result = AssertUnwindSafe(test_body(openai)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// The `x-request-id` response header each interaction of an OpenAI cassette
/// recorded, in wire order — one entry per interaction, `None` for an
/// interaction whose response carried no such header.
///
/// The header is the transport id OpenAI support asks for, and the harness
/// keeps it (placeholder-scrubbed) precisely so a cell can prove the wire
/// reported one. Reading it back from the fixture is how the raw-capture and
/// parity matrices assert that premise instead of assuming it. Parsed with
/// the same YAML decoder the harness writes with, so a layout change in the
/// fixture format fails here loudly rather than silently reading `None`.
pub(super) fn recorded_request_id_headers(scenario: &str) -> Vec<Option<String>> {
    #[derive(serde::Deserialize)]
    struct Interaction {
        then: Response,
    }
    #[derive(serde::Deserialize)]
    struct Response {
        #[serde(default)]
        header: Vec<NameValue>,
    }
    #[derive(serde::Deserialize)]
    struct NameValue {
        name: String,
        value: String,
    }

    let path = crate::cassettes::cassette_path("openai", scenario);
    let contents = std::fs::read_to_string(&path)
        .unwrap_or_else(|err| panic!("cassette {} should be readable: {err}", path.display()));

    serde_yaml::Deserializer::from_str(&contents)
        .map(|document| {
            let interaction = <Interaction as serde::Deserialize>::deserialize(document)
                .unwrap_or_else(|err| {
                    panic!("cassette {} should deserialize: {err}", path.display())
                });
            interaction
                .then
                .header
                .into_iter()
                .find(|header| header.name.eq_ignore_ascii_case("x-request-id"))
                .map(|header| header.value)
        })
        .collect()
}

/// JSON `data:` frames of one recorded SSE body, excluding `[DONE]`.
///
/// `crate::cassettes::recorded_sse_json_frames` reads only a scenario's first
/// interaction; multi-interaction streamed cells (agent tool runs) need the
/// frames of *each* interaction, which they get by pairing this with
/// `crate::cassettes::recorded_interaction_bodies`.
pub(super) fn sse_json_frames(body: &str) -> Vec<serde_json::Value> {
    body.lines()
        .filter_map(|line| line.trim().strip_prefix("data:"))
        .map(str::trim)
        .filter(|payload| *payload != "[DONE]")
        .map(|payload| {
            serde_json::from_str(payload)
                .unwrap_or_else(|err| panic!("recorded SSE frame should be JSON: {err}"))
        })
        .collect()
}

/// Cassette wrapper for the OpenAI Responses prompt-caching matrix
/// (`crates/rig-cassette/fixtures/cassettes/openai/prompt_caching/`).
///
/// Delegates to [`with_openai_cassette`] — the behavior is identical, and
/// deliberately shared so the two cannot drift apart when the base wrapper gains
/// policy. What the separate name buys is a per-suite entry in the
/// cassette-safety registry, so the cache fixtures are auditable as one
/// concern's evidence.
pub(super) async fn with_openai_prompt_caching_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

/// `options` as OpenAI's typed provider options.
pub(super) fn openai_options(
    options: rig::providers::openai::extension::OpenAiOptions,
) -> rig::completion::ProviderOptions {
    rig::completion::ProviderOptions::new().set(options)
}

/// OpenAI's shared provider options: `store`, and `prompt_cache_key` when
/// given.
pub(super) fn shared_options(
    store: Option<bool>,
    prompt_cache_key: Option<&str>,
) -> rig::completion::ProviderOptions {
    let mut options = rig::providers::openai::extension::OpenAiOptions::new();
    if let Some(store) = store {
        options = options.store(store);
    }
    if let Some(key) = prompt_cache_key {
        options = options.prompt_cache_key(key);
    }
    openai_options(options)
}

/// `store: false` as a typed provider option.
pub(super) fn stateless() -> rig::completion::ProviderOptions {
    shared_options(Some(false), None)
}

/// A reasoning effort as typed generation options.
pub(super) fn effort(effort: rig::completion::Effort) -> rig::completion::GenerationOptions {
    rig::completion::GenerationOptions::default().reasoning(effort)
}
