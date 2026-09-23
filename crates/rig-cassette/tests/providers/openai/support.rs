use rig::driver::{Bind, Bound};
use rig::http_client::{BoxedHttpClient, ReqwestClient};
use rig::providers::openai::{OpenAI, Route};
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
/// field it reads: `client.openai.agent(model)` beside
/// `client.chat.agent(model)`, with `.chat(model)` / `.responses(model)`
/// still naming a typed wire when a cell reads the native reply.
pub(super) struct OpenAiCassette<H = BoxedHttpClient> {
    /// The configuration on its flagship route, over the cassette's socket.
    pub(super) openai: Bound<OpenAI, H>,
    /// The same configuration routed to Chat Completions.
    pub(super) chat: Bound<OpenAI, H>,
}

impl<H: Clone> OpenAiCassette<H> {
    /// The configuration for `api_key` at `base_url`, over `http`.
    fn new(api_key: impl Into<String>, base_url: impl Into<String>, http: H) -> Self {
        let openai = OpenAI::new(api_key).with_base_url(base_url).bind(http);
        let chat = openai
            .clone()
            .map_wire(|openai| openai.with_route(Route::Chat));
        Self { openai, chat }
    }
}

/// The bundled transport, erased — the socket the deleted client built.
fn bundled() -> BoxedHttpClient {
    ReqwestClient::default().boxed()
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
        bundled(),
    );

    (cassette, openai)
}

async fn openai_completions_cassette(
    spec: impl Into<CassetteSpec>,
) -> (ProviderCassette, Bound<OpenAI>) {
    let (cassette, openai) = openai_cassette(spec).await;
    (cassette, openai.chat)
}

/// Like [`with_openai_cassette`], but the client sends through the erased
/// [`BoxedHttpClient`] wrapping the same bundled transport — the client type
/// names no concrete `H`. Replaying a recorded scenario through it proves the
/// erasure is byte-transparent: the replay server matches on body bytes.
pub(super) async fn with_openai_boxed_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
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
    let openai = OpenAiCassette::new(
        cassette.api_key("OPENAI_API_KEY"),
        cassette.base_url(),
        ReqwestClient::default().boxed(),
    );
    let result = AssertUnwindSafe(test_body(openai)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// Cassette wrapper for the run-lifecycle matrix (PR #2407): the client sends
/// through a [`BoxedHttpClient`] carrying the supplied [`HttpMiddleware`], so
/// the same recorded exchange exercises the transport middleware seam and the
/// run lifecycle hooks together (see
/// `crates/rig-cassette/fixtures/cassettes/openai/lifecycle_matrix/`).
pub(super) async fn with_openai_lifecycle_cassette<M, F, Fut>(
    spec: impl Into<CassetteSpec>,
    middleware: M,
    test_body: F,
) where
    M: rig::http_client::HttpMiddleware + 'static,
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
    let openai = OpenAiCassette::new(
        cassette.api_key("OPENAI_API_KEY"),
        cassette.base_url(),
        ReqwestClient::default().boxed().with_middleware(middleware),
    );
    let result = AssertUnwindSafe(test_body(openai)).catch_unwind().await;
    cassette.finish_after_test(result).await;
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

/// Per-bug wrapper for the model-turn termination-metadata matrix
/// (`crates/rig-cassette/fixtures/cassettes/openai/turn_termination_matrix/`), rig#2184.
pub(super) async fn with_openai_turn_metadata_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(OpenAiCassette) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_cassette(spec, test_body).await;
}

pub(super) async fn with_openai_completions_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<OpenAI>) -> Fut,
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
    F: FnOnce(Bound<OpenAI>) -> Fut,
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

/// Per-bug wrapper for the truncated-turn matrix
/// (`crates/rig-cassette/fixtures/cassettes/openai/truncated_turn_matrix/`).
pub(super) async fn with_openai_truncation_cassette<F, Fut>(
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
    F: FnOnce(OpenAiCassette<DirectRecordingHttpClient>) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start_via(
        rig_cassette::http::Transport::Direct,
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
        bundled(),
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
    let openai = OpenAiCassette::new("sk-invalid-edge-matrix-key", cassette.base_url(), bundled());
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

/// [`with_openai_prompt_caching_cassette`] for the chat-completions wire.
///
/// OpenAI's two surfaces are two *different* cache paths with two different
/// usage mappings (`prompt_tokens_details.cached_tokens` versus
/// `input_tokens_details.cached_tokens`), so each is recorded separately rather
/// than assumed to behave like its sibling.
pub(super) async fn with_openai_completions_prompt_caching_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(Bound<OpenAI>) -> Fut,
    Fut: Future<Output = ()>,
{
    with_openai_completions_cassette(spec, test_body).await;
}
