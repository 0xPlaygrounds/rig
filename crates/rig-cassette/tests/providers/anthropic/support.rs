use futures::FutureExt;
use rig::http_client::{DynHttpClient, ReqwestClient};
use rig::providers::anthropic::completion::ANTHROPIC_VERSION_2023_06_01;
use rig::providers::anthropic::wire::AnthropicConfig;
use rig_test_support::cassette_models::AnthropicModels;
use rig_test_support::model_session::Session;
use serde::Deserialize;
use std::future::Future;
use std::panic::{AssertUnwindSafe, resume_unwind};

use crate::cassettes::{CassetteSpec, ProviderCassette};

async fn anthropic_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, AnthropicConfig) {
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "anthropic",
        spec,
        "https://api.anthropic.com",
    )
    .await;
    let bound = AnthropicConfig::new(cassette.api_key("ANTHROPIC_API_KEY"))
        .with_base_url(cassette.base_url());

    (cassette, bound)
}

/// Cassette wrapper for the run-lifecycle matrix: the client sends
/// through a [`DynHttpClient`] carrying the supplied [`HttpMiddleware`], so
/// the same recorded exchange exercises the transport middleware seam and the
/// run lifecycle hooks together (see
/// `crates/rig-cassette/fixtures/cassettes/anthropic/lifecycle_matrix/`).
pub(super) async fn with_anthropic_lifecycle_cassette<M, F, Fut>(
    spec: impl Into<CassetteSpec>,
    middleware: M,
    test_body: F,
) where
    M: rig::http_client::HttpMiddleware + 'static,
    F: FnOnce(AnthropicModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "anthropic",
        spec,
        "https://api.anthropic.com",
    )
    .await;
    let provider = AnthropicConfig::new(cassette.api_key("ANTHROPIC_API_KEY"))
        .with_base_url(cassette.base_url());
    let http = DynHttpClient::new(ReqwestClient::default()).with_middleware(middleware);
    let result = AssertUnwindSafe(test_body(AnthropicModels::new(provider, http)))
        .catch_unwind()
        .await;
    cassette.finish_after_test(result).await;
}

pub(super) async fn with_anthropic_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(AnthropicModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    let (cassette, bound) = anthropic_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(AnthropicModels::new(
        bound,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    crate::cassettes::checkpoint_attempt(&cassette, "anthropic", spec.scenario()).await;
    cassette.finish_after_test(result).await;
}

/// [`with_anthropic_cassette`], with `check` run on what the session
/// recorded before the recording is written (`cassettes::finish_checked`).
pub(super) async fn with_anthropic_checked_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
    check: impl FnOnce(&std::path::Path, &str),
) where
    F: FnOnce(AnthropicModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let spec = spec.into();
    let (cassette, bound) = anthropic_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(AnthropicModels::new(
        bound,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    crate::cassettes::checkpoint_attempt(&cassette, "anthropic", spec.scenario()).await;
    crate::cassettes::finish_checked(cassette, "anthropic", spec.scenario(), result, check).await;
}

/// Long-run caching recordings
/// (`crates/rig-cassette/fixtures/cassettes/anthropic/long_run_caching/`). The
/// body gets the models and the session's clock, whose record-only pause
/// spaces a retried turn; it returns whatever the test asserts on after the
/// session is finished.
pub(super) async fn with_anthropic_long_run_cassette<F, Fut, R>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> R
where
    F: FnOnce(AnthropicModels, crate::cassettes::CassetteClock) -> Fut,
    Fut: Future<Output = R>,
{
    let spec = spec.into();
    let (cassette, bound) = anthropic_cassette(spec).await;
    let models = AnthropicModels::new(bound, rig_test_support::cassettes::local_http());
    let result = AssertUnwindSafe(test_body(models, cassette.clock()))
        .catch_unwind()
        .await;
    crate::cassettes::checkpoint_attempt(&cassette, "anthropic", spec.scenario()).await;
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

/// Drive rig's Anthropic client against an Anthropic-*compatible* gateway
/// rather than `api.anthropic.com`.
///
/// The code under test is still the Anthropic provider — only the endpoint
/// differs. Gateways that reimplement the Messages API do not always reproduce
/// Anthropic's own wire choices, and where they diverge the Anthropic adapter
/// is what has to cope. Recording that divergence needs traffic from the
/// gateway itself, so this wrapper points upstream at OpenRouter's Messages
/// endpoint and records with `OPENROUTER_API_KEY`. Replay needs no key, like
/// every other cassette.
///
/// Cassettes recorded through here live under `crates/rig-cassette/fixtures/cassettes/anthropic/`
/// with
/// the rest of the provider's scenarios; the gateway is an implementation
/// detail of how the fixture was obtained, not a separate provider suite.
pub(super) async fn with_anthropic_gateway_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(AnthropicModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "anthropic",
        spec,
        OPENROUTER_MESSAGES_BASE_URL,
    )
    .await;
    let bound = AnthropicConfig::new(cassette.api_key("OPENROUTER_API_KEY"))
        .with_base_url(cassette.base_url());

    let result = AssertUnwindSafe(test_body(AnthropicModels::new(
        bound,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test(result).await;
}

/// OpenRouter's Anthropic Messages endpoint, minus the `/v1/messages` suffix
/// the Anthropic client appends itself.
const OPENROUTER_MESSAGES_BASE_URL: &str = "https://openrouter.ai/api";

/// One model's recorded session
/// (`crates/rig-cassette/fixtures/cassettes/anthropic/models/<model>/`).
/// The body gets the models, the session's clock and, when `upload_pdf`, a
/// Files-API client with the verifier PDF uploaded; the upload is deleted
/// after the body, in the same recording.
pub(super) async fn with_anthropic_model_session_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    upload_pdf: bool,
    test_body: F,
) -> Session
where
    F: FnOnce(
        AnthropicModels,
        Option<(AnthropicModels, String)>,
        crate::cassettes::CassetteClock,
    ) -> Fut,
    Fut: Future<Output = Session>,
{
    let spec = spec.into();
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "anthropic",
        spec,
        "https://api.anthropic.com",
    )
    .await;
    let base_url = normalize_anthropic_base_url(&cassette.base_url());
    let api_key = cassette.api_key("ANTHROPIC_API_KEY");
    let config = AnthropicConfig::new(api_key.as_str()).with_base_url(&base_url);
    let models = AnthropicModels::new(config.clone(), rig_test_support::cassettes::local_http());
    let clock = cassette.clock();
    let result = AssertUnwindSafe(async {
        let uploaded = if upload_pdf {
            Some(upload_pdf_for_file_id_test(&base_url, &api_key).await.id)
        } else {
            None
        };
        let files = uploaded.clone().map(|file_id| {
            (
                AnthropicModels::new(
                    config.clone().with_beta(ANTHROPIC_FILES_BETA),
                    rig_test_support::cassettes::local_http(),
                ),
                file_id,
            )
        });
        let outcome = AssertUnwindSafe(test_body(models, files, clock))
            .catch_unwind()
            .await;
        if let Some(file_id) = &uploaded {
            delete_uploaded_file(&base_url, &api_key, file_id).await;
        }
        match outcome {
            Ok(session) => session,
            Err(payload) => resume_unwind(payload),
        }
    })
    .catch_unwind()
    .await;
    crate::cassettes::checkpoint_attempt(&cassette, "anthropic", spec.scenario()).await;
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

fn normalize_anthropic_base_url(base_url: &str) -> String {
    let trimmed = base_url.trim_end_matches('/');

    if let Some(stripped) = trimmed.strip_suffix("/v1/messages") {
        stripped.to_string()
    } else if let Some(stripped) = trimmed.strip_suffix("/messages") {
        stripped.to_string()
    } else if let Some(stripped) = trimmed.strip_suffix("/v1") {
        stripped.to_string()
    } else {
        trimmed.to_string()
    }
}

/// Like [`with_anthropic_cassette`], but the client authenticates with a
/// deliberately invalid API key — for recording real 401 responses without a
/// secret anywhere near the fixture (auth headers are neither recorded nor
/// matched by the harness).
pub(super) async fn with_anthropic_cassette_bogus_key<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(AnthropicModels) -> Fut,
    Fut: Future<Output = ()>,
{
    let cassette = ProviderCassette::start(
        &crate::cassettes::cassette_root(),
        "anthropic",
        spec,
        "https://api.anthropic.com",
    )
    .await;
    // The rejected credential is this wrapper's subject.
    cassette.expect_account_failure(crate::cassettes::AccountFailure::Auth);
    let bound =
        AnthropicConfig::new("sk-invalid-edge-matrix-key").with_base_url(cassette.base_url());
    let result = AssertUnwindSafe(test_body(AnthropicModels::new(
        bound,
        rig_test_support::cassettes::local_http(),
    )))
    .catch_unwind()
    .await;
    cassette.finish_after_test(result).await;
}

/// The `request-id` response header each interaction of an Anthropic cassette
/// recorded, in wire order — one entry per interaction, `None` for an
/// interaction whose response carried no such header.
///
/// The header is the transport id Anthropic support asks for, and the harness
/// keeps it (placeholder-scrubbed) precisely so a cell can prove the wire
/// reported one. Reading it back from the fixture is how the raw-capture and
/// parity matrices assert that premise instead of assuming it.
pub(super) fn recorded_request_id_headers(scenario: &str) -> Vec<Option<String>> {
    crate::cassettes::recorded_response_header_pairs("anthropic", scenario)
        .into_iter()
        .map(|headers| {
            headers
                .into_iter()
                .find(|(name, _)| name == "request-id")
                .map(|(_, value)| value)
        })
        .collect()
}

/// Assert that a sequence of ids the code under test observed is the sequence
/// the fixture recorded — in both cassette modes.
///
/// On replay the observed values are the recorded ids (or, in a fixture
/// recorded before ids were kept verbatim, their numbered placeholders), so
/// the two sequences must be identical. In record mode the observed values
/// are fresh live ids, so only their equality structure is comparable: which positions repeat, and
/// which are new. Checking that
/// structure in both modes (and exact equality on replay) means the same cell
/// proves "each attempt reports its own id, in recorded order" whether it is
/// being recorded or replayed.
pub(super) fn assert_ids_match_recording(
    observed: &[Option<String>],
    recorded: &[Option<String>],
    context: &str,
) {
    fn ranks(ids: &[Option<String>]) -> Vec<Option<usize>> {
        let mut seen: Vec<&str> = Vec::new();
        ids.iter()
            .map(|id| {
                id.as_deref().map(|id| {
                    seen.iter()
                        .position(|known| *known == id)
                        .unwrap_or_else(|| {
                            seen.push(id);
                            seen.len() - 1
                        })
                })
            })
            .collect()
    }

    assert_eq!(
        observed.len(),
        recorded.len(),
        "{context}: observed {observed:?} and recorded {recorded:?} must have one id per interaction"
    );
    assert_eq!(
        ranks(observed),
        ranks(recorded),
        "{context}: observed {observed:?} must repeat/differ exactly as the recording {recorded:?}"
    );
    if crate::cassettes::CassetteMode::current() == crate::cassettes::CassetteMode::Replay {
        assert_eq!(
            observed, recorded,
            "{context}: on replay the observed ids are the fixture's own"
        );
    }
}

/// The Files API beta a request naming an uploaded file needs.
pub(super) const ANTHROPIC_FILES_BETA: &str = "files-api-2025-04-14";
const VERIFIER_FIXTURE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tests/data/file-id-verifiers.pdf"
);
const VERIFIER_FIXTURE_FILENAME: &str = "rig-file-id-verifiers.pdf";

/// What the Files API answers an upload with.
#[derive(Debug, Deserialize)]
pub(super) struct UploadedFile {
    pub(super) id: String,
    #[serde(rename = "type")]
    file_type: String,
    filename: String,
    mime_type: String,
    size_bytes: u64,
    downloadable: bool,
}

/// Upload the verifier PDF through the Files API (a test-side request rig
/// does not make) and check what Anthropic stored.
pub(super) async fn upload_pdf_for_file_id_test(base_url: &str, api_key: &str) -> UploadedFile {
    let bytes = tokio::fs::read(VERIFIER_FIXTURE_PATH)
        .await
        .expect("verifier fixture PDF should be readable");
    let file_part = reqwest::multipart::Part::bytes(bytes)
        .file_name(VERIFIER_FIXTURE_FILENAME)
        .mime_str("application/pdf")
        .expect("verifier fixture PDF MIME should be valid");
    let form = reqwest::multipart::Form::new().part("file", file_part);

    let response = reqwest::Client::new()
        .post(format!("{base_url}/v1/files"))
        .header("x-api-key", api_key)
        .header("anthropic-version", ANTHROPIC_VERSION_2023_06_01)
        .header("anthropic-beta", ANTHROPIC_FILES_BETA)
        .multipart(form)
        .send()
        .await
        .expect("file upload request should be sent");

    let status = response.status();
    let body = response
        .text()
        .await
        .expect("file upload response body should be readable");
    assert!(
        status.is_success(),
        "file upload failed with {status}: {body}"
    );

    let uploaded: UploadedFile =
        serde_json::from_str(&body).expect("file upload response should deserialize");
    assert_uploaded_file_metadata(&uploaded);
    uploaded
}

fn assert_uploaded_file_metadata(uploaded: &UploadedFile) {
    assert!(
        uploaded.id.starts_with("file_"),
        "expected Anthropic file id to start with file_, got {}",
        uploaded.id
    );
    assert_eq!(uploaded.file_type, "file");
    assert_eq!(uploaded.filename, VERIFIER_FIXTURE_FILENAME);
    assert_eq!(uploaded.mime_type, "application/pdf");
    assert!(
        uploaded.size_bytes > 0,
        "uploaded file size should be nonzero"
    );
    assert!(
        !uploaded.downloadable,
        "user-uploaded Anthropic file should not be downloadable"
    );
}

/// Delete an uploaded file; a failure is reported, not fatal.
pub(super) async fn delete_uploaded_file(base_url: &str, api_key: &str, file_id: &str) {
    let response = reqwest::Client::new()
        .delete(format!("{base_url}/v1/files/{file_id}"))
        .header("x-api-key", api_key)
        .header("anthropic-version", ANTHROPIC_VERSION_2023_06_01)
        .header("anthropic-beta", ANTHROPIC_FILES_BETA)
        .send()
        .await;

    if let Ok(response) = response
        && !response.status().is_success()
    {
        eprintln!(
            "cleanup failed for uploaded Anthropic file {file_id}: {}",
            response.status()
        );
    }
}

/// A response serialized without its `raw` document or its blocks'
/// provider items: what rig normalized, in rig's own spelling.
pub(super) fn canonical_without_raw(
    mut response: rig::completion::CompletionResponse,
) -> serde_json::Value {
    response.choice = response
        .choice
        .iter()
        .map(rig::message::AssistantContent::canonical)
        .collect();
    crate::support::normalized_without_raw(response)
}
