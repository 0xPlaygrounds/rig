//! Rig repository paths for the reusable cassette engine.

pub use rig_cassette::http::*;
use std::path::PathBuf;

/// The HTTP client every cassette test sends through, one per process. A
/// replay or a recording talks only to the local cassette server, so the
/// client skips the system proxy lookup: on macOS that lookup reads every
/// file beside the test binary, which costs seconds in each test process
/// once `target/debug/deps` grows large.
pub fn local_http() -> rig_core::http_client::DynHttpClient {
    rig_core::http_client::DynHttpClient::new(local_reqwest())
}

/// [`local_http`] before it is erased.
pub fn local_reqwest() -> rig_reqwest::ReqwestClient {
    static CLIENT: std::sync::LazyLock<rig_reqwest::ReqwestClient> =
        std::sync::LazyLock::new(|| {
            rig_reqwest::reqwest::Client::builder()
                .no_proxy()
                .build()
                .map_or_else(
                    |_| rig_reqwest::ReqwestClient::default(),
                    rig_reqwest::ReqwestClient::from,
                )
        });
    CLIENT.clone()
}

/// Locate this workspace's provider cassette directory from the crate manifest.
pub fn cassette_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(std::path::Path::parent)
        .expect("test-support crate is two directories below the repository")
        .join("crates/rig-cassette/fixtures/cassettes")
}

/// Resolve a provider scenario's YAML path under the workspace cassette directory.
pub fn cassette_path(provider: &str, scenario: &str) -> PathBuf {
    rig_cassette::http::cassette_path(&cassette_root(), provider, scenario)
}

/// Read recorded request/response bodies in wire order; panic on invalid fixtures.
pub fn recorded_interaction_bodies(provider: &str, scenario: &str) -> Vec<(String, String)> {
    rig_cassette::http::recorded_interaction_bodies(&cassette_root(), provider, scenario)
}

/// Parse the first recorded request as JSON; panic on a missing or invalid fixture.
pub fn recorded_json_request(provider: &str, scenario: &str) -> serde_json::Value {
    rig_cassette::http::recorded_json_request(&cassette_root(), provider, scenario)
}

/// Parse the first recorded response as JSON; panic on a missing or invalid fixture.
pub fn recorded_json_response(provider: &str, scenario: &str) -> serde_json::Value {
    rig_cassette::http::recorded_json_response(&cassette_root(), provider, scenario)
}

/// Read recorded lowercase request-header pairs in wire order.
pub fn recorded_request_header_pairs(provider: &str, scenario: &str) -> Vec<Vec<(String, String)>> {
    rig_cassette::http::recorded_request_header_pairs(&cassette_root(), provider, scenario)
}

/// Read recorded lowercase response-header pairs in wire order.
pub fn recorded_response_header_pairs(
    provider: &str,
    scenario: &str,
) -> Vec<Vec<(String, String)>> {
    rig_cassette::http::recorded_response_header_pairs(&cassette_root(), provider, scenario)
}

/// The recorded value of one response header of one interaction, if it was
/// recorded at all. `name` must be lowercase.
pub fn recorded_response_header(
    provider: &str,
    scenario: &str,
    interaction: usize,
    name: &str,
) -> Option<String> {
    recorded_response_header_pairs(provider, scenario)
        .get(interaction)
        .unwrap_or_else(|| {
            panic!("cassette {provider}/{scenario} should record interaction {interaction}")
        })
        .iter()
        .find(|(recorded, _)| recorded == name)
        .map(|(_, value)| value.clone())
}

/// Parse every recorded request/response body pair as JSON, in wire order.
pub fn recorded_json_turns(
    provider: &str,
    scenario: &str,
) -> Vec<(serde_json::Value, serde_json::Value)> {
    rig_cassette::http::recorded_json_turns(&cassette_root(), provider, scenario)
}

/// Parse the single recorded turn of a single-turn scenario as JSON.
pub fn recorded_json_turn(
    provider: &str,
    scenario: &str,
) -> (serde_json::Value, serde_json::Value) {
    rig_cassette::http::recorded_json_turn(&cassette_root(), provider, scenario)
}

/// Read each recorded request's query parameters in wire order.
pub fn recorded_request_query_pairs(provider: &str, scenario: &str) -> Vec<Vec<(String, String)>> {
    rig_cassette::http::recorded_request_query_pairs(&cassette_root(), provider, scenario)
}

/// Read recorded request paths in wire order.
pub fn recorded_request_paths(provider: &str, scenario: &str) -> Vec<String> {
    rig_cassette::http::recorded_request_paths(&cassette_root(), provider, scenario)
}

/// Read recorded response statuses and bodies in wire order.
pub fn recorded_statuses_and_bodies(provider: &str, scenario: &str) -> Vec<(u16, String)> {
    rig_cassette::http::recorded_statuses_and_bodies(&cassette_root(), provider, scenario)
}

/// Parse JSON data frames from the first recorded SSE response.
pub fn recorded_sse_json_frames(provider: &str, scenario: &str) -> Vec<serde_json::Value> {
    rig_cassette::http::recorded_sse_json_frames(&cassette_root(), provider, scenario)
}

/// Save completed matrix exchanges before a failing body unwinds.
/// This is attempt evidence, never a replacement for cassette finalization.
pub async fn checkpoint_attempt(cassette: &ProviderCassette, provider: &str, scenario: &str) {
    if !scenario.starts_with("checkpoint_matrix")
        && !scenario.starts_with("long_loop_matrix")
        && !scenario.starts_with("long_task_matrix")
        && !scenario.starts_with("history_survival_matrix")
        && !scenario.starts_with("portability_matrix")
    {
        return;
    }
    let Some(directory) = std::env::var_os("RIG_LONG_TASK_ATTEMPT_DIR")
        .or_else(|| std::env::var_os("RIG_CHECKPOINT_ATTEMPT_DIR"))
        .or_else(|| std::env::var_os("RIG_LONG_LOOP_ATTEMPT_DIR"))
    else {
        return;
    };
    let path = PathBuf::from(directory)
        .join(provider)
        .join(format!("{scenario}.yaml"));
    let written = cassette.checkpoint_recording(&path).await;
    eprintln!(
        "CHECKPOINT_ATTEMPT cassette={} completed_exchanges_saved={written}",
        path.display()
    );
}

/// Finish `cassette` after a test body's `result`, checking what it recorded
/// first. `check` reads the session's exchanges under the cassette root and
/// for the scenario it is given: the recording so far, checkpointed under the attempt root, when
/// recording, and the committed fixture on replay. A check that fails while
/// recording leaves the committed fixture as it was and keeps the attempt
/// aside, as a failing body does.
pub async fn finish_checked(
    cassette: ProviderCassette,
    provider: &str,
    scenario: &str,
    result: Result<(), Box<dyn std::any::Any + Send + 'static>>,
    check: impl FnOnce(&std::path::Path, &str),
) {
    let result = match result {
        Ok(()) => {
            let root = if CassetteMode::current() == CassetteMode::Record {
                let root = attempt_root()
                    .join("checked")
                    .join(std::process::id().to_string());
                let path = rig_cassette::http::cassette_path(&root, provider, scenario);
                cassette.checkpoint_recording(&path).await;
                root
            } else {
                cassette_root()
            };
            std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| check(&root, scenario)))
        }
        failed => failed,
    };
    cassette.finish_after_test(result).await;
}
