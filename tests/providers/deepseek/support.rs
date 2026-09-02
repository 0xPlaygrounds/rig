use rig::client::DefaultTransportBuilder as _;
use rig::message::AssistantContent;
use std::future::Future;
use std::panic::AssertUnwindSafe;

use futures::FutureExt;
use rig::providers::deepseek;

use crate::cassettes::{CassetteSpec, ProviderCassette};

async fn deepseek_cassette(spec: impl Into<CassetteSpec>) -> (ProviderCassette, deepseek::Client) {
    let cassette = ProviderCassette::start("deepseek", spec, "https://api.deepseek.com").await;
    let client = deepseek::Client::builder()
        .api_key(cassette.api_key("DEEPSEEK_API_KEY"))
        .base_url(cassette.base_url())
        .build()
        .expect("DeepSeek client should build");

    (cassette, client)
}

pub(super) async fn with_deepseek_cassette<F, Fut>(spec: impl Into<CassetteSpec>, test_body: F)
where
    F: FnOnce(deepseek::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = deepseek_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}

/// The body every result-returning wrapper shares.
///
/// Deliberately *not* one of the registered wrapper names: the safety scan
/// reads each cassette scenario out of the AST at the wrapper's call site, so a
/// wrapper that forwarded to another registered wrapper would hand it a
/// variable instead of a string literal and register nothing.
async fn run_deepseek_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(deepseek::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let (cassette, client) = deepseek_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

pub(super) async fn with_deepseek_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(deepseek::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    run_deepseek_cassette_result(spec, test_body).await
}

/// Bogus-key variant for recording real 401s: the shared model-listing fetch
/// must classify a rejected listing with provider, path and status context
/// (rig#2079), and only a real rejection proves it.
pub(super) async fn with_deepseek_cassette_bogus_key_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(deepseek::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    let cassette = ProviderCassette::start("deepseek", spec, "https://api.deepseek.com").await;
    let client = deepseek::Client::builder()
        .api_key("sk-invalid-edge-matrix-key")
        .base_url(cassette.base_url())
        .build()
        .expect("DeepSeek client should build");
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test_result(result).await
}

/// Truncation-matrix variant: one recorded fixture directory per bug keeps the
/// matrices auditable, and the safety scan pairs a fixture with the wrapper
/// whose call site names it.
pub(super) async fn with_deepseek_truncation_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(deepseek::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    run_deepseek_cassette_result(spec, test_body).await
}

/// Reasoning-block-order matrix variant.
pub(super) async fn with_deepseek_block_order_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(deepseek::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    run_deepseek_cassette_result(spec, test_body).await
}

/// Request-wire-shape matrix variant (content parts, forced tool choice,
/// completion-path error envelopes).
pub(super) async fn with_deepseek_wire_shape_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(deepseek::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    run_deepseek_cassette_result(spec, test_body).await
}

/// Follow-up hunt census: request/response fields that the first sweep did
/// not exercise (stop sequences, reasoning effort and content filtering).
/// This remains separate from per-bug matrix wrappers: a confirmed defect,
/// such as streamed log-probability loss, graduates into its own fixture
/// directory and wrapper.
pub(super) async fn with_deepseek_followup_hunt_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(deepseek::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    run_deepseek_cassette_result(spec, test_body).await
}

/// Per-bug matrix for streamed Chat Completions log-probability loss.
pub(super) async fn with_deepseek_stream_logprobs_cassette_result<F, Fut, E>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) -> Result<(), E>
where
    F: FnOnce(deepseek::Client) -> Fut,
    Fut: Future<Output = Result<(), E>>,
{
    run_deepseek_cassette_result(spec, test_body).await
}

/// Every interaction recorded under `scenario`, as `(request, response)` body
/// strings in cassette order.
///
/// Matrix cells assert their *premise* against these bytes — that the provider
/// turn really did carry (or really did not carry) the shape the cell is about
/// — so a cell whose recorded turn quietly changed shape fails loudly instead
/// of passing while covering nothing.
pub(super) fn recorded_interactions(scenario: &str) -> Vec<(String, String)> {
    #[derive(serde::Deserialize)]
    struct RecordedInteraction {
        when: RecordedSide,
        then: RecordedSide,
    }

    #[derive(serde::Deserialize)]
    struct RecordedSide {
        body: Option<String>,
    }

    let cassette_path = crate::cassettes::cassette_path("deepseek", scenario);
    let contents = std::fs::read_to_string(&cassette_path).unwrap_or_else(|error| {
        panic!(
            "provider cassette {} should be readable after recording: {error}",
            cassette_path.display()
        )
    });

    serde_yaml::Deserializer::from_str(&contents)
        .map(|document| {
            let interaction = <RecordedInteraction as serde::Deserialize>::deserialize(document)
                .expect("cassette interaction should deserialize");
            (
                interaction.when.body.unwrap_or_default(),
                interaction.then.body.unwrap_or_default(),
            )
        })
        .collect()
}

/// The first recorded request body under `scenario`, parsed as JSON.
pub(super) fn recorded_request(scenario: &str) -> serde_json::Value {
    let (request, _) = recorded_interactions(scenario)
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("cassette {scenario} should contain an interaction"));
    serde_json::from_str(&request)
        .unwrap_or_else(|error| panic!("cassette {scenario} request should be JSON: {error}"))
}

/// The first recorded response body under `scenario`, parsed as JSON.
pub(super) fn recorded_response(scenario: &str) -> serde_json::Value {
    let (_, response) = recorded_interactions(scenario)
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("cassette {scenario} should contain an interaction"));
    serde_json::from_str(&response)
        .unwrap_or_else(|error| panic!("cassette {scenario} response should be JSON: {error}"))
}

/// The decoded `data:` frames of the first recorded SSE response under
/// `scenario`, `[DONE]` excluded.
pub(super) fn recorded_stream_chunks(scenario: &str) -> Vec<serde_json::Value> {
    let (_, response) = recorded_interactions(scenario)
        .into_iter()
        .next()
        .unwrap_or_else(|| panic!("cassette {scenario} should contain an interaction"));

    response
        .lines()
        .filter_map(|line| line.trim().strip_prefix("data:"))
        .map(str::trim)
        .filter(|payload| *payload != "[DONE]")
        .map(|payload| {
            serde_json::from_str(payload)
                .unwrap_or_else(|error| panic!("cassette {scenario} chunk should be JSON: {error}"))
        })
        .collect()
}

/// Everything one raw DeepSeek stream produced, in arrival order.
///
/// The shared `collect_raw_stream_observation` drops the terminal record and
/// the block ordering; both are the subject of these matrices, so the matrix
/// cells collect their own.
pub(super) struct RawStreamOutcome {
    pub(super) text: String,
    pub(super) tool_calls: Vec<rig::message::ToolCall>,
    pub(super) reasoning: String,
    /// `"reasoning"`, `"text"`, `"tool_call"` in the order the stream emitted
    /// them, deltas collapsed into their first occurrence's kind.
    pub(super) order: Vec<&'static str>,
    pub(super) final_record: Option<rig::streaming::StreamFinal>,
    pub(super) errors: Vec<String>,
}

impl RawStreamOutcome {
    pub(super) fn finish_reason(&self) -> Option<rig::completion::FinishReason> {
        self.final_record
            .as_ref()
            .and_then(|record| record.finish_reason.clone())
    }

    pub(super) fn tool_call_names(&self) -> Vec<&str> {
        self.tool_calls
            .iter()
            .map(|call| call.function.name.as_str())
            .collect()
    }
}

pub(super) async fn collect_raw_stream_outcome(
    mut stream: rig::streaming::StreamingCompletionResponse,
) -> RawStreamOutcome {
    use futures::StreamExt as _;
    use rig::streaming::{Delta, StreamEvent};

    let mut outcome = RawStreamOutcome {
        text: String::new(),
        tool_calls: Vec::new(),
        reasoning: String::new(),
        order: Vec::new(),
        final_record: None,
        errors: Vec::new(),
    };

    fn note(order: &mut Vec<&'static str>, kind: &'static str) {
        if order.last() != Some(&kind) {
            order.push(kind);
        }
    }

    while let Some(item) = stream.next().await {
        match item {
            Ok(StreamEvent::BlockDelta {
                delta: Delta::Text { text },
                ..
            }) => {
                outcome.text.push_str(&text);
                note(&mut outcome.order, "text");
            }
            Ok(StreamEvent::BlockEnd {
                block: Some(AssistantContent::ToolCall(tool_call)),
                ..
            }) => {
                outcome.tool_calls.push(tool_call);
                note(&mut outcome.order, "tool_call");
            }
            Ok(StreamEvent::BlockDelta {
                delta: Delta::ToolName { .. } | Delta::ToolArguments { .. },
                ..
            }) => {}
            Ok(StreamEvent::BlockEnd {
                block: Some(AssistantContent::Reasoning(reasoning)),
                ..
            }) => {
                outcome.reasoning.push_str(&reasoning.display_text());
                note(&mut outcome.order, "reasoning");
            }
            Ok(StreamEvent::BlockDelta {
                delta: Delta::Reasoning { text: reasoning },
                ..
            }) => {
                outcome.reasoning.push_str(&reasoning);
                note(&mut outcome.order, "reasoning");
            }
            Ok(StreamEvent::Final(record)) => outcome.final_record = Some(record),
            Ok(
                StreamEvent::Unknown(_)
                | StreamEvent::BlockStart { .. }
                | StreamEvent::BlockEnd { .. }
                | StreamEvent::BlockDelta {
                    delta: Delta::TextMeta { .. },
                    ..
                },
            ) => {}
            Err(error) => outcome.errors.push(error.to_string()),
        }
    }

    outcome
}

/// Compare a generated token (response id, system fingerprint, request id)
/// observed by a test with the value its fixture holds.
///
/// Fixtures are placeholder-scrubbed (`chatcmpl-REDACTED_1`, `fp_REDACTED_1`),
/// so on the recording pass the live token cannot equal the fixture's; both
/// are then required to be present and non-empty. On replay the harness
/// serves the scrubbed bytes back, so equality is exact — which is what CI
/// runs. Presence must agree in both modes.
pub(super) fn assert_matches_recorded_token(
    actual: Option<&str>,
    recorded: Option<&str>,
    context: &str,
) {
    match crate::cassettes::CassetteMode::current() {
        crate::cassettes::CassetteMode::Replay => {
            assert_eq!(
                actual, recorded,
                "{context}: replay serves the fixture's token back"
            );
        }
        crate::cassettes::CassetteMode::Record => {
            assert_eq!(
                actual.is_some(),
                recorded.is_some(),
                "{context}: live and recorded token presence must agree"
            );
            if let (Some(actual), Some(recorded)) = (actual, recorded) {
                assert!(
                    !actual.trim().is_empty() && !recorded.trim().is_empty(),
                    "{context}: live and recorded token must both be non-empty"
                );
            }
        }
    }
}

/// Cassette wrapper for the deepseek prompt-caching matrix
/// (`tests/cassettes/deepseek/prompt_caching/`).
///
/// Builds the cassette directly rather than delegating to [`with_deepseek_cassette`]: this
/// provider's cassette-safety `source_dir` covers `support.rs` itself, and the
/// scan requires every call to a *registered* wrapper to pass a string-literal
/// scenario. A delegating wrapper passes its `spec` variable through, which the
/// scan reports as an unregistered scenario. The duplication is three lines and
/// the alternative is an unscannable suite.
pub(super) async fn with_deepseek_prompt_caching_cassette<F, Fut>(
    spec: impl Into<CassetteSpec>,
    test_body: F,
) where
    F: FnOnce(deepseek::Client) -> Fut,
    Fut: Future<Output = ()>,
{
    let (cassette, client) = deepseek_cassette(spec).await;
    let result = AssertUnwindSafe(test_body(client)).catch_unwind().await;
    cassette.finish_after_test(result).await;
}
