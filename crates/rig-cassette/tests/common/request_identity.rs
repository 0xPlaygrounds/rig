//! Transport request ids and `system_fingerprint` reach every surface exactly
//! as the provider sent them: the unary response, the streamed final, a
//! provider error, and the completion span.
//!
//! Each cell runs one unary call, one streamed call and one call the provider
//! rejects, in that order, then compares what Rig reported with the recorded
//! response headers and bodies of the same interactions.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use rig::completion::CompletionRequest;
use rig::error::ProviderError;
use rig::test_utils::TraceCapture;
use serde_json::Value;

const FIELD: &str = rig::telemetry::PROVIDER_REQUEST_ID_FIELD;

/// What Rig reported for the three calls, in call order.
pub struct Observed {
    pub unary: Option<String>,
    pub unary_raw: Value,
    pub streamed: Option<String>,
    pub streamed_raw: Value,
    pub error: Option<String>,
    pub spans: Vec<String>,
}

/// Where a cassette body leaves its observation for the checks that run
/// after the cassette is written.
pub type Slot = Arc<Mutex<Option<Observed>>>;

/// Take the observation a cassette body stored.
pub fn take(slot: &Slot) -> Observed {
    slot.lock()
        .ok()
        .and_then(|mut observed| observed.take())
        .expect("the cell stored its observation")
}

/// Run the three calls with `params` and store what Rig reported in
/// `slot`. The provider must refuse `rejected` once `reject` shapes the
/// request: an unknown model, or an argument out of range where the
/// unknown-model error names the account.
pub async fn run<W, T, Wm, Tr>(
    slot: Slot,
    model: rig::driver::Model<W, T>,
    rejected: rig::driver::Model<Wm, Tr>,
    params: Option<Value>,
    reject: impl FnOnce(CompletionRequest) -> CompletionRequest,
) where
    W: rig::wire::Wire<Op = rig::operation::Completion>,
    T: rig::driver::Transport<W>,
    Wm: rig::wire::Wire<Op = rig::operation::Completion>,
    Tr: rig::driver::Transport<Wm>,
{
    let capture = TraceCapture::default();
    let _guard = tracing::subscriber::set_default(capture.subscriber());

    let unary = model
        .call(
            CompletionRequest::new("Reply with exactly: identity probe")
                .max_tokens(64)
                .additional_params(params.clone()),
        )
        .await
        .expect("unary call");
    let mut stream = model
        .stream(
            CompletionRequest::new("Reply with exactly: stream identity probe")
                .max_tokens(64)
                .additional_params(params.clone()),
        )
        .expect("stream opens");
    while let Some(item) = stream.next().await {
        item.expect("stream item");
    }
    let terminal = stream
        .finish()
        .await
        .expect("the stream ends with a final record");
    let error: ProviderError = rejected
        .call(reject(
            CompletionRequest::new("Reply with exactly: rejected")
                .max_tokens(64)
                .additional_params(params.clone()),
        ))
        .await
        .expect_err("the provider rejects the model");

    let spans = capture
        .values_of(FIELD)
        .iter()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect();
    let observed = Observed {
        unary: unary.provider_request_id,
        unary_raw: unary.raw,
        streamed: terminal.provider_request_id,
        streamed_raw: terminal.raw,
        error: error.provider_request_id().map(str::to_owned),
        spans,
    };
    if let Ok(mut slot) = slot.lock() {
        *slot = Some(observed);
    }
}

/// Every reported id is the verbatim value of `header` on its interaction,
/// the span recorded the same three ids in order, and a recorded
/// `system_fingerprint` reaches the unary response and streamed final.
pub fn assert_recorded(provider: &str, scenario: &str, header: &str, observed: &Observed) {
    let headers = crate::cassettes::recorded_response_header_pairs(provider, scenario);
    assert_eq!(
        headers.len(),
        3,
        "unary, streamed and rejected interactions"
    );
    let recorded: Vec<Option<String>> = headers
        .iter()
        .map(|pairs| {
            pairs
                .iter()
                .find(|(name, _)| name == header)
                .map(|(_, value)| value.clone())
        })
        .collect();
    for (surface, reported, recorded) in [
        ("unary response", &observed.unary, &recorded[0]),
        ("streamed final", &observed.streamed, &recorded[1]),
        ("provider error", &observed.error, &recorded[2]),
    ] {
        // Successful replies must carry the header; an error reply may omit
        // it, and then Rig must not invent one.
        assert!(
            recorded.is_some() || surface == "provider error",
            "[{provider}] the {surface} interaction records {header}"
        );
        assert!(
            !recorded
                .as_deref()
                .is_some_and(|value| value.contains("REDACTED")),
            "[{provider}] {header} is verbatim"
        );
        assert_eq!(
            reported, recorded,
            "[{provider}] the {surface} carries {header} exactly"
        );
    }
    let expected: Vec<String> = recorded.into_iter().flatten().collect();
    assert_eq!(
        observed.spans, expected,
        "[{provider}] the spans record each reported request id once"
    );

    let bodies = crate::cassettes::recorded_interaction_bodies(provider, scenario);
    let unary: Value = serde_json::from_str(&bodies[0].1).expect("unary reply is JSON");
    if let Some(fingerprint) = unary
        .get("system_fingerprint")
        .filter(|value| !value.is_null())
    {
        assert_eq!(
            observed.unary_raw.get("system_fingerprint"),
            Some(fingerprint),
            "[{provider}] the unary response keeps system_fingerprint"
        );
        let streamed = crate::cassettes::recorded_sse_json_frames(provider, scenario)
            .into_iter()
            .filter_map(|frame| frame.get("system_fingerprint").cloned())
            .rfind(|value| !value.is_null());
        if let Some(streamed) = streamed {
            assert_eq!(
                observed.streamed_raw.get("system_fingerprint"),
                Some(&streamed),
                "[{provider}] the streamed final keeps system_fingerprint"
            );
        }
    }
}
