//! Transport request ids and `system_fingerprint` reach every surface exactly
//! as the provider sent them: the unary response, the streamed final, a
//! provider error, and the completion span.
//!
//! Each cell runs one unary call, one streamed call and one call the provider
//! rejects, in that order, then compares what Rig reported with the recorded
//! response headers and bodies of the same interactions.

use std::sync::{Arc, Mutex};

use futures::StreamExt;
use rig::completion::{CompletionModel, CompletionRequestBuilder};
use rig::error::ProviderError;
use rig::streaming::StreamEvent;
use serde_json::Value;
use tracing::field::{Field, Visit};
use tracing::span::{Attributes, Id, Record};
use tracing_subscriber::layer::{Context, Layer, SubscriberExt};

const FIELD: &str = rig::telemetry::PROVIDER_REQUEST_ID_FIELD;

#[derive(Clone, Default)]
struct Recorded(Arc<Mutex<Vec<String>>>);

struct Visitor<'a>(&'a Recorded);

impl Visit for Visitor<'_> {
    fn record_str(&mut self, field: &Field, value: &str) {
        if field.name() == FIELD
            && let Ok(mut values) = self.0.0.lock()
        {
            values.push(value.to_owned());
        }
    }

    fn record_debug(&mut self, _field: &Field, _value: &dyn std::fmt::Debug) {}
}

struct Capture(Recorded);

impl<S: tracing::Subscriber> Layer<S> for Capture {
    fn on_new_span(&self, attrs: &Attributes<'_>, _id: &Id, _ctx: Context<'_, S>) {
        attrs.record(&mut Visitor(&self.0));
    }

    fn on_record(&self, _span: &Id, values: &Record<'_>, _ctx: Context<'_, S>) {
        values.record(&mut Visitor(&self.0));
    }
}

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
pub async fn run<M, R>(
    slot: Slot,
    model: M,
    rejected: R,
    params: Option<Value>,
    reject: impl FnOnce(CompletionRequestBuilder<R>) -> CompletionRequestBuilder<R>,
) where
    M: CompletionModel + Clone,
    R: CompletionModel + Clone,
{
    let recorded = Recorded::default();
    let subscriber = tracing_subscriber::registry().with(Capture(recorded.clone()));
    let _guard = tracing::subscriber::set_default(subscriber);

    let unary = model
        .completion_request("Reply with exactly: identity probe")
        .max_tokens(64)
        .additional_params(params.clone())
        .send()
        .await
        .expect("unary call");
    let mut stream = model
        .completion_request("Reply with exactly: stream identity probe")
        .max_tokens(64)
        .additional_params(params.clone())
        .stream()
        .await
        .expect("stream opens");
    let mut terminal = None;
    while let Some(item) = stream.next().await {
        if let StreamEvent::Final(record) = item.expect("stream item") {
            terminal = Some(record);
        }
    }
    let terminal = terminal.expect("the stream ends with a final record");
    let error: ProviderError = reject(
        rejected
            .completion_request("Reply with exactly: rejected")
            .max_tokens(64)
            .additional_params(params.clone()),
    )
    .send()
    .await
    .expect_err("the provider rejects the model");

    let spans = recorded
        .0
        .lock()
        .map(|values| values.clone())
        .unwrap_or_default();
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
