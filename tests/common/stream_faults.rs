//! Scripted stream faults for the runner/native fault matrix.
//!
//! A fault cell serves a real provider adapter a stream that ends badly and
//! asserts what each runtime does with it: the error it surfaces, the record
//! it keeps, the history it commits, the tools it runs. The bodies are cut
//! from committed recordings ([`recorded_sse_frames`]) or, where a wire only
//! records single terminal frames, assembled from frames a real capture or
//! the adapter's own unit tests pin; every synthetic frame is a labelled
//! constant beside its cell. The scripted transport replaces the cassette
//! proxy for those cells, so their request boundary is pinned by the
//! recording's owning test, not here. Setup failures replay the committed
//! error recordings through the ordinary cassette wrappers.
#![allow(dead_code)]

use std::sync::{
    Arc,
    atomic::{AtomicUsize, Ordering},
};

use bevy_app::App;
use bevy_ecs::prelude::*;
use bytes::Bytes;
use futures::StreamExt;
use rig::agent::{MultiTurnStreamItem, StreamingError, StreamingResult};
use rig::completion::CompletionModel;
use rig::completion::PromptError;
use rig::effect::EffectFamily;
use rig::error::{ErrorKind, ErrorReport};
use rig::observe::{Action, AdapterEvent, ObservationLog};
use rig::streaming::{Delta, StreamEvent};
use rig::test_utils::SequencedStreamingHttpClient;
use rig::tool::{Tool, ToolContext};
use rig_ecs::{
    agent::{Failure, Order, Role, Utterance},
    bus::{Streamed, Witnessing},
    systems::spawn_run,
};
use rig_effect_log::EffectLog;

use crate::{
    ecs_agent::EcsAgent,
    goldens::families,
    support::{MathError, OperationArgs, Subtract},
};

/// The recorded SSE frames of one interaction of a scenario: the response
/// body split at its blank-line delimiters, in wire order.
pub(crate) fn recorded_sse_frames(
    provider: &str,
    scenario: &str,
    interaction: usize,
) -> Vec<String> {
    let recorded = crate::cassettes::recorded_statuses_and_bodies(provider, scenario);
    let (status, body) = recorded
        .get(interaction)
        .unwrap_or_else(|| panic!("{provider}/{scenario} has no interaction {interaction}"));
    assert_eq!(
        *status, 200,
        "{provider}/{scenario}[{interaction}] is not a successful stream"
    );
    let frames = sse_frames(body);
    assert!(
        frames.len() > 1,
        "{provider}/{scenario}[{interaction}] streams one frame; nothing can be cut from it"
    );
    frames
}

/// A body's SSE frames, without their delimiters.
pub(crate) fn sse_frames(body: &str) -> Vec<String> {
    body.split("\n\n")
        .filter(|frame| !frame.trim().is_empty())
        .map(str::to_owned)
        .collect()
}

/// The frames before the first one `is_terminal` accepts: a recording cut
/// short of its ending. At least one frame must precede the ending, or the
/// cut would not be a truncation of anything.
pub(crate) fn frames_before(frames: &[String], is_terminal: impl Fn(&str) -> bool) -> Vec<String> {
    let end = frames
        .iter()
        .position(|frame| is_terminal(frame))
        .expect("the recording carries the terminal frame to cut before");
    assert!(end > 0, "no frame precedes the terminal");
    frames[..end].to_vec()
}

/// The `data:` payload of an SSE frame, parsed.
pub(crate) fn frame_data(frame: &str) -> serde_json::Value {
    let data = frame
        .lines()
        .find_map(|line| {
            line.strip_prefix("data: ")
                .or_else(|| line.strip_prefix("data:"))
        })
        .unwrap_or_else(|| panic!("frame carries no data line: {frame}"));
    serde_json::from_str(data).unwrap_or_else(|error| panic!("frame data is JSON: {error}: {data}"))
}

/// The wire bytes of `frames`, each followed by its delimiter.
pub(crate) fn sse_bytes(frames: &[String]) -> Bytes {
    let mut body = frames.join("\n\n");
    body.push_str("\n\n");
    Bytes::from(body)
}

/// A transport that answers its one streaming request with `chunks`, then
/// EOF. A second request fails: every cell scripts exactly one exchange.
pub(crate) fn scripted(chunks: Vec<Bytes>) -> SequencedStreamingHttpClient {
    SequencedStreamingHttpClient::new(chunks.into_iter().map(Ok).collect())
}

/// What the runner's stream delivered before it ended.
#[derive(Debug, Default)]
pub(crate) struct Drained {
    /// The text deltas, concatenated.
    pub text: String,
    /// Provider terminal records forwarded to the consumer.
    pub terminals: usize,
    /// Committed model tool calls.
    pub tool_calls: usize,
    /// Final responses: a successful run yields exactly one.
    pub finals: usize,
    /// Every error item, as a report.
    pub errors: Vec<ErrorReport>,
}

/// Drain a runner stream to EOF.
pub(crate) async fn drain(stream: &mut StreamingResult) -> Drained {
    let mut drained = Drained::default();
    while let Some(item) = stream.next().await {
        match item {
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamEvent::BlockDelta {
                delta: Delta::Text { text },
                ..
            })) => drained.text.push_str(&text),
            Ok(MultiTurnStreamItem::StreamAssistantItem(StreamEvent::Final(_))) => {
                drained.terminals += 1;
            }
            Ok(MultiTurnStreamItem::ToolCall { .. }) => drained.tool_calls += 1,
            Ok(MultiTurnStreamItem::FinalResponse(_)) => drained.finals += 1,
            Ok(_) => {}
            Err(error) => drained.errors.push(report_of(&error)),
        }
    }
    drained
}

/// The report a runner stream error carries.
pub(crate) fn report_of(error: &StreamingError) -> ErrorReport {
    match error {
        StreamingError::Completion(error) => ErrorReport::from(error),
        StreamingError::Report(report) => report.clone(),
        StreamingError::Prompt(error) => match &**error {
            PromptError::Report(report) => report.clone(),
            PromptError::CompletionError(error) => ErrorReport::from(error),
            PromptError::PromptCancelled { reason, .. } => {
                ErrorReport::new(ErrorKind::Cancelled, reason.clone())
            }
            other => panic!("a provider-shaped failure, not {other:?}"),
        },
    }
}

/// The log as JSON with its delivery batches removed: batch numbers count
/// schedule passes, which a witness's extra work can shift by a tick, and
/// the comparison guide already excludes them from golden equality.
pub(crate) fn log_json_without_deliveries(log: &EffectLog) -> String {
    let mut log = log.clone();
    log.header.deliveries = None;
    serde_json::to_string(&log).expect("the log serializes")
}

/// The error the one completion record holds. A fault cell records exactly
/// one completion and nothing after it: no tool, no memory, no retry.
pub(crate) fn sole_failed_completion(log: &EffectLog) -> &ErrorReport {
    assert_eq!(
        families(log),
        [EffectFamily::Completion],
        "one completion record and no effect after the fault"
    );
    log.records[0]
        .outcome
        .as_ref()
        .expect_err("the completion record holds the fault")
}

/// The recorded stream error items of the one completion, as
/// `(position, report)`.
pub(crate) fn recorded_stream_errors(log: &EffectLog) -> Vec<(usize, ErrorReport)> {
    log.header
        .stream_errors
        .values()
        .flatten()
        .map(|error| (error.item, error.error.clone()))
        .collect()
}

/// A setup failure: the provider refused the request before any frame.
/// Both runtimes surface the recorded status and body under the wire's own
/// classification (`kind`), record the one completion as that failure and
/// stream nothing.
pub(crate) fn assert_setup_failure(report: &ErrorReport, kind: ErrorKind, status: u16) {
    assert_eq!(report.kind, kind, "{report:?}");
    assert_eq!(report.http_status, Some(status), "{report:?}");
    let body = report
        .provider_response_body()
        .expect("the provider's body travels with the failure");
    let body: serde_json::Value =
        serde_json::from_str(body).expect("the recorded error body is JSON");
    assert!(
        body.get("error").is_some(),
        "the provider's error envelope is preserved: {body}"
    );
}

/// Counts a tool's executions.
#[derive(Clone, Default)]
pub(crate) struct Invocations(Arc<AtomicUsize>);

impl Invocations {
    pub(crate) fn count(&self) -> usize {
        self.0.load(Ordering::SeqCst)
    }
}

/// `subtract`, counting its executions: a fault cell's proof that a call
/// the model issued before the fault never ran.
#[derive(Clone)]
pub(crate) struct CountedSubtract(pub(crate) Invocations);

impl Tool for CountedSubtract {
    const NAME: &'static str = Subtract::NAME;
    type Error = MathError;
    type Args = OperationArgs;
    type Output = i32;

    fn description(&self) -> String {
        Subtract.description()
    }

    fn parameters(&self) -> serde_json::Value {
        Subtract.parameters()
    }

    async fn call(
        &self,
        context: &mut ToolContext,
        args: Self::Args,
    ) -> Result<Self::Output, Self::Error> {
        self.0.0.fetch_add(1, Ordering::SeqCst);
        Subtract.call(context, args).await
    }
}

/// One native run and everything a cell asserts on afterwards.
pub(crate) struct NativeRun {
    /// The answer, or the failure.
    pub outcome: Result<String, Failure>,
    /// The recorder's log.
    pub log: EffectLog,
    /// The witness's log, when one was installed.
    pub trace: Option<Arc<ObservationLog>>,
    /// The committed history, in order: who spoke.
    pub roles: Vec<Role>,
    /// The one completion stream's fold, while its effect survives: a
    /// despawned in-flight effect takes its stream with it.
    pub stream: Option<Streamed>,
}

impl NativeRun {
    pub(crate) fn stream(&self) -> &Streamed {
        self.stream
            .as_ref()
            .expect("the stream's effect survived the run")
    }

    pub(crate) fn failure(&self) -> &Failure {
        self.outcome.as_ref().expect_err("the fault fails the run")
    }

    pub(crate) fn provider_report(&self) -> &ErrorReport {
        match self.failure() {
            Failure::Provider(report) => report,
            other => panic!("a provider failure, not {other:?}"),
        }
    }

    pub(crate) fn trace(&self) -> &ObservationLog {
        self.trace.as_deref().expect("a witnessed run")
    }

    /// The log without its scheduling-dependent delivery batches.
    pub(crate) fn log_json(&self) -> String {
        log_json_without_deliveries(&self.log)
    }
}

/// Drive one streamed native run of `prompt` on a fresh world over `model`
/// with an agent-level budget of two turns; `configure` shapes the agent
/// before the run is spawned.
pub(crate) async fn native_run(
    model: impl CompletionModel + 'static,
    preamble: &str,
    prompt: &str,
    witness: bool,
    configure: impl FnOnce(&mut EcsAgent),
) -> NativeRun {
    let mut ecs = EcsAgent::new(model, preamble, 2);
    configure(&mut ecs);
    let trace = witness.then(|| witnessed(&mut ecs.app));
    let run = spawn_run(ecs.app.world_mut(), ecs.agent, &[], prompt, true, None);
    let outcome = ecs.wait_for_outcome(run).await;
    // The run's ending lands before a despawned or dropped handler has
    // necessarily closed its record; give the owned task the same window
    // the anthropic cancellation cells do before reading the log.
    for _ in 0..64 {
        ecs.app.update();
        tokio::task::yield_now().await;
    }
    let log = ecs.effect_log();
    let roles = utterance_roles(ecs.app.world_mut(), run);
    let stream = sole_stream(ecs.app.world_mut());
    NativeRun {
        outcome,
        log,
        trace,
        roles,
        stream,
    }
}

/// A failure with the one field that varies between two replays of the same
/// recording removed: the replay server stamps a live `date` header on every
/// response, and a provider report keeps the response's headers. Everything
/// else — kind, status, message, body, the other headers — compares whole.
pub(crate) fn comparable_failure(failure: &Failure) -> Failure {
    let mut failure = failure.clone();
    let report = match &mut failure {
        Failure::Provider(report)
        | Failure::Cancelled(report)
        | Failure::Tool(report)
        | Failure::Memory(report) => report,
        _ => return failure,
    };
    if let Some(headers) = report
        .provider_response
        .as_mut()
        .and_then(|response| response.headers.as_mut())
    {
        headers.remove("date");
    }
    failure
}

/// The fault must read the same with and without a witness: same failure,
/// same record, same history; and `secret` must not reach the trace.
pub(crate) fn assert_witness_is_a_side_channel(
    observed: &NativeRun,
    plain: &NativeRun,
    secret: &str,
) {
    assert_eq!(
        comparable_failure(observed.failure()),
        comparable_failure(plain.failure()),
        "the witness does not change the failure"
    );
    assert_eq!(
        observed.log_json(),
        plain.log_json(),
        "the witness does not change the record"
    );
    assert_eq!(observed.roles, plain.roles, "nor the history");
    assert!(
        !trace_json(observed.trace()).contains(secret),
        "the credential never reaches the trace"
    );
}

/// The run's committed history, in order: who spoke.
pub(crate) fn utterance_roles(world: &mut World, run: Entity) -> Vec<Role> {
    let mut query = world.query_filtered::<(&ChildOf, &Role, &Order), With<Utterance>>();
    let mut rows: Vec<_> = query
        .iter(world)
        .filter(|(parent, ..)| parent.parent() == run)
        .map(|(_, role, order)| (order.0, *role))
        .collect();
    rows.sort_by_key(|(order, _)| *order);
    rows.into_iter().map(|(_, role)| role).collect()
}

/// The one stream's fold: its text so far, its error items with their
/// positions, and its outcome. `None` once its effect is gone.
pub(crate) fn sole_stream(world: &mut World) -> Option<Streamed> {
    let mut query = world.query::<&Streamed>();
    let streams: Vec<_> = query.iter(world).cloned().collect();
    assert!(
        streams.len() <= 1,
        "at most one streamed effect: {streams:?}"
    );
    streams.into_iter().next()
}

/// Install a witness over a fresh log.
pub(crate) fn witnessed(app: &mut App) -> Arc<ObservationLog> {
    let log = Arc::new(ObservationLog::default());
    Witnessing::install(app.world_mut(), log.clone());
    log
}

/// Every program ending the witness saw, by code.
pub(crate) fn endings(log: &ObservationLog) -> Vec<String> {
    log.trace()
        .observations
        .iter()
        .filter_map(|observation| match &observation.action {
            Action::Ended { ending } => Some(ending.code.clone()),
            _ => None,
        })
        .collect()
}

/// Every provider-boundary fact, in order.
pub(crate) fn adapter_events(log: &ObservationLog) -> Vec<AdapterEvent> {
    log.trace()
        .observations
        .iter()
        .filter_map(|observation| match &observation.action {
            Action::Adapter { observation } => Some(observation.event.clone()),
            _ => None,
        })
        .collect()
}

/// Every truncation the bus witnessed: items delivered and error items.
pub(crate) fn truncations(log: &ObservationLog) -> Vec<(usize, usize)> {
    log.trace()
        .observations
        .iter()
        .filter_map(|observation| match &observation.action {
            Action::StreamTruncated {
                delivered, errors, ..
            } => Some((*delivered, errors.len())),
            _ => None,
        })
        .collect()
}

/// The bus's decisions in order, by name.
pub(crate) fn bus_actions(log: &ObservationLog) -> Vec<&'static str> {
    log.trace()
        .observations
        .iter()
        .filter_map(|observation| match &observation.action {
            Action::Issued => Some("issued"),
            Action::Landed { outcome } => Some(match outcome {
                rig::observe::OutcomeSummary::Err { .. } => "landed_err",
                _ => "landed_ok",
            }),
            Action::Denied { .. } => Some("denied"),
            Action::Refused { .. } => Some("refused"),
            Action::Cancelled { .. } => Some("cancelled"),
            Action::StreamTruncated { .. } => Some("truncated"),
            Action::Held { .. } => Some("held"),
            Action::Released => Some("released"),
            Action::Replaced { .. } => Some("replaced"),
            Action::Adapter { .. } | Action::Ended { .. } | Action::Host { .. } => None,
        })
        .collect()
}

/// The whole trace, serialized: what an analysis sink would persist.
pub(crate) fn trace_json(log: &ObservationLog) -> String {
    serde_json::to_string(&log.trace()).expect("the trace serializes")
}
