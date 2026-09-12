//! Native counterparts of `stream_faults.rs`: the same recorded setup
//! failure, the same drop of the recorded stream and the same scripted
//! faults through the ECS runtime and the real Responses adapter, with the
//! world's witness installed. Each cell runs with and without a witness:
//! observation is a side channel, so the failure, the record and the
//! history must not change with it, and the trace carries the bus's facts
//! beside the Responses adapter's own boundary facts.

use bevy_ecs::prelude::*;
use bytes::Bytes;
use rig::error::ErrorKind;
use rig::observe::{AdapterEnding, AdapterErrorBoundary, AdapterEvent};
use rig::prelude::*;
use rig::providers::openai::{self, GPT_4O};
use rig::test_utils::SequencedStreamingHttpClient;
use rig_ecs::{
    agent::{Failure, MaxTokens, Preamble, Role},
    bus::{BusSet, EffectOutcome, RigSchedule, Streamed},
    systems::RigSet,
};

use super::super::support::with_openai_cassette;
use super::stream_faults::{
    ERROR_EVENT, MISSING_MODEL, SCRIPTED_KEY, SETUP_KIND, SETUP_MAX_TOKENS, SETUP_PROMPT,
    delta_text, scripted_client, text_prefix_frames, tool_call_prefix_frames,
};
use crate::{
    ecs_agent::EcsAgent,
    stream_faults::{
        CountedSubtract, Invocations, NativeRun, adapter_events, assert_setup_failure, bus_actions,
        comparable_failure, endings, native_run, sole_failed_completion, sse_bytes, trace_json,
        truncations,
    },
    support::{
        Adder, STREAMING_PREAMBLE, STREAMING_PROMPT, STREAMING_TOOLS_PREAMBLE,
        STREAMING_TOOLS_PROMPT,
    },
};

/// A scripted-transport model: one streaming exchange, then EOF.
fn scripted_model(
    chunks: Vec<Bytes>,
) -> openai::responses_api::ResponsesCompletionModel<SequencedStreamingHttpClient> {
    scripted_client(chunks).completion_model(GPT_4O)
}

/// The scripted cells' witness check, over this module's credential.
fn assert_witness_is_a_side_channel(observed: &NativeRun, plain: &NativeRun) {
    crate::stream_faults::assert_witness_is_a_side_channel(observed, plain, SCRIPTED_KEY);
}

/// The bus's facts for a failed run, in order, and the Responses adapter's
/// boundary facts: the request it sent and how the attempt closed.
fn assert_failure_facts(
    run: &NativeRun,
    actions: &[&str],
    ending: AdapterEnding,
) -> Vec<AdapterEvent> {
    let trace = run.trace();
    assert_eq!(bus_actions(trace), actions);
    assert_eq!(endings(trace), ["provider"]);
    let events = adapter_events(trace);
    assert!(
        matches!(events.first(), Some(AdapterEvent::Started { route, .. }) if route == "/responses"),
        "the adapter announces its request: {events:?}"
    );
    assert_eq!(
        events.last(),
        Some(&AdapterEvent::Finished { ending }),
        "the attempt closes as the fault: {events:?}"
    );
    events
}

/// The recorded 400 through the native runtime: the run fails as the
/// provider's response, streams nothing, commits only the prompt.
#[tokio::test]
async fn setup_failure_fails_the_run_with_the_recorded_status() {
    // The recording's request carries no system instruction at all, which
    // strict matching distinguishes from an empty one: `Preamble(None)`,
    // not the helper's `Some("")`.
    let configure = |ecs: &mut EcsAgent| {
        ecs.app
            .world_mut()
            .entity_mut(ecs.agent)
            .insert((Preamble(None), MaxTokens(Some(SETUP_MAX_TOKENS))));
    };
    let mut runs = Vec::new();
    for witness in [true, false] {
        let runs = &mut runs;
        with_openai_cassette(
            "error_envelope/nonexistent_model_streaming_error_preserves_status_and_body",
            |client| async move {
                let run = native_run(
                    client.completion_model(MISSING_MODEL),
                    "",
                    SETUP_PROMPT,
                    witness,
                    configure,
                )
                .await;
                assert_setup_failure(run.provider_report(), SETUP_KIND, 400);
                assert_setup_failure(sole_failed_completion(&run.log), SETUP_KIND, 400);
                assert_eq!(run.roles, [Role::User], "only the prompt is history");
                assert!(run.stream().text.is_empty(), "{:?}", run.stream);
                assert_eq!(run.stream().errors.len(), 1, "{:?}", run.stream().errors);
                assert_eq!(run.stream().errors[0].0, 0, "the error is the first item");
                runs.push(run);
            },
        )
        .await;
    }
    let (observed, plain) = (&runs[0], &runs[1]);
    assert_eq!(
        comparable_failure(observed.failure()),
        comparable_failure(plain.failure())
    );
    assert_eq!(observed.log_json(), plain.log_json());
    let events = assert_failure_facts(
        observed,
        &["issued", "landed_err"],
        AdapterEnding::Error {
            boundary: AdapterErrorBoundary::ProviderResponse,
            kind: "provider_response".into(),
            status: Some(400),
            retryable: false,
        },
    );
    assert!(
        events.contains(&AdapterEvent::Response { status: 400 }),
        "{events:?}"
    );
    assert!(
        events.iter().any(|event| matches!(
            event,
            AdapterEvent::ErrorEnvelope { error }
                if error.status.as_deref() == Some("invalid_request_error")
        )),
        "the envelope's own fields are projected: {events:?}"
    );
    assert!(
        !trace_json(observed.trace()).contains(SETUP_PROMPT),
        "the request body never reaches the trace"
    );
}

/// EOF after content through the native runtime: the run fails as a
/// truncation, the stream keeps the prefix, history keeps only the prompt,
/// and the bus witnesses the truncation with what was delivered.
#[tokio::test]
async fn truncation_after_content_fails_the_run_and_keeps_the_prefix() {
    let frames = text_prefix_frames();
    let prefix = delta_text(&frames);
    let mut runs = Vec::new();
    for witness in [true, false] {
        let run = native_run(
            scripted_model(vec![sse_bytes(&frames)]),
            STREAMING_PREAMBLE,
            STREAMING_PROMPT,
            witness,
            |_| {},
        )
        .await;
        let report = run.provider_report();
        assert_eq!(report.kind, ErrorKind::Response, "{report:?}");
        assert_eq!(report.message, rig::serve::stream_truncated().message);
        assert_eq!(
            sole_failed_completion(&run.log).message,
            rig::serve::stream_truncated().message
        );
        assert_eq!(run.stream().text, prefix);
        assert!(run.stream().errors.is_empty(), "EOF is not an item");
        assert!(run.stream().outcome.is_none(), "{:?}", run.stream().outcome);
        assert_eq!(run.roles, [Role::User], "the cut turn is not history");
        runs.push(run);
    }
    let (observed, plain) = (&runs[0], &runs[1]);
    assert_witness_is_a_side_channel(observed, plain);
    let events = assert_failure_facts(
        observed,
        &["issued", "truncated", "landed_err"],
        AdapterEnding::Eof {
            after: frames.len(),
        },
    );
    assert!(
        events.contains(&AdapterEvent::Response { status: 200 }),
        "{events:?}"
    );
    let delivered = observed.stream().events.len();
    assert_eq!(truncations(observed.trace()), [(delivered, 0)]);
}

/// EOF after a complete tool call through the native runtime: the call is
/// never dispatched, the tool never runs, no assistant turn is committed,
/// and the run fails as a truncation.
#[tokio::test]
async fn truncation_after_a_complete_tool_call_never_runs_the_tool() {
    let frames = tool_call_prefix_frames();
    let mut runs = Vec::new();
    for witness in [true, false] {
        let invocations = Invocations::default();
        let counted = invocations.clone();
        let run = native_run(
            scripted_model(vec![sse_bytes(&frames)]),
            STREAMING_TOOLS_PREAMBLE,
            STREAMING_TOOLS_PROMPT,
            witness,
            move |ecs| {
                ecs.tool(Adder);
                ecs.tool(CountedSubtract(counted));
            },
        )
        .await;
        let report = run.provider_report();
        assert_eq!(report.kind, ErrorKind::Response, "{report:?}");
        assert_eq!(
            sole_failed_completion(&run.log).message,
            rig::serve::stream_truncated().message,
            "one completion and no tool effect"
        );
        assert_eq!(invocations.count(), 0, "the tool never ran");
        assert_eq!(run.roles, [Role::User], "the call is not history");
        assert!(
            run.stream()
                .events
                .iter()
                .any(|event| matches!(event, rig::streaming::StreamEvent::BlockEnd { .. })),
            "the call streamed to its end before the cut: {:?}",
            run.stream().events
        );
        runs.push(run);
    }
    let (observed, plain) = (&runs[0], &runs[1]);
    assert_witness_is_a_side_channel(observed, plain);
    assert_failure_facts(
        observed,
        &["issued", "truncated", "landed_err"],
        AdapterEnding::Eof {
            after: frames.len(),
        },
    );
    assert_eq!(truncations(observed.trace()).len(), 1);
}

/// An error event after content through the native runtime: the run fails
/// with the provider's error, the stream keeps the prefix and the error
/// item at its position, and history keeps only the prompt.
#[tokio::test]
async fn error_event_after_content_fails_with_the_provider_error() {
    let mut frames = text_prefix_frames();
    let prefix = delta_text(&frames);
    frames.push(ERROR_EVENT.to_owned());
    let mut runs = Vec::new();
    for witness in [true, false] {
        let run = native_run(
            scripted_model(vec![sse_bytes(&frames)]),
            STREAMING_PREAMBLE,
            STREAMING_PROMPT,
            witness,
            |_| {},
        )
        .await;
        let report = run.provider_report();
        assert_eq!(report.kind, ErrorKind::ProviderResponse, "{report:?}");
        assert!(
            report
                .provider_response_body()
                .is_some_and(|body| body.contains("boom")),
            "{report:?}"
        );
        assert_eq!(
            sole_failed_completion(&run.log).kind,
            ErrorKind::ProviderResponse
        );
        assert_eq!(run.stream().text, prefix);
        assert_eq!(run.stream().errors.len(), 1, "{:?}", run.stream().errors);
        assert_eq!(
            run.stream().errors[0].0,
            run.stream().events.len(),
            "the error item follows the delivered content"
        );
        assert_eq!(run.roles, [Role::User]);
        runs.push(run);
    }
    let (observed, plain) = (&runs[0], &runs[1]);
    assert_witness_is_a_side_channel(observed, plain);
    let events = assert_failure_facts(
        observed,
        &["issued", "landed_err"],
        AdapterEnding::Error {
            boundary: AdapterErrorBoundary::ProviderResponse,
            kind: "provider_response".into(),
            status: None,
            retryable: false,
        },
    );
    assert!(
        events.iter().any(|event| matches!(
            event,
            AdapterEvent::ErrorEnvelope { error }
                if error.code.as_deref() == Some("server_error")
                    && error.status.as_deref() == Some("server_error")
                    && error.message.as_deref() == Some("boom")
        )),
        "the event's envelope is projected: {events:?}"
    );
    assert!(
        truncations(observed.trace()).is_empty(),
        "an error, not EOF"
    );
}

/// Despawn the issued completion once its stream has delivered text: the
/// consumer dropping the runner's stream, on this runtime. A run-level
/// `Cancelled` would instead leave the in-flight stream to its handler
/// (CONTRACT §9.1); the despawn is what ends the exchange mid-answer.
fn despawn_at_first_text(
    mut commands: Commands,
    streams: Query<(Entity, &Streamed), Without<EffectOutcome>>,
) {
    for (entity, stream) in &streams {
        if !stream.text.is_empty() {
            commands.entity(entity).despawn();
        }
    }
}

/// The issued completion is despawned at the recorded stream's first text
/// delta: the run fails as cancelled, the completion is recorded as
/// cancelled and no answer is committed. The replay marked the interaction
/// consumed when it matched the request, so `finish` still passes; the
/// despawn lands on the response body, which the replay does not track.
#[tokio::test]
async fn despawning_the_stream_at_the_first_delta_records_a_cancel() {
    let mut runs = Vec::new();
    for witness in [true, false] {
        let runs = &mut runs;
        with_openai_cassette("streaming/streaming_smoke", |client| async move {
            let run = native_run(
                client.completion_model(GPT_4O),
                STREAMING_PREAMBLE,
                STREAMING_PROMPT,
                witness,
                |ecs| {
                    ecs.app.add_systems(
                        RigSchedule,
                        despawn_at_first_text
                            .after(BusSet::Collect)
                            .before(RigSet::Fold),
                    );
                },
            )
            .await;
            assert!(
                matches!(run.failure(), Failure::Cancelled(_)),
                "a cancelled run, not {:?}",
                run.failure()
            );
            let recorded = sole_failed_completion(&run.log);
            assert_eq!(recorded.kind, ErrorKind::Cancelled, "{recorded:?}");
            assert!(run.stream.is_none(), "the despawned effect took its stream");
            assert_eq!(run.roles, [Role::User], "no answer is committed");
            runs.push(run);
        })
        .await;
    }
    let (observed, plain) = (&runs[0], &runs[1]);
    assert_eq!(
        comparable_failure(observed.failure()),
        comparable_failure(plain.failure())
    );
    assert_eq!(observed.log_json(), plain.log_json());
    let trace = observed.trace();
    assert_eq!(endings(trace), ["cancelled"]);
    assert_eq!(bus_actions(trace), ["issued", "cancelled"]);
}
