//! Native counterparts of `stream_faults.rs`: the same recorded setup
//! failure, the same drop of the recorded stream and the same scripted
//! faults through the ECS runtime and the real Responses adapter, with the
//! world's witness installed. Each cell runs with and without a witness:
//! observation is a side channel, so the failure, the record and the
//! history must not change with it, and the trace carries the bus's facts
//! beside the Responses adapter's own boundary facts.

use bevy_ecs::prelude::*;
use rig::error::ErrorKind;
use rig::providers::openai::GPT_4O;
use rig::streaming::StreamEvent;
use rig_cassette::effect_log::EffectLog;
use rig_ecs::{
    agent::{Failure, Role},
    bus::{BusSet, EffectOutcome, RigSchedule, Streamed},
    systems::RigSet,
};

use super::super::support::with_openai_cassette;
use crate::{
    stream_faults::{
        bus_actions, comparable_failure, endings, log_json_without_deliveries, native_run_serving,
        sole_failed_completion,
    },
    support::{STREAMING_PREAMBLE, STREAMING_PROMPT},
};

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
    crate::goldens::capture_world_programs(async {
        let mut runs = Vec::new();
        for witness in [true, false] {
            let runs = &mut runs;
            with_openai_cassette("streaming/streaming_smoke", |client| async move {
                let model = client.openai.completion(GPT_4O);
                let run = native_run_serving(
                    |label| crate::ecs_matrix::world::FirstDelta {
                        inner: rig::serve::adapters::ModelAdapter::new(label, model),
                        tool: false,
                        release: std::sync::Arc::new(tokio::sync::Semaphore::new(0)),
                    },
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
        crate::goldens::world_golden_effects(
            "openai_stream_faults_despawning_the_stream_at_the_first_delta_records_a_cancel",
            &observed.log,
        );
        assert_eq!(
            comparable_failure(observed.failure()),
            comparable_failure(plain.failure())
        );
        // The delivery gate keeps later deltas pending until despawn drops
        // the stream, so the cancellation cut is independent of polling speed.
        assert_eq!(
            log_json_through_the_first_text_delta(&observed.log),
            log_json_through_the_first_text_delta(&plain.log)
        );
        let trace = observed.trace();
        assert_eq!(endings(trace), ["cancelled"]);
        assert_eq!(bus_actions(trace), ["issued", "cancelled"]);
    })
    .await;
}

/// The log with its one record's kept events cut after the first text
/// delta, without its delivery batches.
fn log_json_through_the_first_text_delta(log: &EffectLog) -> String {
    let mut log = log.clone();
    let [record] = log.records.as_mut_slice() else {
        panic!("one completion record, not {}", log.records.len());
    };
    let events = record.events.as_mut().expect("the recorder keeps events");
    let first_text = events
        .items()
        .iter()
        .position(|item| matches!(item, rig::streaming::Item::Event(StreamEvent::Text { .. })))
        .expect("the despawn waited for a text delta");
    *events = rig::streaming::Transcript::from_items(events.items()[..=first_text].to_vec())
        .expect("a prefix of a transcript");
    log_json_without_deliveries(&log)
}
