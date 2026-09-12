#![allow(clippy::unwrap_used)]
use super::*;
use rig_core::serve::Observe;

#[test]
fn cancellation_and_terminal_observation_have_one_recording_boundary() {
    for terminal_first in [false, true] {
        let recorder = rig_effect_log::EffectLogRecorder::keeping_stream_events();
        let recording = Recording::new(recorder.clone());
        let id = EffectId::from_raw(0);
        let kind = EffectKind::Custom {
            kind: "test".into(),
            payload: serde_json::Value::Null,
        };
        recording.begin(id, "test".into(), kind.clone(), Origin::default());
        let observed = Arc::new(ObservedState::default());
        let mut observer = WorldObserver {
            adapter: None,
            id,
            recording: Some(recording.clone()),
            published: None,
            witness: None,
            observed: observed.clone(),
        };
        let final_item = Ok(StreamEvent::Final(rig_core::streaming::StreamFinal::new(
            "test",
            Default::default(),
        )));
        let answer = rig_core::serve::StreamTap::new()
            .observe(&final_item)
            .unwrap();
        if terminal_first {
            observer.stream_item(&final_item, Some(&answer));
        }
        let original = observed.take_outcome();
        assert_eq!(original.is_some(), terminal_first);
        recording.resolve(id, original.unwrap_or_else(|| Err(cancelled())));
        let closed = serde_json::to_value(recorder.log()).unwrap();
        observer.stream_item(&final_item, Some(&answer));
        observer.event(final_item.as_ref().unwrap());
        observer.stream_error(&cancelled());
        observer.outcome(&answer);
        observer.patch(&kind);
        observer.discard("layer");
        assert_eq!(serde_json::to_value(recorder.log()).unwrap(), closed);
        assert!(!observed.is_discarded());
    }
}

/// The readiness signal the module names — an `On<Add, EffectOutcome>`
/// observer — may consume the answer and despawn the effect right there.
/// The record still closes with the answer the handler gave: the despawn
/// reaches `Remove<InFlight>` with the outcome landed but not yet settled,
/// and that outcome, not a cancellation and not silence, is what the
/// record takes.
#[test]
fn a_despawn_from_the_outcome_observer_still_closes_the_record() {
    use crate::bus::{Handlers, InFlight, PendingEffect, WorldOutcome, run_to_quiescence};
    use rig_core::effect::FamilyDescriptor;

    let mut world = World::new();
    crate::bus::install_bus(&mut world, Default::default());
    let recorder = rig_effect_log::EffectLogRecorder::new();
    Recording::install(&mut world, recorder.clone());
    Handlers::with(&mut world, |handlers| {
        handlers.register_open(
            "open",
            FamilyDescriptor::Custom {
                kind: "test".into(),
            },
        )
    })
    .unwrap()
    .unwrap();
    world.add_observer(|added: On<Add, EffectOutcome>, mut commands: Commands| {
        commands.entity(added.event().entity).despawn();
    });
    let kind = EffectKind::Custom {
        kind: "test".into(),
        payload: serde_json::Value::Null,
    };
    let effect = world.spawn(PendingEffect::new("open", kind)).id();
    run_to_quiescence(&mut world);
    assert!(
        world.get::<InFlight>(effect).is_some(),
        "issued to the open key"
    );
    let answer = Outcome::Custom {
        payload: serde_json::json!("answered"),
    };
    world
        .entity_mut(effect)
        .insert(WorldOutcome::new(Ok(answer.clone())));
    run_to_quiescence(&mut world);
    assert!(
        world.get_entity(effect).is_err(),
        "the observer despawned the effect as its outcome landed"
    );
    let log = recorder.log();
    assert_eq!(log.len(), 1, "the record closed: {log:?}");
    let record = log.first().unwrap();
    assert_eq!(
        serde_json::to_value(&record.outcome).unwrap(),
        serde_json::to_value(Ok::<_, ErrorReport>(answer)).unwrap(),
        "the record holds the handler's answer, not a cancellation"
    );
}
