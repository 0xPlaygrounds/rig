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
            id,
            recording: Some(recording.clone()),
            published: None,
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
        observer.discard();
        assert_eq!(serde_json::to_value(recorder.log()).unwrap(), closed);
        assert!(!observed.is_discarded());
    }
}
