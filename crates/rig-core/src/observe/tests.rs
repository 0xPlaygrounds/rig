use super::*;
use crate::error::ErrorKind;

fn denied(order: u64) -> Observation {
    Observation::new(
        Subject {
            scope: Some("app/run#1".into()),
            order: Some(order),
            key: Some(HandlerKey::from("tool:bash")),
            family: Some(EffectFamily::Tool),
            ..Subject::default()
        },
        Stage::Gate,
        Emitter::versioned("app/steer", "1"),
        Action::Denied {
            reason: Reason::with_detail("denied", "refusing to delete /"),
        },
    )
}

#[test]
fn a_trace_round_trips_through_json() {
    let log = ObservationLog::default().with_session("trial-7");
    log.observe(denied(3));
    log.observe(Observation::new(
        Subject {
            effect: Some(EffectId::from_raw(4)),
            parent: Some(EffectId::from_raw(1)),
            ..Subject::default()
        },
        Stage::Collect,
        Emitter::named("rig-ecs/bus"),
        Action::StreamTruncated {
            delivered: 2,
            tail: Vec::new(),
            errors: vec![Reason::from_report(&ErrorReport::new(
                ErrorKind::Response,
                "the stream ended before its terminal record",
            ))],
        },
    ));
    let trace = log.trace();
    let json = serde_json::to_string(&trace).unwrap();
    let back: ObservationTrace = serde_json::from_str(&json).unwrap();
    assert_eq!(back, trace);
    assert_eq!(back.session.as_deref(), Some("trial-7"));
    assert!(json.contains("\"stage\":\"gate\""));
    assert!(json.contains("\"action\":\"stream_truncated\""));
}

#[derive(Debug, PartialEq, Serialize, Deserialize)]
struct Approval {
    operation: String,
    approved: bool,
}

impl HostAction for Approval {
    const KIND: &'static str = "app/approval";
}

#[test]
fn a_host_action_travels_typed_and_named() {
    let fact = Approval {
        operation: "op-1".into(),
        approved: false,
    };
    let action = fact.action().unwrap();
    assert!(matches!(&action, Action::Host { kind, .. } if &**kind == "app/approval"));
    assert_eq!(Approval::from_action(&action).unwrap().unwrap(), fact);
    let other = Action::Host {
        kind: "app/other".into(),
        payload: serde_json::json!({}),
    };
    assert!(Approval::from_action(&other).is_none());
}

#[test]
fn a_later_fact_reopens_a_finalized_capture_even_when_dropped() {
    for capacity in [0, 8] {
        let log = ObservationLog::with_capacity(capacity);
        log.finalize();
        assert!(log.trace().finalized);
        log.observe(denied(0));
        assert!(
            !log.trace().finalized,
            "a later producer fact invalidates finalization"
        );
        log.finalize();
        assert!(log.trace().finalized);
    }
}

/// The two sink shapes across the capacity boundary: a capture keeps the
/// first facts and counts the rest, a ring keeps the last facts and counts
/// what it let go; both say when something was lost, and a drain starts
/// either over.
#[test]
fn a_capture_keeps_the_first_facts_and_a_ring_the_last() {
    for (ring, fill, kept, dropped, first_seq) in [
        (false, 2, 2, 0, 0),
        (false, 3, 3, 0, 0),
        (false, 5, 3, 2, 0),
        (true, 2, 2, 0, 0),
        (true, 3, 3, 0, 0),
        (true, 5, 3, 2, 2),
    ] {
        let log = if ring {
            ObservationLog::ring(3)
        } else {
            ObservationLog::with_capacity(3)
        };
        for i in 0..fill {
            log.observe(denied(i));
        }
        let trace = log.trace();
        assert_eq!(trace.observations.len(), kept, "ring={ring} fill={fill}");
        assert_eq!(trace.dropped, dropped, "ring={ring} fill={fill}");
        assert_eq!(trace.is_complete(), dropped == 0);
        assert_eq!(
            trace.observations.first().map(|o| o.seq),
            Some(first_seq),
            "ring={ring} fill={fill}: a ring keeps the latest, a capture the earliest"
        );
        let drained = log.drain();
        assert_eq!(drained.observations.len(), kept);
        assert_eq!(drained.dropped, dropped);
        let after = log.trace();
        assert!(
            after.observations.is_empty() && after.dropped == 0,
            "a drain starts over"
        );
        log.observe(denied(9));
        assert_eq!(
            log.trace().observations[0].seq,
            fill,
            "the sequence continues across a drain"
        );
    }
}
