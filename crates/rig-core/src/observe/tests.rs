use super::*;

fn started(order: u64) -> Observation {
    Observation::new(
        Subject {
            scope: Some("app/run#1".into()),
            order: Some(order),
            key: Some(HandlerKey::from("tool:bash")),
            family: Some(EffectFamily::Tool),
            ..Subject::default()
        },
        Stage::Handler,
        Emitter::versioned("app/steer", "1"),
        Action::Adapter {
            observation: AdapterObservation {
                operation: "op-1".into(),
                attempt: Some(1),
                event: AdapterEvent::Started {
                    method: "POST".into(),
                    route: "/completion".into(),
                },
                analysis: None,
            },
        },
    )
}

#[test]
fn a_trace_round_trips_through_json() {
    let log = ObservationLog::default();
    log.observe(started(3));
    let trace = log.trace();
    let json = serde_json::to_string(&trace).unwrap();
    let back: ObservationTrace = serde_json::from_str(&json).unwrap();
    assert_eq!(back, trace);
    assert!(json.contains("\"stage\":\"handler\""));
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
        log.observe(started(0));
        assert!(
            !log.trace().finalized,
            "a later producer fact invalidates finalization"
        );
        log.finalize();
        assert!(log.trace().finalized);
    }
}

/// A capture keeps the first facts and counts the rest as dropped.
#[test]
fn a_capture_keeps_the_first_facts() {
    for (fill, kept, dropped) in [(2, 2, 0), (3, 3, 0), (5, 3, 2)] {
        let log = ObservationLog::with_capacity(3);
        for i in 0..fill {
            log.observe(started(i));
        }
        let trace = log.trace();
        assert_eq!(trace.observations.len(), kept, "fill={fill}");
        assert_eq!(trace.dropped, dropped, "fill={fill}");
        assert_eq!(trace.is_complete(), dropped == 0);
        assert_eq!(trace.observations.first().map(|o| o.seq), Some(0));
    }
}
