use std::{
    sync::{
        Arc,
        atomic::{AtomicU64, Ordering},
    },
    time::Duration,
};

use super::*;
use crate::error::ErrorKind;

/// A clock a test advances by hand.
struct Ticks(AtomicU64);

impl Clock for Ticks {
    fn elapsed(&self) -> Duration {
        Duration::from_millis(self.0.load(Ordering::SeqCst))
    }
}

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
fn a_sink_orders_facts_and_stamps_the_host_clock() {
    let ticks = Arc::new(Ticks(AtomicU64::new(5)));
    let log = ObservationLog::with_capacity(8).with_clock(ticks.clone());
    log.observe(denied(0));
    ticks.0.store(9, Ordering::SeqCst);
    log.observe(Observation::new(
        Subject::scoped("app/run#1"),
        Stage::Runtime,
        Emitter::named("rig-ecs/agent"),
        Action::Ended {
            ending: Reason::code("settled"),
        },
    ));
    let trace = log.trace();
    assert_eq!(
        trace.observations.iter().map(|o| o.seq).collect::<Vec<_>>(),
        [0, 1]
    );
    assert_eq!(
        trace.observations.iter().map(|o| o.at).collect::<Vec<_>>(),
        [
            Some(Duration::from_millis(5)),
            Some(Duration::from_millis(9))
        ]
    );
    assert!(trace.is_complete());
    assert!(!trace.finalized);
    log.finalize();
    assert!(log.trace().finalized);
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

#[test]
fn comparison_ignores_measurements_and_keys_on_semantics() {
    let fast = ObservationLog::default().with_clock(Arc::new(Ticks(AtomicU64::new(1))));
    let slow = ObservationLog::default().with_clock(Arc::new(Ticks(AtomicU64::new(900))));
    for log in [&fast, &slow] {
        log.observe(denied(0));
    }
    assert_eq!(compare(&fast.trace(), &slow.trace()), Comparison::Equal);

    let other = ObservationLog::default();
    other.observe(denied(0));
    other.observe(Observation::new(
        Subject::scoped("app/run#1"),
        Stage::Runtime,
        Emitter::named("rig-ecs/agent"),
        Action::Ended {
            ending: Reason::code("cancelled"),
        },
    ));
    let Comparison::Diverged(divergence) = compare(&fast.trace(), &other.trace()) else {
        panic!("a longer trace diverges at its extra fact");
    };
    assert_eq!(divergence.index, 1);
    assert!(divergence.expected.is_none());
    assert!(matches!(
        divergence.actual,
        Some(Observation {
            action: Action::Ended { .. },
            ..
        })
    ));

    let mut changed = other.trace();
    changed.observations.truncate(1);
    changed.observations[0].action = Action::Denied {
        reason: Reason::with_detail("denied", "another reason"),
    };
    let Comparison::Diverged(divergence) = compare(&fast.trace(), &changed) else {
        panic!("a different reason diverges");
    };
    assert_eq!(divergence.index, 0);
}

#[test]
fn a_full_sink_counts_what_it_drops_and_is_never_equal() {
    let log = ObservationLog::with_capacity(2);
    for order in 0..5 {
        log.observe(denied(order));
    }
    let trace = log.trace();
    assert_eq!(trace.observations.len(), 2);
    assert_eq!(trace.dropped, 3);
    assert!(!trace.is_complete());
    // Sequence numbers keep counting past the drop so a reader can tell
    // where the gap is.
    assert_eq!(trace.observations.last().unwrap().seq, 1);
    let complete = ObservationLog::default();
    complete.observe(denied(0));
    complete.observe(denied(1));
    assert!(matches!(
        compare(&trace, &complete.trace()),
        Comparison::Incomparable { reason } if &*reason.code == "incomplete_expected"
    ));
    assert!(matches!(
        compare(&complete.trace(), &trace),
        Comparison::Incomparable { reason } if &*reason.code == "incomplete_actual"
    ));
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
fn reasons_and_summaries_come_from_reports_without_inventing_detail() {
    let report = ErrorReport::new(ErrorKind::Cancelled, "stopped").with_retryable(false);
    let reason = Reason::from_report(&report);
    assert_eq!(&*reason.code, "cancelled");
    assert_eq!(reason.detail.as_deref(), Some("stopped"));
    assert_eq!(
        OutcomeSummary::of(&Err(report)),
        OutcomeSummary::Err {
            reason,
            retryable: false
        }
    );
    assert!(Emitter::unknown().is_unknown());
    assert!(!Emitter::named("rig-ecs/bus").is_unknown());
    assert_eq!(&*Reason::unknown().code, "unknown");
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
