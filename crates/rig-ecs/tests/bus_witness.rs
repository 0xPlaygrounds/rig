//! The witness: decisions observed where they happen, beside the record.
//!
//! | contract | test |
//! |---|---|
//! | a gate denial is a fact with its reason and no exchange record | `a_gate_denial_is_witnessed_with_its_reason_and_leaves_no_record` |
//! | hold/release and hold/deny stay distinct sequences | `hold_release_and_hold_deny_are_distinct_sequences` |
//! | a judge replacement exposes the recorded and the consumed answer | `a_judge_replacement_exposes_both_answers` |
//! | a layer's patch keeps its before; a layer's denial is a serve-side fact | `a_layer_patch_and_discard_are_witnessed_at_the_handler_side` |
//! | a stream that ends before its terminal keeps its tail | `a_stream_that_ends_before_its_terminal_keeps_its_tail` |
//! | driver refusals and deferrals are facts, once | `driver_refusals_and_deferrals_are_witnessed` |
//! | cancellation in flight is a fact | `cancelling_in_flight_is_witnessed` |
//! | a host policy names itself | `a_host_policy_emits_typed_facts_under_its_own_name` |
//! | the witness changes nothing about the program | `a_witness_changes_nothing` |
//! | a full sink is incomplete, never silently equal | `a_full_sink_reports_incompleteness` |

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::panic,
    clippy::indexing_slicing,
    clippy::type_complexity
)]

mod bus_support;

use std::sync::Arc;

use bevy_ecs::prelude::*;
use bus_support::*;
use rig_core::{
    effect::{EffectFamily, EffectId, EffectKind, Outcome},
    error::{ErrorKind, ErrorReport},
    observe::{
        Action, Comparison, Emitter, HostAction, ObservationLog, ObservationTrace, OutcomeSummary,
        Stage, Subject, compare,
    },
    serve::{Decision, Dispatch, ErasedHandler, Intercept, Reply, Serve, ServingPolicy, Verdict},
};
use rig_ecs::bus::{
    BusSet, EffectLogResource, EffectOutcome, Handlers, Held, InFlight, Issued, PendingEffect,
    RigSchedule, Scope, Witnessing, WorldOutcome,
};
use rig_effect_log::EffectLogRecorder;

fn witnessed(app: &mut bevy_app::App) -> Arc<ObservationLog> {
    let log = Arc::new(ObservationLog::default());
    Witnessing::install(app.world_mut(), log.clone());
    log
}

fn name_of(action: &Action) -> &'static str {
    match action {
        Action::Held { .. } => "held",
        Action::Released => "released",
        Action::Denied { .. } => "denied",
        Action::Approved { .. } => "approved",
        Action::Patched { .. } => "patched",
        Action::Issued => "issued",
        Action::Deferred { .. } => "deferred",
        Action::Refused { .. } => "refused",
        Action::Landed { .. } => "landed",
        Action::StreamTruncated { .. } => "stream_truncated",
        Action::Replaced { .. } => "replaced",
        Action::Cancelled { .. } => "cancelled",
        Action::CancelRequested { .. } => "cancel_requested",
        Action::Retry { .. } => "retry",
        Action::InvalidCall { .. } => "invalid_call",
        Action::Ended { .. } => "ended",
        Action::Host { .. } => "host",
    }
}

fn actions(trace: &ObservationTrace) -> Vec<(Stage, String)> {
    trace
        .observations
        .iter()
        .map(|observation| (observation.stage, name_of(&observation.action).to_owned()))
        .collect()
}

fn deny_model_calls(
    fresh: Query<(Entity, &PendingEffect), (Without<Issued>, Without<EffectOutcome>)>,
    mut commands: Commands,
) {
    for (entity, effect) in &fresh {
        if effect.key.as_str() == "model" {
            commands
                .entity(entity)
                .insert(EffectOutcome(Err(ErrorReport::new(
                    ErrorKind::Denied,
                    "no models today",
                )
                .with_retryable(false))));
        }
    }
}

#[test]
fn a_gate_denial_is_witnessed_with_its_reason_and_leaves_no_record() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(app.world_mut(), recorder.clone());
    let log = witnessed(&mut app);
    register(&mut app, "model", MockModel::new(&counters));
    app.add_systems(RigSchedule, deny_model_calls.in_set(BusSet::Gate));
    let scope = app.world_mut().spawn(Scope("app/run#1".into())).id();
    let effect = app
        .world_mut()
        .spawn((PendingEffect::new("model", completion()), ChildOf(scope)))
        .id();
    tick_until(&mut app, "denied", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    tick(&mut app, 2);

    assert_eq!(
        counters
            .unary_started
            .load(std::sync::atomic::Ordering::SeqCst),
        0
    );
    assert!(recorder.log().records.is_empty(), "a denial is no record");
    let trace = log.trace();
    assert_eq!(actions(&trace), [(Stage::Gate, "denied".to_owned())]);
    let denial = &trace.observations[0];
    assert_eq!(denial.subject.scope.as_deref(), Some("app/run#1"));
    assert_eq!(denial.subject.order, Some(0));
    assert_eq!(denial.subject.effect, None, "never issued");
    assert_eq!(
        denial.subject.key.as_ref().map(|k| k.as_str()),
        Some("model")
    );
    assert_eq!(denial.subject.family, Some(EffectFamily::Completion));
    assert!(
        denial.emitter.is_unknown(),
        "a plain component write names no policy"
    );
    let Action::Denied { reason } = &denial.action else {
        panic!("{denial:?}");
    };
    assert_eq!(reason.code, "denied");
    assert_eq!(reason.detail.as_deref(), Some("no models today"));
}

#[test]
fn hold_release_and_hold_deny_are_distinct_sequences() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    let log = witnessed(&mut app);
    register(&mut app, "model", MockModel::new(&counters));
    let released = app
        .world_mut()
        .spawn((PendingEffect::new("model", completion()), Held))
        .id();
    let denied = app
        .world_mut()
        .spawn((PendingEffect::new("model", completion()), Held))
        .id();
    tick(&mut app, 2);
    assert!(app.world().get::<Issued>(released).is_none(), "held");
    app.world_mut().entity_mut(released).remove::<Held>();
    app.world_mut()
        .entity_mut(denied)
        .insert(EffectOutcome(Err(ErrorReport::new(
            ErrorKind::Denied,
            "reviewer said no",
        ))))
        .remove::<Held>();
    tick_until(&mut app, "answered", |world| {
        world.get::<EffectOutcome>(released).is_some()
    });
    tick(&mut app, 2);

    let trace = log.trace();
    let of = |entity: Entity| -> Vec<String> {
        let order = app.world().get::<rig_ecs::bus::Seq>(entity).unwrap().0;
        trace
            .observations
            .iter()
            .filter(|o| o.subject.order == Some(order))
            .map(|o| format!("{:?}:{}", o.stage, name_of(&o.action)))
            .collect()
    };
    assert_eq!(
        of(released),
        [
            "Gate:held",
            "Gate:released",
            "Dispatch:issued",
            "Collect:landed"
        ]
    );
    assert_eq!(of(denied), ["Gate:held", "Gate:denied"]);
}

fn replace_answers(
    mut answered: Query<(Entity, &EffectOutcome), (With<Issued>, Without<InFlight>)>,
    mut commands: Commands,
    mut done: Local<bool>,
) {
    if *done {
        return;
    }
    for (entity, outcome) in &mut answered {
        if outcome.0.is_ok() {
            commands
                .entity(entity)
                .insert(EffectOutcome(Err(ErrorReport::new(
                    ErrorKind::Internal,
                    "judged unfit",
                ))));
            *done = true;
        }
    }
}

#[test]
fn a_judge_replacement_exposes_both_answers() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(app.world_mut(), recorder.clone());
    let log = witnessed(&mut app);
    register(&mut app, "model", MockModel::new(&counters));
    app.add_systems(RigSchedule, replace_answers.in_set(BusSet::Judge));
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(&mut app, "judged", |world| {
        world
            .get::<EffectOutcome>(effect)
            .is_some_and(|outcome| outcome.0.is_err())
    });
    tick(&mut app, 2);

    assert!(
        recorder.log().records[0].outcome.is_ok(),
        "the record keeps the handler's answer"
    );
    let trace = log.trace();
    assert_eq!(
        actions(&trace),
        [
            (Stage::Dispatch, "issued".to_owned()),
            (Stage::Collect, "landed".to_owned()),
            (Stage::Judge, "replaced".to_owned()),
        ]
    );
    let Action::Replaced { recorded, consumed } = &trace.observations[2].action else {
        panic!("{:?}", trace.observations[2]);
    };
    assert_eq!(
        recorded,
        &OutcomeSummary::Ok {
            family: EffectFamily::Completion
        }
    );
    assert!(matches!(consumed, OutcomeSummary::Err { reason, .. } if reason.code == "internal"));
    assert_eq!(
        trace.observations[2].subject.effect,
        Some(EffectId::from_raw(0))
    );
}

struct Warmer;

impl Intercept for Warmer {
    fn name(&self) -> String {
        "warmer".into()
    }

    async fn before(&self, _: EffectId, kind: &EffectKind) -> Decision {
        let EffectKind::Completion { request, stream } = kind else {
            return Decision::Proceed;
        };
        let mut request = request.clone();
        request.temperature = Some(0.7);
        Decision::Patch(EffectKind::Completion {
            request,
            stream: *stream,
        })
    }

    async fn after(
        &self,
        _: EffectId,
        _: &EffectKind,
        _: &Result<Outcome, ErrorReport>,
    ) -> Verdict {
        Verdict::Keep
    }
}

/// A layer that withdraws every answer on its way out.
struct Withdrawer;

impl Intercept for Withdrawer {
    fn name(&self) -> String {
        "withdrawer".into()
    }

    async fn before(&self, _: EffectId, _: &EffectKind) -> Decision {
        Decision::Proceed
    }

    async fn after(
        &self,
        _: EffectId,
        _: &EffectKind,
        _: &Result<Outcome, ErrorReport>,
    ) -> Verdict {
        Verdict::Replace(Err(ErrorReport::new(ErrorKind::Denied, "withdrawn")))
    }
}

struct Bouncer;

impl Intercept for Bouncer {
    fn name(&self) -> String {
        "bouncer".into()
    }

    async fn before(&self, _: EffectId, _: &EffectKind) -> Decision {
        Decision::deny("not on the list")
    }

    async fn after(
        &self,
        _: EffectId,
        _: &EffectKind,
        _: &Result<Outcome, ErrorReport>,
    ) -> Verdict {
        Verdict::Keep
    }
}

#[test]
fn a_layer_patch_and_discard_are_witnessed_at_the_handler_side() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(app.world_mut(), recorder.clone());
    let log = witnessed(&mut app);
    register(
        &mut app,
        "warm",
        ErasedHandler::new(MockModel::new(&counters)).layered(Warmer),
    );
    register(
        &mut app,
        "bounced",
        ErasedHandler::new(MockModel::new(&counters)).layered(Bouncer),
    );
    register(
        &mut app,
        "withdrawn",
        ErasedHandler::new(MockModel::new(&counters)).layered(Withdrawer),
    );
    let warm = app
        .world_mut()
        .spawn(PendingEffect::new("warm", completion()))
        .id();
    let bounced = app
        .world_mut()
        .spawn(PendingEffect::new("bounced", completion()))
        .id();
    let withdrawn = app
        .world_mut()
        .spawn(PendingEffect::new("withdrawn", completion()))
        .id();
    tick_until(&mut app, "all answered", |world| {
        world.get::<EffectOutcome>(warm).is_some()
            && world.get::<EffectOutcome>(bounced).is_some()
            && world.get::<EffectOutcome>(withdrawn).is_some()
    });
    tick(&mut app, 2);

    let records = recorder.log();
    assert_eq!(
        records.records.len(),
        2,
        "the bounced dispatch is no record; the withdrawn one keeps the handler's answer"
    );
    let warm_record = records
        .records
        .iter()
        .find(|r| r.key.as_str() == "warm")
        .expect("the warm record");
    let EffectKind::Completion { request, .. } = &warm_record.kind else {
        panic!()
    };
    assert_eq!(
        request.temperature,
        Some(0.7),
        "the record holds the patched request"
    );

    let trace = log.trace();
    let patched = trace
        .observations
        .iter()
        .find(|o| matches!(o.action, Action::Patched { .. }))
        .expect("the patch is observed");
    assert_eq!(patched.stage, Stage::Handler);
    assert_eq!(
        patched.emitter.name, "warmer",
        "a layer's patch names the layer"
    );
    let Action::Patched { before, after } = &patched.action else {
        panic!("matched above")
    };
    assert_eq!(before["request"]["temperature"], serde_json::Value::Null);
    assert_eq!(after["request"]["temperature"], 0.7);
    let denied = trace
        .observations
        .iter()
        .find(
            |o| matches!(&o.action, Action::Denied { reason } if reason.code == "layer_discarded"),
        )
        .expect("the discard is observed");
    assert_eq!(denied.stage, Stage::Handler);
    assert_eq!(
        denied.emitter.name, "bouncer",
        "a layer's denial names the layer"
    );
    assert_eq!(
        denied.subject.key.as_ref().map(|k| k.as_str()),
        Some("bounced")
    );
    let replaced = trace
        .observations
        .iter()
        .find(|o| {
            matches!(o.action, Action::Replaced { .. })
                && o.subject
                    .key
                    .as_ref()
                    .is_some_and(|k| k.as_str() == "withdrawn")
        })
        .expect("the layer's replacement is observed");
    assert_eq!(replaced.stage, Stage::Handler);
    assert_eq!(
        replaced.emitter.name, "withdrawer",
        "a layer's replacement names the layer"
    );
    let Action::Replaced { recorded, consumed } = &replaced.action else {
        panic!("matched above")
    };
    assert!(
        matches!(recorded, OutcomeSummary::Ok { .. }),
        "{recorded:?}"
    );
    assert!(
        matches!(consumed, OutcomeSummary::Err { reason, .. } if reason.code == "denied"),
        "{consumed:?}"
    );
    assert!(
        recorder
            .log()
            .records
            .iter()
            .find(|r| r.key.as_str() == "withdrawn")
            .is_some_and(|r| r.outcome.is_ok()),
        "the record keeps the handler's answer"
    );
    assert!(
        !trace.observations.iter().any(|o| {
            o.subject
                .key
                .as_ref()
                .is_some_and(|k| k.as_str() == "bounced")
                && matches!(o.action, Action::Landed { .. })
        }),
        "a discarded dispatch never lands as a record"
    );
}

/// A stream that writes one delta and returns without its terminal record.
struct Truncating;

impl Serve for Truncating {
    type Family = rig_core::effect::family::Completion;

    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        rig_core::effect::HandlerDescriptor {
            key: rig_core::effect::HandlerKey::from("cut"),
            family: rig_core::effect::FamilyDescriptor::Completion {
                model: rig_core::completion::ModelRef::new("cut"),
                capabilities: rig_core::completion::ProviderCapabilities::default(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
        Reply::written(|mut out| async move {
            let _ = out.text("partial ").await;
            let _ = out.text("answer").await;
        })
    }
}

#[test]
fn a_stream_that_ends_before_its_terminal_keeps_its_tail() {
    let mut app = app();
    let log = witnessed(&mut app);
    register(&mut app, "cut", Truncating);
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new("cut", streaming()))
        .id();
    tick_until(&mut app, "truncated", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    tick(&mut app, 2);

    let outcome = app.world().get::<EffectOutcome>(effect).unwrap();
    assert!(matches!(&outcome.0, Err(report) if report.kind == ErrorKind::Response));
    let trace = log.trace();
    let truncated = trace
        .observations
        .iter()
        .find(|o| matches!(o.action, Action::StreamTruncated { .. }))
        .expect("the truncation is observed");
    assert_eq!(truncated.stage, Stage::Collect);
    let Action::StreamTruncated {
        delivered,
        tail,
        errors,
    } = &truncated.action
    else {
        panic!("matched above")
    };
    assert!(*delivered >= 2, "{delivered}");
    assert!(!tail.is_empty(), "the last frames are kept");
    assert!(errors.is_empty());
    let landed = trace
        .observations
        .iter()
        .find(|o| matches!(o.action, Action::Landed { .. }))
        .expect("the outcome lands after the truncation");
    assert!(
        matches!(&landed.action, Action::Landed { outcome: OutcomeSummary::Err { reason, .. } } if reason.code == "response")
    );
}

#[test]
fn driver_refusals_and_deferrals_are_witnessed() {
    let counters = Arc::new(Counters::default());
    let mut app = app_with(ServingPolicy {
        command_capacity: 1,
        ..ServingPolicy::default()
    });
    let log = witnessed(&mut app);
    register(&mut app, "model", MockModel::new(&counters));
    let nobody = app
        .world_mut()
        .spawn(PendingEffect::new("nobody", completion()))
        .id();
    let first = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    let second = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(&mut app, "all answered", |world| {
        [nobody, first, second]
            .iter()
            .all(|e| world.get::<EffectOutcome>(*e).is_some())
    });
    tick(&mut app, 2);

    let trace = log.trace();
    let refused = trace
        .observations
        .iter()
        .find(|o| matches!(o.action, Action::Refused { .. }))
        .expect("the unbound key is refused");
    assert_eq!(refused.stage, Stage::Dispatch);
    assert_eq!(refused.emitter.name, "rig-ecs/bus");
    assert!(
        matches!(&refused.action, Action::Refused { reason } if reason.code == "handler_unavailable")
    );
    assert!(
        !trace.observations.iter().any(|o| o
            .subject
            .key
            .as_ref()
            .is_some_and(|k| k.as_str() == "nobody")
            && matches!(o.action, Action::Denied { .. })),
        "the driver's refusal is not double-counted as a gate denial"
    );
    let deferred: Vec<_> = trace
        .observations
        .iter()
        .filter(|o| matches!(o.action, Action::Deferred { .. }))
        .collect();
    assert_eq!(deferred.len(), 1, "one deferral, once: {deferred:?}");
    assert!(
        matches!(&deferred[0].action, Action::Deferred { reason } if reason.code == "intake_bound")
    );
    assert_eq!(
        trace
            .observations
            .iter()
            .filter(|o| matches!(o.action, Action::Issued))
            .count(),
        2
    );
}

#[test]
fn cancelling_in_flight_is_witnessed() {
    let counters = Arc::new(Counters::default());
    counters.hold.hold();
    let mut app = app();
    let log = witnessed(&mut app);
    register(&mut app, "model", MockModel::new(&counters));
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(&mut app, "in flight", |world| {
        world.get::<InFlight>(effect).is_some()
    });
    app.world_mut().entity_mut(effect).despawn();
    tick(&mut app, 2);
    counters.hold.release();

    let trace = log.trace();
    assert_eq!(
        actions(&trace),
        [
            (Stage::Dispatch, "issued".to_owned()),
            (Stage::Collect, "cancelled".to_owned())
        ]
    );
    let Action::Cancelled { reason } = &trace.observations[1].action else {
        panic!("matched above")
    };
    assert_eq!(reason.code, "cancelled");
    assert_eq!(
        trace.observations[1].subject.effect,
        Some(EffectId::from_raw(0))
    );
}

#[derive(Debug, PartialEq, serde::Serialize, serde::Deserialize)]
struct Approval {
    operation: String,
    approved: bool,
}

impl HostAction for Approval {
    const KIND: &'static str = "app/approval";
}

#[test]
fn a_host_policy_emits_typed_facts_under_its_own_name() {
    let mut app = app();
    let log = witnessed(&mut app);
    let witness = app.world().resource::<Witnessing>().clone();
    witness.emit(
        Subject::scoped("app/run#1"),
        Stage::Host,
        Emitter::versioned("app/approvals", "3"),
        Approval {
            operation: "op-1".into(),
            approved: false,
        }
        .action()
        .unwrap(),
    );
    let trace = log.trace();
    assert_eq!(trace.observations.len(), 1);
    let fact = &trace.observations[0];
    assert_eq!(fact.emitter.name, "app/approvals");
    assert_eq!(fact.emitter.version.as_deref(), Some("3"));
    assert_eq!(
        Approval::from_action(&fact.action).unwrap().unwrap(),
        Approval {
            operation: "op-1".into(),
            approved: false
        }
    );
}

/// A completion key served by a system, within the schedule: the passes an
/// answer takes are the program's, not a task pool's timing.
fn answer_open(
    asked: Query<
        (Entity, &PendingEffect),
        (
            With<InFlight>,
            Without<WorldOutcome>,
            Without<EffectOutcome>,
        ),
    >,
    mut commands: Commands,
) {
    for (entity, effect) in &asked {
        if effect.key.as_str() == "open" {
            commands
                .entity(entity)
                .insert(WorldOutcome::new(Ok(Outcome::Completion(
                    rig_core::completion::CompletionResponse::new(
                        vec![rig_core::message::AssistantContent::text(
                            "served by a system",
                        )],
                        rig_core::completion::Usage::new(),
                        "open",
                    ),
                ))));
        }
    }
}

fn program(
    with_witness: bool,
) -> (
    rig_effect_log::EffectLog,
    Vec<String>,
    Option<ObservationTrace>,
) {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(app.world_mut(), recorder.clone());
    let log = with_witness.then(|| witnessed(&mut app));
    register(&mut app, "model", MockModel::new(&counters));
    app.add_systems(RigSchedule, deny_model_calls.in_set(BusSet::Gate));
    Handlers::with(app.world_mut(), |handlers| {
        handlers.register_open(
            "open",
            rig_core::effect::FamilyDescriptor::Completion {
                model: rig_core::completion::ModelRef::new("open"),
                capabilities: rig_core::completion::ProviderCapabilities::default(),
            },
        )
    })
    .expect("a bus")
    .expect("a fresh key");
    app.add_systems(RigSchedule, answer_open.after(BusSet::Judge));
    let denied = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    let served = app
        .world_mut()
        .spawn(PendingEffect::new("open", completion()))
        .id();
    let again = app
        .world_mut()
        .spawn(PendingEffect::new("open", completion()))
        .id();
    tick_until(&mut app, "all answered", |world| {
        [denied, served, again]
            .iter()
            .all(|e| world.get::<EffectOutcome>(*e).is_some())
    });
    tick(&mut app, 2);
    let outcomes = [denied, served, again]
        .iter()
        .map(|e| {
            format!(
                "{:?}",
                OutcomeSummary::of(&app.world().get::<EffectOutcome>(*e).unwrap().0)
            )
        })
        .collect();
    (recorder.log(), outcomes, log.map(|log| log.trace()))
}

#[test]
fn a_witness_changes_nothing() {
    let (plain_log, plain_outcomes, none) = program(false);
    let (witnessed_log, witnessed_outcomes, trace) = program(true);
    assert!(none.is_none());
    assert_eq!(plain_outcomes, witnessed_outcomes);
    assert_eq!(
        serde_json::to_value(&plain_log).unwrap(),
        serde_json::to_value(&witnessed_log).unwrap(),
        "the exchange record, delivery batches included, is identical with and without a witness"
    );
    assert!(
        plain_log
            .header
            .deliveries
            .as_ref()
            .is_some_and(|d| !d.is_empty())
    );
    let trace = trace.unwrap();
    assert!(trace.is_complete());
    assert!(trace.observations.len() >= 5, "{:?}", actions(&trace));
}

#[test]
fn a_full_sink_reports_incompleteness() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    let log = Arc::new(ObservationLog::with_capacity(1));
    Witnessing::install(app.world_mut(), log.clone());
    register(&mut app, "model", MockModel::new(&counters));
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    tick_until(&mut app, "answered", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    tick(&mut app, 2);
    let trace = log.trace();
    assert_eq!(trace.observations.len(), 1);
    assert!(trace.dropped >= 1);
    assert!(!trace.is_complete());
    let complete = ObservationLog::default().trace();
    assert!(matches!(
        compare(&complete, &trace),
        Comparison::Incomparable { .. }
    ));
    assert_eq!(
        trace.observations[0]
            .subject
            .key
            .as_ref()
            .map(|k| k.as_str()),
        Some("model")
    );
}

#[test]
fn despawning_a_held_intent_is_a_cancellation_not_a_release() {
    let counters = Arc::new(Counters::default());
    let mut app = app();
    let log = witnessed(&mut app);
    register(&mut app, "model", MockModel::new(&counters));
    let held = app
        .world_mut()
        .spawn((PendingEffect::new("model", completion()), Held))
        .id();
    tick(&mut app, 2);
    app.world_mut().entity_mut(held).despawn();
    tick(&mut app, 2);
    let trace = log.trace();
    assert_eq!(
        actions(&trace),
        [
            (Stage::Gate, "held".to_owned()),
            (Stage::Dispatch, "cancelled".to_owned())
        ],
        "{trace:?}"
    );
    let Action::Cancelled { reason } = &trace.observations[1].action else {
        panic!("matched above")
    };
    assert_eq!(reason.code, "despawned_before_dispatch");
    // The despawn bookkeeping does not outlive the despawn.
    assert!(
        format!(
            "{:?}",
            app.world().resource::<rig_ecs::bus::witness::Despawning>()
        )
        .contains("Despawning({})")
    );
}
