//! The witness: decisions observed where they happen, beside the record.
//!
//! | contract | test |
//! |---|---|
//! | a gate denial is a fact with its reason and no exchange record | `a_gate_denial_is_witnessed_with_its_reason_and_leaves_no_record` |
//! | hold/release and hold/deny stay distinct sequences | `hold_release_and_hold_deny_are_distinct_sequences` |
//! | named owners release independently with observation enabled or disabled | `named_owners_release_independently_without_changing_dispatch` |
//! | denial and despawn do not emit ordinary owner releases | `named_holds_do_not_emit_releases_for_denial_or_despawn` |
//! | scene restore retains a barrier with explicitly unknown ownership | `restored_hold_is_unknown_until_the_host_reevaluates_it` |
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

struct HandlerClock(std::sync::atomic::AtomicU64);

impl rig_core::observe::Clock for HandlerClock {
    fn elapsed(&self) -> std::time::Duration {
        std::time::Duration::from_millis(self.0.load(std::sync::atomic::Ordering::SeqCst))
    }
}

#[test]
fn handler_intervals_pin_landing_first_item_and_cancellation() {
    use rig_core::observe::{HandlerInterval, HandlerTiming};
    for (streamed, first, cancel) in [
        (false, false, false),
        (false, false, true),
        (true, false, true),
        (true, true, true),
    ] {
        let counters = Arc::new(Counters::default());
        counters.hold.hold();
        let mut app = app();
        let clock = Arc::new(HandlerClock(std::sync::atomic::AtomicU64::new(10)));
        let log = Arc::new(ObservationLog::default().with_clock(clock.clone()));
        Witnessing::install(app.world_mut(), log.clone());
        register(&mut app, "model", MockModel::endless(&counters));
        let effect = app
            .world_mut()
            .spawn(PendingEffect::new(
                "model",
                if streamed { streaming() } else { completion() },
            ))
            .id();
        tick_until(&mut app, "issued", |world| {
            world.get::<InFlight>(effect).is_some()
        });
        if first {
            clock.0.store(23, std::sync::atomic::Ordering::SeqCst);
            counters.hold.release();
            tick_until(&mut app, "first delivered item", |world| {
                world
                    .get::<rig_ecs::bus::Streamed>(effect)
                    .is_some_and(|stream| !stream.events.is_empty())
            });
            counters.hold.hold();
        }
        clock.0.store(60, std::sync::atomic::Ordering::SeqCst);
        if cancel {
            app.world_mut().despawn(effect);
        } else {
            counters.hold.release();
            tick_until(&mut app, "landed", |world| {
                world.get::<EffectOutcome>(effect).is_some()
            });
        }
        counters.hold.release();
        let trace = log.trace();
        let timings: Vec<_> = trace
            .observations
            .iter()
            .filter_map(|o| o.handler_timing.as_ref())
            .collect();
        assert_eq!(timings.len(), 1);
        assert_eq!(
            *timings[0],
            HandlerTiming {
                interval: if streamed {
                    HandlerInterval::TimeToFirstItem
                } else {
                    HandlerInterval::Execution
                },
                duration: if first {
                    Some(std::time::Duration::from_millis(13))
                } else if !cancel {
                    Some(std::time::Duration::from_millis(50))
                } else {
                    None
                },
                complete: first || !cancel,
            }
        );
        let mut untimed = trace.clone();
        for observation in &mut untimed.observations {
            observation.handler_timing = None;
        }
        assert_eq!(compare(&trace, &untimed), Comparison::Equal);
    }
}

struct EmptyStream {
    error: bool,
}

impl Serve for EmptyStream {
    type Family = rig_core::effect::family::Completion;
    fn descriptor(&self) -> rig_core::effect::HandlerDescriptor {
        Truncating.descriptor()
    }
    async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
        let error = self.error;
        Reply::written(move |mut out| async move {
            if error {
                let _ = out
                    .error(ErrorReport::new(
                        ErrorKind::Response,
                        "first item is an error",
                    ))
                    .await;
            }
        })
    }
}

#[test]
fn empty_and_error_first_streams_distinguish_no_item_from_zero_duration() {
    for error in [false, true] {
        let mut baseline = None;
        for timed in [false, true] {
            let mut app = app();
            let log = Arc::new(if timed {
                ObservationLog::default().with_clock(Arc::new(HandlerClock(
                    std::sync::atomic::AtomicU64::new(10),
                )))
            } else {
                ObservationLog::default()
            });
            Witnessing::install(app.world_mut(), log.clone());
            register(&mut app, "cut", EmptyStream { error });
            let effect = app
                .world_mut()
                .spawn(PendingEffect::new("cut", streaming()))
                .id();
            tick_until(&mut app, "empty stream closure", |world| {
                world.get::<EffectOutcome>(effect).is_some()
            });
            let trace = log.trace();
            let landed = trace
                .observations
                .iter()
                .find(|o| matches!(o.action, Action::Landed { .. }))
                .unwrap();
            assert!(matches!(
                landed.action,
                Action::Landed {
                    outcome: OutcomeSummary::Err { .. }
                }
            ));
            if timed {
                assert_eq!(
                    landed.handler_timing,
                    Some(rig_core::observe::HandlerTiming {
                        interval: rig_core::observe::HandlerInterval::TimeToFirstItem,
                        duration: error.then_some(std::time::Duration::ZERO),
                        complete: error,
                    })
                );
                assert_eq!(
                    compare(baseline.as_ref().unwrap(), &trace),
                    Comparison::Equal
                );
            } else {
                assert!(landed.handler_timing.is_none());
                baseline = Some(trace);
            }
        }
    }
}

#[test]
fn explicit_operations_keep_retry_identity_and_current_dispatch_subjects() {
    use rig_core::{
        client::CompletionClient,
        completion::CompletionModel as _,
        observe::{AdapterContext, AdapterEvent},
        serve::adapters::CompletionAdapter,
        test_utils::RecordingHttpClient,
    };
    let mut app = app();
    let log = witnessed(&mut app);
    let http = RecordingHttpClient::new(
        r#"{"candidates":[{"content":{"parts":[{"text":"pong"}],"role":"model"},"finishReason":"STOP"}]}"#,
    );
    let client = rig_core::providers::gemini::Client::builder()
        .api_key("test-key")
        .http_client(http.clone())
        .build()
        .unwrap();
    let model = client.completion_model("test-model");
    let request = model.completion_request("identical call").build();
    register(
        &mut app,
        "model",
        CompletionAdapter::new("test-model", model),
    );
    let operation = AdapterContext::new(log.clone(), Subject::default(), "logical-call");
    let mut entities = Vec::new();
    for host in [1u64, 2] {
        let entity = app
            .world_mut()
            .spawn((
                PendingEffect::new(
                    "model",
                    EffectKind::Completion {
                        request: request.clone(),
                        stream: false,
                    },
                ),
                Scope(format!("host/{host}")),
                rig_ecs::bus::AdapterOperation {
                    context: operation.clone(),
                    host_attempt: host.try_into().unwrap(),
                },
            ))
            .id();
        entities.push(entity);
        tick_until(&mut app, "provider attempt completed", |world| {
            world.get::<EffectOutcome>(entity).is_some()
        });
    }
    assert_eq!(http.requests().len(), 2);
    assert_eq!(http.requests()[0], http.requests()[1]);
    let trace = log.trace();
    for (index, entity) in entities.into_iter().enumerate() {
        let id = app.world().get::<Issued>(entity).unwrap().0;
        let facts: Vec<_> = trace
            .observations
            .iter()
            .filter_map(|o| {
                let Action::Adapter { observation } = &o.action else {
                    return None;
                };
                if o.subject.effect != Some(id) {
                    return None;
                }
                assert_eq!(
                    o.subject.scope.as_deref(),
                    Some(format!("host/{}", index + 1).as_str())
                );
                assert_eq!(observation.operation, "logical-call");
                assert_eq!(observation.attempt, Some((index + 1) as u64));
                assert_eq!(
                    observation.host_attempt.map(std::num::NonZeroU64::get),
                    Some((index + 1) as u64)
                );
                Some(observation)
            })
            .collect();
        assert_eq!(facts.len(), 4);
        assert!(matches!(facts[0].event, AdapterEvent::Started { .. }));
        assert!(matches!(facts[3].event, AdapterEvent::Finished { .. }));
    }
    let parallel: Vec<_> = ["duplicate/1", "duplicate/2"]
        .into_iter()
        .map(|name| {
            let entity = app
                .world_mut()
                .spawn((
                    PendingEffect::new(
                        "model",
                        EffectKind::Completion {
                            request: request.clone(),
                            stream: false,
                        },
                    ),
                    Scope("parallel".into()),
                    rig_ecs::bus::AdapterOperation {
                        context: AdapterContext::new(log.clone(), Subject::default(), name),
                        host_attempt: 1.try_into().unwrap(),
                    },
                ))
                .id();
            (entity, name)
        })
        .collect();
    tick_until(&mut app, "parallel identical requests completed", |world| {
        parallel
            .iter()
            .all(|(entity, _)| world.get::<EffectOutcome>(*entity).is_some())
    });
    assert_eq!(http.requests().len(), 4);
    assert!(http.requests().iter().all(|r| r == &http.requests()[0]));
    for (entity, name) in parallel {
        let id = app.world().get::<Issued>(entity).unwrap().0;
        let trace = log.trace();
        let facts: Vec<_> = trace
            .observations
            .iter()
            .filter_map(|o| {
                let Action::Adapter { observation } = &o.action else {
                    return None;
                };
                (o.subject.effect == Some(id)).then_some(observation)
            })
            .collect();
        assert_eq!(facts.len(), 4);
        assert!(facts.iter().all(|f| f.operation == name
            && f.attempt == Some(1)
            && f.host_attempt == Some(1.try_into().unwrap())));
    }
}

fn name_of(action: &Action) -> &'static str {
    match action {
        Action::Adapter { .. } => "adapter",
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

#[test]
fn hold_lifecycle_despawn_does_not_emit_uncorrelated_transitions() {
    let mut app = app();
    let log = witnessed(&mut app);
    let entity = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    app.world_mut()
        .add_observer(|event: On<Add, Held>, mut commands: Commands| {
            commands.entity(event.entity).despawn();
        });
    assert!(rig_ecs::bus::acquire_hold(
        app.world_mut(),
        entity,
        Emitter::named("policy/a")
    ));
    assert!(app.world().get_entity(entity).is_err());
    assert!(
        !log.trace()
            .observations
            .iter()
            .any(|fact| matches!(fact.action, Action::Held { .. } | Action::Released))
    );
}

#[test]
fn ownership_component_observers_preserve_fact_order() {
    use rig_ecs::bus::{HoldOwners, acquire_hold, release_hold};
    let mut app = app();
    let log = witnessed(&mut app);
    let entity = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    app.world_mut()
        .add_observer(|event: On<Add, HoldOwners>, mut commands: Commands| {
            let entity = event.entity;
            commands.queue(move |world: &mut World| {
                release_hold(world, entity, "policy/a");
            });
        });
    assert!(acquire_hold(
        app.world_mut(),
        entity,
        Emitter::named("policy/a")
    ));
    assert!(app.world().get::<Held>(entity).is_none());
    assert!(app.world().get::<HoldOwners>(entity).is_none());
    let trace = log.trace();
    let transitions: Vec<_> = trace
        .observations
        .iter()
        .filter_map(|fact| match fact.action {
            Action::Held { .. } => Some(true),
            Action::Released => Some(false),
            _ => None,
        })
        .collect();
    assert_eq!(transitions, [true, false]);
}

#[test]
fn hold_lifecycle_observers_preserve_owners_and_fact_order() {
    use rig_ecs::bus::{HoldOwners, acquire_hold, release_hold};
    let mut app = app();
    let log = witnessed(&mut app);
    let entity = app
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    app.world_mut()
        .add_observer(|event: On<Add, Held>, mut commands: Commands| {
            let entity = event.entity;
            commands.queue(move |world: &mut World| {
                release_hold(world, entity, "policy/a");
            });
        });
    app.world_mut()
        .add_observer(|event: On<Remove, Held>, mut commands: Commands| {
            let entity = event.entity;
            commands.queue(move |world: &mut World| {
                acquire_hold(world, entity, Emitter::named("policy/b"));
            });
        });
    assert!(acquire_hold(
        app.world_mut(),
        entity,
        Emitter::named("policy/a")
    ));
    assert!(app.world().get::<Held>(entity).is_some());
    let owners = app.world().get::<HoldOwners>(entity).unwrap();
    assert_eq!(
        owners
            .owners()
            .map(|owner| owner.name.as_str())
            .collect::<Vec<_>>(),
        ["policy/b"]
    );
    let trace = log.trace();
    let transitions: Vec<_> = trace
        .observations
        .iter()
        .filter_map(|fact| match fact.action {
            Action::Held { .. } => Some((fact.emitter.name.as_str(), true)),
            Action::Released => Some((fact.emitter.name.as_str(), false)),
            _ => None,
        })
        .collect();
    assert_eq!(
        transitions,
        [("policy/a", true), ("policy/a", false), ("policy/b", true)]
    );
    assert!(release_hold(app.world_mut(), entity, "policy/b"));
    assert!(app.world().get::<Held>(entity).is_some());
    assert_eq!(
        app.world()
            .get::<HoldOwners>(entity)
            .unwrap()
            .owners()
            .count(),
        1
    );
    let trace = log.trace();
    let transitions: Vec<_> = trace
        .observations
        .iter()
        .filter_map(|fact| match fact.action {
            Action::Held { .. } => Some((fact.emitter.name.as_str(), true)),
            Action::Released => Some((fact.emitter.name.as_str(), false)),
            _ => None,
        })
        .collect();
    assert_eq!(
        transitions,
        [
            ("policy/a", true),
            ("policy/a", false),
            ("policy/b", true),
            ("policy/b", false),
            ("policy/b", true),
        ]
    );
}

#[test]
fn restored_hold_is_unknown_until_the_host_reevaluates_it() {
    use rig_ecs::bus::{HoldOwners, acquire_hold, release_hold, scene::Scene};
    let mut original = app();
    let effect = original
        .world_mut()
        .spawn(PendingEffect::new("model", completion()))
        .id();
    acquire_hold(
        original.world_mut(),
        effect,
        Emitter::named("policy/original"),
    );
    let scene = Scene::save(original.world_mut());
    let mut restored = app();
    let log = witnessed(&mut restored);
    let counters = Arc::new(Counters::default());
    register(&mut restored, "model", MockModel::new(&counters));
    let effect = scene.load(restored.world_mut()).unwrap()[0];
    tick(&mut restored, 2);
    assert!(restored.world().get::<Held>(effect).is_some());
    assert!(restored.world().get::<HoldOwners>(effect).is_none());
    assert!(restored.world().get::<Issued>(effect).is_none());
    assert!(!release_hold(
        restored.world_mut(),
        effect,
        "policy/original"
    ));
    assert!(acquire_hold(
        restored.world_mut(),
        effect,
        Emitter::named("policy/current")
    ));
    assert!(release_hold(
        restored.world_mut(),
        effect,
        &Emitter::unknown().name
    ));
    tick(&mut restored, 2);
    assert!(restored.world().get::<Held>(effect).is_some());
    assert!(restored.world().get::<Issued>(effect).is_none());
    assert!(release_hold(restored.world_mut(), effect, "policy/current"));
    tick_until(&mut restored, "restored effect answered", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    let trace = log.trace();
    let owners: Vec<_> = trace
        .observations
        .iter()
        .filter_map(|fact| {
            matches!(fact.action, Action::Held { .. }).then_some(fact.emitter.name.as_str())
        })
        .collect();
    assert_eq!(owners, [Emitter::unknown().name.as_str(), "policy/current"]);
}

#[test]
fn named_owners_release_independently_without_changing_dispatch() {
    use rig_ecs::bus::{HoldOwners, acquire_hold, release_hold};
    for enabled in [false, true] {
        for first in ["policy/a", "policy/b"] {
            let counters = Arc::new(Counters::default());
            let mut app = app();
            let log = enabled.then(|| witnessed(&mut app));
            register(&mut app, "model", MockModel::new(&counters));
            let entity = app
                .world_mut()
                .spawn(PendingEffect::new("model", completion()))
                .id();
            for owner in ["policy/a", "policy/b"] {
                assert!(acquire_hold(app.world_mut(), entity, Emitter::named(owner)));
                assert!(!acquire_hold(
                    app.world_mut(),
                    entity,
                    Emitter::named(owner)
                ));
            }
            tick(&mut app, 2);
            assert!(app.world().get::<Issued>(entity).is_none());
            assert!(release_hold(app.world_mut(), entity, first));
            assert!(!release_hold(app.world_mut(), entity, first));
            tick(&mut app, 2);
            assert!(app.world().get::<Held>(entity).is_some());
            assert!(app.world().get::<Issued>(entity).is_none());
            assert_eq!(
                app.world()
                    .get::<HoldOwners>(entity)
                    .unwrap()
                    .owners()
                    .count(),
                1
            );
            let last = if first == "policy/a" {
                "policy/b"
            } else {
                "policy/a"
            };
            assert!(release_hold(app.world_mut(), entity, last));
            assert!(app.world().get::<Held>(entity).is_none());
            assert!(app.world().get::<HoldOwners>(entity).is_none());
            tick_until(&mut app, "answered", |world| {
                world.get::<EffectOutcome>(entity).is_some()
            });
            assert!(app.world().get::<EffectOutcome>(entity).unwrap().0.is_ok());
            if let Some(log) = log {
                let trace = log.trace();
                let owners: Vec<_> = trace
                    .observations
                    .iter()
                    .filter_map(|fact| match fact.action {
                        Action::Held { .. } => Some((fact.emitter.name.as_str(), "held")),
                        Action::Released => Some((fact.emitter.name.as_str(), "released")),
                        _ => None,
                    })
                    .collect();
                assert_eq!(
                    owners,
                    [
                        ("policy/a", "held"),
                        ("policy/b", "held"),
                        (first, "released"),
                        (last, "released")
                    ]
                );
            }
        }
    }
}

#[test]
fn named_holds_do_not_emit_releases_for_denial_or_despawn() {
    use rig_ecs::bus::{acquire_hold, release_hold};
    let mut app = app();
    let log = witnessed(&mut app);
    for despawn in [false, true] {
        let entity = app
            .world_mut()
            .spawn(PendingEffect::new("model", completion()))
            .id();
        for owner in ["policy/a", "policy/b"] {
            acquire_hold(app.world_mut(), entity, Emitter::named(owner));
        }
        if despawn {
            app.world_mut().despawn(entity);
        } else {
            app.world_mut()
                .entity_mut(entity)
                .insert(EffectOutcome(Err(ErrorReport::new(
                    ErrorKind::Denied,
                    "denied",
                ))));
            release_hold(app.world_mut(), entity, "policy/a");
            release_hold(app.world_mut(), entity, "policy/b");
        }
    }
    assert!(
        !log.trace()
            .observations
            .iter()
            .any(|fact| matches!(fact.action, Action::Released))
    );
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
    let log = Arc::new(ObservationLog::default().with_clock(Arc::new(HandlerClock(
        std::sync::atomic::AtomicU64::new(10),
    ))));
    Witnessing::install(app.world_mut(), log.clone());
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
    assert!(
        denied.handler_timing.is_none(),
        "a layer-discarded call has no handler landing interval"
    );
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
