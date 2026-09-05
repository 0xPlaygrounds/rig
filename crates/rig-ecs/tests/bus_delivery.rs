//! Replay preserves outcome visibility and stream collection batches.

#![allow(
    clippy::expect_used,
    clippy::unwrap_used,
    clippy::indexing_slicing,
    clippy::panic
)]

mod bus_support;

use bevy_ecs::prelude::*;
use futures::channel::oneshot;
use rig_core::{
    completion::{ModelRef, ProviderCapabilities, Usage},
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    serve::{OutcomeSink, Serve},
    streaming::StreamFinal,
};
use rig_ecs::bus::{
    BusSet, EffectLogResource, EffectOutcome, Handlers, InFlight, PendingEffect, Replay,
    RigSchedule, Streamed,
};
use rig_effect_log::{EffectLog, EffectLogRecorder};
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicUsize, Ordering},
    },
    time::Instant,
};

#[derive(Resource, Default)]
struct First(Option<String>);

fn first_answer(
    event: On<Add, EffectOutcome>,
    effects: Query<&PendingEffect>,
    mut first: ResMut<First>,
) {
    if first.0.is_none() {
        first.0 = effects
            .get(event.event().entity)
            .ok()
            .map(|effect| effect.key.to_string());
    }
}

fn ordered_live(reverse: bool) -> (EffectLog, String) {
    ordered_live_with_error(reverse, false)
}

fn ordered_live_with_error(reverse: bool, recorded_divergence: bool) -> (EffectLog, String) {
    let mut app = bus_support::app();
    app.init_resource::<First>().add_observer(first_answer);
    for key in ["a", "b"] {
        Handlers::with(app.world_mut(), |handlers| {
            handlers.register_open(
                key,
                FamilyDescriptor::Custom {
                    kind: "ordering".into(),
                },
            )
        })
        .unwrap()
        .unwrap();
    }
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(app.world_mut(), recorder.clone());
    let entities: Vec<_> = ["a", "b"]
        .into_iter()
        .map(|key| {
            app.world_mut()
                .spawn(PendingEffect::new(
                    key,
                    EffectKind::Custom {
                        kind: "ordering".into(),
                        payload: serde_json::json!({}),
                    },
                ))
                .id()
        })
        .collect();
    app.update();
    assert!(
        entities
            .iter()
            .all(|entity| app.world().get::<InFlight>(*entity).is_some())
    );
    for index in if reverse { [1, 0] } else { [0, 1] } {
        app.world_mut()
            .entity_mut(entities[index])
            .insert(rig_ecs::bus::WorldOutcome::new(
                if recorded_divergence && index == 1 {
                    Err(rig_core::error::ErrorReport::new(
                        rig_core::error::ErrorKind::Divergence,
                        "nested replay failed",
                    ))
                } else {
                    Ok(Outcome::Custom {
                        payload: serde_json::json!({"value": "ok"}),
                    })
                },
            ));
        app.update();
    }
    (
        serde_json::from_str(&serde_json::to_string(&recorder.log()).unwrap()).unwrap(),
        app.world().resource::<First>().0.clone().unwrap(),
    )
}

#[test]
fn opposite_answer_orders_have_distinct_logs_and_replay_their_own_winner() {
    let (forward, first_forward) = ordered_live(false);
    let (reverse, first_reverse) = ordered_live(true);
    assert_ne!(
        serde_json::to_value(&forward).unwrap(),
        serde_json::to_value(&reverse).unwrap()
    );
    assert_ne!(first_forward, first_reverse);
    for (log, expected) in [(forward, first_forward), (reverse, first_reverse)] {
        let mut app = bus_support::app();
        app.init_resource::<First>().add_observer(first_answer);
        Handlers::with(app.world_mut(), |handlers| {
            Replay::policy_visible().register(handlers, &log)
        })
        .unwrap()
        .unwrap();
        let loaded = Replay::load(app.world_mut(), &log);
        bus_support::tick_until(&mut app, "both replayed", |world| {
            loaded
                .iter()
                .all(|entity| world.get::<EffectOutcome>(*entity).is_some())
        });
        assert_eq!(app.world().resource::<First>().0.as_ref(), Some(&expected));
    }
}

#[test]
fn a_recorded_divergence_is_delivered_without_overwriting_other_results() {
    for reverse in [false, true] {
        let (log, first) = ordered_live_with_error(reverse, true);
        let mut replay = bus_support::app();
        replay.init_resource::<First>().add_observer(first_answer);
        Handlers::with(replay.world_mut(), |handlers| {
            Replay::policy_visible().register(handlers, &log)
        })
        .unwrap()
        .unwrap();
        let loaded = Replay::load(replay.world_mut(), &log);
        bus_support::tick_until(
            &mut replay,
            "recorded success and divergence replayed",
            |world| {
                loaded
                    .iter()
                    .all(|entity| world.get::<EffectOutcome>(*entity).is_some())
            },
        );
        assert!(
            replay
                .world()
                .get_resource::<rig_ecs::bus::ReplayFailure>()
                .is_none()
        );
        assert_eq!(replay.world().resource::<First>().0.as_ref(), Some(&first));
        for (entity, record) in loaded.into_iter().zip(&log.records) {
            assert_eq!(
                serde_json::to_value(&replay.world().get::<EffectOutcome>(entity).unwrap().0)
                    .unwrap(),
                serde_json::to_value(&record.outcome).unwrap()
            );
        }
    }
}

#[test]
fn bounded_intake_preserves_deliveries_for_reserved_and_new_effects() {
    let (log, _) = ordered_live(false);
    for reserved in [false, true] {
        let mut app = bus_support::app();
        app.world_mut()
            .resource_mut::<rig_ecs::bus::Policy>()
            .0
            .command_capacity = 1;
        Handlers::with(app.world_mut(), |handlers| {
            Replay::policy_visible().register(handlers, &log)
        })
        .unwrap()
        .unwrap();
        let effects: Vec<_> = if reserved {
            Replay::load(app.world_mut(), &log)
        } else {
            log.records
                .iter()
                .map(|record| {
                    app.world_mut()
                        .spawn(PendingEffect::new(record.key.clone(), record.kind.clone()))
                        .id()
                })
                .collect()
        };
        bus_support::tick_until(
            &mut app,
            "queued effect receives its own delivery",
            |world| {
                effects
                    .iter()
                    .all(|entity| world.get::<EffectOutcome>(*entity).is_some())
            },
        );
        assert!(
            !app.world()
                .contains_resource::<rig_ecs::bus::ReplayFailure>()
        );
        assert!(
            effects.iter().all(|entity| app
                .world()
                .get::<EffectOutcome>(*entity)
                .unwrap()
                .0
                .is_ok())
        );
    }
}

#[test]
fn serial_serving_replays_two_streams_on_the_same_key() {
    let mut live = bus_support::serial_app();
    let recorder = EffectLogRecorder::keeping_stream_events();
    EffectLogResource::install(live.world_mut(), recorder.clone());
    let counters = Arc::new(bus_support::Counters::default());
    bus_support::register(&mut live, "model", bus_support::MockModel::new(&counters));
    let effects: Vec<_> = (0..2)
        .map(|_| {
            live.world_mut()
                .spawn(PendingEffect::new("model", bus_support::streaming()))
                .id()
        })
        .collect();
    bus_support::tick_until(&mut live, "both serial streams collected", |world| {
        effects
            .iter()
            .all(|entity| world.get::<EffectOutcome>(*entity).is_some())
    });
    let log: EffectLog =
        serde_json::from_str(&serde_json::to_string(&recorder.log()).unwrap()).unwrap();
    let mut replay = bus_support::serial_app();
    Handlers::with(replay.world_mut(), |handlers| {
        Replay::policy_visible().register(handlers, &log)
    })
    .unwrap()
    .unwrap();
    let loaded: Vec<_> = (0..2)
        .map(|_| {
            replay
                .world_mut()
                .spawn(PendingEffect::new("model", bus_support::streaming()))
                .id()
        })
        .collect();
    bus_support::tick_until(&mut replay, "both serial streams replayed", |world| {
        loaded
            .iter()
            .all(|entity| world.get::<EffectOutcome>(*entity).is_some())
    });
    assert!(
        !replay
            .world()
            .contains_resource::<rig_ecs::bus::ReplayFailure>()
    );
    for (before, after) in effects.into_iter().zip(loaded) {
        assert_eq!(
            serde_json::to_value(live.world().get::<Streamed>(before).unwrap()).unwrap(),
            serde_json::to_value(replay.world().get::<Streamed>(after).unwrap()).unwrap(),
        );
        assert!(
            replay
                .world()
                .get::<EffectOutcome>(after)
                .unwrap()
                .0
                .is_ok()
        );
    }
    let mut impossible = log;
    for delivery in impossible.header.deliveries.as_mut().unwrap() {
        delivery.batch = 1;
    }
    let mut replay = bus_support::serial_app();
    Handlers::with(replay.world_mut(), |handlers| {
        Replay::policy_visible().register(handlers, &impossible)
    })
    .unwrap()
    .unwrap();
    Replay::load(replay.world_mut(), &impossible);
    bus_support::tick_until(
        &mut replay,
        "impossible serial batch is diagnosed",
        |world| world.contains_resource::<rig_ecs::bus::ReplayFailure>(),
    );
    assert!(
        replay
            .world()
            .resource::<rig_ecs::bus::ReplayFailure>()
            .0
            .message
            .contains("serial key")
    );
}

#[derive(Resource, Default)]
struct Visible(Vec<Vec<String>>);

fn observe_visible(effects: Query<&PendingEffect, With<EffectOutcome>>, mut seen: ResMut<Visible>) {
    let mut keys: Vec<_> = effects
        .iter()
        .map(|effect| effect.key.to_string())
        .collect();
    keys.sort();
    if !keys.is_empty() && seen.0.last() != Some(&keys) {
        seen.0.push(keys);
    }
}

#[allow(clippy::type_complexity)]
fn world_answer<const LATE: bool>(
    effects: Query<
        (Entity, &PendingEffect),
        (
            With<InFlight>,
            Without<EffectOutcome>,
            Without<rig_ecs::bus::WorldOutcome>,
        ),
    >,
    mut commands: Commands,
) {
    let key = if LATE { "b" } else { "a" };
    for (entity, effect) in &effects {
        if effect.key.as_str() == key {
            commands
                .entity(entity)
                .insert(rig_ecs::bus::WorldOutcome::new(Ok(Outcome::Custom {
                    payload: serde_json::json!(key),
                })));
        }
    }
}

#[test]
fn world_answers_on_either_side_of_collect_preserve_policy_observations() {
    let mut live = bus_support::app();
    live.init_resource::<Visible>();
    live.world_mut().resource_mut::<Schedules>().add_systems(
        RigSchedule,
        (
            world_answer::<false>
                .after(BusSet::Dispatch)
                .before(BusSet::Collect),
            observe_visible.after(BusSet::Collect).before(BusSet::Judge),
            world_answer::<true>.after(BusSet::Judge),
        ),
    );
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(live.world_mut(), recorder.clone());
    for key in ["a", "b"] {
        Handlers::with(live.world_mut(), |handlers| {
            handlers.register_open(
                key,
                FamilyDescriptor::Custom {
                    kind: "phase".into(),
                },
            )
        })
        .unwrap()
        .unwrap();
        live.world_mut().spawn(PendingEffect::new(
            key,
            EffectKind::Custom {
                kind: "phase".into(),
                payload: serde_json::Value::Null,
            },
        ));
    }
    bus_support::tick_until(&mut live, "late world answer collected", |world| {
        world.resource::<Visible>().0.len() == 2
    });
    assert_eq!(
        live.world().resource::<Visible>().0,
        [vec!["a".to_owned()], vec!["a".to_owned(), "b".to_owned()]]
    );
    let log: EffectLog =
        serde_json::from_str(&serde_json::to_string(&recorder.log()).unwrap()).unwrap();
    let mut replay = bus_support::app();
    replay.init_resource::<Visible>();
    replay.world_mut().resource_mut::<Schedules>().add_systems(
        RigSchedule,
        observe_visible.after(BusSet::Collect).before(BusSet::Judge),
    );
    Handlers::with(replay.world_mut(), |handlers| {
        Replay::policy_visible().register(handlers, &log)
    })
    .unwrap()
    .unwrap();
    Replay::load(replay.world_mut(), &log);
    bus_support::tick_until(&mut replay, "same observed answer sets", |world| {
        world.resource::<Visible>().0.len() == 2
    });
    assert_eq!(
        replay.world().resource::<Visible>().0,
        live.world().resource::<Visible>().0
    );
}

#[test]
fn direct_in_flight_outcomes_refuse_policy_delivery_claims() {
    let mut live = bus_support::app();
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(live.world_mut(), recorder.clone());
    Handlers::with(live.world_mut(), |handlers| {
        handlers.register_open(
            "open",
            FamilyDescriptor::Custom {
                kind: "open".into(),
            },
        )
    })
    .unwrap()
    .unwrap();
    let effect = live
        .world_mut()
        .spawn(PendingEffect::new(
            "open",
            EffectKind::Custom {
                kind: "open".into(),
                payload: serde_json::Value::Null,
            },
        ))
        .id();
    live.update();
    live.world_mut()
        .entity_mut(effect)
        .insert(EffectOutcome(Ok(Outcome::Custom {
            payload: serde_json::json!("bypassed collection"),
        })));
    live.update();
    let log = recorder.log();
    let mut replay = bus_support::app();
    let error = Handlers::with(replay.world_mut(), |handlers| {
        Replay::policy_visible().register(handlers, &log)
    })
    .unwrap()
    .unwrap_err();
    assert!(error.message.contains("WorldOutcome"));
    Handlers::with(replay.world_mut(), |handlers| {
        Replay::default().register(handlers, &log)
    })
    .unwrap()
    .unwrap();
}

struct BatchedStream {
    gates: Mutex<VecDeque<oneshot::Receiver<Vec<&'static str>>>>,
    produced: Arc<AtomicUsize>,
    error: Option<rig_core::error::ErrorReport>,
}

impl Serve for BatchedStream {
    type Family = rig_core::effect::family::Completion;
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("model"),
            family: FamilyDescriptor::Completion {
                model: ModelRef::new("batched"),
                capabilities: ProviderCapabilities::default(),
            },
            layers: vec![],
        }
    }
    async fn serve(&self, _kind: EffectKind, sink: OutcomeSink) {
        let mut writer = sink.writer();
        loop {
            let gate = self.gates.lock().unwrap().pop_front();
            let Some(gate) = gate else {
                break;
            };
            let pieces = gate.await.unwrap();
            for piece in pieces {
                writer.text(piece).await.unwrap();
            }
            if self.gates.lock().unwrap().is_empty() {
                if let Some(error) = &self.error {
                    writer.error(error.clone()).await.unwrap();
                    drop(writer);
                } else {
                    writer
                        .finish(StreamFinal::new("batched", Usage::new()))
                        .await
                        .unwrap();
                }
                self.produced.fetch_add(1, Ordering::SeqCst);
                return;
            }
            self.produced.fetch_add(1, Ordering::SeqCst);
        }
    }
}

#[derive(Resource, Default)]
struct Snapshots(Vec<String>);

fn snapshot(streams: Query<&Streamed, Changed<Streamed>>, mut snapshots: ResMut<Snapshots>) {
    for stream in &streams {
        if !stream.text.is_empty() {
            snapshots.0.push(stream.text.clone());
        }
    }
}

fn observing_app() -> bevy_app::App {
    let mut app = bus_support::app();
    app.init_resource::<Snapshots>();
    app.world_mut()
        .resource_mut::<Schedules>()
        .add_systems(RigSchedule, snapshot.after(BusSet::Judge));
    app
}

fn live_stream(groups: Vec<Vec<&'static str>>, keep: bool) -> (EffectLog, Vec<String>) {
    let mut app = observing_app();
    let recorder = if keep {
        EffectLogRecorder::keeping_stream_events()
    } else {
        EffectLogRecorder::new()
    };
    EffectLogResource::install(app.world_mut(), recorder.clone());
    let (senders, receivers): (Vec<_>, Vec<_>) =
        (0..groups.len()).map(|_| oneshot::channel()).unzip();
    let produced = Arc::new(AtomicUsize::new(0));
    bus_support::register(
        &mut app,
        "model",
        BatchedStream {
            gates: Mutex::new(receivers.into()),
            produced: produced.clone(),
            error: None,
        },
    );
    let effect = app
        .world_mut()
        .spawn(PendingEffect::new("model", bus_support::streaming()))
        .id();
    app.update();
    for (index, (sender, group)) in senders.into_iter().zip(groups).enumerate() {
        sender.send(group).unwrap();
        // The handler acknowledges the entire batch before collection. No
        // sleeps or races decide whether two deltas belong to the same pass.
        let start = Instant::now();
        while produced.load(Ordering::SeqCst) <= index {
            assert!(
                start.elapsed() < bus_support::GUARD,
                "producer did not acknowledge batch"
            );
            std::thread::yield_now();
        }
        app.update();
    }
    bus_support::tick_until(&mut app, "stream closed", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    (
        serde_json::from_str(&serde_json::to_string(&recorder.log()).unwrap()).unwrap(),
        app.world().resource::<Snapshots>().0.clone(),
    )
}

#[test]
fn kept_streams_replay_single_and_multi_event_policy_batches() {
    for groups in [
        vec![vec!["a"], vec!["b"], vec!["c"]],
        vec![vec!["a", "b"], vec!["c"]],
    ] {
        let expected: Vec<_> = groups
            .iter()
            .scan(String::new(), |text, group| {
                text.push_str(&group.concat());
                Some(text.clone())
            })
            .collect();
        let (log, live) = live_stream(groups, true);
        assert_eq!(live, expected);
        let mut app = observing_app();
        Handlers::with(app.world_mut(), |handlers| {
            Replay::policy_visible().register(handlers, &log)
        })
        .unwrap()
        .unwrap();
        let effect = Replay::load(app.world_mut(), &log)[0];
        bus_support::tick_until(&mut app, "replayed stream", |world| {
            world.get::<EffectOutcome>(effect).is_some()
        });
        assert_eq!(app.world().resource::<Snapshots>().0, live);
        assert_eq!(
            serde_json::to_value(&app.world().get::<Streamed>(effect).unwrap().events).unwrap(),
            serde_json::to_value(log.records[0].events.as_ref().unwrap()).unwrap()
        );
    }
}

#[test]
fn folded_stream_refuses_policy_mode_but_replays_a_final_answer() {
    let (log, live) = live_stream(vec![vec!["a"], vec!["b"], vec!["c"]], false);
    assert_eq!(live, ["a", "ab", "abc"]);
    let mut app = observing_app();
    let error = Handlers::with(app.world_mut(), |handlers| {
        Replay::policy_visible().register(handlers, &log)
    })
    .unwrap()
    .expect_err("the recording omitted event bytes");
    assert!(error.message.contains("kept stream events"));
    Handlers::with(app.world_mut(), |handlers| {
        Replay::default().register(handlers, &log)
    })
    .unwrap()
    .unwrap();
    let effect = Replay::load(app.world_mut(), &log)[0];
    bus_support::tick_until(&mut app, "folded replay", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    assert_eq!(app.world().resource::<Snapshots>().0, ["abc"]);
}

#[test]
fn request_shape_mismatch_is_a_terminal_divergence_instead_of_a_delivery_wait() {
    let (streamed, _) = live_stream(vec![vec!["a"]], true);
    let counters = Arc::new(bus_support::Counters::default());
    let mut live = bus_support::app();
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(live.world_mut(), recorder.clone());
    bus_support::register(&mut live, "model", bus_support::MockModel::new(&counters));
    let effect = live
        .world_mut()
        .spawn(PendingEffect::new("model", bus_support::completion()))
        .id();
    bus_support::tick_until(&mut live, "unary recorded", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    for log in [streamed, recorder.log()] {
        let mut app = bus_support::app();
        Handlers::with(app.world_mut(), |handlers| {
            Replay::policy_visible().register(handlers, &log)
        })
        .unwrap()
        .unwrap();
        let effect = Replay::load(app.world_mut(), &log)[0];
        if let EffectKind::Completion { stream, .. } = &mut app
            .world_mut()
            .get_mut::<PendingEffect>(effect)
            .unwrap()
            .kind
        {
            *stream = !*stream;
        }
        bus_support::tick_until(&mut app, "request mismatch terminal", |world| {
            world.get::<EffectOutcome>(effect).is_some()
        });
        assert_eq!(
            app.world()
                .get::<EffectOutcome>(effect)
                .unwrap()
                .0
                .as_ref()
                .unwrap_err()
                .kind,
            rig_core::error::ErrorKind::Divergence
        );
    }
}

#[test]
fn malformed_delivery_metadata_is_refused_before_handlers_are_registered() {
    let (log, _) = live_stream(vec![vec!["a"]], true);
    for variant in 0..4 {
        let mut malformed = log.clone();
        let deliveries = malformed.header.deliveries.as_mut().unwrap();
        match variant {
            0 => deliveries[0].id = rig_core::effect::EffectId::from_raw(999),
            1 => deliveries.push(deliveries.last().unwrap().clone()),
            2 => deliveries[0].kind = rig_core::effect::DeliveryKind::Stream { items: usize::MAX },
            _ => deliveries.clear(),
        }
        let mut app = bus_support::app();
        let result = Handlers::with(app.world_mut(), |handlers| {
            Replay::policy_visible().register(handlers, &malformed)
        })
        .unwrap();
        assert!(result.is_err());
        assert_eq!(
            app.world_mut()
                .query::<&rig_ecs::bus::Bound>()
                .iter(app.world())
                .count(),
            0
        );
    }
}

#[test]
fn replay_tail_and_restored_subset_do_not_wait_for_already_answered_effects() {
    use rig_ecs::bus::{Issued, Scene};
    let (log, _) = ordered_live(false);
    let mut first = log.clone();
    first.records.truncate(1);
    first
        .header
        .deliveries
        .as_mut()
        .unwrap()
        .retain(|step| step.id == first.records[0].id);
    let mut head = bus_support::app();
    Handlers::with(head.world_mut(), |handlers| {
        Replay::policy_visible().register(handlers, &first)
    })
    .unwrap()
    .unwrap();
    let answered = Replay::load(head.world_mut(), &first)[0];
    bus_support::tick_until(&mut head, "head complete", |world| {
        world.get::<EffectOutcome>(answered).is_some()
    });
    let tail_effect = head
        .world_mut()
        .spawn((
            PendingEffect::new(log.records[1].key.clone(), log.records[1].kind.clone()),
            rig_ecs::bus::Reserved(log.records[1].id),
        ))
        .id();
    assert!(head.world().get::<Issued>(tail_effect).is_none());
    let scene = Scene::save(head.world_mut());
    for replay_log in [log.clone(), log.tail(1)] {
        let mut restored = bus_support::app();
        Handlers::with(restored.world_mut(), |handlers| {
            Replay::policy_visible().register(handlers, &replay_log)
        })
        .unwrap()
        .unwrap();
        let loaded = scene.load(restored.world_mut()).unwrap();
        bus_support::tick_until(&mut restored, "resumed subset", |world| {
            loaded
                .iter()
                .all(|entity| world.get::<EffectOutcome>(*entity).is_some())
        });
        assert!(loaded.iter().all(|entity| {
            restored
                .world()
                .get::<EffectOutcome>(*entity)
                .unwrap()
                .0
                .is_ok()
        }));
        assert_eq!(
            restored.world().get::<Issued>(loaded[1]).unwrap().0,
            log.records[1].id
        );
    }
}

fn cancel_loser(
    event: On<Add, EffectOutcome>,
    effects: Query<(Entity, &PendingEffect)>,
    mut first: ResMut<First>,
    mut commands: Commands,
) {
    let (_, effect) = effects.get(event.event().entity).unwrap();
    if first.0.is_none() {
        first.0 = Some(effect.key.to_string());
    }
    if effect.key == HandlerKey::from("b") {
        for (entity, effect) in &effects {
            if effect.key == HandlerKey::from("a") {
                commands.entity(entity).despawn();
            }
        }
    }
}

#[test]
fn policy_replay_does_not_insert_an_outcome_for_the_cancelled_loser() {
    let mut live = bus_support::app();
    live.init_resource::<First>().add_observer(cancel_loser);
    let recorder = EffectLogRecorder::new();
    EffectLogResource::install(live.world_mut(), recorder.clone());
    for key in ["a", "b"] {
        Handlers::with(live.world_mut(), |handlers| {
            handlers.register_open(
                key,
                FamilyDescriptor::Custom {
                    kind: "race".into(),
                },
            )
        })
        .unwrap()
        .unwrap();
    }
    let entities: Vec<_> = ["a", "b"]
        .into_iter()
        .map(|key| {
            live.world_mut()
                .spawn(PendingEffect::new(
                    key,
                    EffectKind::Custom {
                        kind: "race".into(),
                        payload: serde_json::Value::Null,
                    },
                ))
                .id()
        })
        .collect();
    live.update();
    live.world_mut()
        .entity_mut(entities[1])
        .insert(rig_ecs::bus::WorldOutcome::new(Ok(Outcome::Custom {
            payload: serde_json::json!("winner"),
        })));
    live.update();
    let log: EffectLog =
        serde_json::from_str(&serde_json::to_string(&recorder.log()).unwrap()).unwrap();
    assert_eq!(
        log.records[0].outcome.as_ref().unwrap_err().kind,
        rig_core::error::ErrorKind::Cancelled
    );
    assert!(live.world().get_entity(entities[0]).is_err());
    let mut replay = bus_support::app();
    replay.init_resource::<First>().add_observer(cancel_loser);
    Handlers::with(replay.world_mut(), |handlers| {
        Replay::policy_visible().register(handlers, &log)
    })
    .unwrap()
    .unwrap();
    let loaded = Replay::load(replay.world_mut(), &log);
    bus_support::tick_until(&mut replay, "winner cancels loser", |world| {
        world.get::<EffectOutcome>(loaded[1]).is_some()
    });
    replay.update();
    assert_eq!(replay.world().resource::<First>().0.as_deref(), Some("b"));
    assert!(replay.world().get_entity(loaded[0]).is_err());
    assert!(
        !replay
            .world()
            .contains_resource::<rig_ecs::bus::ReplayFailure>()
    );
}

fn cancel_partial(streams: Query<(Entity, &Streamed)>, mut commands: Commands) {
    for (entity, stream) in &streams {
        if stream.text == "ab" {
            commands.entity(entity).despawn();
        }
    }
}

#[test]
fn policy_cancels_at_the_same_partial_stream_state() {
    let mut live = observing_app();
    live.world_mut()
        .resource_mut::<Schedules>()
        .add_systems(RigSchedule, cancel_partial.after(snapshot));
    let recorder = EffectLogRecorder::keeping_stream_events();
    EffectLogResource::install(live.world_mut(), recorder.clone());
    let (send, first) = oneshot::channel();
    let (_unused, second) = oneshot::channel();
    let produced = Arc::new(AtomicUsize::new(0));
    bus_support::register(
        &mut live,
        "model",
        BatchedStream {
            gates: Mutex::new(vec![first, second].into()),
            produced: produced.clone(),
            error: None,
        },
    );
    let effect = live
        .world_mut()
        .spawn(PendingEffect::new("model", bus_support::streaming()))
        .id();
    live.update();
    send.send(vec!["a", "b"]).unwrap();
    let start = Instant::now();
    while produced.load(Ordering::SeqCst) == 0 {
        assert!(start.elapsed() < bus_support::GUARD);
        std::thread::yield_now();
    }
    live.update();
    assert!(live.world().get_entity(effect).is_err());
    assert_eq!(live.world().resource::<Snapshots>().0, ["ab"]);
    let log: EffectLog =
        serde_json::from_str(&serde_json::to_string(&recorder.log()).unwrap()).unwrap();
    let mut replay = observing_app();
    replay
        .world_mut()
        .resource_mut::<Schedules>()
        .add_systems(RigSchedule, cancel_partial.after(snapshot));
    Handlers::with(replay.world_mut(), |handlers| {
        Replay::policy_visible().register(handlers, &log)
    })
    .unwrap()
    .unwrap();
    let effect = Replay::load(replay.world_mut(), &log)[0];
    bus_support::tick_until(&mut replay, "same partial cancel", |world| {
        world.get_entity(effect).is_err()
    });
    assert_eq!(replay.world().resource::<Snapshots>().0, ["ab"]);
    assert!(
        !replay
            .world()
            .contains_resource::<rig_ecs::bus::ReplayFailure>()
    );
}

#[derive(Resource, Default)]
struct Interleaved(Vec<Vec<(String, String, Option<rig_core::error::ErrorKind>)>>);

fn observe_interleaved(
    streams: Query<(&PendingEffect, &Streamed), Changed<Streamed>>,
    mut observed: ResMut<Interleaved>,
) {
    let mut changed: Vec<_> = streams
        .iter()
        .filter(|(_, stream)| !stream.text.is_empty())
        .map(|(effect, stream)| {
            (
                effect.key.to_string(),
                stream.text.clone(),
                stream
                    .outcome
                    .as_ref()
                    .and_then(|outcome| outcome.as_ref().err().map(|error| error.kind)),
            )
        })
        .collect();
    changed.sort_by(|a, b| a.0.cmp(&b.0));
    if !changed.is_empty() {
        observed.0.push(changed);
    }
}

fn interleaved_app() -> bevy_app::App {
    let mut app = bus_support::app();
    app.init_resource::<Interleaved>();
    app.world_mut()
        .resource_mut::<Schedules>()
        .add_systems(RigSchedule, observe_interleaved.after(BusSet::Judge));
    app
}

#[test]
fn interleaved_streams_preserve_partial_states_and_provider_errors() {
    use rig_core::error::{ErrorKind, ErrorReport};

    for fail_second in [false, true] {
        let mut live = interleaved_app();
        let recorder = EffectLogRecorder::keeping_stream_events();
        EffectLogResource::install(live.world_mut(), recorder.clone());
        let mut controls = Vec::new();
        let mut effects = Vec::new();
        for key in ["a", "b"] {
            let (first_send, first) = oneshot::channel();
            let (last_send, last) = oneshot::channel();
            let produced = Arc::new(AtomicUsize::new(0));
            bus_support::register(
                &mut live,
                key,
                BatchedStream {
                    gates: Mutex::new(vec![first, last].into()),
                    produced: produced.clone(),
                    error: (fail_second && key == "b").then(|| {
                        ErrorReport::new(ErrorKind::Provider, "scripted provider failure")
                    }),
                },
            );
            controls.push((VecDeque::from([first_send, last_send]), produced));
            effects.push(
                live.world_mut()
                    .spawn(PendingEffect::new(key, bus_support::streaming()))
                    .id(),
            );
        }
        live.update();
        for (index, batch, pieces) in [
            (1, 1, vec!["B"]),
            (0, 1, vec!["a", "b"]),
            (1, 2, vec!["C"]),
            (0, 2, vec!["c"]),
        ] {
            let (senders, produced) = &mut controls[index];
            senders.pop_front().unwrap().send(pieces).unwrap();
            let start = Instant::now();
            while produced.load(Ordering::SeqCst) < batch {
                assert!(start.elapsed() < bus_support::GUARD);
                std::thread::yield_now();
            }
            live.update();
        }
        bus_support::tick_until(&mut live, "interleaved streams closed", |world| {
            effects
                .iter()
                .all(|entity| world.get::<EffectOutcome>(*entity).is_some())
        });
        let log: EffectLog =
            serde_json::from_str(&serde_json::to_string(&recorder.log()).unwrap()).unwrap();
        let mut replay = interleaved_app();
        Handlers::with(replay.world_mut(), |handlers| {
            Replay::policy_visible().register(handlers, &log)
        })
        .unwrap()
        .unwrap();
        let loaded = Replay::load(replay.world_mut(), &log);
        bus_support::tick_until(&mut replay, "interleaved streams replayed", |world| {
            loaded
                .iter()
                .all(|entity| world.get::<EffectOutcome>(*entity).is_some())
        });
        assert_eq!(
            replay.world().resource::<Interleaved>().0,
            live.world().resource::<Interleaved>().0
        );
        for (before, after) in effects.into_iter().zip(loaded) {
            assert_eq!(
                serde_json::to_value(live.world().get::<Streamed>(before).unwrap()).unwrap(),
                serde_json::to_value(replay.world().get::<Streamed>(after).unwrap()).unwrap(),
            );
            assert_eq!(
                serde_json::to_value(live.world().get::<EffectOutcome>(before).unwrap()).unwrap(),
                serde_json::to_value(replay.world().get::<EffectOutcome>(after).unwrap()).unwrap(),
            );
        }
    }
}

struct TerminalErrors {
    error_first: bool,
    error_kind: rig_core::error::ErrorKind,
    produced: Arc<AtomicUsize>,
}

impl Serve for TerminalErrors {
    type Family = rig_core::effect::family::Completion;
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: "model".into(),
            family: FamilyDescriptor::Completion {
                model: ModelRef::new("terminal-errors"),
                capabilities: ProviderCapabilities::default(),
            },
            layers: vec![],
        }
    }
    async fn serve(&self, _kind: EffectKind, mut sink: OutcomeSink) {
        let error = rig_core::error::ErrorReport::new(self.error_kind, "original error");
        let terminal =
            rig_core::streaming::StreamEvent::Final(StreamFinal::new("test", Usage::new()));
        let first = if self.error_first {
            vec![Err(error), Ok(terminal)]
        } else {
            vec![Ok(terminal), Err(error)]
        };
        for item in first
            .into_iter()
            .chain([Err(rig_core::error::ErrorReport::new(
                rig_core::error::ErrorKind::Provider,
                "late error",
            ))])
        {
            sink.send(item).await.unwrap();
        }
        drop(sink);
        self.produced.store(1, Ordering::SeqCst);
    }
}

#[test]
fn errors_before_and_after_final_keep_their_positions_and_first_outcome() {
    for (error_first, error_kind) in [false, true].into_iter().flat_map(|first| {
        [
            rig_core::error::ErrorKind::Response,
            rig_core::error::ErrorKind::Divergence,
        ]
        .map(|kind| (first, kind))
    }) {
        let mut live = bus_support::app();
        let recorder = EffectLogRecorder::keeping_stream_events();
        EffectLogResource::install(live.world_mut(), recorder.clone());
        let produced = Arc::new(AtomicUsize::new(0));
        bus_support::register(
            &mut live,
            "model",
            TerminalErrors {
                error_first,
                error_kind,
                produced: produced.clone(),
            },
        );
        let effect = live
            .world_mut()
            .spawn(PendingEffect::new("model", bus_support::streaming()))
            .id();
        live.update();
        let start = Instant::now();
        while produced.load(Ordering::SeqCst) == 0 {
            assert!(start.elapsed() < bus_support::GUARD);
            std::thread::yield_now();
        }
        bus_support::tick_until(&mut live, "terminal sequence collected", |world| {
            world.get::<EffectOutcome>(effect).is_some()
        });
        let expected = serde_json::to_value(live.world().get::<Streamed>(effect).unwrap()).unwrap();
        assert_eq!(
            live.world()
                .get::<EffectOutcome>(effect)
                .unwrap()
                .0
                .is_err(),
            error_first
        );
        let log: EffectLog =
            serde_json::from_str(&serde_json::to_string(&recorder.log()).unwrap()).unwrap();
        let errors = &log.header.stream_errors[&log.records[0].id];
        assert_eq!(errors.len(), 2);
        assert_eq!(errors[0].item, usize::from(!error_first));
        assert_eq!(errors[1].item, 2);
        // A valid three-item batch cannot omit either error from its count.
        let mut short = log.clone();
        let first_batch = short
            .header
            .deliveries
            .as_mut()
            .unwrap()
            .iter_mut()
            .find(|delivery| matches!(delivery.kind, rig_core::effect::DeliveryKind::Stream { .. }))
            .unwrap();
        first_batch.kind = rig_core::effect::DeliveryKind::Stream { items: 2 };
        let error = rig_ecs::bus::delivery::ReplayDelivery::new(&short, true)
            .err()
            .expect("all error items must be represented");
        assert!(error.message.contains("counts disagree"));
        if error_first {
            // Legacy successful-event bytes plus the folded error do not
            // identify whether that error preceded Final. Refuse policy replay.
            let mut missing = log.clone();
            missing.header.stream_errors.clear();
            missing
                .header
                .deliveries
                .as_mut()
                .unwrap()
                .iter_mut()
                .find(|delivery| {
                    matches!(delivery.kind, rig_core::effect::DeliveryKind::Stream { .. })
                })
                .unwrap()
                .kind = rig_core::effect::DeliveryKind::Stream { items: 2 };
            let error = rig_ecs::bus::delivery::ReplayDelivery::new(&missing, true)
                .err()
                .expect("first outcome is not reconstructible");
            assert!(error.message.contains("first stream outcome"));
            assert!(error.message.contains("error positions"));
        }
        let mut replay = bus_support::app();
        Handlers::with(replay.world_mut(), |handlers| {
            Replay::policy_visible().register(handlers, &log)
        })
        .unwrap()
        .unwrap();
        let loaded = Replay::load(replay.world_mut(), &log)[0];
        bus_support::tick_until(&mut replay, "terminal sequence replayed", |world| {
            world.get::<EffectOutcome>(loaded).is_some()
        });
        assert_eq!(
            serde_json::to_value(replay.world().get::<Streamed>(loaded).unwrap()).unwrap(),
            expected
        );
        assert_eq!(
            serde_json::to_value(replay.world().get::<EffectOutcome>(loaded).unwrap()).unwrap(),
            serde_json::to_value(live.world().get::<EffectOutcome>(effect).unwrap()).unwrap()
        );
    }
}
