//! Independent live consumers and policy replay observe actual delivery batches.
use crate::bus_support;
use crate::run_support;

use bevy_ecs::prelude::*;
use rig_core::{
    completion::{ModelRef, ProviderCapabilities},
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor},
    error::{ErrorKind, ErrorReport},
    serve::{Dispatch, Reply, Serve},
    streaming::{Item, Relayed, StreamEvent},
};
use rig_ecs::bus::{EffectOutcome, PendingEffect, StreamItemsDelivered, Streamed};
use std::sync::{Arc, Mutex};

type Trace = Arc<Mutex<Vec<(usize, serde_json::Value)>>>;

/// A delivered batch, as the trace keeps it: each item's rendering.
fn batch(items: &[Result<Item<StreamEvent>, ErrorReport>]) -> serde_json::Value {
    items.iter().map(|item| format!("{item:?}")).collect()
}

fn observe(app: &mut bevy_app::App) -> Trace {
    let trace = Trace::default();
    let output = trace.clone();
    app.add_observer(
        move |delivery: On<StreamItemsDelivered>,
              states: Query<&Streamed>,
              outcomes: Query<&EffectOutcome>| {
            let state = states.get(delivery.effect).unwrap();
            assert!(
                outcomes.get(delivery.effect).is_err(),
                "delivery precedes outcome"
            );
            assert!(
                state.events.len() + state.errors.len() >= delivery.start + delivery.items.len()
            );
            output
                .lock()
                .unwrap()
                .push((delivery.start, batch(&delivery.items)));
        },
    );
    trace
}

struct WithErrors;

impl Serve for WithErrors {
    type Family = rig_core::effect::family::Completion;
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: "model".into(),
            family: FamilyDescriptor::Completion {
                model: ModelRef::new("delivery-errors"),
                capabilities: ProviderCapabilities::default(),
            },
            layers: vec![],
        }
    }
    async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
        Reply::Stream(Box::pin(futures::stream::iter([
            Err(ErrorReport::new(ErrorKind::Response, "before final")),
            bus_support::done("mock"),
            Err(ErrorReport::new(ErrorKind::Provider, "after final")),
        ])))
    }
}

/// Spawn a streaming completion effect on `key`.
fn spawn_stream(app: &mut bevy_app::App, key: &str) -> Entity {
    app.world_mut()
        .spawn(PendingEffect::new(key, bus_support::streaming()))
        .id()
}

fn run(app: &mut bevy_app::App) -> Entity {
    bus_support::register(app, "model", WithErrors);
    let effect = spawn_stream(app, "model");
    bus_support::tick_until(app, "stream closed", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    effect
}

fn flattened(trace: &Trace) -> Vec<serde_json::Value> {
    let mut items = Vec::new();
    for (start, batch) in trace.lock().unwrap().iter() {
        assert_eq!(*start, items.len());
        items.extend(batch.as_array().unwrap().iter().cloned());
    }
    items
}

#[test]
fn independent_consumers_preserve_interleaved_errors_without_a_recorder() {
    let mut app = bus_support::app();
    let first = observe(&mut app);
    let second = observe(&mut app);
    let effect = run(&mut app);
    assert_eq!(*first.lock().unwrap(), *second.lock().unwrap());
    let items = flattened(&first);
    // The response between the errors is the outcome, not an item.
    assert_eq!(items.len(), 2);
    assert!(items[0].as_str().unwrap().contains("before final"));
    assert!(items[1].as_str().unwrap().contains("after final"));
    assert_eq!(
        app.world()
            .get::<Streamed>(effect)
            .unwrap()
            .errors
            .iter()
            .map(|(index, _)| *index)
            .collect::<Vec<_>>(),
        vec![0, 1]
    );
    assert!(app.world().get::<EffectOutcome>(effect).unwrap().0.is_err());
}

#[test]
fn error_delivery_remains_readable_when_an_observer_despawns_the_effect() {
    let mut app = bus_support::app();
    let mut traces = Vec::new();
    for remove in [true, false] {
        let trace = Trace::default();
        traces.push(trace.clone());
        app.add_observer(
            move |event: On<StreamItemsDelivered>, mut commands: Commands| {
                trace
                    .lock()
                    .unwrap()
                    .push((event.start, batch(&event.items)));
                if remove && event.items.iter().any(Result::is_err) {
                    commands.entity(event.effect).despawn();
                }
            },
        );
    }
    bus_support::register(&mut app, "model", WithErrors);
    let effect = spawn_stream(&mut app, "model");
    bus_support::tick_until(&mut app, "consumer removed effect", |world| {
        world.get_entity(effect).is_err()
    });
    assert_eq!(*traces[0].lock().unwrap(), *traces[1].lock().unwrap());
    assert!(flattened(&traces[0]).len() >= 2);
}

struct Controlled(Mutex<Option<futures::channel::mpsc::UnboundedReceiver<bus_support::Relay>>>);

impl Serve for Controlled {
    type Family = rig_core::effect::family::Completion;
    fn descriptor(&self) -> HandlerDescriptor {
        WithErrors.descriptor()
    }
    async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
        Reply::Stream(Box::pin(self.0.lock().unwrap().take().unwrap()))
    }
}

fn text(value: &str) -> bus_support::Relay {
    bus_support::text_item(value)
}

#[test]
fn delivery_precedes_run_settlement_and_terminal_graph_cleanup() {
    use rig_ecs::{agent::Settled, systems::RunCommands};
    let mut app = run_support::app();
    let trace = observe(&mut app);
    let final_seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let observed = final_seen.clone();
    app.add_observer(move |event: On<StreamItemsDelivered>| {
        if event
            .items
            .iter()
            .any(|item| matches!(item, Ok(Item::Event(StreamEvent::Text { .. }))))
        {
            observed.store(true, std::sync::atomic::Ordering::SeqCst);
        }
    });
    app.add_observer(move |event: On<Add, Settled>, mut commands: Commands| {
        assert!(final_seen.load(std::sync::atomic::Ordering::SeqCst));
        commands.entity(event.event().entity).despawn();
    });
    let counters = Arc::new(bus_support::Counters::default());
    let model = run_support::register(
        &mut app,
        "model",
        bus_support::MockModel {
            cap: 2,
            ..bus_support::MockModel::new(&counters)
        },
    );
    let agent = run_support::spawn_agent(app.world_mut(), "test", model);
    let run = app.world_mut().spawn_run(agent, &[], "hello", true, None);
    run_support::tick_until(&mut app, "settled run removed", |world| {
        world.get_entity(run).is_err()
    });
    assert!(!flattened(&trace).is_empty());
}

#[test]
fn cancellation_and_truncated_closure_do_not_fabricate_delivery_items() {
    for cancel in [false, true] {
        let mut app = bus_support::app();
        let trace = observe(&mut app);
        let (sender, receiver) = futures::channel::mpsc::unbounded();
        bus_support::register(&mut app, "model", Controlled(Mutex::new(Some(receiver))));
        let effect = spawn_stream(&mut app, "model");
        sender
            .unbounded_send(bus_support::text_start_item())
            .unwrap();
        sender.unbounded_send(text("partial")).unwrap();
        bus_support::tick_until(&mut app, "prefix delivered", |_| {
            flattened(&trace).len() == 2
        });
        if cancel {
            app.world_mut().despawn(effect);
        }
        drop(sender);
        if cancel {
            app.update();
            assert!(app.world().get_entity(effect).is_err());
        } else {
            bus_support::tick_until(&mut app, "truncated closure", |world| {
                world.get::<EffectOutcome>(effect).is_some()
            });
            assert!(app.world().get::<EffectOutcome>(effect).unwrap().0.is_err());
        }
        let delivered: Vec<_> = [bus_support::text_start_item(), text("partial")]
            .into_iter()
            .map(|item| match item {
                Ok(Relayed::Item(item)) => Ok(item),
                other => panic!("an item: {other:?}"),
            })
            .collect();
        assert_eq!(
            flattened(&trace),
            batch(&delivered).as_array().unwrap().clone()
        );
    }
}

#[test]
fn replay_retains_both_accepted_batches_when_one_observer_removes_the_other_effect() {
    use futures::StreamExt;
    use rig_cassette::ecs::{EffectLogResource, Replay};
    use rig_cassette::effect_log::EffectLogRecorder;
    use rig_ecs::bus::Streaming;
    use std::sync::atomic::{AtomicBool, Ordering};

    struct ReadyStream {
        gate: Arc<bus_support::Hold>,
        ready: Arc<AtomicBool>,
        text: &'static str,
    }
    impl Serve for ReadyStream {
        type Family = rig_core::effect::family::Completion;
        fn descriptor(&self) -> HandlerDescriptor {
            WithErrors.descriptor()
        }
        async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
            let gate = self.gate.clone();
            let ready = self.ready.clone();
            let value = self.text;
            let first = futures::stream::once(async move {
                gate.wait().await;
            })
            .flat_map(move |()| {
                futures::stream::iter([bus_support::text_start_item(), text(value)])
            });
            let open = futures::stream::poll_fn(move |_| {
                // The worker polls again only after placing the first item in
                // its consumer queue. Keep EOF gated to isolate batch delivery.
                ready.store(true, Ordering::SeqCst);
                std::task::Poll::Pending
            });
            Reply::Stream(Box::pin(first.chain(open)))
        }
    }

    type Batches = Arc<Mutex<Vec<(u64, usize, serde_json::Value)>>>;
    fn consumers(app: &mut bevy_app::App, a: Entity, b: Entity) -> [Batches; 2] {
        let traces: [Batches; 2] = Default::default();
        for (index, trace) in traces.iter().enumerate() {
            let trace = trace.clone();
            app.add_observer(
                move |event: On<StreamItemsDelivered>, mut commands: Commands| {
                    trace.lock().unwrap().push((
                        event.id.as_u64(),
                        event.start,
                        batch(&event.items),
                    ));
                    if index == 0 {
                        commands
                            .entity(if event.effect == a { b } else { a })
                            .despawn();
                    }
                },
            );
        }
        traces
    }

    let mut live = bus_support::app();
    let recorder = EffectLogRecorder::keeping_stream_events();
    EffectLogResource::install(live.world_mut(), recorder.clone());
    let gate = Arc::new(bus_support::Hold::default());
    gate.hold();
    let ready = [
        Arc::new(AtomicBool::new(false)),
        Arc::new(AtomicBool::new(false)),
    ];
    for (index, key) in ["a", "b"].into_iter().enumerate() {
        bus_support::register(
            &mut live,
            key,
            ReadyStream {
                gate: gate.clone(),
                ready: ready[index].clone(),
                text: key,
            },
        );
    }
    let effects: Vec<_> = ["a", "b"]
        .into_iter()
        .map(|key| spawn_stream(&mut live, key))
        .collect();
    let original = consumers(&mut live, effects[0], effects[1]);
    bus_support::tick_until(&mut live, "both workers installed", |world| {
        effects
            .iter()
            .all(|entity| world.get::<Streaming>(*entity).is_some())
    });
    gate.release();
    let started = std::time::Instant::now();
    while !ready.iter().all(|ready| ready.load(Ordering::SeqCst)) {
        assert!(started.elapsed() < bus_support::GUARD);
        std::thread::yield_now();
    }
    live.update();
    assert_eq!(original[0].lock().unwrap().len(), 2);
    assert_eq!(*original[0].lock().unwrap(), *original[1].lock().unwrap());
    assert!(live.world().get_entity(effects[1]).is_err());
    assert!(live.world().get_entity(effects[0]).is_err());
    let log = recorder.log();

    let mut replay = bus_support::app();
    let replay_recorder = EffectLogRecorder::keeping_stream_events();
    EffectLogResource::install(replay.world_mut(), replay_recorder.clone());
    Replay::policy_visible()
        .register(replay.world_mut(), &log)
        .unwrap();
    let effects: Vec<_> = ["a", "b"]
        .into_iter()
        .map(|key| spawn_stream(&mut replay, key))
        .collect();
    let replayed = consumers(&mut replay, effects[0], effects[1]);
    bus_support::tick_until(&mut replay, "replayed accepted deliveries", |_| {
        replayed[0].lock().unwrap().len() == 2
    });
    assert!(replay.world().get_entity(effects[1]).is_err());
    assert!(replay.world().get_entity(effects[0]).is_err());
    replay.update();
    assert!(
        !replay
            .world()
            .contains_resource::<rig_cassette::ecs::ReplayFailure>()
    );
    assert_eq!(*original[0].lock().unwrap(), *replayed[0].lock().unwrap());
    assert_eq!(*replayed[0].lock().unwrap(), *replayed[1].lock().unwrap());
    let replay_log = replay_recorder.log();
    // Absolute collection pass numbers include worker readiness. Both streams
    // must retain their single shared visible batch and cancelled prefixes.
    for recorded in [&log, &replay_log] {
        let batches = recorded.header.deliveries.as_ref().unwrap();
        assert_eq!(batches.len(), 2);
        assert_eq!(batches[0].batch, batches[1].batch);
        assert_eq!(recorded.records.len(), 2);
        for record in &recorded.records {
            assert_eq!(
                record.outcome.as_ref().unwrap_err().kind,
                ErrorKind::Cancelled
            );
            // The text part's start, then its fragment.
            assert_eq!(record.events.as_ref().unwrap().len(), 2);
        }
    }
    assert_eq!(
        log.header
            .deliveries
            .unwrap()
            .into_iter()
            .map(|delivery| (delivery.id, delivery.kind))
            .collect::<Vec<_>>(),
        replay_log
            .header
            .deliveries
            .unwrap()
            .into_iter()
            .map(|delivery| (delivery.id, delivery.kind))
            .collect::<Vec<_>>()
    );
}
