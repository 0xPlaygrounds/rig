//! Independent live consumers and policy replay observe actual delivery batches.
use crate::bus_support;
use crate::run_support;

use bevy_ecs::prelude::*;
use rig_core::{
    completion::{ModelRef, ProviderCapabilities, Usage},
    effect::{EffectKind, FamilyDescriptor, HandlerDescriptor},
    error::{ErrorKind, ErrorReport},
    serve::{Dispatch, Reply, Serve},
    streaming::{StreamEvent, StreamFinal},
};
use rig_ecs::bus::{EffectOutcome, PendingEffect, StreamItemsDelivered, Streamed};
use std::sync::{Arc, Mutex};

type Trace = Arc<Mutex<Vec<(usize, serde_json::Value)>>>;

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
            output.lock().unwrap().push((
                delivery.start,
                serde_json::to_value(&delivery.items).unwrap(),
            ));
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
            Ok(StreamEvent::Final(StreamFinal::new(
                "mock",
                Usage::default(),
                serde_json::json!({}),
            ))),
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
    assert_eq!(items.len(), 3);
    assert_eq!(items[0]["Err"]["message"], "before final");
    assert!(items[1].get("Ok").is_some());
    assert_eq!(items[2]["Err"]["message"], "after final");
    assert_eq!(
        app.world()
            .get::<Streamed>(effect)
            .unwrap()
            .errors
            .iter()
            .map(|(index, _)| *index)
            .collect::<Vec<_>>(),
        vec![0, 2]
    );
    assert!(app.world().get::<EffectOutcome>(effect).unwrap().0.is_err());
}
#[test]
fn policy_replay_preserves_notification_batches_and_checkpoint_load_emits_none() {
    use rig_cassette::ecs::EffectLogResource;
    use rig_cassette::ecs::Replay;
    use rig_cassette::effect_log::EffectLogRecorder;
    use rig_cassette::effect_log::EffectLogReplayer;
    use rig_core::serve::ErasedHandler;
    use rig_ecs::checkpoint::{RestoreMode, load_world};
    let mut live = bus_support::app();
    let recorder = EffectLogRecorder::keeping_stream_events();
    EffectLogResource::install(live.world_mut(), recorder.clone());
    let original = observe(&mut live);
    run(&mut live);
    let mut replay = bus_support::app();
    let replayed = observe(&mut replay);
    Replay::policy_visible()
        .register(replay.world_mut(), &recorder.log())
        .unwrap();
    let effect = spawn_stream(&mut replay, "model");
    bus_support::tick_until(&mut replay, "replayed stream", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    assert_eq!(*original.lock().unwrap(), *replayed.lock().unwrap());
    let saved = bus_support::checkpoint(&mut replay);
    drop(replay);
    let mut restored = bus_support::app();
    let restored_trace = observe(&mut restored);
    // The saved handler is the replayer that answered the stream; the restored
    // world is served by the same recorded implementation, supplied here
    // rather than assembled live. No delivery plan travels with it.
    let log = recorder.log();
    let replayers = EffectLogReplayer::for_log_by_id(&log)
        .unwrap()
        .into_iter()
        .map(|replayer| (replayer.key().clone(), ErasedHandler::new(replayer)));
    load_world(&saved, restored.world_mut(), RestoreMode::Strict, replayers).unwrap();
    restored.update();
    assert!(restored_trace.lock().unwrap().is_empty());
    assert_eq!(
        restored
            .world_mut()
            .query::<&Streamed>()
            .single(restored.world())
            .unwrap()
            .errors
            .len(),
        2
    );
}

#[test]
fn terminal_delivery_remains_readable_when_an_observer_despawns_the_effect() {
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
                    .push((event.start, serde_json::to_value(&event.items).unwrap()));
                if remove
                    && event
                        .items
                        .iter()
                        .any(|item| matches!(item, Ok(StreamEvent::Final(_))))
                {
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

type Item = Result<StreamEvent, ErrorReport>;

struct Controlled(Mutex<Option<futures::channel::mpsc::UnboundedReceiver<Item>>>);

impl Serve for Controlled {
    type Family = rig_core::effect::family::Completion;
    fn descriptor(&self) -> HandlerDescriptor {
        WithErrors.descriptor()
    }
    async fn serve(&self, _: EffectKind, _: Dispatch) -> Reply {
        Reply::Stream(Box::pin(self.0.lock().unwrap().take().unwrap()))
    }
}

fn text(value: &str) -> Item {
    Ok(StreamEvent::BlockDelta {
        id: rig_core::streaming::BlockId::Wire("text".into()),
        delta: rig_core::streaming::Delta::Text { text: value.into() },
    })
}

#[test]
fn late_and_reenabled_consumers_hydrate_without_a_backlog() {
    use rig_core::streaming::Delta;
    let mut app = bus_support::app();
    let (sender, receiver) = futures::channel::mpsc::unbounded();
    bus_support::register(&mut app, "model", Controlled(Mutex::new(Some(receiver))));
    let effect = spawn_stream(&mut app, "model");
    sender.unbounded_send(text("early α")).unwrap();
    bus_support::tick_until(&mut app, "initial prefix", |world| {
        world
            .get::<Streamed>(effect)
            .is_some_and(|s| s.text == "early α")
    });
    let shown = Arc::new(Mutex::new(
        app.world().get::<Streamed>(effect).unwrap().text.clone(),
    ));
    let enabled = Arc::new(std::sync::atomic::AtomicBool::new(true));
    let output = shown.clone();
    let active = enabled.clone();
    app.add_observer(move |event: On<StreamItemsDelivered>| {
        if active.load(std::sync::atomic::Ordering::SeqCst) {
            for item in &event.items {
                if let Ok(StreamEvent::BlockDelta {
                    delta: Delta::Text { text },
                    ..
                }) = item
                {
                    output.lock().unwrap().push_str(text);
                }
            }
        }
    });
    sender.unbounded_send(text(" β")).unwrap();
    bus_support::tick_until(&mut app, "future delivery", |_| {
        *shown.lock().unwrap() == "early α β"
    });
    enabled.store(false, std::sync::atomic::Ordering::SeqCst);
    sender.unbounded_send(text(" skipped")).unwrap();
    bus_support::tick_until(&mut app, "disabled observer", |world| {
        world
            .get::<Streamed>(effect)
            .unwrap()
            .text
            .ends_with(" skipped")
    });
    assert_eq!(*shown.lock().unwrap(), "early α β");
    *shown.lock().unwrap() = app.world().get::<Streamed>(effect).unwrap().text.clone();
    enabled.store(true, std::sync::atomic::Ordering::SeqCst);
    sender.unbounded_send(text(" ω")).unwrap();
    sender
        .unbounded_send(Ok(StreamEvent::Final(StreamFinal::new(
            "mock",
            Usage::default(),
            serde_json::json!({}),
        ))))
        .unwrap();
    drop(sender);
    bus_support::tick_until(&mut app, "hydrated stream closed", |world| {
        world.get::<EffectOutcome>(effect).is_some()
    });
    assert_eq!(*shown.lock().unwrap(), "early α β skipped ω");
}

#[test]
fn burst_delivery_is_bounded_and_does_not_starve_another_effect() {
    use std::collections::BTreeMap;
    let mut app = bus_support::app();
    let batches = Arc::new(Mutex::new(BTreeMap::<Entity, Vec<usize>>::new()));
    let output = batches.clone();
    app.add_observer(move |event: On<StreamItemsDelivered>| {
        assert!(!event.items.is_empty() && event.items.len() <= 64);
        output
            .lock()
            .unwrap()
            .entry(event.effect)
            .or_default()
            .push(event.items.len());
    });
    let counters = Arc::new(bus_support::Counters::default());
    bus_support::register(
        &mut app,
        "model",
        bus_support::MockModel {
            cap: 5000,
            ..bus_support::MockModel::new(&counters)
        },
    );
    let effects: Vec<_> = (0..2).map(|_| spawn_stream(&mut app, "model")).collect();
    bus_support::tick_until(&mut app, "both streams deliver", |world| {
        effects.iter().all(|e| {
            world
                .get::<Streamed>(*e)
                .is_some_and(|s| !s.events.is_empty())
        })
    });
    bus_support::tick_until(&mut app, "both streams end", |world| {
        effects
            .iter()
            .all(|e| world.get::<EffectOutcome>(*e).is_some())
    });
    for effect in effects {
        let state = app.world().get::<Streamed>(effect).unwrap();
        assert_eq!(
            batches.lock().unwrap()[&effect].iter().sum::<usize>(),
            state.events.len()
        );
        assert!(batches.lock().unwrap()[&effect].len() > 1);
    }
}
#[test]
fn live_visibility_is_independent_of_recorder_event_retention() {
    use rig_cassette::ecs::EffectLogResource;
    use rig_cassette::effect_log::EffectLogRecorder;
    let mut expected = None;
    for keep in [false, true] {
        let recorder = if keep {
            EffectLogRecorder::keeping_stream_events()
        } else {
            EffectLogRecorder::new()
        };
        let mut app = bus_support::app();
        EffectLogResource::install(app.world_mut(), recorder.clone());
        let trace = observe(&mut app);
        run(&mut app);
        let actual = flattened(&trace);
        if let Some(expected) = &expected {
            assert_eq!(&actual, expected);
        }
        expected = Some(actual);
        assert_eq!(recorder.log().records.len(), 1);
    }
}

#[test]
fn final_delivery_precedes_run_settlement_and_terminal_graph_cleanup() {
    use rig_ecs::{agent::Settled, systems::RunCommands};
    let mut app = run_support::app();
    let trace = observe(&mut app);
    let final_seen = Arc::new(std::sync::atomic::AtomicBool::new(false));
    let observed = final_seen.clone();
    app.add_observer(move |event: On<StreamItemsDelivered>| {
        if event
            .items
            .iter()
            .any(|item| matches!(item, Ok(StreamEvent::Final(_))))
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
        sender.unbounded_send(text("partial")).unwrap();
        bus_support::tick_until(&mut app, "prefix delivered", |_| {
            flattened(&trace).len() == 1
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
        assert_eq!(
            flattened(&trace),
            vec![serde_json::to_value(text("partial")).unwrap()]
        );
    }
}

#[test]
fn empty_final_and_unary_stream_fold_have_distinct_delivery_contracts() {
    for streaming in [false, true] {
        let mut app = bus_support::app();
        let trace = observe(&mut app);
        let (sender, receiver) = futures::channel::mpsc::unbounded();
        bus_support::register(&mut app, "model", Controlled(Mutex::new(Some(receiver))));
        let mut kind = bus_support::streaming();
        if let EffectKind::Completion { stream, .. } = &mut kind {
            *stream = streaming;
        }
        let effect = app
            .world_mut()
            .spawn(PendingEffect::new("model", kind))
            .id();
        let terminal = Ok(StreamEvent::Final(StreamFinal::new(
            "mock",
            Usage::default(),
            serde_json::json!({}),
        )));
        sender.unbounded_send(terminal.clone()).unwrap();
        drop(sender);
        bus_support::tick_until(&mut app, "empty stream closed", |world| {
            world.get::<EffectOutcome>(effect).is_some()
        });
        assert!(app.world().get::<EffectOutcome>(effect).unwrap().0.is_ok());
        assert_eq!(
            flattened(&trace),
            if streaming {
                vec![serde_json::to_value(terminal).unwrap()]
            } else {
                vec![]
            }
        );
    }
}

struct RetryingStream(Arc<Mutex<Vec<rig_core::completion::CompletionRequest>>>);

impl Serve for RetryingStream {
    type Family = rig_core::effect::family::Completion;
    fn descriptor(&self) -> HandlerDescriptor {
        WithErrors.descriptor()
    }
    async fn serve(&self, kind: EffectKind, _: Dispatch) -> Reply {
        let EffectKind::Completion { request, .. } = kind else {
            return Reply::Outcome(Err(ErrorReport::new(
                ErrorKind::Request,
                "expected completion",
            )));
        };
        let first = {
            let mut requests = self.0.lock().unwrap();
            requests.push(request);
            requests.len() == 1
        };
        Reply::written(move |mut writer| async move {
            writer
                .text(if first { "partial" } else { "done" })
                .await
                .unwrap();
            if !first {
                writer
                    .finish(StreamFinal::new(
                        "mock",
                        Usage::default(),
                        serde_json::json!({}),
                    ))
                    .await
                    .unwrap();
            }
        })
    }
}

#[test]
fn retried_streams_have_distinct_delivery_identities_and_identical_requests() {
    use rig_core::streaming::Delta;
    use rig_ecs::{
        agent::{MaxTurns, RunResult, Settled},
        systems::RunCommands,
    };
    let mut app = run_support::app();
    let requests = Arc::new(Mutex::new(Vec::new()));
    let model = run_support::register(&mut app, "model", RetryingStream(requests.clone()));
    let agent = run_support::spawn_agent(app.world_mut(), "test", model);
    app.world_mut().entity_mut(agent).insert(MaxTurns(4));
    let observations = Arc::new(Mutex::new(
        std::collections::BTreeMap::<u64, (usize, String)>::new(),
    ));
    let observed = observations.clone();
    app.add_observer(move |event: On<StreamItemsDelivered>| {
        let mut observed = observed.lock().unwrap();
        let (count, text) = observed.entry(event.id.as_u64()).or_default();
        assert_eq!(*count, event.start);
        *count += event.items.len();
        for item in &event.items {
            if let Ok(StreamEvent::BlockDelta {
                delta: Delta::Text { text: piece },
                ..
            }) = item
            {
                text.push_str(piece);
            }
        }
    });
    let run = app.world_mut().spawn_run(agent, &[], "hello", true, None);
    run_support::tick_until(&mut app, "retry completed", |world| {
        world.get::<Settled>(run).is_some()
    });
    assert_eq!(app.world().get::<RunResult>(run).unwrap().0, "done");
    let requests = requests.lock().unwrap();
    assert_eq!(requests.len(), 2);
    assert_eq!(
        serde_json::to_value(&requests[0]).unwrap(),
        serde_json::to_value(&requests[1]).unwrap()
    );
    let observed = observations.lock().unwrap();
    assert_eq!(observed.len(), 2);
    assert_eq!(
        observed
            .values()
            .map(|(_, text)| text.as_str())
            .collect::<Vec<_>>(),
        vec!["partial", "done"]
    );
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
                text(value)
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
                        serde_json::to_value(&event.items).unwrap(),
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
            assert_eq!(record.events.as_ref().unwrap().len(), 1);
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
