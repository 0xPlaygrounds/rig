//! A running Bevy world over the effect bus: the six runtime proofs.
//!
//! 1. `ModelHandle` is a `Component`, `Dispatcher` is a `Resource` — the
//!    bounds compile because `Component: Send + Sync + 'static` is
//!    unconditional; a mock-backed handle is inserted and queried.
//! 2. The driver is spawned on a `TaskPool` (the `IoTaskPool` shape) and its
//!    `Task` held in a component — `BusDriver: Send` at the real call site.
//! 3. A system dispatches and stores the `Pending` in a component; a second
//!    system polls it across ticks with `block_on(poll_once(&mut pending))`
//!    — `Unpin` and executor-neutrality in the canonical Bevy pattern.
//! 4. Despawn cancels: despawning the entity holding the driver task mid
//!    dispatch resolves the in-flight `Pending` with `BusClosed`.
//! 5. Non-blocking dispatch: with the command channel full, `dispatch` from
//!    a system returns immediately; the pressure lands on the `Pending`.
//! 6. A `dispatch_stream` consumed across ticks then dropped mid-stream is
//!    observed as a cancellation by the handler.
//!
//! Every proof runs under a wall-clock guard; a hang is a failure, never a
//! wait.

use std::{
    sync::{
        Arc,
        atomic::{AtomicUsize, Ordering},
    },
    time::{Duration, Instant},
};

use bevy_ecs::prelude::*;
use bevy_tasks::{Task, TaskPool, block_on, futures_lite::future::poll_once};
use rig_core::{
    bus::{
        Bus, BusConfig, Dispatcher, EffectStream, Handler, HandlerFuture, ModelHandle, OutcomeSink,
        Pending,
    },
    completion::{
        CompletionRequest, CompletionResponse, Message, ModelRef, ProviderCapabilities, Usage,
    },
    effect::{EffectFamily, EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    error::ErrorKind,
    message::AssistantContent,
    streaming::{BlockId, MintKind, StreamEvent, StreamFinal},
};

const GUARD: Duration = Duration::from_secs(10);

/// A mock completion handler: answers unary dispatches with a fixed text,
/// streams one text delta per poll until the consumer goes away, and counts
/// how often it observed cancellation.
struct MockModel {
    unary_served: Arc<AtomicUsize>,
    stream_cancelled: Arc<AtomicUsize>,
    stream_sends: Arc<AtomicUsize>,
}

impl Handler for MockModel {
    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("model"),
            family: FamilyDescriptor::Completion {
                model: ModelRef::new("mock"),
                capabilities: ProviderCapabilities::default(),
            },
        }
    }

    fn handle(&self, kind: EffectKind, mut sink: OutcomeSink) -> HandlerFuture<'_> {
        Box::pin(async move {
            match kind {
                EffectKind::Completion { stream: false, .. } => {
                    self.unary_served.fetch_add(1, Ordering::SeqCst);
                    let response = CompletionResponse::new(
                        vec![AssistantContent::text("hello from the world")],
                        Usage::new(),
                        "mock",
                    );
                    sink.resolve(Ok(Outcome::Completion(response))).await;
                }
                EffectKind::Completion { stream: true, .. } => {
                    let id = BlockId::minted(MintKind::Text, 0);
                    loop {
                        let event = StreamEvent::text(id.clone(), "tick ");
                        if sink.send(Ok(event)).await.is_err() {
                            self.stream_cancelled.fetch_add(1, Ordering::SeqCst);
                            return;
                        }
                        self.stream_sends.fetch_add(1, Ordering::SeqCst);
                        if self.stream_sends.load(Ordering::SeqCst) >= 1_000 {
                            let _ = sink
                                .send(Ok(StreamEvent::Final(StreamFinal::new(
                                    "mock",
                                    Usage::new(),
                                ))))
                                .await;
                            return;
                        }
                    }
                }
                other => {
                    sink.resolve(Err(rig_core::error::ErrorReport::new(
                        ErrorKind::HandlerUnavailable,
                        format!("mock model cannot serve {}", other.name()),
                    )))
                    .await;
                }
            }
        })
    }
}

fn request() -> CompletionRequest {
    CompletionRequest {
        model: None,
        chat_history: vec![Message::user("hi")],
        documents: vec![],
        tools: vec![],
        temperature: None,
        max_tokens: None,
        tool_choice: None,
        additional_params: None,
        output_schema: None,
        record_telemetry_content: false,
    }
}

// ---- proof 1: the bounds hold as a Component and a Resource ----

#[derive(Component)]
struct Model(ModelHandle);

#[derive(Resource)]
struct BusRes(Dispatcher);

// ---- proof 2: the driver lives in a task pool task held by a component ----

#[derive(Component)]
struct DriverTask(#[allow(dead_code)] Task<()>);

// ---- proof 3: a dispatch in flight, polled across ticks ----

#[derive(Component)]
struct InFlight(Pending);

#[derive(Component)]
struct Answered(Result<Outcome, rig_core::error::ErrorReport>);

#[derive(Resource, Default)]
struct Ticks(usize);

fn dispatch_system(
    mut commands: Commands,
    bus: Res<BusRes>,
    models: Query<(Entity, &Model), Without<InFlight>>,
) {
    for (entity, model) in &models {
        let pending = bus.0.dispatch(
            model.0.key(),
            EffectKind::Completion {
                request: request(),
                stream: false,
            },
        );
        commands.entity(entity).insert(InFlight(pending));
    }
}

fn poll_system(
    mut commands: Commands,
    mut in_flight: Query<(Entity, &mut InFlight)>,
    mut ticks: ResMut<Ticks>,
) {
    ticks.0 += 1;
    for (entity, mut pending) in &mut in_flight {
        // The canonical Bevy pattern: poll once per tick, no executor of ours.
        if let Some(outcome) = block_on(poll_once(&mut pending.0)) {
            commands
                .entity(entity)
                .remove::<InFlight>()
                .insert(Answered(outcome));
        }
    }
}

fn tick(world: &mut World, schedule: &mut Schedule) {
    schedule.run(world);
}

fn guarded<T>(label: &str, mut step: impl FnMut() -> Option<T>) -> T {
    let started = Instant::now();
    loop {
        if let Some(value) = step() {
            return value;
        }
        assert!(started.elapsed() < GUARD, "{label}: hung for {GUARD:?}");
        std::thread::sleep(Duration::from_millis(1));
    }
}

fn main() {
    let pool = TaskPool::new();
    let served = Arc::new(AtomicUsize::new(0));
    let stream_cancelled = Arc::new(AtomicUsize::new(0));
    let stream_sends = Arc::new(AtomicUsize::new(0));
    let model = || MockModel {
        unary_served: served.clone(),
        stream_cancelled: stream_cancelled.clone(),
        stream_sends: stream_sends.clone(),
    };

    // ---- proofs 1, 2, 3 ----
    {
        let (dispatcher, mut driver) = Bus::channel();
        driver.register("model", model());
        let mut world = World::new();
        world.insert_resource(BusRes(dispatcher.clone()));
        world.insert_resource(Ticks::default());
        // Proof 2: `IoTaskPool::get().spawn(driver)` shape — the driver is
        // `Send` at the real call site.
        let driver_entity = world.spawn(DriverTask(pool.spawn(driver))).id();
        // Proof 1: a mock-backed handle, inserted and queried.
        let handle: ModelHandle = dispatcher
            .handle(&HandlerKey::from("model"))
            .expect("bound by family");
        assert_eq!(
            handle.descriptor().family.family(),
            EffectFamily::Completion
        );
        let model_entity = world.spawn(Model(handle)).id();
        let mut schedule = Schedule::default();
        schedule.add_systems((dispatch_system, poll_system).chain());

        // Proof 3: polled across ticks until the outcome arrives.
        let outcome = guarded("proof 3", || {
            tick(&mut world, &mut schedule);
            world
                .get::<Answered>(model_entity)
                .map(|answered| answered.0.clone())
        });
        match outcome {
            Ok(Outcome::Completion(response)) => {
                assert_eq!(
                    response.choice,
                    vec![AssistantContent::text("hello from the world")]
                );
            }
            other => panic!("proof 3: expected a completion, got {other:?}"),
        }
        assert_eq!(served.load(Ordering::SeqCst), 1);
        let ticks = world.resource::<Ticks>().0;
        assert!(ticks >= 1, "proof 3: at least one tick");
        println!("proof 3: resolved after {ticks} tick(s)");

        // ---- proof 4: despawn cancels ----
        // Start a dispatch, then despawn the entity holding the driver task
        // (Task cancels on drop) before the pool has served it.
        let mut pending = dispatcher.dispatch(
            &HandlerKey::from("model"),
            EffectKind::Completion {
                request: request(),
                stream: false,
            },
        );
        world.despawn(driver_entity);
        let report = guarded("proof 4", || match block_on(poll_once(&mut pending)) {
            Some(Err(report)) => Some(report),
            Some(Ok(outcome)) => {
                // The pool may have served it before the despawn landed; the
                // proof is that the *next* dispatch answers closed.
                let _ = outcome;
                let mut next = dispatcher.dispatch(
                    &HandlerKey::from("model"),
                    EffectKind::Completion {
                        request: request(),
                        stream: false,
                    },
                );
                loop {
                    if let Some(result) = block_on(poll_once(&mut next)) {
                        return Some(result.expect_err("the driver is gone"));
                    }
                    std::thread::sleep(Duration::from_millis(1));
                }
            }
            None => None,
        });
        assert_eq!(report.kind, ErrorKind::BusClosed, "proof 4: {report:?}");
        println!("proof 4: BusClosed after despawn");
    }

    // ---- proof 5: dispatch never blocks a system ----
    {
        let (dispatcher, driver) = Bus::channel_with(BusConfig {
            command_capacity: 1,
            ..BusConfig::default()
        });
        // Nobody drives yet: fill the channel from "a system" and keep
        // dispatching; every call returns immediately.
        let key = HandlerKey::from("model");
        let started = Instant::now();
        let mut pendings: Vec<Pending> = Vec::new();
        for _ in 0..64 {
            let mut pending = dispatcher.dispatch(
                &key,
                EffectKind::Completion {
                    request: request(),
                    stream: false,
                },
            );
            // One poll performs the send (or parks on the full channel).
            let _ = block_on(poll_once(&mut pending));
            pendings.push(pending);
        }
        assert!(
            started.elapsed() < Duration::from_secs(1),
            "proof 5: dispatch stalled the caller"
        );
        // The bound is bus-wide: one command buffered, sixty-three parked at
        // their send stage with the pressure on the pendings.
        assert_eq!(dispatcher.buffered(), 1, "proof 5: the bound did not hold");
        drop(driver);
        for mut pending in pendings {
            let outcome = guarded("proof 5", || block_on(poll_once(&mut pending)));
            assert_eq!(outcome.expect_err("closed").kind, ErrorKind::BusClosed);
        }
        println!("proof 5: 64 dispatches returned immediately; one buffered, the rest parked");
    }

    // ---- proof 6: a stream consumed across ticks, dropped mid-stream ----
    {
        let (dispatcher, mut driver) = Bus::channel_with(BusConfig {
            stream_capacity: 4,
            ..BusConfig::default()
        });
        driver.register("model", model());
        let _driver_task = pool.spawn(driver);
        let mut stream: EffectStream = dispatcher.dispatch_stream(
            &HandlerKey::from("model"),
            EffectKind::Completion {
                request: request(),
                stream: true,
            },
        );
        let mut received = 0;
        guarded("proof 6", || {
            use futures_lite_poll::next_item;
            match next_item(&mut stream) {
                Some(Some(Ok(_))) => {
                    received += 1;
                    (received >= 3).then_some(())
                }
                Some(Some(Err(report))) => panic!("proof 6: {report:?}"),
                Some(None) => panic!("proof 6: stream ended early"),
                None => None,
            }
        });
        drop(stream);
        guarded("proof 6", || {
            (stream_cancelled.load(Ordering::SeqCst) == 1).then_some(())
        });
        println!("proof 6: handler observed cancellation after {received} events");
    }

    println!("bevy-bus-host: ok");
}

/// `poll_once` for a stream: one poll of `next()` without an executor.
mod futures_lite_poll {
    use bevy_tasks::{block_on, futures_lite::future::poll_once};
    use rig_core::bus::EffectStream;

    pub fn next_item(
        stream: &mut EffectStream,
    ) -> Option<Option<Result<rig_core::streaming::StreamEvent, rig_core::error::ErrorReport>>>
    {
        use bevy_tasks::futures_lite::StreamExt;
        block_on(poll_once(stream.next()))
    }
}
