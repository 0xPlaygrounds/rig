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
//! 7. Registration from a system: the `Registrar` is a `NonSend` resource —
//!    the same spelling natively and in the browser — and a system installs
//!    a handler while the driver task is running; the next tick dispatches
//!    to it; a later system removes it and a dispatch answers
//!    `HandlerUnavailable`. The `Dispatcher` stays a plain `Resource`.
//!
//! Every proof runs under a wall-clock guard; a hang is a failure, never a
//! wait.

use std::{
    sync::{
        Arc,
        atomic::{AtomicBool, AtomicUsize, Ordering},
    },
    task::Poll,
    time::{Duration, Instant},
};

use bevy_ecs::prelude::*;
use bevy_tasks::{Task, TaskPool, block_on, futures_lite::future::poll_once};
use rig_core::{
    bus::{
        Bus, BusConfig, Dispatcher, EffectStream, ModelHandle, OutcomeSink, Pending, Registrar,
        Serve,
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

/// What one proof's mock observed: its own counters, so no proof's traffic
/// leaks into another's assertions (the stream cap below is per proof too).
#[derive(Default)]
struct Counters {
    /// Unary dispatches the handler entered.
    unary_started: AtomicUsize,
    /// Unary dispatches the handler answered.
    unary_served: AtomicUsize,
    stream_cancelled: AtomicUsize,
    stream_sends: AtomicUsize,
    /// While set, a unary dispatch stays in flight inside the handler (it
    /// yields to the executor instead of answering) — proof 4 despawns the
    /// driver while a dispatch is genuinely being served.
    hold: AtomicBool,
}

/// Cap on the deltas one proof's stream emits before its terminal record.
const STREAM_CAP: usize = 1_000;

/// A mock completion handler: answers unary dispatches with a fixed text
/// (once the hold is released), streams one text delta per poll until the
/// consumer goes away, and counts what it observed.
struct MockModel {
    counters: Arc<Counters>,
}

/// Yield to the executor once (a `Pending` that wakes itself), so a held
/// handler stays cancellable rather than spinning.
struct YieldNow(bool);

impl Future for YieldNow {
    type Output = ();

    fn poll(mut self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<()> {
        if self.0 {
            Poll::Ready(())
        } else {
            self.0 = true;
            cx.waker().wake_by_ref();
            Poll::Pending
        }
    }
}

impl Serve for MockModel {
    type Family = rig_core::effect::family::Completion;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("model"),
            family: FamilyDescriptor::Completion {
                model: ModelRef::new("mock"),
                capabilities: ProviderCapabilities::default(),
            },
        }
    }

    async fn serve(&self, kind: EffectKind, mut sink: OutcomeSink) {
        match kind {
            EffectKind::Completion { stream: false, .. } => {
                self.counters.unary_started.fetch_add(1, Ordering::SeqCst);
                while self.counters.hold.load(Ordering::SeqCst) {
                    YieldNow(false).await;
                }
                self.counters.unary_served.fetch_add(1, Ordering::SeqCst);
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
                        self.counters
                            .stream_cancelled
                            .fetch_add(1, Ordering::SeqCst);
                        return;
                    }
                    let sent = self.counters.stream_sends.fetch_add(1, Ordering::SeqCst) + 1;
                    if sent >= STREAM_CAP {
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

// ---- proof 7: registration from a system, through a NonSend registrar ----

/// The registrar is `NonSend` on every target: one spelling, both targets.
struct RegistrarRes(Registrar);

#[derive(Resource)]
struct RuntimeModel {
    key: HandlerKey,
    counters: Arc<Counters>,
    in_flight: Option<Pending>,
    answered: Option<Result<Outcome, rig_core::error::ErrorReport>>,
}

fn register_runtime_model(registrar: NonSendMut<RegistrarRes>, model: Res<RuntimeModel>) {
    registrar
        .0
        .register(
            model.key.clone(),
            MockModel {
                counters: model.counters.clone(),
            },
        )
        .expect("a fresh key");
}

fn dispatch_runtime_model(bus: Res<BusRes>, mut model: ResMut<RuntimeModel>) {
    if model.in_flight.is_none() && model.answered.is_none() {
        let pending = bus.0.dispatch(
            &model.key,
            EffectKind::Completion {
                request: request(),
                stream: false,
            },
        );
        model.in_flight = Some(pending);
    }
}

fn poll_runtime_model(mut model: ResMut<RuntimeModel>) {
    let outcome = model
        .in_flight
        .as_mut()
        .and_then(|pending| block_on(poll_once(pending)));
    if let Some(outcome) = outcome {
        model.in_flight = None;
        model.answered = Some(outcome);
    }
}

fn deregister_runtime_model(registrar: NonSendMut<RegistrarRes>, model: Res<RuntimeModel>) {
    assert!(
        registrar.0.deregister(&model.key),
        "proof 7: was registered"
    );
}

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

    // ---- proofs 1, 2, 3, 7 ----
    {
        let counters = Arc::new(Counters::default());
        let (dispatcher, registrar, mut driver) = Bus::channel();
        driver
            .register(
                "model",
                MockModel {
                    counters: counters.clone(),
                },
            )
            .expect("register");
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
        assert_eq!(counters.unary_served.load(Ordering::SeqCst), 1);
        let ticks = world.resource::<Ticks>().0;
        // The first tick performs the send; the outcome is seen by a later
        // tick at the earliest — a same-tick resolution would mean the
        // dispatch blocked the system.
        assert!(ticks >= 2, "proof 3: resolved within one tick ({ticks})");
        println!("proof 3: resolved after {ticks} tick(s)");

        // ---- proof 7: a system registers, the next tick dispatches ----
        // The registrar is a `NonSend` resource — the same two lines on
        // native and wasm — and the driver task is still running.
        world.insert_non_send(RegistrarRes(registrar));
        let runtime_counters = Arc::new(Counters::default());
        world.insert_resource(RuntimeModel {
            key: HandlerKey::from("runtime"),
            counters: runtime_counters.clone(),
            in_flight: None,
            answered: None,
        });
        let mut register = Schedule::default();
        register.add_systems(register_runtime_model);
        register.run(&mut world);
        assert!(
            dispatcher
                .descriptor(&HandlerKey::from("runtime"))
                .is_some(),
            "proof 7: the descriptor is visible the moment the system registered"
        );
        let mut serve = Schedule::default();
        serve.add_systems((dispatch_runtime_model, poll_runtime_model).chain());
        let outcome = guarded("proof 7", || {
            serve.run(&mut world);
            world.resource::<RuntimeModel>().answered.clone()
        });
        match outcome {
            Ok(Outcome::Completion(response)) => {
                assert_eq!(
                    response.choice,
                    vec![AssistantContent::text("hello from the world")]
                );
            }
            other => panic!("proof 7: expected a completion, got {other:?}"),
        }
        assert_eq!(runtime_counters.unary_served.load(Ordering::SeqCst), 1);
        assert_eq!(
            counters.unary_served.load(Ordering::SeqCst),
            1,
            "proof 7: the runtime handler served it, not the pre-spawn one"
        );
        let mut remove = Schedule::default();
        remove.add_systems(deregister_runtime_model);
        remove.run(&mut world);
        assert!(
            dispatcher
                .descriptor(&HandlerKey::from("runtime"))
                .is_none()
        );
        world.resource_mut::<RuntimeModel>().answered = None;
        let outcome = guarded("proof 7", || {
            serve.run(&mut world);
            world.resource::<RuntimeModel>().answered.clone()
        });
        let report = outcome.expect_err("proof 7: deregistered");
        assert_eq!(
            report.kind,
            ErrorKind::HandlerUnavailable,
            "proof 7: {report:?}"
        );
        println!(
            "proof 7: a system registered and removed a handler through the NonSend registrar"
        );

        // ---- proof 4: despawn cancels ----
        // Hold the handler so the dispatch is genuinely in flight — entered,
        // not answered — then despawn the entity holding the driver task
        // (Task cancels on drop). The in-flight `Pending` resolves with
        // `BusClosed`; an answer is impossible, so `Ok` is a failure.
        counters.hold.store(true, Ordering::SeqCst);
        let mut pending = dispatcher.dispatch(
            &HandlerKey::from("model"),
            EffectKind::Completion {
                request: request(),
                stream: false,
            },
        );
        guarded("proof 4", || {
            // One poll performs the send; the pool serves it into the hold.
            let _ = block_on(poll_once(&mut pending));
            (counters.unary_started.load(Ordering::SeqCst) == 2).then_some(())
        });
        world.despawn(driver_entity);
        let report = guarded("proof 4", || match block_on(poll_once(&mut pending)) {
            Some(Err(report)) => Some(report),
            Some(Ok(outcome)) => {
                panic!("proof 4: a held handler answered after the despawn: {outcome:?}")
            }
            None => None,
        });
        assert_eq!(report.kind, ErrorKind::BusClosed, "proof 4: {report:?}");
        assert_eq!(
            counters.unary_served.load(Ordering::SeqCst),
            1,
            "proof 4: the held dispatch was never answered"
        );
        println!("proof 4: BusClosed after despawn of an in-flight dispatch");
    }

    // ---- proof 5: dispatch never blocks a system ----
    {
        let (dispatcher, _registrar, driver) = Bus::channel_with(BusConfig {
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
        let counters = Arc::new(Counters::default());
        let (dispatcher, _registrar, mut driver) = Bus::channel_with(BusConfig {
            stream_capacity: 4,
            ..BusConfig::default()
        });
        driver
            .register(
                "model",
                MockModel {
                    counters: counters.clone(),
                },
            )
            .expect("register");
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
            (counters.stream_cancelled.load(Ordering::SeqCst) == 1).then_some(())
        });
        assert!(
            counters.stream_sends.load(Ordering::SeqCst) < STREAM_CAP,
            "proof 6: the stream was cancelled, not capped"
        );
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
