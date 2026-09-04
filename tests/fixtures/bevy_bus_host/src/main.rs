//! A running Bevy world over the effect bus: the six runtime proofs.
//!
//! 1. `ModelHandle` is a `Component`, `Dispatcher` is a `Resource` — the
//!    bounds compile because `Component: Send + Sync + 'static` is
//!    unconditional; a mock-backed handle is inserted and queried.
//! 2. The driver is spawned on a `TaskPool` (the `IoTaskPool` shape) and its
//!    `Task` held in a component — `BusDriver: Send` at the real call site.
//! 3. A system dispatches and stores the `Pending` in a component; a second
//!    system probes it across ticks with `Pending::poll_outcome` — no
//!    executor, no waker minted per frame. (Proof 7 keeps the older
//!    `block_on(poll_once(&mut pending))` spelling, which stays legal.)
//! 4. Despawn cancels: despawning the entity holding the driver task mid
//!    dispatch resolves the in-flight `Pending` with `BusClosed`; and a
//!    `Pending` dropped after its send but before the driver ran is never
//!    served — its handler is not entered.
//! 5. Non-blocking dispatch: with the command channel full, `dispatch` from
//!    a system returns immediately; the pressure lands on the `Pending`.
//! 6. A `dispatch_stream` held in a component, probed across ticks with
//!    `EffectStream::poll_item`, then dropped mid-stream is observed as a
//!    cancellation by the handler.
//! 7. Registration from a system: the `Registrar` is a `NonSend` resource —
//!    the same spelling natively and in the browser — and a system installs
//!    a handler while the driver task is running; the next tick dispatches
//!    to it; a later system removes it and a dispatch answers
//!    `HandlerUnavailable`. The `Dispatcher` stays a plain `Resource`.
//!
//! 8. A tool that is a system: a handler detaches its sink into a resource
//!    and returns; a system with `World` access answers it on a later
//!    tick. Under serial serving the key stays busy until the answer, so
//!    a second dispatch to it waits — the log's order is the serve order.
//!
//! 9. The driver restarts: the entity holding the driver task is despawned
//!    (a state transition), `Bus::reopen` gives the bus a new driver on a
//!    later tick, the model is re-registered, and a `ModelHandle` component
//!    bound before the restart completes again — nothing re-inserted.
//!
//! 10. Scene round-trip: every `PendingEffect { key, kind }` (serde) and the
//!     bus's descriptors (`Dispatcher::descriptors`) are serialized, the
//!     world is cleared and rebuilt from that text, handles are re-bound
//!     with `Handle::rebind` and the effects re-dispatched under fresh ids
//!     (`mint_id` + `dispatch_with_id`); they resolve.
//!
//! 11. A layer that suspends: an `Intercept` whose `before` hands the
//!     dispatch to a resource and waits; a system with `Query` access
//!     approves or denies next tick; a denied dispatch resolves `Denied`
//!     on the consumer's `Pending`, a system sees it on the tick it lands,
//!     and the log holds no record for it; despawning the consumer
//!     mid-suspend closes the slot the system holds without a panic.
//!
//! 12. A system-resolved tool dispatches a nested completion through its
//!     sink's dispatcher: the child's effect entity is spawned as a child
//!     of the tool's (`ChildOf`), the record carries `parent`; under serial
//!     serving a nested dispatch to the tool's own key is refused, not
//!     hung; despawning the parent entity cancels the child and its record
//!     says `Cancelled`.
//!
//! 13. A checkpointed scene: the pending effects and a log position are a
//!     `Checkpoint` beside the log's tail; the world is rebuilt over a
//!     fresh bus with replayers for the tail; re-dispatch from the
//!     position resolves to the records.
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
use rig_bus::{Bus, Dispatcher, EffectStream, ModelHandle, Pending, Registrar, SinkDispatch};
use rig_core::serve::ServingPolicy;
use rig_core::serve::{Decision, DetachedSink, Intercept, OutcomeSink, Serve, Verdict};
use rig_core::{
    completion::{
        CompletionRequest, CompletionResponse, Message, ModelRef, ProviderCapabilities, Usage,
    },
    effect::{EffectFamily, EffectKind, FamilyDescriptor, HandlerDescriptor, HandlerKey, Outcome},
    error::ErrorKind,
    message::AssistantContent,
    streaming::StreamFinal,
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
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
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
                let mut out = sink.writer();
                loop {
                    if out.text("tick ").await.is_err() {
                        self.counters
                            .stream_cancelled
                            .fetch_add(1, Ordering::SeqCst);
                        return;
                    }
                    let sent = self.counters.stream_sends.fetch_add(1, Ordering::SeqCst) + 1;
                    if sent >= STREAM_CAP {
                        let _ = out.finish(StreamFinal::new("mock", Usage::new())).await;
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

// ---- proof 6: a stream in a component, probed per tick ----

#[derive(Component)]
struct InFlightStream(EffectStream);

#[derive(Resource, Default)]
struct Ticks(usize);

// ---- proof 10: scene round-trip ----

/// In-flight intent as data: what a scene stores for a dispatch, whether
/// or not it had been sent.
#[derive(Component, serde::Serialize, serde::Deserialize, Clone, Debug)]
struct PendingEffect {
    key: HandlerKey,
    kind: EffectKind,
}

/// What a scene stores for the bus side: the descriptors, so a load can
/// re-bind handles before the handlers are re-registered.
#[derive(serde::Serialize, serde::Deserialize)]
struct Scene {
    handlers: Vec<HandlerDescriptor>,
    effects: Vec<PendingEffect>,
}

// ---- proof 8: a tool that is a system ----

/// Effects a `WorldTool` handler handed to the world, with their sinks:
/// a plain `Resource` — the sink is `Send + Sync` on every target.
#[derive(Resource, Default, Clone)]
struct WorldToolMailbox(Arc<std::sync::Mutex<Vec<(EffectKind, DetachedSink)>>>);

/// A handler that never answers in its own future.
struct WorldTool {
    mailbox: WorldToolMailbox,
}

impl Serve for WorldTool {
    type Family = rig_core::effect::family::Dynamic;

    fn descriptor(&self) -> HandlerDescriptor {
        HandlerDescriptor {
            key: HandlerKey::from("world-tool"),
            family: FamilyDescriptor::Custom {
                kind: "host:world-tool".into(),
            },
            layers: Vec::new(),
        }
    }

    async fn serve(&self, kind: EffectKind, sink: OutcomeSink) {
        self.mailbox
            .0
            .lock()
            .expect("mailbox")
            .push((kind, sink.detach()));
    }
}

/// World state the answering system reads.
#[derive(Component)]
struct WorldState;

#[derive(Resource)]
struct WorldToolRuns {
    key: HandlerKey,
    in_flight: Vec<Pending>,
    answered: Vec<Result<Outcome, rig_core::error::ErrorReport>>,
    /// Model entities seen by the answering system, per answer.
    seen: Vec<usize>,
}

fn dispatch_world_tool(bus: Res<BusRes>, mut runs: ResMut<WorldToolRuns>) {
    if runs.in_flight.is_empty() && runs.answered.is_empty() {
        for n in 0..2u64 {
            let pending = bus.0.dispatch(
                &runs.key,
                EffectKind::Custom {
                    kind: Arc::from("host:world-tool"),
                    payload: serde_json::json!({ "n": n }),
                },
            );
            runs.in_flight.push(pending);
        }
    }
}

/// The answering system: `World` access (a query over the model entities)
/// and the mailbox. One answer per tick, so the serial order is visible.
fn answer_world_tool(
    mailbox: Res<WorldToolMailbox>,
    state: Query<(), With<WorldState>>,
    mut runs: ResMut<WorldToolRuns>,
) {
    let next = mailbox.0.lock().expect("mailbox").pop();
    if let Some((kind, sink)) = next {
        let seen = state.iter().count();
        runs.seen.push(seen);
        let payload = match kind {
            EffectKind::Custom { payload, .. } => payload,
            other => panic!("proof 8: {other:?}"),
        };
        block_on(sink.resolve(Ok(Outcome::Custom(
            serde_json::json!({ "answered": payload, "models": seen }),
        ))));
    }
}

fn poll_world_tool(mut runs: ResMut<WorldToolRuns>) {
    let mut still = Vec::new();
    for mut pending in std::mem::take(&mut runs.in_flight) {
        match pending.poll_outcome() {
            Some(outcome) => runs.answered.push(outcome),
            None => still.push(pending),
        }
    }
    runs.in_flight = still;
}

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
        // Once per tick, no executor: the probe.
        if let Some(outcome) = pending.0.poll_outcome() {
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

// ---- proof 11: a layer that suspends, decided by a system ----

/// A decision the world will make: the layer awaits it, the system fills
/// it. `Arc::strong_count == 1` on the world's side means the layer's
/// future is gone (the consumer cancelled), so nothing waits for the
/// decision any more.
#[derive(Clone)]
struct DecisionSlot(Arc<std::sync::Mutex<(Option<Decision>, Option<std::task::Waker>)>>);

impl DecisionSlot {
    fn new() -> Self {
        Self(Arc::new(std::sync::Mutex::new((None, None))))
    }

    fn decide(&self, decision: Decision) {
        let mut slot = self.0.lock().expect("slot");
        slot.0 = Some(decision);
        if let Some(waker) = slot.1.take() {
            waker.wake();
        }
    }

    /// Whether the layer that asked is gone.
    fn is_canceled(&self) -> bool {
        Arc::strong_count(&self.0) == 1
    }
}

impl Future for DecisionSlot {
    type Output = Decision;

    fn poll(self: std::pin::Pin<&mut Self>, cx: &mut std::task::Context<'_>) -> Poll<Decision> {
        let mut slot = self.0.lock().expect("slot");
        match slot.0.take() {
            Some(decision) => Poll::Ready(decision),
            None => {
                slot.1 = Some(cx.waker().clone());
                Poll::Pending
            }
        }
    }
}

/// The dispatches waiting for the world's decision.
#[derive(Resource, Default, Clone)]
struct Approvals(Arc<std::sync::Mutex<Vec<(rig_core::effect::EffectId, DecisionSlot)>>>);

/// The layer: every dispatch waits for the world.
struct Gate {
    approvals: Approvals,
}

impl Intercept for Gate {
    fn name(&self) -> String {
        "Gate".to_owned()
    }

    async fn before(&self, id: rig_core::effect::EffectId, _kind: &EffectKind) -> Decision {
        let slot = DecisionSlot::new();
        self.approvals
            .0
            .lock()
            .expect("approvals")
            .push((id, slot.clone()));
        slot.await
    }

    async fn after(
        &self,
        _id: rig_core::effect::EffectId,
        _kind: &EffectKind,
        _outcome: &Result<Outcome, rig_core::error::ErrorReport>,
    ) -> Verdict {
        Verdict::Keep
    }
}

/// What the world decided, and the slot it still holds for the third.
#[derive(Resource, Default)]
struct Decided {
    approved: usize,
    denied: usize,
    held: Option<DecisionSlot>,
    /// Denials a system saw land on a consumer, on the tick they landed.
    denied_seen: usize,
}

/// A system with `Query` access decides: the first dispatch proceeds if
/// the world has its three markers, the second is denied, the third is
/// held (the consumer will go).
fn decide_system(
    approvals: Res<Approvals>,
    world_state: Query<&WorldState>,
    mut decided: ResMut<Decided>,
) {
    let asked: Vec<_> = approvals.0.lock().expect("approvals").drain(..).collect();
    for (_, slot) in asked {
        let n = decided.approved + decided.denied + usize::from(decided.held.is_some());
        match n {
            0 if world_state.iter().count() == 3 => {
                slot.decide(Decision::Proceed);
                decided.approved += 1;
            }
            0 => panic!("proof 11: the world has three markers"),
            1 => {
                slot.decide(Decision::deny("blocked by the world"));
                decided.denied += 1;
            }
            _ => decided.held = Some(slot),
        }
    }
}

/// The observer's seat: a denial is seen on the tick it lands.
fn see_denials(answered: Query<&Answered, Added<Answered>>, mut decided: ResMut<Decided>) {
    for outcome in &answered {
        if matches!(&outcome.0, Err(report) if report.kind == ErrorKind::Denied) {
            decided.denied_seen += 1;
        }
    }
}

// ---- proof 12: a nested dispatch from a system-resolved tool ----

/// A tool call the world is resolving: its sink, and the child completion
/// it dispatched through the sink's dispatcher.
#[derive(Resource, Default)]
struct Nested {
    parents: Vec<(Entity, DetachedSink)>,
    answered: Vec<Result<Outcome, rig_core::error::ErrorReport>>,
    /// The refusal a same-key nested dispatch got under serial serving.
    refused: Option<rig_core::error::ErrorReport>,
}

/// Pops the tool's sink from the mailbox and dispatches a child completion
/// through it; the child's entity is a child of the tool's.
fn nest_system(
    mut commands: Commands,
    mailbox: Res<WorldToolMailbox>,
    mut nested: ResMut<Nested>,
    parents: Query<(Entity, &InFlight)>,
) {
    let mut taken = mailbox.0.lock().expect("mailbox");
    while let Some((_, sink)) = taken.pop() {
        let scoped = sink
            .dispatcher()
            .expect("proof 12: served by the bus driver");
        assert_eq!(
            scoped.parent(),
            Some(sink.id()),
            "proof 12: scoped to the tool's dispatch"
        );
        // The tool's own key under serial serving: refused at the send,
        // never queued behind the call that waits on it.
        let mut own = scoped.dispatch(
            &HandlerKey::from("world-tool"),
            EffectKind::Custom {
                kind: Arc::from("host:nested"),
                payload: serde_json::json!({"leaf": true}),
            },
        );
        match own.poll_outcome() {
            Some(Err(report)) => nested.refused = Some(report),
            other => panic!("proof 12: the same-key nested dispatch was not refused: {other:?}"),
        }
        let child = scoped.dispatch(
            &HandlerKey::from("model"),
            EffectKind::Completion {
                request: request(),
                stream: false,
            },
        );
        assert_eq!(child.parent(), Some(sink.id()));
        // The tool's entity is the one whose `InFlight` id is the sink's.
        let (parent_entity, _) = parents
            .iter()
            .find(|(_, in_flight)| in_flight.0.id() == sink.id())
            .expect("proof 12: the tool's entity");
        commands.spawn((InFlight(child), ChildOf(parent_entity)));
        nested.parents.push((parent_entity, sink));
    }
}

/// The world's duty with a detached sink: a consumer that went away is a
/// closed sink, and the world drops it — the record says `Cancelled`
/// through the sink's drop, and the dispatch ends.
fn drop_cancelled_sinks(mut nested: ResMut<Nested>) {
    nested.parents.retain(|(_, sink)| !sink.is_closed());
}

/// When a child completion answered, the tool answers with it.
fn finish_nested(
    mut commands: Commands,
    mut nested: ResMut<Nested>,
    children: Query<(Entity, &ChildOf, &Answered)>,
) {
    for (entity, child_of, answered) in &children {
        let Some(position) = nested
            .parents
            .iter()
            .position(|(parent, _)| *parent == child_of.parent())
        else {
            continue;
        };
        let (_, sink) = nested.parents.remove(position);
        let text = match &answered.0 {
            Ok(Outcome::Completion(response)) => response
                .choice
                .iter()
                .filter_map(|content| match content {
                    AssistantContent::Text(text) => Some(text.text.clone()),
                    _ => None,
                })
                .collect::<String>(),
            other => panic!("proof 12: a completion, not {other:?}"),
        };
        let outcome = Ok(Outcome::Custom(serde_json::json!({ "nested": text })));
        block_on(sink.resolve(outcome.clone()));
        nested.answered.push(outcome);
        commands.entity(entity).despawn();
    }
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

/// `--replay <golden.effects.json>`: the effect corpus's log-as-script
/// replay in a Bevy world. The host registers a replayer for every key
/// the golden names, then a system dispatches each record's effect to
/// its key in the golden's order and compares the answer with the
/// record's. No agent runs here and nothing is interpreted: what it
/// proves is the bus plumbing and the order — a host that persisted a
/// run's effects can dispatch them again from its systems and get the
/// recorded answers back.
fn replay(path: &str) {
    let pool = TaskPool::new();
    let text = std::fs::read_to_string(path).expect("the golden reads");
    let log: rig_effect_log::EffectLog = serde_json::from_str(&text).expect("the golden decodes");
    let (dispatcher, _registrar, mut driver) = Bus::channel();
    rig_effect_log::EffectLogReplayer::register_all(&log, &mut driver).expect("the golden's keys");
    let mut world = World::new();
    world.insert_resource(BusRes(dispatcher.clone()));
    world.insert_resource(ReplayScript {
        records: log.records.clone().into(),
        in_flight: None,
        replayed: 0,
    });
    world.spawn(DriverTask(pool.spawn(driver)));
    let mut schedule = Schedule::default();
    schedule.add_systems((dispatch_next_record, poll_next_record).chain());
    let total = log.records.len();
    let replayed = guarded("replay", || {
        tick(&mut world, &mut schedule);
        let script = world.resource::<ReplayScript>();
        (script.records.is_empty() && script.in_flight.is_none()).then_some(script.replayed)
    });
    assert_eq!(replayed, total, "every record dispatched and compared");
    println!("replay: {total} record(s) of {path} replayed in the golden's order");
    println!("bevy-bus-host: ok");
}

/// The golden's records still to dispatch, and the one in flight.
#[derive(Resource)]
struct ReplayScript {
    records: std::collections::VecDeque<rig_core::effect::EffectRecord>,
    in_flight: Option<(rig_core::effect::EffectRecord, Pending)>,
    replayed: usize,
}

fn dispatch_next_record(bus: Res<BusRes>, mut script: ResMut<ReplayScript>) {
    if script.in_flight.is_some() {
        return;
    }
    if let Some(record) = script.records.pop_front() {
        let pending = bus.0.dispatch(&record.key, record.kind.clone());
        script.in_flight = Some((record, pending));
    }
}

fn poll_next_record(mut script: ResMut<ReplayScript>) {
    let Some((record, pending)) = script.in_flight.as_mut() else {
        return;
    };
    let Some(outcome) = block_on(poll_once(pending)) else {
        return;
    };
    let got = serde_json::to_value(&outcome).expect("an outcome serializes");
    let want = serde_json::to_value(&record.outcome).expect("a record serializes");
    assert_eq!(
        got,
        want,
        "record {} (`{}`) replayed differently",
        record.id.as_u64(),
        record.key
    );
    script.replayed += 1;
    script.in_flight = None;
}

fn main() {
    let mut args = std::env::args().skip(1);
    if let Some(flag) = args.next() {
        assert_eq!(flag, "--replay", "the only flag is --replay <golden>");
        let path = args.next().expect("--replay takes the golden's path");
        replay(&path);
        return;
    }
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
        let (dispatcher, _registrar, driver) = Bus::channel_with(ServingPolicy {
            command_capacity: 1,
            ..ServingPolicy::default()
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
        let (dispatcher, _registrar, mut driver) = Bus::channel_with(ServingPolicy {
            stream_capacity: 4,
            ..ServingPolicy::default()
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
        // The stream lives in a component — `EffectStream: Send` on every
        // target — and a system probes it once per tick.
        let mut world = World::new();
        let stream: EffectStream = dispatcher.dispatch_stream(
            &HandlerKey::from("model"),
            EffectKind::Completion {
                request: request(),
                stream: true,
            },
        );
        let entity = world.spawn(InFlightStream(stream)).id();
        let mut received = 0;
        guarded("proof 6", || {
            let mut stream = world.get_mut::<InFlightStream>(entity).expect("the stream");
            match stream.0.poll_item() {
                Some(Some(Ok(_))) => {
                    received += 1;
                    (received >= 3).then_some(())
                }
                Some(Some(Err(report))) => panic!("proof 6: {report:?}"),
                Some(None) => panic!("proof 6: stream ended early"),
                None => None,
            }
        });
        // Despawn = drop = cancel.
        world.despawn(entity);
        guarded("proof 6", || {
            (counters.stream_cancelled.load(Ordering::SeqCst) == 1).then_some(())
        });
        assert!(
            counters.stream_sends.load(Ordering::SeqCst) < STREAM_CAP,
            "proof 6: the stream was cancelled, not capped"
        );
        println!("proof 6: handler observed cancellation after {received} events");
    }

    // ---- proof 8: a tool that is a system ----
    {
        let mailbox = WorldToolMailbox::default();
        let (dispatcher, _registrar, mut driver) = Bus::channel_with(ServingPolicy {
            serial_per_handler: true,
            ..ServingPolicy::default()
        });
        driver
            .register(
                "world-tool",
                WorldTool {
                    mailbox: mailbox.clone(),
                },
            )
            .expect("register");
        let _driver_task = pool.spawn(driver);
        let mut world = World::new();
        world.insert_resource(BusRes(dispatcher.clone()));
        world.insert_resource(mailbox.clone());
        world.insert_resource(WorldToolRuns {
            key: HandlerKey::from("world-tool"),
            in_flight: Vec::new(),
            answered: Vec::new(),
            seen: Vec::new(),
        });
        // World state the answering system reads: three marker entities.
        for _ in 0..3 {
            world.spawn(WorldState);
        }
        let mut schedule = Schedule::default();
        schedule.add_systems((dispatch_world_tool, answer_world_tool, poll_world_tool).chain());
        // Tick until the first dispatch is with the world and answered;
        // the second must still be unserved — the serial key is busy
        // until the detached sink answered, not until the handler returned.
        guarded("proof 8", || {
            schedule.run(&mut world);
            let runs = world.resource::<WorldToolRuns>();
            (runs.answered.len() == 1).then_some(())
        });
        fn custom_payload(
            outcome: &Result<Outcome, rig_core::error::ErrorReport>,
        ) -> serde_json::Value {
            match outcome {
                Ok(Outcome::Custom(payload)) => payload.clone(),
                other => panic!("proof 8: expected a custom outcome, got {other:?}"),
            }
        }
        {
            let runs = world.resource::<WorldToolRuns>();
            assert_eq!(runs.in_flight.len(), 1, "proof 8: the second waits");
            assert_eq!(
                custom_payload(&runs.answered[0]),
                serde_json::json!({ "answered": { "n": 0 }, "models": 3 }),
                "proof 8: answered from world state, in serve order"
            );
        }
        guarded("proof 8", || {
            schedule.run(&mut world);
            let runs = world.resource::<WorldToolRuns>();
            (runs.answered.len() == 2).then_some(())
        });
        let runs = world.resource::<WorldToolRuns>();
        assert_eq!(
            custom_payload(&runs.answered[1]),
            serde_json::json!({ "answered": { "n": 1 }, "models": 3 })
        );
        assert!(runs.in_flight.is_empty());
        println!("proof 8: a system answered a detached sink; the serial key waited for it");
    }

    // ---- proof 9: the driver restarts ----
    {
        let counters = Arc::new(Counters::default());
        let (dispatcher, _registrar, mut driver) = Bus::channel();
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
        let driver_entity = world.spawn(DriverTask(pool.spawn(driver))).id();
        let handle: ModelHandle = dispatcher
            .handle(&HandlerKey::from("model"))
            .expect("bound by family");
        let model_entity = world.spawn(Model(handle)).id();
        let mut schedule = Schedule::default();
        schedule.add_systems((dispatch_system, poll_system).chain());
        guarded("proof 9", || {
            tick(&mut world, &mut schedule);
            world.get::<Answered>(model_entity).map(|_| ())
        });
        assert_eq!(counters.unary_served.load(Ordering::SeqCst), 1);

        // The state transition: the driver task goes with its entity. The
        // executor drops the cancelled task on its own time, so the reopen
        // is retried across ticks until the old driver is really gone.
        world.despawn(driver_entity);
        let (registrar, new_driver) = guarded("proof 9", || Bus::reopen(&dispatcher).ok());
        assert!(
            dispatcher.descriptor(&HandlerKey::from("model")).is_none(),
            "proof 9: the handlers died with the driver"
        );
        let restarted = Arc::new(Counters::default());
        registrar
            .register(
                "model",
                MockModel {
                    counters: restarted.clone(),
                },
            )
            .expect("a fresh table");
        world.spawn(DriverTask(pool.spawn(new_driver)));
        // The same `Model(handle)` component, never re-inserted, completes
        // again: a handle is a key over the bus, not over a driver.
        world.entity_mut(model_entity).remove::<Answered>();
        let outcome = guarded("proof 9", || {
            tick(&mut world, &mut schedule);
            world
                .get::<Answered>(model_entity)
                .map(|answered| answered.0.clone())
        });
        assert!(outcome.is_ok(), "proof 9: {outcome:?}");
        assert_eq!(restarted.unary_served.load(Ordering::SeqCst), 1);
        assert_eq!(counters.unary_served.load(Ordering::SeqCst), 1);
        println!("proof 9: the driver restarted under a handle bound before the restart");
    }

    // ---- proof 10: scene round-trip ----
    {
        let counters = Arc::new(Counters::default());
        let (dispatcher, _registrar, mut driver) = Bus::channel();
        driver
            .register(
                "model",
                MockModel {
                    counters: counters.clone(),
                },
            )
            .expect("register");
        let _driver_task = pool.spawn(driver);
        // The world before the save: two effects, one in flight and one
        // never sent; a handle component.
        let mut world = World::new();
        let handle: ModelHandle = dispatcher
            .handle(&HandlerKey::from("model"))
            .expect("bound");
        world.spawn(Model(handle));
        for n in 0..2u64 {
            let effect = PendingEffect {
                key: HandlerKey::from("model"),
                kind: EffectKind::Custom {
                    kind: Arc::from("host:scene"),
                    payload: serde_json::json!({ "n": n }),
                },
            };
            world.spawn(effect);
        }
        // Save: the descriptors in one snapshot, every pending effect.
        let scene = Scene {
            handlers: dispatcher.descriptors(),
            effects: world
                .query::<&PendingEffect>()
                .iter(&world)
                .cloned()
                .collect(),
        };
        let text = serde_json::to_string(&scene).expect("proof 10: the scene serializes");
        drop(world);

        // Load: a fresh world and a fresh bus (a new process), no handler
        // registered yet — the handle is re-bound from the stored
        // descriptor first, the handlers arrive later.
        let loaded: Scene = serde_json::from_str(&text).expect("proof 10: the scene loads");
        let (dispatcher, _registrar, mut driver) = Bus::channel();
        let mut world = World::new();
        let stored = loaded
            .handlers
            .iter()
            .find(|d| d.key == HandlerKey::from("model"))
            .cloned()
            .expect("proof 10: the model's descriptor was stored");
        let handle: ModelHandle = ModelHandle::rebind(dispatcher.clone(), stored);
        world.spawn(Model(handle));
        let mut re_dispatched = Vec::new();
        for effect in loaded.effects {
            // Fresh ids: an `EffectId` is never persisted.
            let id = dispatcher.mint_id();
            re_dispatched.push((
                id,
                dispatcher.dispatch_with_id(id, &effect.key, effect.kind.clone()),
            ));
            world.spawn(effect);
        }
        assert_eq!(world.query::<&PendingEffect>().iter(&world).count(), 2);
        // Now the handlers: a mock that answers the custom kind.
        driver
            .register(
                "model",
                MockModel {
                    counters: counters.clone(),
                },
            )
            .expect("register");
        let _driver_task = pool.spawn(driver);
        let mut outcomes = Vec::new();
        for (id, mut pending) in re_dispatched {
            assert_eq!(pending.id(), id);
            let outcome = guarded("proof 10", || pending.poll_outcome());
            outcomes.push(outcome);
        }
        // The mock answers a custom kind with `HandlerUnavailable` — the
        // point is that the re-dispatched effects reached the handler that
        // arrived after the load, through the re-bound view's key.
        for outcome in &outcomes {
            let report = outcome
                .as_ref()
                .expect_err("proof 10: the mock refuses custom kinds");
            assert_eq!(
                report.kind,
                ErrorKind::HandlerUnavailable,
                "proof 10: {report:?}"
            );
            assert!(
                report.message.contains("mock model cannot serve"),
                "proof 10: {report:?}"
            );
        }
        let model = world
            .query::<&Model>()
            .single(&world)
            .expect("proof 10: one model");
        assert_eq!(model.0.key(), &HandlerKey::from("model"));
        println!(
            "proof 10: a scene of pending effects and descriptors round-tripped and re-dispatched"
        );
    }

    // ---- proof 4 (buffered): a dispatch dropped before the driver ran ----
    // The same-frame despawn: a system dispatches (one poll sends), the
    // entity is despawned before the driver task ever runs. The handler
    // must not be entered at all — one poll of a provider call would be an
    // HTTP request nobody wants.
    {
        let counters = Arc::new(Counters::default());
        let (dispatcher, _registrar, mut driver) = Bus::channel();
        driver
            .register(
                "model",
                MockModel {
                    counters: counters.clone(),
                },
            )
            .expect("register");
        let key = HandlerKey::from("model");
        let mut dropped = dispatcher.dispatch(
            &key,
            EffectKind::Completion {
                request: request(),
                stream: false,
            },
        );
        let _ = block_on(poll_once(&mut dropped));
        assert_eq!(dispatcher.buffered(), 1, "proof 4: sent, not yet taken");
        drop(dropped);
        let _driver_task = pool.spawn(driver);
        // A later dispatch on the same bus is served in order after the
        // cancelled one was taken and skipped, so its answer bounds the test.
        let mut later = dispatcher.dispatch(
            &key,
            EffectKind::Completion {
                request: request(),
                stream: false,
            },
        );
        let outcome = guarded("proof 4", || block_on(poll_once(&mut later)));
        assert!(outcome.is_ok(), "proof 4: {outcome:?}");
        assert_eq!(
            counters.unary_started.load(Ordering::SeqCst),
            1,
            "proof 4: the dispatch dropped before the driver ran was served"
        );
        println!("proof 4: a dispatch cancelled while buffered never entered its handler");
    }

    // ---- proof 11: a layer that suspends, decided by a system ----
    {
        let counters = Arc::new(Counters::default());
        let approvals = Approvals::default();
        let (dispatcher, _registrar, mut driver) = Bus::channel();
        let layered = rig_core::serve::ErasedHandler::new(MockModel {
            counters: counters.clone(),
        })
        .layered(Gate {
            approvals: approvals.clone(),
        });
        driver
            .register_erased(HandlerKey::from("model"), layered)
            .expect("register");
        assert_eq!(
            dispatcher
                .descriptor(&HandlerKey::from("model"))
                .expect("published")
                .layers,
            ["Gate"],
            "proof 11: the descriptor names the layer"
        );
        let recorder = rig_effect_log::EffectLogRecorder::new();
        driver.record_to(recorder.clone());
        let _driver_task = pool.spawn(driver);
        let mut world = World::new();
        world.insert_resource(approvals.clone());
        world.insert_resource(Decided::default());
        world.insert_resource(Ticks::default());
        for _ in 0..3 {
            world.spawn(WorldState);
        }
        let key = HandlerKey::from("model");
        let mut consumers = Vec::new();
        for _ in 0..3 {
            let pending = dispatcher.dispatch(
                &key,
                EffectKind::Completion {
                    request: request(),
                    stream: false,
                },
            );
            consumers.push(world.spawn(InFlight(pending)).id());
        }
        let mut schedule = Schedule::default();
        schedule.add_systems((decide_system, poll_system, see_denials).chain());
        guarded("proof 11", || {
            tick(&mut world, &mut schedule);
            let decided = world.resource::<Decided>();
            (decided.approved == 1
                && decided.denied == 1
                && decided.held.is_some()
                && world.get::<Answered>(consumers[0]).is_some()
                && world.get::<Answered>(consumers[1]).is_some())
            .then_some(())
        });
        assert!(
            world
                .get::<Answered>(consumers[0])
                .expect("answered")
                .0
                .is_ok(),
            "proof 11: the approved dispatch was served"
        );
        let denied = &world.get::<Answered>(consumers[1]).expect("answered").0;
        assert!(
            matches!(denied, Err(report) if report.kind == ErrorKind::Denied && !report.retryable),
            "proof 11: {denied:?}"
        );
        assert_eq!(
            world.resource::<Decided>().denied_seen,
            1,
            "proof 11: a system saw the denial land"
        );
        assert_eq!(
            counters.unary_served.load(Ordering::SeqCst),
            1,
            "proof 11: the denied one never reached the model"
        );
        // The third suspends still; its consumer goes: the world's slot
        // closes, and deciding it is a no-op, never a panic.
        assert!(world.get::<InFlight>(consumers[2]).is_some());
        world.despawn(consumers[2]);
        let held = world.resource_mut::<Decided>().held.take().expect("held");
        guarded("proof 11", || held.is_canceled().then_some(()));
        held.decide(Decision::Proceed);
        // The log: one record (the approved), none for the denial, and the
        // cancelled one resolved by the consumer's drop.
        guarded("proof 11", || (recorder.in_flight() == 0).then_some(()));
        let log = recorder.take();
        let kinds: Vec<_> = log
            .iter()
            .map(|record| match &record.outcome {
                Ok(_) => "ok".to_owned(),
                Err(report) => format!("{:?}", report.kind),
            })
            .collect();
        assert_eq!(kinds, ["ok", "Cancelled"], "proof 11: {kinds:?}");
        println!("proof 11: a suspended layer was approved, denied and cancelled by the world");
    }

    // ---- proof 12: a nested dispatch from a system-resolved tool ----
    {
        let counters = Arc::new(Counters::default());
        let mailbox = WorldToolMailbox::default();
        let (dispatcher, _registrar, mut driver) = Bus::channel_with(ServingPolicy {
            serial_per_handler: true,
            ..ServingPolicy::default()
        });
        driver
            .register(
                "world-tool",
                WorldTool {
                    mailbox: mailbox.clone(),
                },
            )
            .expect("register");
        driver
            .register(
                "model",
                MockModel {
                    counters: counters.clone(),
                },
            )
            .expect("register");
        let recorder = rig_effect_log::EffectLogRecorder::new();
        driver.record_to(recorder.clone());
        let _driver_task = pool.spawn(driver);
        let mut world = World::new();
        world.insert_resource(BusRes(dispatcher.clone()));
        world.insert_resource(mailbox.clone());
        world.insert_resource(Nested::default());
        world.insert_resource(Ticks::default());
        let key = HandlerKey::from("world-tool");
        let tool_call = |n: u64| EffectKind::Custom {
            kind: Arc::from("host:tool"),
            payload: serde_json::json!({ "n": n }),
        };
        let first = world
            .spawn(InFlight(dispatcher.dispatch(&key, tool_call(0))))
            .id();
        let mut schedule = Schedule::default();
        schedule.add_systems(
            (
                nest_system,
                poll_system,
                finish_nested,
                drop_cancelled_sinks,
            )
                .chain(),
        );
        let outcome = guarded("proof 12 (first answered)", || {
            tick(&mut world, &mut schedule);
            world
                .get::<Answered>(first)
                .map(|answered| answered.0.clone())
        });
        assert!(
            matches!(&outcome, Ok(Outcome::Custom(payload)) if payload["nested"] == "hello from the world"),
            "proof 12: {outcome:?}"
        );
        let refused = world
            .resource::<Nested>()
            .refused
            .clone()
            .expect("proof 12: the same-key dispatch was refused");
        assert_eq!(refused.kind, ErrorKind::Request);
        assert!(
            refused.message.contains("re-entrant"),
            "proof 12: {}",
            refused.message
        );
        // The chain in the record: the child names the tool's dispatch.
        guarded("proof 12 (records)", || {
            (recorder.in_flight() == 0).then_some(())
        });
        let log = recorder.take();
        assert_eq!(
            log.len(),
            2,
            "proof 12: the tool and its child, not the refused one"
        );
        assert_eq!(
            log[1].parent,
            Some(log[0].id),
            "proof 12: the child carries its parent"
        );
        assert_eq!(log[0].key, key);
        // The parent goes while the child is in flight: the child is
        // cancelled with it, and its record says so.
        counters.hold.store(true, Ordering::SeqCst);
        let second = world
            .spawn(InFlight(dispatcher.dispatch(&key, tool_call(1))))
            .id();
        guarded("proof 12 (second started)", || {
            tick(&mut world, &mut schedule);
            (counters.unary_started.load(Ordering::SeqCst) == 2).then_some(())
        });
        let children = world
            .query::<&ChildOf>()
            .iter(&world)
            .filter(|child_of| child_of.parent() == second)
            .count();
        assert_eq!(
            children, 1,
            "proof 12: the child's entity is a child of the tool's"
        );
        world.despawn(second);
        // The world keeps ticking: its system drops the closed sink, and
        // the child's cancel lands on the driver's next poll.
        guarded("proof 12 (cancelled records)", || {
            tick(&mut world, &mut schedule);
            (recorder.in_flight() == 0).then_some(())
        });
        let log = recorder.take();
        let kinds: Vec<_> = log
            .iter()
            .map(|record| match &record.outcome {
                Ok(_) => "ok".to_owned(),
                Err(report) => format!("{:?}", report.kind),
            })
            .collect();
        assert_eq!(kinds, ["Cancelled", "Cancelled"], "proof 12: {kinds:?}");
        assert_eq!(log[1].parent, Some(log[0].id));
        counters.hold.store(false, Ordering::SeqCst);
        assert!(
            world.resource::<Nested>().parents.is_empty(),
            "proof 12: the world dropped the cancelled tool's sink"
        );
        println!(
            "proof 12: a system-resolved tool nested a completion through its sink's dispatcher; the chain refused itself and cancelled together"
        );
    }

    // ---- proof 13: a checkpointed scene ----
    {
        let path = concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../../crates/rig-verify/fixtures/anthropic_tool_call_turn.effects.json"
        );
        let text = std::fs::read_to_string(path).expect("proof 13: the golden reads");
        let log: rig_effect_log::EffectLog =
            serde_json::from_str(&text).expect("proof 13: the golden decodes");
        // The world had performed the first record; the rest is pending
        // intent, stored as the checkpoint's state beside the tail.
        let pending: Vec<PendingEffect> = log
            .records
            .iter()
            .skip(1)
            .map(|record| PendingEffect {
                key: record.key.clone(),
                kind: record.kind.clone(),
            })
            .collect();
        let (checkpoint, tail) = log.checkpoint(
            1,
            serde_json::to_value(&pending).expect("proof 13: the intent serializes"),
        );
        let saved = (
            serde_json::to_string(&checkpoint).expect("proof 13: the checkpoint serializes"),
            serde_json::to_string(&tail).expect("proof 13: the tail serializes"),
        );
        // A fresh process image: the checkpoint and the tail loaded, the
        // continuation named, a fresh bus with replayers for the tail, the
        // effects re-dispatched from the position.
        let checkpoint: rig_effect_log::Checkpoint =
            serde_json::from_str(&saved.0).expect("proof 13: the checkpoint loads");
        let tail: rig_effect_log::EffectLog =
            serde_json::from_str(&saved.1).expect("proof 13: the tail loads");
        let refused = rig_effect_log::EffectLog::from_checkpoint(&checkpoint, log.clone())
            .expect_err("proof 13: the full log is not the tail");
        assert!(
            refused.message.starts_with("resume refused"),
            "proof 13: {}",
            refused.message
        );
        let continuation = rig_effect_log::EffectLog::from_checkpoint(&checkpoint, tail)
            .expect("proof 13: the tail follows");
        let (dispatcher, _registrar, mut driver) = Bus::channel();
        rig_effect_log::EffectLogReplayer::register_all(&continuation, &mut driver)
            .expect("proof 13: the tail's keys");
        let mut world = World::new();
        world.insert_resource(BusRes(dispatcher.clone()));
        let intent: Vec<PendingEffect> =
            serde_json::from_value(checkpoint.state.clone()).expect("proof 13: the intent loads");
        assert_eq!(intent.len(), continuation.len());
        world.insert_resource(ReplayScript {
            records: continuation.records.clone().into(),
            in_flight: None,
            replayed: 0,
        });
        for effect in intent {
            world.spawn(effect);
        }
        world.spawn(DriverTask(pool.spawn(driver)));
        let mut schedule = Schedule::default();
        schedule.add_systems((dispatch_next_record, poll_next_record).chain());
        let total = continuation.len();
        let replayed = guarded("proof 13", || {
            tick(&mut world, &mut schedule);
            let script = world.resource::<ReplayScript>();
            (script.records.is_empty() && script.in_flight.is_none()).then_some(script.replayed)
        });
        assert_eq!(replayed, total);
        assert_eq!(checkpoint.at, 1);
        println!(
            "proof 13: a checkpointed scene re-dispatched {total} record(s) from position {} and they resolved",
            checkpoint.at
        );
    }

    println!("bevy-bus-host: ok");
}
