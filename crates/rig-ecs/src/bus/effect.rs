//! Components for effect intent, ordering, identity, execution, and outcomes.
//!
//! ```
//! use rig_ecs::bus::Held;
//! let mut world = bevy_ecs::world::World::new();
//! let effect = world.spawn(Held).id();
//! assert!(world.get::<Held>(effect).is_some());
//! ```

use bevy_reflect::Reflect;
use std::sync::Arc;

use bevy_ecs::{lifecycle::HookContext, prelude::*, world::DeferredWorld};
use bevy_tasks::Task;
use rig_core::{
    effect::{CustomEffect, EffectId, EffectKind, Family, HandlerKey, Key, Outcome},
    error::ErrorReport,
    streaming::{StreamEvent, StreamEvents},
    tool::{PublishedContext, ToolContext},
};
use serde::{Deserialize, Serialize};

/// A dispatch intent containing its handler key and request.
/// Streaming completions accumulate in [`Streamed`]; other requests are unary.
/// Adding this component requires the bus's [`SeqCounter`] resource and stamps
/// [`Seq`] in spawn order, replacing any supplied sequence value.
#[derive(Component, Debug, Clone, Serialize, Deserialize, Reflect)]
#[require(Seq)]
#[component(on_add = stamp_seq)]
#[reflect(Component)]
pub struct PendingEffect {
    /// The handler key the effect is routed to.
    #[reflect(remote = crate::bus::reflect::HandlerKeyReflect)]
    pub key: HandlerKey,
    /// The effect.
    #[reflect(remote = crate::bus::reflect::EffectKindReflect)]
    pub kind: EffectKind,
}

impl PendingEffect {
    /// A pending effect for `key`.
    pub fn new(key: impl Into<HandlerKey>, kind: EffectKind) -> Self {
        Self {
            key: key.into(),
            kind,
        }
    }

    /// Wrap a typed request for `key`, returning any family conversion error.
    pub fn typed<F: Family>(key: &Key<F>, request: F::Request) -> Result<Self, ErrorReport> {
        Ok(Self {
            key: key.raw().clone(),
            kind: F::wrap(request)?,
        })
    }

    /// Build a custom request for `key` with `E::KIND`, returning a request error
    /// if its payload cannot be serialized.
    pub fn custom<E: CustomEffect>(
        key: impl Into<HandlerKey>,
        effect: &E,
    ) -> Result<Self, ErrorReport> {
        let payload = serde_json::to_value(effect).map_err(|error| {
            ErrorReport::new(
                rig_core::error::ErrorKind::Request,
                format!("the custom effect `{}` did not serialize: {error}", E::KIND),
            )
        })?;
        Ok(Self {
            key: key.into(),
            kind: EffectKind::Custom {
                kind: Arc::from(E::KIND),
                payload,
            },
        })
    }

    /// Whether this effect is dispatched as a stream.
    pub const fn is_stream(&self) -> bool {
        matches!(self.kind, EffectKind::Completion { stream: true, .. })
    }
}

/// Global dispatch order, stamped from [`SeqCounter`] whenever [`PendingEffect`]
/// is added, replacing manually supplied values. Scene loading preserves relative
/// order by spawning effects in saved order.
#[derive(
    Component,
    Debug,
    Clone,
    Copy,
    Default,
    PartialEq,
    Eq,
    PartialOrd,
    Ord,
    Hash,
    Serialize,
    Deserialize,
    Reflect,
)]
#[reflect(Component)]
pub struct Seq(pub u64);

/// The world's one dispatch-order counter (see [`Seq`]).
#[derive(Resource, Debug, Default, Reflect)]
#[reflect(Resource)]
pub struct SeqCounter(pub u64);

fn stamp_seq(mut world: DeferredWorld<'_>, context: HookContext) {
    let next = {
        let mut counter = world.resource_mut::<SeqCounter>();
        let next = counter.0;
        counter.0 += 1;
        next
    };
    if let Some(mut seq) = world.get_mut::<Seq>(context.entity) {
        seq.0 = next;
    } else {
        world.commands().entity(context.entity).insert(Seq(next));
    }
}

/// The world's one effect-id counter: `Dispatch` mints ids from it, strictly
/// increasing, unless the entity carries a [`Reserved`] id. Every
/// `Reserved` or `Issued` inserted anywhere (a scene load, a log load, a
/// host's own) bumps it past that id, so a minted id never collides with
/// a saved one. `u64::MAX` is the exhausted counter sentinel: fresh dispatch
/// then returns a request error without minting or recording an effect. The
/// maximum allocatable ID is `u64::MAX - 1`; invalid direct component insertion
/// saturates the counter rather than wrapping it.
#[derive(Resource, Debug, Default, Reflect)]
#[reflect(Resource)]
pub struct IdCounter(pub u64);

/// An id the effect must be dispatched under: a scene's saved id, a
/// replayed record's. Consumed by `Dispatch`, which bumps [`IdCounter`]
/// past it so a minted id never collides with a reserved one.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[component(on_insert = bump_ids_past_reserved)]
#[reflect(Component)]
pub struct Reserved(#[reflect(remote = crate::bus::reflect::EffectIdReflect)] pub EffectId);

fn bump_ids_past_reserved(mut world: DeferredWorld<'_>, context: HookContext) {
    if let Some(Reserved(id)) = world.get::<Reserved>(context.entity).copied() {
        let mut counter = world.resource_mut::<IdCounter>();
        counter.0 = counter.0.max(id.as_u64().saturating_add(1));
    }
}

fn bump_ids_past_issued(mut world: DeferredWorld<'_>, context: HookContext) {
    if let Some(Issued(id)) = world.get::<Issued>(context.entity).copied() {
        let mut counter = world.resource_mut::<IdCounter>();
        counter.0 = counter.0.max(id.as_u64().saturating_add(1));
    }
}

/// The id the effect was dispatched under. Inserted by `Dispatch` and kept
/// for the entity's life: what a child's record names as its `parent`, what
/// a scene saves.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[component(on_insert = bump_ids_past_issued)]
#[reflect(Component)]
pub struct Issued(#[reflect(remote = crate::bus::reflect::EffectIdReflect)] pub EffectId);

/// A pending effect a user system is still deciding about: `Dispatch`
/// leaves it alone until the marker is removed (approve), the effect is
/// denied (`EffectOutcome(Err(..))` inserted) or the entity despawned. The
/// world-side spelling of a layer that suspends in `before`.
/// Policies with independent ownership use [`super::acquire_hold`] and
/// [`super::release_hold`]; removing this marker directly bypasses all owners
/// (the batch's own marker follows: a call the batch held and a host
/// approved is dispatched and counted as active).
#[derive(Component, Debug, Clone, Copy, Default, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct Held;

/// The effect was taken: a handler is serving it. Present from `Dispatch`
/// until `settle` closes the record; for a stream, until the returned
/// stream reaches EOF. Carries the key it occupies so serial serving is a
/// query over this component. Never serialized: a scene stores intent.
#[derive(Component, Debug, Clone, Reflect)]
#[reflect(Component)]
pub struct InFlight {
    /// The key the effect occupies.
    #[reflect(remote = crate::bus::reflect::HandlerKeyReflect)]
    pub key: HandlerKey,
}

/// The marker of an initial handler task, held for the entity in the
/// world's [`Executions`] table.
#[derive(Component)]
pub struct Serving;

/// Delivery receiver and fold for a stream driven by an effect-owned worker
/// (held in [`Executions`]). Dropping the component drops the receiver, and
/// the worker returns at its next send.
#[derive(Component)]
pub struct Streaming {
    /// Bounded worker delivery, exclusively consumed by Collect.
    pub events: futures::channel::mpsc::Receiver<
        Result<rig_core::streaming::StreamEvent, rig_core::error::ErrorReport>,
    >,
    /// The shared core fold of delivered events.
    pub fold: rig_core::serve::StreamTap,
    /// Items consumed from this receiver, across collection passes.
    pub delivered: usize,
}

/// Drive one owned stream on the pool with bounded delivery to Collect;
/// every delivered item raises `wake`. Returns the receiver half and the
/// worker task.
fn spawn_stream_worker(
    mut stream: StreamEvents,
    capacity: usize,
    wake: super::plugin::Wake,
) -> (
    futures::channel::mpsc::Receiver<
        Result<rig_core::streaming::StreamEvent, rig_core::error::ErrorReport>,
    >,
    Task<()>,
) {
    use futures::{SinkExt, StreamExt};
    let (mut sender, events) = futures::channel::mpsc::channel(capacity.max(1));
    let task = bevy_tasks::IoTaskPool::get().spawn(async move {
        while let Some(item) = stream.next().await {
            if sender.send(item).await.is_err() {
                return;
            }
            wake.signal();
        }
        wake.signal();
    });
    (events, task)
}

/// The tasks behind [`Serving`] and [`Streaming`], rows in the non-send
/// [`Executions`] table: a task is `!Send` on wasm and cannot be a
/// component, and one storage keeps one code path on both targets.
#[derive(bevy_ecs::system::SystemParam)]
pub struct Tasks<'w> {
    executions: NonSendMut<'w, Executions>,
}

impl Tasks<'_> {
    /// The [`Serving`] component for `task`.
    pub fn serving(&mut self, entity: Entity, task: Task<rig_core::serve::Reply>) -> Serving {
        self.executions.tasks.insert(entity, task);
        Serving
    }

    /// Whether the serving task finished: its reply, once.
    pub fn poll(&mut self, entity: Entity) -> Option<rig_core::serve::Reply> {
        let task = self.executions.tasks.get_mut(&entity)?;
        let reply = bevy_tasks::futures::check_ready(task)?;
        self.executions.tasks.remove(&entity);
        Some(reply)
    }

    /// A [`Streaming`] component driving `stream` on the pool.
    pub fn streaming(
        &mut self,
        entity: Entity,
        stream: StreamEvents,
        capacity: usize,
        wake: super::plugin::Wake,
    ) -> Streaming {
        let (events, task) = spawn_stream_worker(stream, capacity, wake);
        self.executions.streams.insert(entity, task);
        Streaming {
            events,
            fold: rig_core::serve::StreamTap::new(),
            delivered: 0,
        }
    }

    /// The stream worker of `entity` is done with: dropped now.
    pub fn forget_stream(&mut self, entity: Entity) {
        self.executions.streams.remove(&entity);
    }

    /// The tasks of `world`, for an exclusive system. `None` when the bus
    /// is not installed (the table is missing).
    pub fn with<T>(world: &mut World, f: impl FnOnce(&mut Tasks<'_>) -> T) -> Option<T> {
        let mut state = bevy_ecs::system::SystemState::<Tasks>::new(world);
        let mut tasks = state.get_mut(world).ok()?;
        Some(f(&mut tasks))
    }
}

/// The tasks of in-flight effects, indexed by their effect entity and
/// dropped by [`drop_execution`] when the effect leaves flight. Non-send
/// because a task is `!Send` on wasm.
#[derive(Default)]
pub struct Executions {
    /// Initial tasks, indexed by their in-flight effect entity.
    pub tasks: std::collections::HashMap<Entity, Task<rig_core::serve::Reply>>,
    /// Stream workers, cancelled when the effect leaves flight.
    pub streams: std::collections::HashMap<Entity, Task<()>>,
}

/// Remove owned execution immediately when an effect leaves flight.
pub fn drop_execution(removed: On<Remove, InFlight>, mut executions: NonSendMut<Executions>) {
    let entity = removed.event().entity;
    executions.tasks.remove(&entity);
    executions.streams.remove(&entity);
}

/// Accumulated stream events, error positions, text, and the first folded outcome.
/// [`EffectOutcome`] is published at EOF, retaining the serial key through stream
/// completion. Observe `Changed<Streamed>` for newly collected data.
///
/// Live collection allows 64 queue checks per effect and 4,096 per pass.
/// Exact replay grouping requires delivery metadata and kept items; event bytes
/// alone establish order, and folded recordings provide only the final answer.
/// Checkpoints restore completed streams but reject unfinished streams with
/// delivered progress because no continuation cursor is saved.
#[derive(Component, Debug, Default, Clone, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct Streamed {
    /// Every error item and its zero-based position among all stream items,
    /// including errors after the first terminal outcome. Live observation is
    /// independent of recorder event retention; `outcome` remains the first fold.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    #[reflect(remote = crate::bus::reflect::StreamErrorsReflect)]
    pub errors: Vec<(usize, ErrorReport)>,
    /// Every event, in order.
    #[reflect(remote = crate::bus::reflect::StreamEventsReflect)]
    pub events: Vec<StreamEvent>,
    /// The text deltas concatenated.
    pub text: String,
    /// The fold's outcome at the terminal record, or the error that ended
    /// the stream.
    #[reflect(remote = crate::bus::reflect::StreamedOutcomeReflect)]
    pub outcome: Option<Result<Outcome, ErrorReport>>,
}

/// The answer. Inserted by `Collect` when a handler's task or stream
/// finished, by a `Gate` system that denies, or by a `Judge` system that
/// replaces. Serde, so a scene keeps answered effects answered.
#[derive(Component, Debug, Clone, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct EffectOutcome(
    #[reflect(remote = crate::bus::reflect::OutcomeReflect)] pub Result<Outcome, ErrorReport>,
);

/// An open world's submitted answer. Insert with [`WorldOutcome::new`];
/// `Collect` publishes it as [`EffectOutcome`] in submission order. Typed
/// [`Answer<E>`] uses the same path. Submission after Collect is visible next
/// pass, so policy observes the same boundaries live and during replay.
/// Like a ready task, this inbox is transient; save after collection to retain
/// its answer in a scene. Applications observe `EffectOutcome`, not this inbox.
#[derive(Component, Debug)]
#[component(on_add = stamp_world_outcome)]
pub struct WorldOutcome {
    /// The submitted answer, published unchanged by the collector.
    pub outcome: Result<Outcome, ErrorReport>,
    order: u64,
}

impl WorldOutcome {
    /// Submit one answer for an in-flight world-served effect.
    pub fn new(outcome: Result<Outcome, ErrorReport>) -> Self {
        Self { outcome, order: 0 }
    }

    /// Submission order within this world, stamped on component insertion.
    pub fn order(&self) -> u64 {
        self.order
    }
}

/// Monotonic submission order for the world's answer inbox.
#[derive(Resource, Default)]
pub struct WorldOutcomeCounter(pub u64);

fn stamp_world_outcome(mut world: DeferredWorld<'_>, context: HookContext) {
    let order = {
        let mut counter = world.resource_mut::<WorldOutcomeCounter>();
        let order = counter.0;
        counter.0 += 1;
        order
    };
    if let Some(mut outcome) = world.get_mut::<WorldOutcome>(context.entity) {
        outcome.order = order;
    }
}

impl EffectOutcome {
    /// Return the family's typed answer, propagating a recorded or conversion error.
    pub fn typed<F: Family>(&self) -> Result<F::Answer, ErrorReport> {
        F::unwrap(self.0.clone()?)
    }

    /// Deserialize a custom answer as `E::Answer`, returning recorded errors or a
    /// response error for a different outcome family or invalid payload.
    pub fn custom<E: CustomEffect>(&self) -> Result<E::Answer, ErrorReport> {
        match self.0.clone()? {
            Outcome::Custom { payload: value } => serde_json::from_value(value).map_err(|error| {
                ErrorReport::new(
                    rig_core::error::ErrorKind::Response,
                    format!("the answer to `{}` did not deserialize: {error}", E::KIND),
                )
            }),
            other => Err(ErrorReport::new(
                rig_core::error::ErrorKind::Response,
                format!(
                    "`{}` was answered with a {} outcome",
                    E::KIND,
                    other.family()
                ),
            )),
        }
    }
}

/// Serializable inbound tool context attached beside the effect payload.
/// Dispatch supplies it to the handler; absent inputs use an empty context.
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct ToolInputs(#[reflect(remote = crate::bus::reflect::ToolContextReflect)] pub ToolContext);

/// What the tool published into its context: read off the dispatch's
/// [`PublishedContext`] when the outcome lands (`Collect`), or inserted by
/// the system that answers an open tool key. Data, beside the outcome.
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct ToolOutputs(
    #[reflect(remote = crate::bus::reflect::ToolContextReflect)] pub ToolContext,
);

/// The slot a task-served tool call publishes into, shared with its dispatch context
/// for the length of the call; `Collect` reads it into [`ToolOutputs`].
/// Never serialized: in-flight state.
#[derive(Component, Clone)]
pub struct Publishing(pub Arc<PublishedContext>);

/// A stable serializable program scope, not a runtime handle.
/// Dispatch records the nearest scope along `ChildOf`, checking the effect first.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize, Reflect)]
#[reflect(Component)]
pub struct Scope(pub String);

/// A [`CustomEffect`] a system can serve: its payload and its answer live
/// in components, so both are `Send + Sync` on every target (plain data
/// is). Blanket-implemented.
pub trait WorldEffect: CustomEffect + Send + Sync {
    /// The answer, as a component field: `E::Answer`.
    type Reply: Serialize + serde::de::DeserializeOwned + Send + Sync + 'static;
}

impl<E> WorldEffect for E
where
    E: CustomEffect + Send + Sync,
    E::Answer: Send + Sync,
{
    type Reply = E::Answer;
}

/// A custom effect a system answers: what a [`WorldHandler`](super::WorldHandler)
/// dispatch lands as on the effect entity, deserialized. A user system with
/// any `World` access reads it and inserts [`Answer<E>`].
#[derive(Component, Debug, Clone)]
pub struct Asked<E: WorldEffect>(pub E);

/// A system's answer to an [`Asked<E>`]: inserting it resolves the effect
/// (the plugin serializes it into the [`EffectOutcome`]).
#[derive(Component, Debug, Clone)]
pub struct Answer<E: WorldEffect>(pub E::Reply);

/// A typed key as a component: the typed view over a handler entity's key,
/// placed wherever a system wants it. `Send + Sync` on every target.
#[derive(Component, Debug, Clone, Serialize, Deserialize)]
pub struct Typed<F: Family>(pub Key<F>);

impl<F: Family> Typed<F> {
    /// Build a pending effect for this key, returning any family conversion error.
    pub fn pending(&self, request: F::Request) -> Result<PendingEffect, ErrorReport> {
        PendingEffect::typed(&self.0, request)
    }

    /// The key.
    pub fn key(&self) -> &HandlerKey {
        self.0.raw()
    }
}

// Every component a system holds is `Send + Sync` on every target: the
// tasks, owned streams, and erased handlers live elsewhere.
const _: () = {
    const fn assert_send_sync<T: Send + Sync + 'static>() {}
    assert_send_sync::<PendingEffect>();
    assert_send_sync::<Seq>();
    assert_send_sync::<Issued>();
    assert_send_sync::<Reserved>();
    assert_send_sync::<Held>();
    assert_send_sync::<InFlight>();
    assert_send_sync::<Streamed>();
    assert_send_sync::<EffectOutcome>();
    assert_send_sync::<Scope>();
    assert_send_sync::<ToolInputs>();
    assert_send_sync::<ToolOutputs>();
    assert_send_sync::<Publishing>();
    assert_send_sync::<Serving>();
    assert_send_sync::<Streaming>();
    assert_send_sync::<Typed<rig_core::effect::family::Completion>>();
};
