//! The effect entity's components: intent, order, identity, state, answer.

use std::sync::Arc;

use bevy_ecs::{lifecycle::HookContext, prelude::*, world::DeferredWorld};
use bevy_tasks::Task;
use futures::channel::mpsc;
use rig_core::{
    effect::{CustomEffect, EffectId, EffectKind, Family, HandlerKey, Key, Outcome},
    error::ErrorReport,
    streaming::StreamEvent,
    tool::{PublishedContext, ToolContext},
};
use serde::{Deserialize, Serialize};

/// The intent: what to dispatch, and to whom. Spawn one to dispatch; the
/// plugin does the rest. Serde, so a scene stores it. A `Completion { stream:
/// true }` kind is dispatched as a stream (the answer accumulates in
/// [`Streamed`]); every other kind unary.
///
/// Requires [`Seq`], stamped on add from the world's [`SeqCounter`] in spawn
/// order, so a plain `commands.spawn(PendingEffect { .. })` is deterministic
/// with no user effort.
#[derive(Component, Debug, Clone, Serialize, Deserialize)]
#[require(Seq)]
#[component(on_add = stamp_seq)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct PendingEffect {
    /// The handler key the effect is routed to.
    #[cfg_attr(feature = "reflect", reflect(remote = crate::bus::reflect::HandlerKeyReflect))]
    pub key: HandlerKey,
    /// The effect.
    #[cfg_attr(feature = "reflect", reflect(remote = crate::bus::reflect::EffectKindReflect))]
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

    /// A typed request for a typed key: the family wraps it.
    pub fn typed<F: Family>(key: &Key<F>, request: F::Request) -> Result<Self, ErrorReport> {
        Ok(Self {
            key: key.raw().clone(),
            kind: F::wrap(request)?,
        })
    }

    /// A custom effect for `key`: `E::KIND` and its serde payload.
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

/// The dispatch order: `Dispatch` takes pending effects in ascending `Seq`,
/// the total order the log reproduces. **Global and reserved**: the
/// component hook stamps it from [`SeqCounter`] on every `PendingEffect`
/// add and overwrites any value set by hand — a per-spawner counter would
/// collide across spawners and break the order the log asserts. A scene
/// loads its effects in their saved order, so their new `Seq`s keep it.
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
)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct Seq(pub u64);

/// The world's one dispatch-order counter (see [`Seq`]).
#[derive(Resource, Debug, Default)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Resource))]
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
#[derive(Resource, Debug, Default)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Resource))]
pub struct IdCounter(pub u64);

/// An id the effect must be dispatched under: a scene's saved id, a
/// replayed record's. Consumed by `Dispatch`, which bumps [`IdCounter`]
/// past it so a minted id never collides with a reserved one.
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[component(on_insert = bump_ids_past_reserved)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct Reserved(
    #[cfg_attr(feature = "reflect", reflect(remote = crate::bus::reflect::EffectIdReflect))]
    pub  EffectId,
);

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
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[component(on_insert = bump_ids_past_issued)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct Issued(
    #[cfg_attr(feature = "reflect", reflect(remote = crate::bus::reflect::EffectIdReflect))]
    pub  EffectId,
);

/// A pending effect a user system is still deciding about: `Dispatch`
/// leaves it alone until the marker is removed (approve), the effect is
/// denied (`EffectOutcome(Err(..))` inserted) or the entity despawned. The
/// world-side spelling of a layer that suspends in `before`.
#[derive(Component, Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct Held;

/// The effect was taken: a handler is serving it. Present from `Dispatch`
/// until `settle` closes the record — for a stream, until the handler's
/// channel closes. Carries the key it occupies so serial serving is a
/// query over this component. Never serialized: a scene stores intent.
#[derive(Component, Debug, Clone)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct InFlight {
    /// The key the effect occupies.
    #[cfg_attr(feature = "reflect", reflect(remote = crate::bus::reflect::HandlerKeyReflect))]
    pub key: HandlerKey,
}

/// The handler's future, running on the task pool, held in the effect
/// entity. Its output is the outcome the handler resolved. Dropping it —
/// a despawn — cancels the handler and its sink.
#[derive(Component)]
pub struct Serving(pub Task<Result<Outcome, ErrorReport>>);

/// A streaming handler's future and the channel it sends on, held in the
/// effect entity; `Collect` drains the channel into [`Streamed`] each pass
/// without blocking. Dropping it cancels the handler.
#[derive(Component)]
pub struct Streaming {
    /// The handler's future.
    pub task: Task<()>,
    /// What the handler has sent and `Collect` has not yet taken.
    pub events: mpsc::Receiver<Result<StreamEvent, ErrorReport>>,
    /// The fold of the events into the outcome (rig-core's one fold).
    pub fold: rig_core::serve::StreamTap,
}

/// The per-tick fold of a stream: every event so far (`Changed<Streamed>`
/// is the delta signal), the text so far, and the folded outcome once the
/// terminal record — or an error — arrived. The [`EffectOutcome`] lands
/// when the handler's channel closes, so a serial key stays busy until the
/// handler is done, as it does on rig-agent's bus.
///
/// A collection pass can deliver several events. With the `replay` feature,
/// `Replay::policy_visible()` preserves recorded delivery batches when the
/// recorder kept event bytes. Keeping bytes alone in a driver without batch
/// tracking promises event order only; folded recordings supply a final
/// answer, not these partial states. Completed scenes restore all three
/// fields. Loading an unfinished stream with delivered progress is refused
/// because the scene has no cursor with which to resume after that prefix.
#[derive(Component, Debug, Default, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct Streamed {
    /// Every event, in order.
    #[cfg_attr(feature = "reflect", reflect(remote = crate::bus::reflect::StreamEventsReflect))]
    pub events: Vec<StreamEvent>,
    /// The text deltas concatenated.
    pub text: String,
    /// The fold's outcome at the terminal record, or the error that ended
    /// the stream.
    #[cfg_attr(feature = "reflect", reflect(remote = crate::bus::reflect::StreamedOutcomeReflect))]
    pub outcome: Option<Result<Outcome, ErrorReport>>,
}

/// The answer. Inserted by `Collect` when a handler's task or channel
/// finished, by a `Gate` system that denies, or by a `Judge` system that
/// replaces. Serde, so a scene keeps answered effects answered.
#[derive(Component, Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct EffectOutcome(
    #[cfg_attr(feature = "reflect", reflect(remote = crate::bus::reflect::OutcomeReflect))]
    pub  Result<Outcome, ErrorReport>,
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
    /// The answer as the family's typed answer.
    pub fn typed<F: Family>(&self) -> Result<F::Answer, ErrorReport> {
        F::unwrap(self.0.clone()?)
    }

    /// A custom answer, as `E::Answer`.
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

/// The context a tool call runs with: the inbound values the driver hands
/// the tool beside the effect (format 5: never in it), as data on the
/// effect entity. `Dispatch` attaches it to the handler's sink; absent,
/// the tool runs under an empty context. A scene saves it.
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct ToolInputs(
    #[cfg_attr(feature = "reflect", reflect(remote = crate::bus::reflect::ToolContextReflect))]
    pub ToolContext,
);

/// What the tool published into its context: read off the sink's
/// [`PublishedContext`] when the outcome lands (`Collect`), or inserted by
/// the system that answers an open tool key. Data, beside the outcome.
#[derive(Component, Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct ToolOutputs(
    #[cfg_attr(feature = "reflect", reflect(remote = crate::bus::reflect::ToolContextReflect))]
    pub ToolContext,
);

/// The slot a task-served tool call publishes into, shared with its sink
/// for the length of the call; `Collect` reads it into [`ToolOutputs`].
/// Never serialized: in-flight state.
#[derive(Component, Clone)]
pub struct Publishing(pub Arc<PublishedContext>);

/// The scope of a program, as data: a stable serde id of the run or agent
/// an effect entity descends from — never a runtime handle. `Dispatch` reads
/// the nearest `Scope` up the `ChildOf` chain (the entity's own first) into
/// the record's `scope`, so one log written by several programs in one
/// world reads per program.
#[derive(Component, Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
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
    /// A pending effect for this key.
    pub fn pending(&self, request: F::Request) -> Result<PendingEffect, ErrorReport> {
        PendingEffect::typed(&self.0, request)
    }

    /// The key.
    pub fn key(&self) -> &HandlerKey {
        self.0.raw()
    }
}

// Every component a system holds is `Send + Sync` on every target: the
// task, the channel, the erased handler live elsewhere.
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
    assert_send_sync::<Typed<rig_core::effect::family::Completion>>();
};
