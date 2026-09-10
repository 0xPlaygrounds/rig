//! The world's [`Witness`]: the analysis sink the bus's own systems feed at
//! their decision sites, and the seam a host system emits through.
//!
//! Beside [`Recording`](super::Recording), not inside it: the record is the
//! replay oracle (what a handler served and answered); a witness sees the
//! program around it — an intent held and released, denied before flight,
//! refused by the driver, replaced after the
//! record closed, cancelled in flight, a stream that ended before its
//! terminal record. Every fact is emitted where it happens, by the system
//! that owns it, or by a lifecycle observer for the transitions a user
//! system makes with a plain component write (`Held`, a `Gate` denial, a
//! `Judge` replacement); those carry [`Emitter::unknown`] unless the policy
//! names itself through [`Witnessing::emit`].
//!
//! Install with [`Witnessing::install`], once per world: a later call
//! replaces the sink and installs no second set of observers. Absent, nothing
//! here runs: every emission site reads `Option<Res<Witnessing>>`, and the
//! observers return at once, so a world without a witness pays a resource
//! lookup per site.
//!
//! Two limits a reader of the trace must know. A scene load restores
//! `Held` and pre-dispatch outcomes by component insert, so a witnessed
//! world observes the restored gate state again (as `Held` / `Denied`) at
//! the load; the facts are true of the restored world, not new decisions.
//! And a `Judge` that rewrites an outcome in place through `Mut<EffectOutcome>`
//! raises no lifecycle event: only an insert is observed, so such a policy
//! must emit for itself.

use std::sync::Arc;

use bevy_ecs::{
    lifecycle::{Despawn, Insert, Remove},
    prelude::*,
};
use rig_core::observe::{
    Action, Emitter, Observation, OutcomeSummary, Reason, Stage, Subject, Witness,
};

use super::effect::{EffectOutcome, Held, InFlight, Issued, PendingEffect, Scope, Seq};

/// The bus's emitter name.
pub const BUS_EMITTER: &str = "rig-ecs/bus";

/// The bus as an emitter: its name and crate version.
pub fn bus_emitter() -> Emitter {
    Emitter::versioned(BUS_EMITTER, env!("CARGO_PKG_VERSION"))
}

/// How far up a `ChildOf` chain a subject is resolved: a host bug that
/// makes a cycle ends here, not in an observer that never returns.
const WALK_LIMIT: usize = 4096;

/// The world's witness. Cloning shares the sink. `Send + Sync` on every
/// target, as a resource must be (the sink trait's compat bounds are
/// no-ops on browser wasm).
#[derive(Resource, Clone)]
pub struct Witnessing(Arc<dyn Witness + Send + Sync>);

/// Host-owned logical provider operation for one pending effect.
///
/// Attach before dispatch. The bus binds the current effect/scope/parent
/// subject while retaining this operation's witness and HTTP send counter.
/// Reuse the context only for retries of the same logical call; unrelated
/// calls (including identical concurrent requests) need distinct contexts.
/// Runtime-only: scene/effect-log replay does not serialize this identity.
#[derive(Component, Clone, Debug)]
pub struct AdapterOperation {
    /// Shared observation context for the logical call.
    pub context: rig_core::observe::AdapterContext,
    /// One-based host dispatch ordinal, independent of HTTP send ordinals.
    pub host_attempt: std::num::NonZeroU64,
}

/// The observers were installed: a second `install` only replaces the sink.
#[derive(Resource, Debug, Default)]
struct WitnessObserversInstalled;

impl Witnessing {
    /// A witness over `sink`.
    pub fn new(sink: impl Witness + Send + Sync) -> Self {
        Self(Arc::new(sink))
    }

    /// A witness over a shared sink.
    pub fn shared(sink: Arc<dyn Witness + Send + Sync>) -> Self {
        Self(sink)
    }

    /// Install `sink` on `world`: the resource and, the first time, the
    /// lifecycle observers for the transitions user systems make by
    /// component write. Calling again replaces the sink only.
    pub fn install(world: &mut World, sink: impl Witness + Send + Sync) {
        world.insert_resource(Self::new(sink));
        if world.contains_resource::<WitnessObserversInstalled>() {
            return;
        }
        world.init_resource::<WitnessObserversInstalled>();
        world.init_resource::<Despawning>();
        world.add_observer(observe_despawn);
        world.add_observer(observe_held);
        world.add_observer(observe_released);
        world.add_observer(observe_hold_transition);
        world.add_observer(observe_preflight_outcome);
        world.add_observer(observe_outcome_replaced);
        world.add_observer(observe_pending_despawned);
    }

    /// One fact.
    pub fn observe(&self, observation: Observation) {
        self.0.observe(observation);
    }

    /// One fact, assembled: the seam a host policy emits through, naming
    /// itself as the emitter.
    pub fn emit(&self, subject: Subject, stage: Stage, emitter: Emitter, action: Action) {
        self.0
            .observe(Observation::new(subject, stage, emitter, action));
    }

    /// The sink, for a host that reads it back.
    pub fn sink(&self) -> &Arc<dyn Witness + Send + Sync> {
        &self.0
    }
}

/// The driver refused this intent before any handler served it: the
/// pre-flight outcome observer leaves it to the driver's own emission.
/// Runtime-only; a scene does not save it.
#[derive(Component, Debug, Clone, Copy, Default)]
pub struct Refused;

/// What the world saw for a settled effect, written by `settle` when a
/// witness is installed: the outcome after any layer verdict, as its
/// summary and a fingerprint of its whole value, so a later `Judge` insert
/// is observed as a replacement even within one family. Runtime-only.
#[derive(Component, Debug, Clone)]
pub struct SeenOutcome {
    /// The outcome, in brief.
    pub summary: OutcomeSummary,
    /// [`fingerprint`] of the whole outcome.
    pub fingerprint: u64,
}

impl SeenOutcome {
    /// The seen outcome of `outcome`.
    pub fn of(outcome: &Result<rig_core::effect::Outcome, rig_core::error::ErrorReport>) -> Self {
        Self {
            summary: OutcomeSummary::of(outcome),
            fingerprint: fingerprint(outcome),
        }
    }
}

/// A 64-bit FNV-1a hash of a value's serde form, for in-process equality
/// of two outcomes without keeping either. Not a stable cross-process id.
pub fn fingerprint<T: serde::Serialize>(value: &T) -> u64 {
    let mut hash: u64 = 0xcbf2_9ce4_8422_2325;
    if let Ok(bytes) = serde_json::to_vec(value) {
        for byte in bytes {
            hash ^= u64::from(byte);
            hash = hash.wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    hash
}

/// Held entities a despawn is in progress for, so the `Held` the despawn
/// removes is not observed as a release. Only held entities enter, and the
/// release observer removes them, so the set is empty between despawns.
#[derive(Resource, Debug, Default)]
pub struct Despawning(std::collections::HashSet<Entity>);

/// The dispatcher's view of the witness: the sink and the subjects.
/// One system parameter, so the
/// dispatch system stays within Bevy's parameter limit.
#[derive(bevy_ecs::system::SystemParam)]
pub struct DispatchWitness<'w, 's> {
    /// The sink, when installed.
    pub witness: Option<Res<'w, Witnessing>>,
    /// Subjects of effect entities.
    pub subjects: Subjects<'w, 's>,
}

/// The tree walk a subject needs: order, id, parents, scopes. Without the
/// intent query, so a system that holds `&mut PendingEffect` can still name
/// a subject through [`SubjectWalk::of_intent`].
#[derive(bevy_ecs::system::SystemParam)]
pub struct SubjectWalk<'w, 's> {
    seqs: Query<'w, 's, &'static Seq>,
    issued: Query<'w, 's, &'static Issued>,
    parents: Query<'w, 's, &'static ChildOf>,
    scopes: Query<'w, 's, &'static Scope>,
}

impl SubjectWalk<'_, '_> {
    /// The subject of an intent whose key and family the caller holds.
    pub fn of_intent(
        &self,
        entity: Entity,
        key: &rig_core::effect::HandlerKey,
        family: rig_core::effect::EffectFamily,
    ) -> Subject {
        let mut subject = self.of_entity(entity);
        subject.key = Some(key.clone());
        subject.family = Some(family);
        subject
    }

    /// The subject of `entity` without its intent: its scope (the nearest
    /// [`Scope`] up the tree, its own first), its dispatch order, its id
    /// once issued, its parent's id.
    pub fn of_entity(&self, entity: Entity) -> Subject {
        let mut subject = Subject {
            order: self.seqs.get(entity).ok().map(|seq| seq.0),
            effect: self.issued.get(entity).ok().map(|issued| issued.0),
            ..Subject::default()
        };
        let mut current = entity;
        for _ in 0..WALK_LIMIT {
            if subject.scope.is_none()
                && let Ok(Scope(scope)) = self.scopes.get(current)
            {
                subject.scope = Some(scope.clone());
            }
            match self.parents.get(current) {
                Ok(parent) => {
                    current = parent.parent();
                    if subject.parent.is_none()
                        && let Ok(Issued(id)) = self.issued.get(current)
                    {
                        subject.parent = Some(*id);
                    }
                }
                Err(_) => break,
            }
        }
        subject
    }

    /// The subject of a program entity (a run): its scope only.
    pub fn of_scope(&self, entity: Entity) -> Subject {
        let mut current = entity;
        for _ in 0..WALK_LIMIT {
            if let Ok(Scope(scope)) = self.scopes.get(current) {
                return Subject::scoped(scope.clone());
            }
            match self.parents.get(current) {
                Ok(parent) => current = parent.parent(),
                Err(_) => break,
            }
        }
        Subject::default()
    }
}

/// What the subject of an effect entity is read from: the walk and the
/// intent.
#[derive(bevy_ecs::system::SystemParam)]
pub struct Subjects<'w, 's> {
    walk: SubjectWalk<'w, 's>,
    pending: Query<'w, 's, &'static PendingEffect>,
}

impl Subjects<'_, '_> {
    /// The subject of `entity`: its scope, dispatch order, id once issued,
    /// parent's id, key and family.
    pub fn of(&self, entity: Entity) -> Subject {
        let mut subject = self.walk.of_entity(entity);
        if let Ok(pending) = self.pending.get(entity) {
            subject.key = Some(pending.key.clone());
            subject.family = Some(pending.kind.family());
        }
        subject
    }

    /// The subject of a program entity (a run): its scope only.
    pub fn of_scope(&self, entity: Entity) -> Subject {
        self.walk.of_scope(entity)
    }
}

/// A despawn of a held intent began (lifecycle events are per component,
/// so this fires for held entities only): the `Held` removal that follows
/// is the despawn, not a release.
fn observe_despawn(despawned: On<Despawn, Held>, mut despawning: ResMut<Despawning>) {
    despawning.0.insert(despawned.event().entity);
}

/// An intent was held before dispatch (a `Gate` system or the agent's
/// batch release wrote `Held`). The holder is unknown unless it emits.
fn observe_held(
    added: On<bevy_ecs::lifecycle::Add, Held>,
    subjects: Subjects,
    witness: Option<Res<Witnessing>>,
    owned: Query<(), With<super::HoldOwners>>,
) {
    if owned.contains(added.event().entity) {
        return;
    }
    let Some(witness) = witness else {
        return;
    };
    witness.emit(
        subjects.of(added.event().entity),
        Stage::Gate,
        Emitter::unknown(),
        Action::Held {
            reason: Reason::unknown(),
        },
    );
}

/// A held intent was released to dispatch. A hold removed by a denial is
/// the denial; one removed by a despawn is the despawn (observed by
/// [`observe_pending_despawned`]).
fn observe_released(
    removed: On<Remove, Held>,
    outcomes: Query<(), With<EffectOutcome>>,
    subjects: Subjects,
    mut despawning: ResMut<Despawning>,
    witness: Option<Res<Witnessing>>,
    owned: Query<(), With<super::HoldOwners>>,
) {
    let entity = removed.event().entity;
    let despawned = despawning.0.remove(&entity);
    let Some(witness) = witness else {
        return;
    };
    if outcomes.get(entity).is_ok() || despawned || owned.contains(entity) {
        return;
    }
    witness.emit(
        subjects.of(entity),
        Stage::Gate,
        Emitter::unknown(),
        Action::Released,
    );
}

fn observe_hold_transition(
    event: On<super::hold::HoldTransition>,
    outcomes: Query<(), With<EffectOutcome>>,
    subjects: Subjects,
    witness: Option<Res<Witnessing>>,
) {
    let event = event.event();
    if !event.acquired && outcomes.contains(event.entity) {
        return;
    }
    if let Some(witness) = witness {
        witness.emit(
            subjects.of(event.entity),
            Stage::Gate,
            event.owner.clone(),
            if event.acquired {
                Action::Held {
                    reason: Reason::unknown(),
                }
            } else {
                Action::Released
            },
        );
    }
}

/// What the pre-flight outcome observer reads: the outcome, whether the
/// intent was issued, whether the driver refused it itself.
pub type PreflightView = (&'static EffectOutcome, Has<Issued>, Has<Refused>);

/// An outcome landed on an intent never issued: a `Gate` denial (or a
/// driver refusal, which the driver emitted itself and marked `Refused`).
fn observe_preflight_outcome(
    added: On<bevy_ecs::lifecycle::Add, EffectOutcome>,
    outcomes: Query<PreflightView, With<PendingEffect>>,
    subjects: Subjects,
    witness: Option<Res<Witnessing>>,
) {
    let Some(witness) = witness else {
        return;
    };
    let entity = added.event().entity;
    let Ok((outcome, issued, refused)) = outcomes.get(entity) else {
        return;
    };
    if issued || refused {
        return;
    }
    let action = match &outcome.0 {
        Err(report) => Action::Denied {
            reason: Reason::from_report(report),
        },
        // A gate that answers an intent itself: observed as a replacement
        // of nothing by an answer, at the gate.
        Ok(outcome) => Action::Replaced {
            recorded: OutcomeSummary::Err {
                reason: Reason::code("never_served"),
                retryable: false,
            },
            consumed: OutcomeSummary::Ok {
                family: outcome.family(),
            },
        },
    };
    witness.emit(subjects.of(entity), Stage::Gate, Emitter::unknown(), action);
}

/// An outcome was written over a settled effect's: a `Judge` decision,
/// observed with what the world saw before and what it sees now. The
/// initial landing (before `settle` wrote `SeenOutcome`) is not a
/// replacement; a rewrite to an identical value is not observed.
fn observe_outcome_replaced(
    inserted: On<Insert, EffectOutcome>,
    mut replaced: Query<(&EffectOutcome, &mut SeenOutcome), Without<InFlight>>,
    subjects: Subjects,
    witness: Option<Res<Witnessing>>,
) {
    let Some(witness) = witness else {
        return;
    };
    let entity = inserted.event().entity;
    let Ok((outcome, mut seen)) = replaced.get_mut(entity) else {
        return;
    };
    let now = SeenOutcome::of(&outcome.0);
    if now.fingerprint == seen.fingerprint {
        return;
    }
    witness.emit(
        subjects.of(entity),
        Stage::Judge,
        Emitter::unknown(),
        Action::Replaced {
            recorded: seen.summary.clone(),
            consumed: now.summary.clone(),
        },
    );
    *seen = now;
}

/// An intent left the world before dispatch (a program's cancel despawned
/// it): never served, no record — the witness is the only trace.
fn observe_pending_despawned(
    removed: On<Remove, PendingEffect>,
    state: Query<(Has<Issued>, Has<EffectOutcome>), With<PendingEffect>>,
    subjects: Subjects,
    witness: Option<Res<Witnessing>>,
) {
    let entity = removed.event().entity;
    let Some(witness) = witness else {
        return;
    };
    let Ok((issued, answered)) = state.get(entity) else {
        return;
    };
    if issued || answered {
        return;
    }
    witness.emit(
        subjects.of(entity),
        Stage::Dispatch,
        Emitter::unknown(),
        Action::Cancelled {
            reason: Reason::code("despawned_before_dispatch"),
        },
    );
}
