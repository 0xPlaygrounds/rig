//! Handler registration and lookup through world entities and bound descriptors.
//!
//! Erased handlers live in non-send storage for WASM compatibility. [`Bound`]
//! hooks maintain the key index, while [`ServedBy`] links effects to handlers.
//!
//! ```
//! let index = rig_ecs::bus::HandlerIndex::default();
//! assert!(index.is_empty());
//! ```

use bevy_reflect::Reflect;
use std::{
    any::TypeId,
    collections::{HashMap, HashSet},
    marker::PhantomData,
};

use bevy_ecs::{
    prelude::*,
    system::{EntityCommands, SystemParam},
};
use rig_core::{
    effect::{EffectKind, Family, FamilyDescriptor, HandlerDescriptor, HandlerKey, Key, Outcome},
    error::{ErrorKind, ErrorReport},
    serve::{ErasedHandler, Serve},
};
use serde::{Deserialize, Serialize};

use super::effect::{Answer, Asked, WorldEffect};

/// A handler's saved key and descriptor, used for typed-key validation.
/// Each key must identify at most one entity. The immutable component requires
/// replacement through insertion so hooks keep [`HandlerIndex`] current.
#[derive(Component, Debug, Clone, Serialize, Deserialize, Reflect)]
#[component(immutable, on_insert = index_bound, on_discard = unindex_bound)]
#[reflect(Component)]
pub struct Bound {
    /// The key the handler serves.
    #[reflect(remote = crate::bus::reflect::HandlerKeyReflect)]
    pub key: HandlerKey,
    /// What it is: the descriptor, with `key` as its key.
    #[reflect(remote = crate::bus::reflect::HandlerDescriptorReflect)]
    pub descriptor: HandlerDescriptor,
}

/// The key → handler entity map, maintained by [`Bound`]'s hooks and read
/// by `Dispatch` to resolve an effect's key to the entity serving it. Kept
/// ahead of the hooks by [`Handlers`], so a registration is visible to the
/// same borrow that made it.
#[derive(Resource, Debug, Default)]
pub struct HandlerIndex {
    keys: HashMap<HandlerKey, (Entity, rig_core::effect::EffectFamily)>,
}

impl HandlerIndex {
    /// The entity bound to `key`.
    pub fn entity(&self, key: &HandlerKey) -> Option<Entity> {
        self.keys.get(key).map(|(entity, _)| *entity)
    }

    /// Bound keys.
    pub fn len(&self) -> usize {
        self.keys.len()
    }

    /// Whether nothing is bound.
    pub fn is_empty(&self) -> bool {
        self.keys.is_empty()
    }
}

fn index_bound(
    mut world: bevy_ecs::world::DeferredWorld<'_>,
    context: bevy_ecs::lifecycle::HookContext,
) {
    let Some(bound) = world.get::<Bound>(context.entity) else {
        return;
    };
    let (key, family) = (bound.key.clone(), bound.family());
    if let Some(mut index) = world.get_resource_mut::<HandlerIndex>() {
        index.keys.insert(key, (context.entity, family));
    }
}

fn unindex_bound(
    mut world: bevy_ecs::world::DeferredWorld<'_>,
    context: bevy_ecs::lifecycle::HookContext,
) {
    let Some(bound) = world.get::<Bound>(context.entity) else {
        return;
    };
    let key = bound.key.clone();
    if let Some(mut index) = world.get_resource_mut::<HandlerIndex>()
        && index
            .keys
            .get(&key)
            .is_some_and(|(entity, _)| *entity == context.entity)
    {
        index.keys.remove(&key);
    }
}

/// The handler entity serving an effect: resolved by `Dispatch` from the
/// effect's key through [`HandlerIndex`], or set at spawn by a system that
/// already holds the entity. Serial serving and reentrancy are queries
/// over it and [`Serves`].
#[derive(Component, Debug, Clone, Copy, PartialEq, Eq, Reflect)]
#[relationship(relationship_target = Serves)]
#[reflect(Component)]
pub struct ServedBy(pub Entity);

/// The effects a handler entity serves or served.
#[derive(Component, Debug, Default, Reflect)]
#[relationship_target(relationship = ServedBy)]
#[reflect(Component)]
pub struct Serves(Vec<Entity>);

impl Bound {
    /// The family the handler serves.
    pub fn family(&self) -> rig_core::effect::EffectFamily {
        self.descriptor.family.family()
    }
}

/// How a bound key is served.
pub enum Served {
    /// By a [`Serve`] future on the task pool: the common case, every
    /// adapter and every replayer.
    Task(ErasedHandler),
    /// By a system: the dispatch stays on its entity, `InFlight`, and a
    /// user system answers it through [`Asked<E>`] and [`Answer<E>`] for
    /// a [`WorldHandler`], or by submitting a [`super::WorldOutcome`]
    /// for a key bound with [`Handlers::register_open`].
    World(WorldServe),
}

/// A handler that is a system, erased: what the world does to the effect
/// entity when `Dispatch` takes it. A plain function pointer, so the table
/// holds no closure and no `E`.
#[derive(Clone)]
pub struct WorldServe {
    /// What the key is bound as: the family a same-borrow re-registration
    /// is checked against.
    pub family: FamilyDescriptor,
    /// What the dispatch lands as: for a [`WorldHandler`], deserialize the
    /// payload and insert `Asked<E>` on the effect entity (or say why the
    /// payload is not an `E`); for an open key, nothing is inserted and the effect
    /// entity itself is the question.
    pub ask: fn(&mut EntityCommands<'_>, &EffectKind) -> Result<(), ErrorReport>,
}

/// The `ask` of an open key: the effect entity is the question, nothing
/// is added to it.
fn open(_entity: &mut EntityCommands<'_>, _kind: &EffectKind) -> Result<(), ErrorReport> {
    Ok(())
}

/// A handler that is a system, for a [`WorldEffect`] `E`: bound with
/// [`Handlers::register_world`]. A dispatch to its key lands on the effect
/// entity as [`Asked<E>`]; a user system with any `World` access reads it
/// and inserts [`Answer<E>`]; the plugin queues a [`super::WorldOutcome`] and
/// publishes the [`super::EffectOutcome`] in `Collect`. Serial serving keeps the key busy until the answer
/// lands. Unary only: a system answers once.
pub struct WorldHandler<E: WorldEffect>(PhantomData<fn() -> E>);

impl<E: WorldEffect> WorldHandler<E> {
    /// The descriptor a world handler for `E` is bound with.
    pub fn descriptor(key: HandlerKey) -> HandlerDescriptor {
        HandlerDescriptor {
            key,
            family: FamilyDescriptor::Custom {
                kind: E::KIND.to_owned(),
            },
            layers: Vec::new(),
        }
    }

    /// How it is served.
    pub fn served() -> Served {
        Served::World(WorldServe {
            family: FamilyDescriptor::Custom {
                kind: E::KIND.to_owned(),
            },
            ask: ask::<E>,
        })
    }
}

fn ask<E: WorldEffect>(
    entity: &mut EntityCommands<'_>,
    kind: &EffectKind,
) -> Result<(), ErrorReport> {
    let EffectKind::Custom {
        kind: label,
        payload,
    } = kind
    else {
        return Err(ErrorReport::new(
            ErrorKind::Request,
            format!(
                "`{}` is served by a system for `{}` effects; a {} effect cannot be asked of it",
                E::KIND,
                E::KIND,
                kind.family()
            ),
        ));
    };
    if &**label != E::KIND {
        return Err(ErrorReport::new(
            ErrorKind::Request,
            format!(
                "a `{label}` effect reached the system serving `{}`",
                E::KIND
            ),
        ));
    }
    let effect: E = serde_json::from_value(payload.clone()).map_err(|error| {
        ErrorReport::new(
            ErrorKind::Request,
            format!("the payload of `{}` did not deserialize: {error}", E::KIND),
        )
    })?;
    entity.insert(Asked(effect));
    Ok(())
}

/// A system's answer becomes the outcome: the observer installed once per
/// `E` by [`Handlers::register_world`].
pub fn answered<E: WorldEffect>(
    added: On<Add, Answer<E>>,
    answers: Query<&Answer<E>, With<Asked<E>>>,
    mut commands: Commands,
) {
    let entity = added.event().entity;
    let Ok(answer) = answers.get(entity) else {
        // Unsolicited or duplicate answers must not become effect outcomes.
        commands.entity(entity).remove::<Answer<E>>();
        return;
    };
    let outcome = serde_json::to_value(&answer.0)
        .map(|payload| Outcome::Custom { payload })
        .map_err(|error| {
            ErrorReport::new(
                ErrorKind::Response,
                format!("the answer to `{}` did not serialize: {error}", E::KIND),
            )
        });
    commands
        .entity(entity)
        .remove::<(Asked<E>, Answer<E>)>()
        .insert(super::effect::WorldOutcome::new(outcome));
}

/// The marker of a handler held for the entity in the world's non-send
/// [`HandlerTable`].
#[derive(Component)]
pub struct Handler;

/// The world's erased handlers, keyed by their [`Bound`] entity. Non-send
/// because an [`ErasedHandler`] is `!Send` on wasm and cannot be a
/// component there; one storage keeps one code path on both targets.
#[derive(Default)]
pub struct HandlerTable {
    served: HashMap<Entity, Served>,
}

impl HandlerTable {
    pub(crate) fn contains(&self, entity: Entity) -> bool {
        self.served.contains_key(&entity)
    }

    /// Installation after checkpoint preflight, without deferred commands or
    /// a second fallible validation step halfway through the transaction.
    pub(crate) fn install(world: &mut World, entity: Entity, handler: ErasedHandler) {
        world
            .non_send_mut::<Self>()
            .served
            .insert(entity, Served::Task(handler));
        world.entity_mut(entity).insert(Handler);
    }
}

/// Remove the stored handler when its [`Handler`] component is removed.
pub fn unbound(removed: On<Remove, Handler>, mut table: NonSendMut<HandlerTable>) {
    table.served.remove(&removed.event().entity);
}

/// How handler entities are served: the [`HandlerTable`].
#[derive(SystemParam)]
pub struct Registry<'w> {
    table: NonSend<'w, HandlerTable>,
}

impl Registry<'_> {
    /// How the handler entity `entity` is served, if it is bound.
    pub fn served(&self, entity: Entity) -> Option<&Served> {
        self.table.served.get(&entity)
    }
}

/// Register and deregister handlers from a system: the registry API over
/// handler entities.
#[derive(SystemParam)]
pub struct Handlers<'w, 's> {
    commands: Commands<'w, 's>,
    index: ResMut<'w, HandlerIndex>,
    world_kinds: ResMut<'w, WorldKinds>,
    bound: Query<'w, 's, &'static Bound>,
    table: NonSendMut<'w, HandlerTable>,
}

/// The `E`s whose answer observer is installed.
#[derive(Resource, Default)]
pub struct WorldKinds(HashSet<TypeId>);

impl Handlers<'_, '_> {
    /// Run `f` with this world's registry, apply its deferred commands, and return
    /// its result. Return an error when required bus resources are unavailable.
    pub fn with<T>(
        world: &mut World,
        f: impl FnOnce(&mut Handlers<'_, '_>) -> T,
    ) -> Result<T, ErrorReport> {
        let mut state = bevy_ecs::system::SystemState::<Handlers>::new(world);
        let out = match state.get_mut(world) {
            Ok(mut handlers) => f(&mut handlers),
            Err(error) => {
                return Err(ErrorReport::new(
                    ErrorKind::Internal,
                    format!("the world has no bus: {error}"),
                ));
            }
        };
        state.apply(world);
        Ok(out)
    }

    /// Register `handler` under `key` and return its entity, replacing an existing
    /// handler of the same family. Return an error if the bound key has a different
    /// family; deregister it before changing families.
    pub fn register(
        &mut self,
        key: impl Into<HandlerKey>,
        handler: impl Serve + 'static,
    ) -> Result<Entity, ErrorReport> {
        self.register_erased(key, ErasedHandler::new(handler))
    }

    /// Register an erased handler with its descriptor key normalized to `key`.
    /// Return its entity, or an error if the key is bound to another family.
    pub fn register_erased(
        &mut self,
        key: impl Into<HandlerKey>,
        handler: ErasedHandler,
    ) -> Result<Entity, ErrorReport> {
        let key = key.into();
        let described = handler.descriptor();
        let descriptor = HandlerDescriptor {
            key: key.clone(),
            family: described.family,
            layers: described.layers,
        };
        self.bind(key, descriptor, Served::Task(handler))
    }

    /// Register a handler and return a typed [`Key`]. Return an error if its
    /// descriptor does not match `F` or the key is bound to another family.
    pub fn register_typed<F: Family>(
        &mut self,
        key: impl Into<HandlerKey>,
        handler: impl Serve + 'static,
    ) -> Result<Key<F>, ErrorReport> {
        let key = key.into();
        let handler = ErasedHandler::new(handler);
        let descriptor = handler.descriptor();
        if descriptor.family.family() != F::FAMILY {
            return Err(ErrorReport::new(
                ErrorKind::HandlerUnavailable,
                format!(
                    "`{key}` was registered as {} but the handler serves {}",
                    F::FAMILY,
                    descriptor.family.family()
                ),
            ));
        }
        self.register_erased(key.clone(), handler)?;
        Ok(Key::new_unchecked(key))
    }

    /// Bind `key` to a handler that is a system for `E`: see
    /// [`WorldHandler`]. Installs the answer observer for `E` once and returns
    /// an error if the key is bound to another family.
    pub fn register_world<E: WorldEffect>(
        &mut self,
        key: impl Into<HandlerKey>,
    ) -> Result<Entity, ErrorReport> {
        let key = key.into();
        let descriptor = WorldHandler::<E>::descriptor(key.clone());
        let entity = self.bind(key, descriptor, WorldHandler::<E>::served())?;
        if self.world_kinds.0.insert(TypeId::of::<E>()) {
            self.commands.add_observer(answered::<E>);
        }
        Ok(entity)
    }

    /// Bind `key` to a world system with advertised `family`, returning its entity
    /// or an error if already bound to another family. The system must submit a
    /// unary [`super::WorldOutcome`]; serial serving stays busy until collection.
    /// The advertised family does not constrain incoming effect kinds.
    pub fn register_open(
        &mut self,
        key: impl Into<HandlerKey>,
        family: FamilyDescriptor,
    ) -> Result<Entity, ErrorReport> {
        let key = key.into();
        let descriptor = HandlerDescriptor {
            key: key.clone(),
            family: family.clone(),
            layers: Vec::new(),
        };
        self.bind(
            key,
            descriptor,
            Served::World(WorldServe { family, ask: open }),
        )
    }

    fn bind(
        &mut self,
        key: HandlerKey,
        descriptor: HandlerDescriptor,
        served: Served,
    ) -> Result<Entity, ErrorReport> {
        let family = descriptor.family.family();
        let entity = match self.index.keys.get(&key).copied() {
            Some((entity, bound)) if bound == family => {
                self.commands.entity(entity).insert((
                    Bound {
                        key: key.clone(),
                        descriptor,
                    },
                    Name::new(key.to_string()),
                ));
                entity
            }
            Some((_, bound)) => {
                return Err(ErrorReport::new(
                    ErrorKind::HandlerUnavailable,
                    format!(
                        "`{key}` is bound to a {bound} handler; a {family} handler cannot take its place while it is bound — deregister it first"
                    ),
                ));
            }
            None => self
                .commands
                .spawn((
                    Bound {
                        key: key.clone(),
                        descriptor,
                    },
                    Name::new(key.to_string()),
                ))
                .id(),
        };
        self.index.keys.insert(key, (entity, family));
        self.serve(entity, served);
        Ok(entity)
    }

    fn serve(&mut self, entity: Entity, served: Served) {
        self.table.served.insert(entity, served);
        self.commands.entity(entity).insert(Handler);
    }

    /// Remove the handler bound to `key`: its entity despawns. Returns
    /// whether one was bound.
    pub fn deregister(&mut self, key: &HandlerKey) -> bool {
        match self.index.keys.remove(key) {
            Some((entity, _)) => {
                self.commands.entity(entity).despawn();
                true
            }
            None => false,
        }
    }

    /// The handler entity bound to `key`.
    pub fn entity(&self, key: &HandlerKey) -> Option<Entity> {
        self.index.entity(key)
    }

    /// The descriptor bound to `key`.
    pub fn descriptor(&self, key: &HandlerKey) -> Option<HandlerDescriptor> {
        let entity = self.index.entity(key)?;
        self.bound
            .get(entity)
            .ok()
            .map(|bound| bound.descriptor.clone())
    }

    /// Every bound key.
    pub fn keys(&self) -> Vec<HandlerKey> {
        let mut keys: Vec<HandlerKey> = self.bound.iter().map(|bound| bound.key.clone()).collect();
        keys.sort();
        keys
    }

    /// Every bound descriptor, by key.
    pub fn descriptors(&self) -> Vec<HandlerDescriptor> {
        let mut described: Vec<HandlerDescriptor> = self
            .bound
            .iter()
            .map(|bound| bound.descriptor.clone())
            .collect();
        described.sort_by(|a, b| a.key.cmp(&b.key));
        described
    }
}
