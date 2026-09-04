//! Handlers are entities: a [`Bound`] component (the key and the
//! descriptor, serde) on an entity, and the erased handler in the world's
//! [`HandlerTable`], keyed by that entity. The registry is a query over
//! `Bound`; registration spawns, deregistration despawns.

use std::{
    any::TypeId,
    collections::{HashMap, HashSet},
    marker::PhantomData,
};

use bevy_ecs::{
    lifecycle::Remove,
    prelude::*,
    system::{EntityCommands, SystemParam},
};
use rig_core::{
    effect::{EffectKind, Family, FamilyDescriptor, HandlerDescriptor, HandlerKey, Key, Outcome},
    error::{ErrorKind, ErrorReport},
    serve::{ErasedHandler, Serve},
};
use serde::{Deserialize, Serialize};

use super::effect::{Answer, Asked, EffectOutcome, WorldEffect};

/// What a handler entity is bound to: its key and its descriptor. The serde
/// twin of the handler; what a scene saves and what a typed key is checked
/// against. One per handler entity; a key is bound to at most one entity.
#[derive(Component, Debug, Clone, Serialize, Deserialize)]
#[cfg_attr(feature = "reflect", derive(bevy_reflect::Reflect), reflect(Component))]
pub struct Bound {
    /// The key the handler serves.
    #[cfg_attr(feature = "reflect", reflect(remote = crate::bus::reflect::HandlerKeyReflect))]
    pub key: HandlerKey,
    /// What it is: the descriptor, with `key` as its key.
    #[cfg_attr(feature = "reflect", reflect(remote = crate::bus::reflect::HandlerDescriptorReflect))]
    pub descriptor: HandlerDescriptor,
}

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
    /// user system answers it — through [`Asked<E>`] and [`Answer<E>`] for
    /// a [`WorldHandler`], or by inserting the [`EffectOutcome`] itself
    /// for a key bound with [`Handlers::register_open`].
    World(WorldServe),
}

/// A handler that is a system, erased: what the world does to the effect
/// entity when `Dispatch` takes it. A plain function pointer, so the table
/// holds no closure and no `E`.
#[derive(Clone)]
pub struct WorldServe {
    /// What the key is bound as: the family a same-borrow re-registration
    /// is checked against (boxed: a descriptor is large next to the task
    /// arm's pointer).
    pub family: Box<FamilyDescriptor>,
    /// What the dispatch lands as: for a [`WorldHandler`], deserialize the
    /// payload and insert `Asked<E>` on the effect entity (or say why the
    /// payload is not an `E`); for an open key, nothing — the effect
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
/// and inserts [`Answer<E>`]; the plugin turns the answer into the
/// [`EffectOutcome`]. Serial serving keeps the key busy until the answer
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
            family: Box::new(FamilyDescriptor::Custom {
                kind: E::KIND.to_owned(),
            }),
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
        // An answer with no question — a second answer, or one on an effect
        // a task is serving — is dropped, never an outcome.
        commands.entity(entity).remove::<Answer<E>>();
        return;
    };
    let outcome = serde_json::to_value(&answer.0)
        .map(Outcome::Custom)
        .map_err(|error| {
            ErrorReport::new(
                ErrorKind::Response,
                format!("the answer to `{}` did not serialize: {error}", E::KIND),
            )
        });
    commands
        .entity(entity)
        .remove::<(Asked<E>, Answer<E>)>()
        .insert(EffectOutcome(outcome));
}

/// The world's erased handlers, keyed by their [`Bound`] entity. `NonSend`
/// on every target, by spelling: an [`ErasedHandler`] is `!Send` on browser
/// wasm (a provider client there is), and one spelling on both targets
/// beats a fork. Systems that dispatch or register therefore run on the
/// main thread; nothing else needs the table.
#[derive(Default)]
pub struct HandlerTable {
    served: HashMap<Entity, Served>,
    /// The `E`s whose answer observer is installed.
    world_kinds: HashSet<TypeId>,
    /// The key each served entity is bound to: what `bind` consults for a
    /// registration made earlier in the same system, whose `Bound` the
    /// query cannot see yet (commands are deferred).
    keys: HashMap<HandlerKey, Entity>,
}

impl HandlerTable {
    /// How the handler entity `entity` is served, if it is bound.
    pub fn served(&self, entity: Entity) -> Option<&Served> {
        self.served.get(&entity)
    }

    /// Bound handler entities.
    pub fn len(&self) -> usize {
        self.served.len()
    }

    /// Whether nothing is bound.
    pub fn is_empty(&self) -> bool {
        self.served.is_empty()
    }

    /// Forget `entity`'s handler: the `Bound` removal observer's call.
    pub fn remove(&mut self, entity: Entity) -> Option<Served> {
        self.keys.retain(|_, bound| *bound != entity);
        self.served.remove(&entity)
    }
}

/// A `Bound` component removed — a deregistration, a despawn — takes the
/// handler out of the table with it.
pub fn unbound(removed: On<Remove, Bound>, mut table: NonSendMut<HandlerTable>) {
    table.remove(removed.event().entity);
}

/// Register and deregister handlers from a system: the registry API over
/// handler entities. Main-thread only ([`HandlerTable`] is `NonSend`).
#[derive(SystemParam)]
pub struct Handlers<'w, 's> {
    commands: Commands<'w, 's>,
    table: NonSendMut<'w, HandlerTable>,
    bound: Query<'w, 's, (Entity, &'static Bound)>,
}

impl Handlers<'_, '_> {
    /// Run `f` with a `Handlers` over `world` and apply what it did: for a
    /// host that registers from outside a system (a test, a scene load).
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

    /// Register `handler` under `key`: a new handler entity, or — when the
    /// key is bound to a handler of the same family — the bound entity
    /// re-served. A key never changes family while bound; a handler of
    /// another family is refused, as the bus refuses it.
    pub fn register(
        &mut self,
        key: impl Into<HandlerKey>,
        handler: impl Serve + 'static,
    ) -> Result<Entity, ErrorReport> {
        self.register_erased(key, ErasedHandler::new(handler))
    }

    /// Register an already-erased handler.
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

    /// [`register`](Self::register), returning a [`Key`] carrying the
    /// family the handler proved by its descriptor.
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
    /// [`WorldHandler`]. Installs the answer observer for `E` once.
    pub fn register_world<E: WorldEffect>(
        &mut self,
        key: impl Into<HandlerKey>,
    ) -> Result<Entity, ErrorReport> {
        let key = key.into();
        let descriptor = WorldHandler::<E>::descriptor(key.clone());
        let entity = self.bind(key, descriptor, WorldHandler::<E>::served())?;
        if self.table.world_kinds.insert(TypeId::of::<E>()) {
            self.commands.add_observer(answered::<E>);
        }
        Ok(entity)
    }

    /// Bind `key` to the world itself, as `family`: a dispatch to it is
    /// taken (`Issued`, `InFlight`, the record opened) and left on its
    /// entity for a user system with any `World` access to answer by
    /// inserting the [`EffectOutcome`] — of any family, `family` being what
    /// the key is advertised as (a tool's definition, a model's
    /// capabilities). What the system dispatches on the way is a
    /// `PendingEffect` it spawns `ChildOf` the effect it serves. Serial
    /// serving keeps the key busy until the outcome lands. Unary only.
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
            Served::World(WorldServe {
                family: Box::new(family),
                ask: open,
            }),
        )
    }

    fn bind(
        &mut self,
        key: HandlerKey,
        descriptor: HandlerDescriptor,
        served: Served,
    ) -> Result<Entity, ErrorReport> {
        let family = descriptor.family.family();
        let known = self
            .table
            .keys
            .get(&key)
            .copied()
            .and_then(|entity| {
                self.bound
                    .get(entity)
                    .ok()
                    .map(|(_, bound)| (entity, bound.clone()))
            })
            .or_else(|| {
                self.bound
                    .iter()
                    .find(|(_, bound)| bound.key == key)
                    .map(|(entity, bound)| (entity, bound.clone()))
            });
        // A registration earlier in this borrow, not yet a `Bound` the query
        // can see: the table knows its entity and its family.
        let known = known.or_else(|| {
            self.table.keys.get(&key).copied().and_then(|entity| {
                self.table.served.get(&entity).map(|served| {
                    let family = match served {
                        Served::Task(handler) => handler.descriptor().family,
                        Served::World(world) => (*world.family).clone(),
                    };
                    (
                        entity,
                        Bound {
                            key: key.clone(),
                            descriptor: HandlerDescriptor {
                                key: key.clone(),
                                family,
                                layers: Vec::new(),
                            },
                        },
                    )
                })
            })
        });
        let entity = match known {
            Some((entity, bound)) if bound.family() == family => {
                self.commands.entity(entity).insert(Bound {
                    key: key.clone(),
                    descriptor,
                });
                entity
            }
            Some((_, bound)) => {
                return Err(ErrorReport::new(
                    ErrorKind::HandlerUnavailable,
                    format!(
                        "`{key}` is bound to a {} handler; a {family} handler cannot take its place while it is bound — deregister it first",
                        bound.family()
                    ),
                ));
            }
            None => self
                .commands
                .spawn(Bound {
                    key: key.clone(),
                    descriptor,
                })
                .id(),
        };
        self.table.keys.insert(key, entity);
        self.table.served.insert(entity, served);
        Ok(entity)
    }

    /// Remove the handler bound to `key`: its entity despawns. Returns
    /// whether one was bound.
    pub fn deregister(&mut self, key: &HandlerKey) -> bool {
        let entity = self.table.keys.get(key).copied().or_else(|| {
            self.bound
                .iter()
                .find(|(_, bound)| &bound.key == key)
                .map(|(entity, _)| entity)
        });
        match entity {
            Some(entity) => {
                self.table.remove(entity);
                self.commands.entity(entity).despawn();
                true
            }
            None => false,
        }
    }

    /// The descriptor bound to `key`.
    pub fn descriptor(&self, key: &HandlerKey) -> Option<HandlerDescriptor> {
        self.bound
            .iter()
            .find(|(_, bound)| &bound.key == key)
            .map(|(_, bound)| bound.descriptor.clone())
    }

    /// Every bound key.
    pub fn keys(&self) -> Vec<HandlerKey> {
        let mut keys: Vec<HandlerKey> = self
            .bound
            .iter()
            .map(|(_, bound)| bound.key.clone())
            .collect();
        keys.sort();
        keys
    }

    /// Every bound descriptor, by key.
    pub fn descriptors(&self) -> Vec<HandlerDescriptor> {
        let mut described: Vec<HandlerDescriptor> = self
            .bound
            .iter()
            .map(|(_, bound)| bound.descriptor.clone())
            .collect();
        described.sort_by(|a, b| a.key.cmp(&b.key));
        described
    }
}
