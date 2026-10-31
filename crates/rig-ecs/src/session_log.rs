//! The session log in the app. Each agent's log, in the session's
//! [`SessionStore`], gets every change of the agent's [`Conversation`] and
//! of its [`ReflectSaved`] components, such as its model, reasoning
//! setting, system prompt, tool access and last usage. The format, its
//! writer and its readers are rig-memory's.
//!
//! Each change waits in the conversation's outbox until the kernel saves
//! it: between the stages of every pass, at the end of the frame, before a
//! tool that may change something starts, and when a turn is stopped.
//! Saving announces each as a [`ConversationChanged`] and queues it, and
//! the queue is written at the end of the frame, a subagent's log before
//! its parent's. Records are only appended; [`restore`](super::restore)
//! reads the logs back at startup.

use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::AppTypeRegistry;
use bevy_ecs::resource::IsResource;
use bevy_ecs::schedule::ScheduleLabel;
use bevy_ecs::system::SystemParam;
use bevy_ecs::world::DeferredWorld;
use bevy_log::error;
use bevy_reflect::serde::TypedReflectSerializer;
use bevy_reflect::{CreateTypeData, Reflect, TypePath};
use rig_core::conversation::ChangeRecord;
use rig_memory::{Repair, SESSION, Session, SessionError, Store, Writer};

use super::agent::{Agent, AgentId, Conversation, Notice, SpawnedBy};
use super::restore::SavedKeys;
use super::{SaveSystems, WriteLogs};

/// Saves an agent component or a resource with the session: derive
/// `Reflect` and add `Saved`, as `#[reflect(Component, Saved)]` or
/// `#[reflect(Resource, Saved)]`.
///
/// - Each change is logged at the end of the frame under its type path, or
///   the key its plugin gives it ([`SavedKeys`]): an agent's in its log, a
///   resource in the [`SESSION`] log. A value that serializes to `null`, or
///   a removed component, is saved as absent.
/// - On restore it is inserted before [`SessionRestored`] exists, on the
///   agent it was saved with, so an `On<Insert<T>>` observer re-arms what
///   it says. A resource replaces the one its plugin inserted while the app
///   was built.
/// - A type reflected opaquely with serde
///   (`#[reflect(opaque, Component, Saved, Clone, Serialize, Deserialize)]`)
///   is saved through serde: the way to save rig-core data such as a
///   [`MessageId`](rig_core::conversation::MessageId).
/// - A generic type is saved only once registered. A logged type no build
///   registers is reported and kept in the log. A value that no longer fits
///   its type is reported and not inserted; its next save replaces it.
/// - A fork ([`ForkAgent`](super::fork::ForkAgent)) copies an agent's
///   value, unless the type says otherwise as `Saved(Reset)` ([`Reset`]) or
///   `Saved(Never)` ([`Never`]); what the fork gets is saved with it, so a
///   restore does not bring back the source's value.
#[derive(Clone)]
pub struct ReflectSaved {
    watch: fn(&mut App),
    /// What a fork gets of the type.
    pub(crate) fork: Fork,
}

/// A saved agent component a fork starts afresh with, as its `Default`,
/// such as a meter of the agent's own calls:
/// `#[reflect(Component, Saved(Reset))]`. The fork has it whether or not
/// the source has.
pub struct Reset;

/// A saved agent component a fork does not get, such as a duty the source
/// owes: `#[reflect(Component, Saved(Never))]`. It is removed from the
/// fork, even when every agent requires it; restored, the fork then gets
/// such a required one at its default, as any restored agent does.
pub struct Never;

/// What a fork gets of a saved component.
#[derive(Clone, Copy)]
pub(crate) enum Fork {
    /// The source's value.
    Copy,
    /// What this leaves on the fork once it is spawned.
    Set(fn(&mut EntityWorldMut)),
}

impl ReflectSaved {
    fn of<T: Component + Reflect + TypePath>(fork: Fork) -> Self {
        // `Changed` also sees an immutable value inserted in place of
        // another.
        let watch = |app: &mut App| {
            app.add_systems(Last, log_changed::<T>.in_set(SaveSystems));
        };
        Self { watch, fork }
    }
}

impl<T: Component + Reflect + TypePath> CreateTypeData<T> for ReflectSaved {
    fn create_type_data(_: ()) -> Self {
        Self::of::<T>(Fork::Copy)
    }
}

impl<T: Component + Reflect + TypePath + Default> CreateTypeData<T, Reset> for ReflectSaved {
    fn create_type_data(_: Reset) -> Self {
        Self::of::<T>(Fork::Set(|fork| {
            fork.insert(T::default());
        }))
    }
}

impl<T: Component + Reflect + TypePath> CreateTypeData<T, Never> for ReflectSaved {
    fn create_type_data(_: Never) -> Self {
        // A required component may have put it on the fork.
        Self::of::<T>(Fork::Set(|fork| {
            fork.remove::<T>();
        }))
    }
}

/// Logs each change of the saved component `T` on the agents or as a
/// resource, and its removal from an agent; reflection runs only for a
/// change. A change starts its agent's log, as a message does, unless it
/// is a value the agent was spawned with, so a session nobody used is not
/// kept.
fn log_changed<T: Component + Reflect + TypePath>(
    changed: Query<
        (Entity, Option<&AgentId>, &T),
        (Or<(With<Agent>, With<IsResource>)>, Changed<T>),
    >,
    mut removed: RemovedComponents<T>,
    agents: Query<&AgentId, With<Agent>>,
    spawned: Query<(), Added<Agent>>,
    log: Res<SessionLog>,
    registry: Res<AppTypeRegistry>,
    keys: Res<SavedKeys>,
) {
    if !log.0.is_live() {
        return;
    }
    let path = T::type_path();
    let name = keys.key(path).unwrap_or(path);
    let registry = registry.read();
    for (entity, id, component) in &changed {
        let value = TypedReflectSerializer::new(component.as_partial_reflect(), &registry);
        let log_name = id.map_or(SESSION, |id| id.0.as_str());
        let starts = id.is_some() && !spawned.contains(entity);
        match serde_json::to_value(value) {
            Ok(value) => log.0.save_value(log_name, name, Some(value), starts),
            Err(failure) => error!("not logging {name}: {failure}"),
        }
    }
    for entity in removed.read() {
        if let Ok(id) = agents.get(entity)
            && !changed.contains(entity)
        {
            log.0
                .save_value(&id.0, name, None, !spawned.contains(entity));
        }
    }
}

/// Where the session is kept, such as a rig-memory `MemoryStore` or
/// `DirStore`: inserted while the app is built, before it runs, by any
/// plugin, or by [`open_session`]. Without one nothing is kept.
#[derive(Resource, Clone)]
pub struct SessionStore(pub Arc<dyn Store>);

impl SessionStore {
    /// The session kept in `store`.
    pub fn new(store: impl Store) -> Self {
        Self(Arc::new(store))
    }
}

/// The session's agent logs: rig-memory's [`Writer`] of the records queued
/// for them and each agent's latest-wins state. Nothing is logged before
/// the session was restored, or without a [`SessionStore`].
#[derive(Resource, Clone, Default)]
pub struct SessionLog(pub(crate) Writer);

impl SessionLog {
    /// Writes every queued record ([`Writer::flush`]); an error once the
    /// logging stopped.
    pub(crate) fn flush(&self) -> Result<(), SessionError> {
        self.0.flush()
    }

    /// Whether a write stored a record ([`Writer::has_written`]), such as
    /// for a host that marks its session saved only once something is.
    pub fn has_written(&self) -> bool {
        self.0.has_written()
    }

    /// Why the logging stopped, if a write failed ([`Writer::failure`]):
    /// for a front end that tells the user at exit, where the [`Notice`]
    /// sent in [`WriteLogs`] comes too late to be read.
    pub fn failure(&self) -> Option<String> {
        self.0.failure()
    }
}

/// Inserted once the session was restored from its [`SessionStore`] and is
/// logged: a choice made afterwards is the user's, such as a model to
/// remember (`run_if(resource_exists::<SessionRestored>)`). It holds what
/// reading the session repaired, such as a last line a crash cut short
/// and set aside, each also shown as a notice. It stays when a log write
/// fails later, which stops only the logging. An app without a store, or
/// whose session was refused, never gets one.
#[derive(Resource, Clone, Debug, Default)]
pub struct SessionRestored {
    /// What reading the session repaired.
    pub repairs: Vec<Repair>,
}

/// The session [`open_session`] read, until the restore takes it.
#[derive(Resource)]
pub(crate) struct Opened(pub(crate) Session);

/// Opens `store` for this app: reads the session it holds, refusing one it
/// cannot read whole (made by another version of rig, or broken) before
/// anything is written, sets each torn last line
/// aside, and keeps what it read for the restore at startup. On success
/// `store` is the app's [`SessionStore`] and its log writes there once the
/// session is restored; on failure the app keeps nothing of it. An app that
/// only inserts a `SessionStore` has it opened by the restore, which shows
/// a refusal as a notice.
///
/// # Errors
///
/// The [`SessionError`] reading the session gave: a session from before
/// `rig-session/2`, a log in another format, a record that cannot be read,
/// or the store failing.
pub fn open_session(world: &mut World, store: Arc<dyn Store>) -> Result<(), SessionError> {
    let (writer, session) = Writer::open(store.clone())?;
    world.insert_resource(SessionStore(store));
    world.insert_resource(SessionLog(writer));
    world.insert_resource(Opened(session));
    Ok(())
}

/// A change saved from an agent's conversation, in the order made. A
/// restore makes none; a change made while nothing is logged (no store, or
/// before the restore) is announced all the same.
#[derive(Message, Clone, Debug)]
pub struct ConversationChanged {
    /// The agent.
    pub agent: Entity,
    /// The change: what changed, who made it and when.
    pub change: ChangeRecord,
}

/// Retires the agent `entity` for good: its log says so, so a restore
/// brings back neither it nor the agents it spawned, and it is despawned
/// with them and their work, its conversation saved first.
#[derive(EntityEvent, Clone, Debug)]
pub struct Retire {
    /// The agent.
    pub entity: Entity,
}

/// Saves the outbox of `conversation`, that of `agent` whose id is `id`,
/// to `log`; one [`ConversationChanged`] per change saved.
pub(crate) fn save_conversation(
    log: &SessionLog,
    agent: Entity,
    id: &AgentId,
    conversation: &mut Conversation,
) -> Vec<ConversationChanged> {
    let saved = log.0.save(&id.0, conversation);
    let committed = |change| ConversationChanged { agent, change };
    saved.into_iter().map(committed).collect()
}

/// Saves every agent conversation changed since it last ran, without
/// marking any changed again.
#[derive(SystemParam)]
pub(crate) struct Drain<'w, 's> {
    agents:
        Query<'w, 's, (Entity, &'static AgentId, &'static mut Conversation), Changed<Conversation>>,
    log: Res<'w, SessionLog>,
    committed: MessageWriter<'w, ConversationChanged>,
}

impl Drain<'_, '_> {
    /// Saves them; an agent whose conversation had changes, if any did.
    pub(crate) fn save(&mut self) -> Option<Entity> {
        let mut changed = None;
        for (agent, id, mut conversation) in &mut self.agents {
            let conversation = conversation.bypass_change_detection();
            let saved = save_conversation(&self.log, agent, id, conversation);
            if !saved.is_empty() {
                changed = Some(agent);
                self.committed.write_batch(saved);
            }
        }
        changed
    }
}

/// The schedule that saves the agent conversations changed since it last
/// ran ([`save_now`]): right before `Update`, between the stages of each
/// agent pass and in `Last` ([`SaveSystems`]).
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub(crate) struct SaveConversations;

/// An agent whose conversation [`save_conversations`] saved changes of,
/// since [`save_now`] last took it.
#[derive(Resource, Default)]
pub(crate) struct SavedOne(Option<Entity>);

/// Saves the conversations changed since it last ran, in
/// [`SaveConversations`].
pub(crate) fn save_conversations(mut drain: Drain, mut saved: ResMut<SavedOne>) {
    if let Some(agent) = drain.save() {
        saved.0 = Some(agent);
    }
}

/// Runs [`SaveConversations`]: an agent whose conversation had changes,
/// if any did; one the run before `Update` saved is not counted.
pub(crate) fn save_now(world: &mut World) -> Option<Entity> {
    world.get_resource_mut::<SavedOne>()?.0 = None;
    world.try_run_schedule(SaveConversations).ok()?;
    world.get_resource_mut::<SavedOne>()?.0.take()
}

/// Saves what a conversation about to be replaced or removed did not save
/// yet, so neither a value inserted over it nor its agent despawned loses
/// a change.
pub(crate) fn save_discarded(mut world: DeferredWorld, context: HookContext) {
    let agent = context.entity;
    let (Some(log), Some(id)) = (
        world.get_resource::<SessionLog>().cloned(),
        world.get::<AgentId>(agent).cloned(),
    ) else {
        return;
    };
    let Some(mut conversation) = world.get_mut::<Conversation>(agent) else {
        return;
    };
    if conversation.unsaved().is_empty() {
        return;
    }
    let conversation = conversation.bypass_change_detection();
    for committed in save_conversation(&log, agent, &id, conversation) {
        world.write_message(committed);
    }
}

/// Retires an agent, as [`Retire`] says.
pub(crate) fn retire(
    retire: On<Retire>,
    mut agents: Query<(&AgentId, &mut Conversation)>,
    log: Res<SessionLog>,
    mut committed: MessageWriter<ConversationChanged>,
    mut commands: Commands,
) {
    let agent = retire.entity;
    let Ok((id, mut conversation)) = agents.get_mut(agent) else {
        return;
    };
    let conversation = conversation.bypass_change_detection();
    committed.write_batch(save_conversation(&log, agent, id, conversation));
    log.0.retire(&id.0);
    commands.entity(agent).try_despawn();
}

/// Restores the session at startup, settles what a crash or restart left
/// half done, and logs everything after.
pub struct SessionLogPlugin;

impl Plugin for SessionLogPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreStartup, super::restore::restore_session)
            .add_systems(Startup, super::restore::reconcile)
            .add_systems(Last, write_logs.in_set(WriteLogs))
            .add_observer(open_child_log)
            .init_resource::<SavedKeys>();
    }

    /// Watches every type registered as [`ReflectSaved`], once every plugin
    /// had its say in the registry.
    fn finish(&self, app: &mut App) {
        let watches: Vec<fn(&mut App)> = app
            .world()
            .get_resource::<AppTypeRegistry>()
            .map(|registry| {
                let registry = registry.read();
                let saved = registry.iter_with_data::<ReflectSaved>();
                saved.map(|(_, saved)| saved.watch).collect()
            })
            .unwrap_or_default();
        for watch in watches {
            watch(app);
        }
    }
}

/// Starts the log of an agent spawned by another, so its header names
/// that agent and a restore links them again.
fn open_child_log(
    spawned: On<Add<SpawnedBy>>,
    agents: Query<(&AgentId, &SpawnedBy)>,
    ids: Query<&AgentId>,
    log: Res<SessionLog>,
) {
    if let Ok((child, parent)) = agents.get(spawned.entity)
        && let Ok(parent) = ids.get(parent.0)
    {
        log.0.open_log(&child.0, Some(&parent.0));
    }
}

/// Writes the frame's records. A failure is shown once.
fn write_logs(log: Res<SessionLog>, mut reported: Local<bool>, mut notices: MessageWriter<Notice>) {
    if let Err(failure) = log.flush()
        && !*reported
    {
        *reported = true;
        error!("the session log stopped: {failure}");
        notices.write(Notice::error(
            None,
            format!("The session is no longer saved: {failure}"),
        ));
    }
}

#[cfg(test)]
pub(crate) mod tests;
