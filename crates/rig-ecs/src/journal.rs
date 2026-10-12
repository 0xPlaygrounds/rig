//! The session's journal in the app. Each agent's log, in the session's
//! [`SessionStore`], gets every change made to the agent's
//! [`Conversation`] and every change of a [`ReflectSaved`] component, such
//! as the agent's model, reasoning setting, system prompt, tool access and
//! last usage. The format, its writer and its readers are rig-memory's.
//!
//! A conversation is changed by calling its methods; each change waits in
//! the conversation's outbox until the kernel saves it: between the stages
//! of every pass, at the end of the frame, before a tool that may change
//! something starts, and when a turn is interrupted. Saving queues the
//! changes for the log, announces each as a [`Committed`], and writes the
//! queue at the end of the frame, a subagent's log before its parent's.
//! Nothing is ever rewritten; [`restore`](super::restore) folds the logs
//! back at startup.

use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::AppTypeRegistry;
use bevy_ecs::resource::IsResource;
use bevy_ecs::system::SystemParam;
use bevy_ecs::world::DeferredWorld;
use bevy_log::error;
use bevy_reflect::serde::TypedReflectSerializer;
use bevy_reflect::{CreateTypeData, Reflect, TypePath};
use rig_core::conversation::Recorded;
use rig_memory::{LoadError, Repair, SESSION, Session, Store, Writer};
use web_time::{SystemTime, UNIX_EPOCH};

use super::agent::{Agent, AgentId, Conversation, Notice, SpawnedBy};
use super::{StopTurns, WriteJournal};

/// Saves an agent component or a resource with the session: derive
/// `Reflect` and add `Saved` to its `#[reflect(..)]`, as
/// `#[reflect(Component, Saved)]` or `#[reflect(Resource, Saved)]`. Each
/// change of it is logged at the end of the frame by its type path (an
/// agent's in its log, a resource in the session's [`SESSION`] log, written
/// once an agent's log is) and restored by reflection, inserted on the
/// agent it was saved with before [`SessionRestored`] exists, so an
/// `On<Insert<T>>` observer re-arms what it says; a value that serializes
/// to `null` is saved as absent, and so is an agent component removed. A
/// resource is restored in place of the one its plugin inserted while the
/// app was built.
///
/// A type reflected opaquely with serde, as
/// `#[reflect(opaque, Component, Saved, Clone, Serialize, Deserialize)]`,
/// is saved and restored through its serde form: the way to save a
/// component holding rig-core data that is not `Reflect`, such as a
/// [`MessageId`](rig_core::conversation::MessageId).
///
/// Registration is a runtime fact: a generic type is saved only once
/// registered (`app.register_type::<T>()`), and a logged type no build
/// registers any more is reported on restore and kept in the log.
#[derive(Clone)]
pub struct ReflectSaved {
    watch: fn(&mut App),
}

impl<T: Component + Reflect + TypePath> CreateTypeData<T> for ReflectSaved {
    fn create_type_data(_: ()) -> Self {
        // `Changed` also sees an immutable value inserted in place of
        // another.
        let watch = |app: &mut App| {
            let log = log_changed::<T>.after(StopTurns).before(WriteJournal);
            app.add_systems(Last, log);
        };
        Self { watch }
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
) {
    if !log.0.is_live() {
        return;
    }
    let name = T::type_path();
    let registry = registry.read();
    for (entity, id, component) in &changed {
        let value = TypedReflectSerializer::new(component.as_partial_reflect(), &registry);
        let key = id.map_or(SESSION, |id| id.0.as_str());
        let starts = id.is_some() && !spawned.contains(entity);
        match serde_json::to_value(value) {
            Ok(value) => log.0.state(key, name, Some(value), starts),
            Err(failure) => error!("not logging {name}: {failure}"),
        }
    }
    for entity in removed.read() {
        if let Ok(id) = agents.get(entity)
            && !changed.contains(entity)
        {
            log.0.state(&id.0, name, None, !spawned.contains(entity));
        }
    }
}

/// Milliseconds since the Unix epoch at `at`; 0 before it.
pub(crate) fn millis(at: SystemTime) -> u64 {
    at.duration_since(UNIX_EPOCH)
        .map(|elapsed| u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or_default()
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
    /// Writes every queued record ([`Writer::flush`]).
    pub(crate) fn flush(&self) {
        self.0.flush();
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

/// Opens `store` for this app: reads the session it holds, refusing an old
/// or broken one before anything is written, sets each torn last line
/// aside, and keeps what it read for the restore at startup. On success
/// `store` is the app's [`SessionStore`] and its log writes there once the
/// session is restored; on failure the app keeps nothing of it. An app that
/// only inserts a `SessionStore` has it opened by the restore, which shows
/// a refusal as a notice.
///
/// # Errors
///
/// The [`LoadError`] reading the session gave: a session saved by an older
/// rig, a log in another format, a record that cannot be read, or the
/// store failing.
pub fn open_session(world: &mut World, store: Arc<dyn Store>) -> Result<(), LoadError> {
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
pub struct Committed {
    /// The agent.
    pub agent: Entity,
    /// The change: what changed, who made it and when.
    pub change: Recorded,
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
/// to `log`; one [`Committed`] per change saved.
pub(crate) fn save_conversation(
    log: &SessionLog,
    agent: Entity,
    id: &AgentId,
    conversation: &mut Conversation,
) -> Vec<Committed> {
    let saved = log.0.save(&id.0, conversation);
    let committed = |change| Committed { agent, change };
    saved.into_iter().map(committed).collect()
}

/// Saves every agent conversation changed since it last ran, without
/// marking any changed again.
#[derive(SystemParam)]
pub(crate) struct Drain<'w, 's> {
    agents:
        Query<'w, 's, (Entity, &'static AgentId, &'static mut Conversation), Changed<Conversation>>,
    log: Res<'w, SessionLog>,
    committed: MessageWriter<'w, Committed>,
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

/// Saves the conversations changed outside the agent loop, such as by
/// commands and observers, in `Last` before the frame's records are
/// written.
pub(crate) fn save_conversations(mut drain: Drain) {
    drain.save();
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
    mut committed: MessageWriter<Committed>,
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
pub struct JournalPlugin;

impl Plugin for JournalPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreStartup, super::restore::restore_session)
            .add_systems(Startup, super::restore::reconcile)
            .add_systems(Last, write_logs.in_set(WriteJournal))
            .add_observer(open_child_log);
    }

    /// Watches every type registered as [`ReflectSaved`], once every
    /// plugin had its say in the registry.
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
    log.flush();
    if !*reported && let Some(failure) = log.0.failure() {
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
