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
//! changes for the log, announces each as a [`ConversationChanged`], and writes the
//! queue at the end of the frame, a subagent's log before its parent's.
//!
//! Records are appended while the session runs. Nothing is rewritten
//! except through rig-memory: a [`RewriteSession`] maps every saved record
//! in place, keeping each record's place and stamp, and the agents'
//! conversations and saved values become the rewritten ones;
//! [`restore`](super::restore) folds the logs back at startup.

use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::lifecycle::HookContext;
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::AppTypeRegistry;
use bevy_ecs::resource::IsResource;
use bevy_ecs::schedule::{ScheduleLabel, SystemSet};
use bevy_ecs::system::SystemParam;
use bevy_ecs::world::DeferredWorld;
use bevy_log::{error, warn};
use bevy_reflect::serde::TypedReflectSerializer;
use bevy_reflect::{CreateTypeData, Reflect, TypePath};
use rig_core::conversation::{Author, ChangeRecord};
use rig_memory::{
    LoadError, Place, Repair, Rewrite, Rewritten, SESSION, Session, Store, Versioned, Writer,
};
use web_time::{SystemTime, UNIX_EPOCH};

use super::WriteLogs;
use super::agent::{Agent, AgentId, Conversation, Notice, SpawnedBy};
use super::follows::{self, Mirroring, ReflectFollows};
use super::restore::{IgnoredSavedKeys, SavedKeys, SavedUpgrades, Shelved};
use super::restore::{SavedReader, resource_entity, saved_type};

/// Saves an agent component or a resource with the session: derive
/// `Reflect` and add `Saved` to its `#[reflect(..)]`, as
/// `#[reflect(Component, Saved)]` or `#[reflect(Resource, Saved)]`. Each
/// change of it is logged at the end of the frame under its type path, or
/// the stable key its plugin gives it ([`SavedKeys`]), with its type's
/// version ([`SavedUpgrades`]; an agent's in its log, a resource in the
/// session's [`SESSION`] log, written once an agent's log is) and restored
/// by reflection, read at its type's version through the steps its plugin
/// registered, inserted on the
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
/// registers any more is reported on restore and kept in the log. A value
/// no step reads, or that no longer fits its type, is reported, listed in
/// [`Shelved`] and not inserted; the next change of it first moves the old
/// value to a side key (rig-memory's `SET_ASIDE`), so nothing is lost. A
/// value a newer rig saved is reported and never written over in this run.
#[derive(Clone)]
pub struct ReflectSaved {
    watch: fn(&mut App),
}

impl<T: Component + Reflect + TypePath> CreateTypeData<T> for ReflectSaved {
    fn create_type_data(_: ()) -> Self {
        // `Changed` also sees an immutable value inserted in place of
        // another.
        let watch = |app: &mut App| {
            app.add_systems(Last, log_changed::<T>.in_set(Saving));
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
    (keys, upgrades): (Res<SavedKeys>, Res<SavedUpgrades>),
) {
    if !log.0.is_live() {
        return;
    }
    let path = T::type_path();
    let name = keys.key(path).unwrap_or(path);
    let v = upgrades.0.version(path);
    let registry = registry.read();
    for (entity, id, component) in &changed {
        let value = TypedReflectSerializer::new(component.as_partial_reflect(), &registry);
        let log_name = id.map_or(SESSION, |id| id.0.as_str());
        let starts = id.is_some() && !spawned.contains(entity);
        match serde_json::to_value(value) {
            Ok(value) => log
                .0
                .state(log_name, name, Some(Versioned { v, value }), starts),
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
/// agent pass and in `Last` ([`Saving`]). Followed components are mirrored
/// into their lines first ([`Mirroring`]), and after the save each agent
/// whose line may have changed is given its line's values
/// (`follows::sync`). So a change of a followed component is kept at the
/// entry it was made after, give or take a stage, and a line changed
/// before `Update`, such as by a key or a command, or before a stage gives
/// its values to `Update` or that stage, not to the next frame.
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

/// Saves the conversations changed outside the agent loop, such as by
/// commands and observers, in `Last` before the frame's records are
/// written.
pub(crate) fn save_in_last(world: &mut World) {
    save_now(world);
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

/// The systems in `Last` that queue what the frame changed for the log:
/// after the turns stopped on exit, before [`apply_rewrites`] and
/// [`WriteLogs`].
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct Saving;

/// Rewrites this session's saved history through rig-memory
/// ([`Writer::rewrite`]) at the end of the frame, once everything the
/// frame saved is queued and before it is written: every record of every
/// log, kept in its place with its stamp, and every file of the store, go
/// through `with`. Then every agent's conversation is the rewritten one
/// (its unsaved asides kept, passed through `with` too), and every saved
/// component or resource whose newest value changed is inserted again
/// from it, as a restore inserts it, so `On<Insert<T>>` observers see it.
/// [`SessionRewritten`] follows in the same frame.
///
/// A running turn keeps running; what it adds after the rewrite was not
/// mapped, nor was what a plugin keeps a copy of: a `Conversation` copied
/// before the rewrite and put back is saved whole, with the old text. Text
/// already sent to a model is not reached. Without a session store there
/// is no saved history, and nothing is rewritten. The map is any
/// [`Rewrite`], such as [`Strings`](rig_memory::Strings) for a redaction.
#[derive(Event, Clone)]
pub struct RewriteSession {
    /// Who rewrites it, kept in each rewritten log's header.
    pub by: Author,
    /// The map.
    pub with: Arc<dyn Rewrite>,
    /// Who asked, such as an entity of the plugin's own:
    /// [`SessionRewritten`] says so.
    pub asker: Option<Entity>,
}

/// A [`RewriteSession`] done, or why not, in the frame it ran, before
/// [`WriteLogs`]. The owners of what the kernel does not keep re-read
/// or map theirs on it, with the same `with` (rig-memory's
/// `rewrite_value`, `rewrite_file`): the effect log's writer, the scratch
/// files, and every plugin's copies, in memory or in files of its own.
#[derive(Message, Clone)]
pub struct SessionRewritten {
    /// Who rewrote it, as [`RewriteSession`] said.
    pub by: Author,
    /// Who asked, as [`RewriteSession`] said.
    pub asker: Option<Entity>,
    /// The map the session was rewritten through.
    pub with: Arc<dyn Rewrite>,
    /// What was rewritten, or why nothing was: nothing is changed when
    /// the session cannot be read, or a rewritten log would not read.
    pub result: Result<Rewritten, String>,
}

/// The rewrites asked for this frame.
#[derive(Resource, Default)]
pub(crate) struct Rewrites(Vec<RewriteSession>);

/// Queues a [`RewriteSession`] for the end of the frame.
pub(crate) fn ask_rewrite(asked: On<RewriteSession>, mut rewrites: ResMut<Rewrites>) {
    rewrites.0.push(asked.event().clone());
}

/// Whether a rewrite was asked for this frame.
pub(crate) fn rewrites_asked(rewrites: Res<Rewrites>) -> bool {
    !rewrites.0.is_empty()
}

/// How many rounds of rewrites asked while rewrites are applied, such as by
/// an observer of a value reread, run in one frame; the rest wait for the
/// next.
const REWRITE_ROUNDS: usize = 8;

/// Runs the rewrites asked for, as [`RewriteSession`] says, each with its
/// own author, and those asked meanwhile, in the same frame.
pub(crate) fn apply_rewrites(world: &mut World) {
    let log = world
        .get_resource::<SessionLog>()
        .cloned()
        .unwrap_or_default();
    for _ in 0..REWRITE_ROUNDS {
        let asked = world
            .get_resource_mut::<Rewrites>()
            .map(|mut rewrites| std::mem::take(&mut rewrites.0))
            .unwrap_or_default();
        if asked.is_empty() {
            return;
        }
        for RewriteSession { by, with, asker } in asked {
            let result = log.0.rewrite(by.clone(), &*with);
            let result = result.map_err(|failure| failure.to_string());
            if let Ok(rewritten) = &result {
                for failure in reread(world, rewritten, &*with) {
                    let failure = format!("After rewriting the session: {failure}.");
                    world.write_message(Notice::error(None, failure));
                }
            }
            world.write_message(SessionRewritten {
                by,
                asker,
                with,
                result,
            });
        }
        // What observers of the reread values asked, such as another rewrite.
        world.flush();
    }
}

/// Makes each live agent's conversation and each saved value the
/// rewritten one, mapping its unsaved asides through `with`; what failed.
fn reread(world: &mut World, rewritten: &Rewritten, with: &dyn Rewrite) -> Vec<String> {
    let mut agents = world.query_filtered::<(Entity, &AgentId), With<Agent>>();
    let agents: Vec<(Entity, String)> = agents
        .iter(world)
        .map(|(agent, id)| (agent, id.0.clone()))
        .collect();
    let registry = world.get_resource::<AppTypeRegistry>().cloned();
    let registry = registry.unwrap_or_default();
    let registry = registry.read();
    let mut reader = SavedReader::new(world);
    let mut failures = Vec::new();
    for log in &rewritten.logs {
        let agent = agents.iter().find(|(_, id)| *id == log.log);
        let agent = match (agent, log.log == SESSION) {
            (Some(&(agent, _)), _) => Some(agent),
            (None, true) => None,
            // Not live, such as a retired agent's.
            (None, false) => continue,
        };
        if let Some(mut conversation) = agent.and_then(|agent| world.get_mut::<Conversation>(agent))
        {
            let at = Place::Unsaved { log: &log.log };
            let unsaved: Vec<_> = conversation
                .asides()
                .filter(|(_, aside)| !aside.saved)
                .map(|(id, aside)| (id, aside.clone()))
                .collect();
            for (id, mut aside) in unsaved {
                if with.aside(&at, &mut aside) {
                    conversation.replace_aside(id, aside.unsaved());
                }
            }
            if let Err(failure) = conversation.reread(log.conversation.clone()) {
                failures.push(format!(
                    "the conversation of {} was not reread: {failure}",
                    log.log
                ));
            }
        }
        let id = agent.map(|_| AgentId(log.log.clone()));
        for key in &log.changed_keys {
            let inserted = match log.state.get(key) {
                Some(value) => reader
                    .read(&registry, &log.log, id.as_ref(), &log.state, key, value)
                    .and_then(|read| {
                        let Some((component, value)) = read else {
                            return Ok(());
                        };
                        let mut entity = match agent {
                            Some(agent) => world
                                .get_entity_mut(agent)
                                .map_err(|failure| failure.to_string())?,
                            None => resource_entity(world, component)?,
                        };
                        component.insert(&mut entity, value.as_partial_reflect(), &registry);
                        Ok(())
                    }),
                // A value the rewrite made null is gone, from an agent.
                None => saved_type(reader.path(key), &registry).map(|(_, component)| {
                    if let Some(mut entity) =
                        agent.and_then(|agent| world.get_entity_mut(agent).ok())
                    {
                        component.remove(&mut entity);
                    }
                }),
            };
            if let Err(failure) = inserted {
                failures.push(format!("the saved value `{key}` was not reread: {failure}"));
            }
        }
    }
    reader.finish(world);
    failures
}

/// Restores the session at startup, settles what a crash or restart left
/// half done, and logs everything after.
pub struct SessionLogPlugin;

impl Plugin for SessionLogPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(PreStartup, super::restore::restore_session)
            .add_systems(Startup, super::restore::reconcile)
            .add_systems(Last, write_logs.in_set(WriteLogs))
            .configure_sets(SaveConversations, Mirroring.before(save_conversations))
            .add_systems(SaveConversations, follows::sync.after(save_conversations))
            .add_observer(open_child_log)
            .init_resource::<SavedUpgrades>()
            .init_resource::<IgnoredSavedKeys>()
            .init_resource::<SavedKeys>()
            .init_resource::<Shelved>();
    }

    /// Watches every type registered as [`ReflectSaved`] or
    /// [`ReflectFollows`], once every plugin had its say in the registry.
    fn finish(&self, app: &mut App) {
        let watches: Vec<fn(&mut App)> = app
            .world()
            .get_resource::<AppTypeRegistry>()
            .map(|registry| {
                let registry = registry.read();
                let saved = registry.iter_with_data::<ReflectSaved>();
                let saved = saved.filter_map(|(registration, saved)| {
                    if registration.data::<ReflectFollows>().is_none() {
                        return Some(saved.watch);
                    }
                    let path = registration.type_info().type_path();
                    warn!("{path} is both Saved and Follows; it follows its line");
                    None
                });
                let follows = registry.iter_with_data::<ReflectFollows>();
                saved
                    .chain(follows.map(|(_, follows)| follows.watch))
                    .collect()
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
