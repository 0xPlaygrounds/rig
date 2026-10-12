//! The session's journal in the app: each agent's log, in the session's
//! [`SessionStore`], gets a record per change a [`Commit`] made (a message,
//! a splice, a halt, the agent retired) and per change of a
//! [`ReflectSaved`] component, such as the agent's model, reasoning
//! setting, system prompt, tool access and last usage. The format, its
//! writer and its readers are rig-cassette's ([`rig_cassette::journal`]):
//! records are queued as they happen and written at the end of the frame,
//! a subagent's before its parent's, and at once before a tool that may
//! change something runs. Nothing is ever rewritten;
//! [`restore`](super::restore) folds the logs back at startup.

use std::ops::{Bound, RangeBounds};
use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::AppTypeRegistry;
use bevy_ecs::resource::IsResource;
use bevy_ecs::system::SystemParam;
use bevy_log::error;
use bevy_reflect::serde::TypedReflectSerializer;
use bevy_reflect::{CreateTypeData, Reflect, TypePath};
use rig_cassette::journal::{
    Entry, Halt, JournalStore, MessageId, Origin, SESSION, Spliced, Writer,
};
use rig_core::completion::Message;
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
            Ok(value) => log.0.component(key, name, Some(value), starts),
            Err(failure) => error!("not logging {name}: {failure}"),
        }
    }
    for entity in removed.read() {
        if let Ok(id) = agents.get(entity)
            && !changed.contains(entity)
        {
            log.0
                .component(&id.0, name, None, !spawned.contains(entity));
        }
    }
}

/// Milliseconds since the Unix epoch at `at`; 0 before it.
pub(crate) fn millis(at: SystemTime) -> u64 {
    at.duration_since(UNIX_EPOCH)
        .map(|elapsed| u64::try_from(elapsed.as_millis()).unwrap_or(u64::MAX))
        .unwrap_or_default()
}

/// Where the session is kept, such as a rig-cassette `MemoryStore` or
/// `JsonlDirStore`: inserted while the app is built, before it runs, by
/// any plugin. Without one nothing is kept.
#[derive(Resource, Clone)]
pub struct SessionStore(pub Arc<dyn JournalStore>);

impl SessionStore {
    /// The session kept in `store`.
    pub fn new(store: impl JournalStore) -> Self {
        Self(Arc::new(store))
    }
}

/// The session's agent logs: rig-cassette's [`Writer`] of the records
/// queued for them and each agent's latest-wins state. Only a [`Commit`]
/// and the saved components' watchers log; nothing is logged before the
/// session was restored, or without a [`SessionStore`].
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
/// remember (`run_if(resource_exists::<SessionRestored>)`). It stays when
/// a log write fails later, which stops only the logging. An app without
/// a store never gets one.
#[derive(Resource, Clone, Copy, Debug, Default)]
pub struct SessionRestored;

/// Changes agents' conversations, the one way they change: each change is
/// logged, with a message's images stored as blobs, and announced as a
/// [`Committed`], so a plugin sees every change the log records.
#[derive(SystemParam)]
pub struct Commit<'w, 's> {
    ids: Query<'w, 's, &'static AgentId>,
    log: Res<'w, SessionLog>,
    committed: MessageWriter<'w, Committed>,
    commands: Commands<'w, 's>,
}

impl Commit<'_, '_> {
    /// Adds `message` to `conversation`, that of `agent`: into its last
    /// message when both are the user's, so user and model keep taking
    /// turns.
    pub fn message(&mut self, agent: Entity, conversation: &mut Conversation, message: Message) {
        self.message_from(agent, conversation, message, None);
    }

    /// [`Self::message`], with where it came from when not from the user,
    /// the model or a tool.
    pub fn message_from(
        &mut self,
        agent: Entity,
        conversation: &mut Conversation,
        message: Message,
        origin: Option<Origin>,
    ) {
        let Ok(id) = self.ids.get(agent) else {
            return;
        };
        self.log.0.message(&id.0, &message, origin.as_ref());
        self.committed.write(Committed::Message {
            agent,
            message: message.clone(),
            origin: origin.clone(),
        });
        conversation.0.append(message, origin);
    }

    /// Replaces messages `range` of `conversation`, that of `agent`, with
    /// `with`, merging a user message into a user message beside it: an
    /// entry taken with [`entry`](rig_cassette::journal::Conversation::entry)
    /// keeps its message's id, and
    /// a merged message keeps the id that was there before. It is logged,
    /// so a restore makes the same change, and announced as a
    /// [`Committed::Splice`]. A cut is `splice(agent, c, len.., [])`; an
    /// edit in place passes `conversation.entry(i)` back changed. A splice
    /// that reaches the end clears the halt; one before it keeps it.
    pub fn splice(
        &mut self,
        agent: Entity,
        conversation: &mut Conversation,
        range: impl RangeBounds<usize>,
        with: impl IntoIterator<Item = Entry>,
    ) {
        let Ok(id) = self.ids.get(agent) else {
            return;
        };
        let len = conversation.messages().len();
        let to = match range.end_bound() {
            Bound::Included(&last) => last.saturating_add(1),
            Bound::Excluded(&end) => end,
            Bound::Unbounded => len,
        }
        .min(len);
        let from = match range.start_bound() {
            Bound::Included(&first) => first,
            Bound::Excluded(&before) => before.saturating_add(1),
            Bound::Unbounded => 0,
        }
        .min(to);
        let with: Vec<Entry> = with.into_iter().collect();
        if from == to && with.is_empty() {
            return;
        }
        self.log.0.splice(&id.0, from, to, &with);
        let Spliced { removed, changed } = conversation.0.splice(from, to, with);
        self.committed.write(Committed::Splice {
            agent,
            removed,
            changed,
        });
    }

    /// Retires `agent` for good: the log says so, so a restore brings back
    /// neither it nor the agents it spawned, and it is despawned with them
    /// and their work.
    pub fn retire(&mut self, agent: Entity) {
        let Ok(id) = self.ids.get(agent) else {
            return;
        };
        self.log.0.retire(&id.0);
        self.committed.write(Committed::Retired { agent });
        self.commands.entity(agent).try_despawn();
    }

    /// Leaves the user's last message in `conversation`, that of `agent`,
    /// unanswered for `reason` when the model owes an answer to it, so a
    /// restore does not send it to the model. Not for a turn the app's
    /// exit stops: the restart carries that one on.
    pub fn halt(&mut self, agent: Entity, conversation: &mut Conversation, reason: Halt) {
        if let Ok(id) = self.ids.get(agent)
            && conversation.0.halt(reason)
        {
            let reason = Some(reason);
            self.log.0.halt(&id.0, reason);
            self.committed.write(Committed::Halt { agent, reason });
        }
    }

    /// Asks the model again for the user's last message, as a retry does:
    /// a halted conversation is no longer halted, which the log records.
    /// Whether the conversation ends in the user's message.
    pub(crate) fn resume(&mut self, agent: Entity, conversation: &mut Conversation) -> bool {
        let halted = conversation.halted().is_some();
        let asks = conversation.0.resume();
        if halted && let Ok(id) = self.ids.get(agent) {
            self.log.0.halt(&id.0, None);
            self.committed.write(Committed::Halt {
                agent,
                reason: None,
            });
        }
        asks
    }
}

/// A change a [`Commit`] made to an agent's conversation, in the order
/// made; a restore makes none.
#[derive(Message, Clone, Debug)]
pub enum Committed {
    /// A message was added, into the last one when both are the user's.
    Message {
        agent: Entity,
        message: Message,
        /// Where it came from, when not from the user, the model or a tool.
        origin: Option<Origin>,
    },
    /// Messages were spliced: `removed` are the ids no longer in the
    /// conversation, `changed` those of the messages the splice made or
    /// changed. Ids stay valid across several splices in one frame; who
    /// spliced is read from the changed messages' item origins.
    Splice {
        agent: Entity,
        removed: Vec<MessageId>,
        changed: Vec<MessageId>,
    },
    /// The user's last message was left unanswered for `reason`, or, with
    /// none, asked for again.
    Halt { agent: Entity, reason: Option<Halt> },
    /// The agent was retired, and despawned.
    Retired { agent: Entity },
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
        log.0.open(&child.0, Some(&parent.0));
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
