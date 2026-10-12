use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::system::RunSystemOnce;
use bevy_reflect::Reflect;
use rig_cassette::journal::{
    Entry, Halt, JournalStore, MemoryStore, MessageId, Origin, STOPPED, SavedSession, fork_session,
    read_session,
};
use rig_core::completion::Message;
use rig_core::message::UserContent;

use super::{Commit, Committed, JournalPlugin, ReflectSaved, SessionStore};
use crate::AgentPlugin;
use crate::agent::{Agent, AgentId, Conversation, SpawnedBy};

/// Adds `message` to the conversation of `agent` as [`Commit::message`]
/// does.
pub(crate) fn commit_message(world: &mut World, agent: Entity, message: Message) {
    let add = |In((agent, message)): In<(Entity, Message)>,
               mut agents: Query<&mut Conversation>,
               mut commit: Commit| {
        if let Ok(mut conversation) = agents.get_mut(agent) {
            commit.message(agent, &mut conversation, message);
        }
    };
    let added = world.run_system_cached_with(add, (agent, message));
    assert!(added.is_ok(), "{added:?}");
}

/// An app on `store`, after its first frame restored the session.
fn app(store: &MemoryStore) -> App {
    let mut app = App::new();
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, JournalPlugin))
        .insert_resource(SessionStore::new(store.clone()));
    app.finish();
    app.update();
    app
}

/// The agents the user started, oldest id first.
fn agents(app: &mut App) -> Vec<Entity> {
    let world = app.world_mut();
    let mut agents =
        world.query_filtered::<(Entity, &AgentId), (With<Agent>, Without<SpawnedBy>)>();
    let mut agents: Vec<(Entity, String)> = agents
        .iter(world)
        .map(|(agent, id)| (agent, id.0.clone()))
        .collect();
    agents.sort_by(|a, b| a.1.cmp(&b.1));
    agents.into_iter().map(|(agent, _)| agent).collect()
}

fn messages(app: &App, agent: Entity) -> Vec<Message> {
    let conversation = app.world().get::<Conversation>(agent);
    conversation.map_or_else(Vec::new, |conversation| conversation.messages().to_vec())
}

/// Runs `edit` with the conversation of `agent` and a [`Commit`].
fn commit(
    app: &mut App,
    agent: Entity,
    edit: impl Fn(Entity, &mut Conversation, &mut Commit) + Send + Sync + 'static,
) {
    let run = move |mut agents: Query<&mut Conversation>, mut commit: Commit| {
        if let Ok(mut conversation) = agents.get_mut(agent) {
            edit(agent, &mut conversation, &mut commit);
        }
    };
    assert!(app.world_mut().run_system_once(run).is_ok());
}

/// Four messages: two questions and their answers.
fn talk(app: &mut App, agent: Entity) {
    for n in 1..=2 {
        commit_message(
            app.world_mut(),
            agent,
            Message::user(format!("question {n}")),
        );
        commit_message(
            app.world_mut(),
            agent,
            Message::assistant(format!("answer {n}")),
        );
    }
    app.update();
}

#[test]
fn a_cut_and_a_retired_agent_are_logged_and_a_restore_follows_them() {
    let store = MemoryStore::default();
    let mut first = app(&store);
    let Some(&agent) = agents(&mut first).first() else {
        return;
    };
    talk(&mut first, agent);
    // A subagent, and the one it spawned: retiring the first takes both.
    let child = first.world_mut().spawn((Agent, SpawnedBy(agent))).id();
    let grandchild = first.world_mut().spawn((Agent, SpawnedBy(child))).id();
    for spawned in [child, grandchild] {
        commit_message(first.world_mut(), spawned, Message::user("work"));
    }
    commit(&mut first, agent, |agent, conversation, commit| {
        commit.splice(agent, conversation, 2.., []);
    });
    commit(&mut first, child, |child, _, commit| commit.retire(child));
    first.update();
    assert_eq!(messages(&first, agent).len(), 2);
    assert!(
        first.world().get_entity(grandchild).is_err(),
        "retired with its parent"
    );
    let committed = first.world().resource::<Messages<Committed>>();
    let mut cursor = committed.get_cursor();
    let changes: Vec<_> = cursor
        .read(committed)
        .filter(|change| !matches!(change, Committed::Message { .. }))
        .map(|change| match change {
            Committed::Splice { removed, .. } => format!("cut {}", removed.len()),
            Committed::Retired { .. } => "retired".to_owned(),
            other => format!("{other:?}"),
        })
        .collect();
    assert_eq!(changes, ["cut 2", "retired"]);
    drop(first);

    let mut second = app(&store);
    let restored = agents(&mut second);
    assert_eq!(restored.len(), 1, "only the agent the user started");
    let Some(&agent) = restored.first() else {
        return;
    };
    let kept = [Message::user("question 1"), Message::assistant("answer 1")];
    assert_eq!(messages(&second, agent), kept);
    let world = second.world_mut();
    assert_eq!(world.query::<&Agent>().iter(world).count(), 1);
}

/// Every message of `conversation` with its origins and id, and its halt.
fn state(conversation: &rig_cassette::journal::Conversation) -> (Vec<Entry>, Option<Halt>) {
    let entries = (0..conversation.messages().len()).filter_map(|at| conversation.entry(at));
    (entries.collect(), conversation.halted())
}

/// The session in `store` as a restore reads it, by agent id.
fn saved_state(store: &MemoryStore, id: &AgentId) -> Option<(Vec<Entry>, Option<Halt>)> {
    let session = read_session(store).ok()?;
    let saved = session.agents.into_iter().find(|saved| saved.id == id.0)?;
    Some(state(&saved.conversation))
}

#[test]
fn splices_keep_ids_origins_and_the_halt_and_a_restore_makes_the_same_changes() {
    let store = MemoryStore::default();
    let mut app = app(&store);
    let Some(&agent) = agents(&mut app).first() else {
        return;
    };
    let id = app.world().get::<AgentId>(agent).cloned();
    let Some(id) = id else { return };
    commit(&mut app, agent, |agent, conversation, commit| {
        commit.message(agent, conversation, Message::user("question 1"));
        commit.message(agent, conversation, Message::assistant("answer 1"));
        let note = Some(Origin::plugin("note"));
        commit.message_from(agent, conversation, Message::user("question 2"), note);
        commit.message(agent, conversation, Message::assistant("answer 2"));
        commit.message(agent, conversation, Message::user("question 3"));
        commit.halt(agent, conversation, Halt::Stopped);
    });
    let ids = |app: &App| {
        let conversation = app.world().get::<Conversation>(agent);
        conversation.map_or_else(Vec::new, |conversation| {
            let len = conversation.messages().len();
            (0..len).filter_map(|at| conversation.id(at)).collect()
        })
    };
    let before: Vec<MessageId> = ids(&app);
    assert_eq!(before.len(), 5);
    let [q1, a1, q2, a2, q3] = before.as_slice() else {
        return;
    };
    // An edit in place, then a splice in the middle, then a summary over
    // both that merges into the prompt it kept.
    commit(&mut app, agent, |agent, conversation, commit| {
        let Some(mut edited) = conversation.entry(2) else {
            return;
        };
        edited.message = Message::user("question 2, edited");
        commit.splice(agent, conversation, 2..3, [edited]);
        let shorter = Message::assistant("answer 1, shorter");
        commit.splice(agent, conversation, 1..2, [shorter.into()]);
        let summary = Message::user("Summary: question 1.");
        let summary = Entry::from(summary, Origin::plugin("compaction"));
        commit.splice(agent, conversation, 0..2, [summary]);
    });
    app.update();
    let after = ids(&app);
    assert_eq!(
        after,
        [*q2, *a2, *q3],
        "the edited, kept prompt keeps its id"
    );
    let live = app.world().get::<Conversation>(agent).map(|c| state(c));
    let Some((entries, halted)) = live.clone() else {
        return;
    };
    assert_eq!(
        halted,
        Some(Halt::Stopped),
        "a splice before the end keeps it"
    );
    let first = entries
        .first()
        .map(|entry| (&entry.message, &entry.origins));
    let texts = ["Summary: question 1.", "question 2, edited"].map(UserContent::text);
    let merged = Message::User {
        content: texts.into(),
    };
    let origins = vec![
        (0, Origin::plugin("compaction")),
        (1, Origin::plugin("note")),
    ];
    assert_eq!(first, Some((&merged, &origins)));
    let committed = app.world().resource::<Messages<Committed>>();
    let mut cursor = committed.get_cursor();
    let last_splice = cursor.read(committed).filter_map(|change| match change {
        Committed::Splice {
            removed, changed, ..
        } => Some((removed.clone(), changed.clone())),
        _ => None,
    });
    let last_splice = last_splice.last();
    assert!(last_splice.is_some(), "splices are committed");
    let Some((removed, changed)) = last_splice else {
        return;
    };
    assert!(!removed.contains(q2) && removed.contains(q1) && !removed.contains(a1));
    assert_eq!(changed, [*q2]);
    assert_eq!(saved_state(&store, &id), live, "restored as made live");

    // A cut of the tail clears the halt, and a restore agrees.
    commit(&mut app, agent, |agent, conversation, commit| {
        let last = conversation.messages().len().saturating_sub(1);
        commit.splice(agent, conversation, last.., []);
    });
    app.update();
    let live = app.world().get::<Conversation>(agent).map(|c| state(c));
    assert_eq!(live.as_ref().map(|(_, halted)| *halted), Some(None));
    assert_eq!(saved_state(&store, &id), live);
    drop(app);
    let mut second = self::app(&store);
    let Some(&agent) = agents(&mut second).first() else {
        return;
    };
    let restored = second.world().get::<Conversation>(agent);
    assert_eq!(restored.map(|c| state(c)), live);
}

#[test]
fn a_retried_request_is_not_said_to_be_stopped_after_a_restore() {
    let store = MemoryStore::default();
    let mut app = app(&store);
    let Some(&agent) = agents(&mut app).first() else {
        return;
    };
    commit(&mut app, agent, |agent, conversation, commit| {
        commit.message(agent, conversation, Message::user("run the tests"));
        commit.halt(agent, conversation, Halt::Stopped);
        // A retry asks again; the user steers before the model answers.
        assert!(commit.resume(agent, conversation));
        commit.message(agent, conversation, Message::user("only the fast ones"));
    });
    app.update();
    let live = messages(&app, agent);
    let stopped = live.iter().any(|message| {
        matches!(message, Message::User { content }
            if content.contains(&UserContent::text(STOPPED)))
    });
    assert!(!stopped, "{live:?}");
    drop(app);
    let mut second = self::app(&store);
    let Some(&agent) = agents(&mut second).first() else {
        return;
    };
    assert_eq!(messages(&second, agent), live);
}

/// A plugin's counter, saved with each agent.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved)]
struct Counter(u32);

#[test]
fn a_saved_change_starts_the_agents_log_and_the_values_it_was_spawned_with_do_not() {
    let store = MemoryStore::default();
    let mut first = app(&store);
    let Some(&agent) = agents(&mut first).first() else {
        return;
    };
    first.update();
    let id = first.world().get::<AgentId>(agent).cloned();
    let log = |id: Option<AgentId>| id.and_then(|id| store.read(&id.0).ok());
    assert_eq!(log(id.clone()), None, "a session nobody used is not kept");
    // A plugin's state, such as a mode the user turned on before any
    // message, is kept for the next start.
    first.world_mut().entity_mut(agent).insert(Counter(1));
    first.update();
    assert!(log(id).is_some());
    drop(first);

    let mut second = app(&store);
    let Some(&agent) = agents(&mut second).first() else {
        return;
    };
    let counter = second
        .world()
        .get::<Counter>(agent)
        .map(|counter| counter.0);
    assert_eq!(counter, Some(1));
}

#[test]
fn a_session_is_read_without_writing_and_forked_as_of_a_message() {
    let store = MemoryStore::default();
    let mut app = app(&store);
    let Some(&agent) = agents(&mut app).first() else {
        return;
    };
    for n in 1..=2 {
        app.world_mut().entity_mut(agent).insert(Counter(n));
        let said = [
            Message::user(format!("question {n}")),
            Message::assistant(format!("answer {n}")),
        ];
        for message in said {
            commit_message(app.world_mut(), agent, message);
        }
        app.update();
    }
    let second_question = app
        .world()
        .get::<Conversation>(agent)
        .and_then(|conversation| conversation.id(2));
    // A summary of the first question, merged into the second, which keeps
    // its id.
    commit(&mut app, agent, |agent, conversation, commit| {
        let summary = Message::user("The first question.");
        let summary = Entry::from(summary, Origin::plugin("compaction"));
        commit.splice(agent, conversation, 0..2, [summary]);
    });
    app.update();
    let id = app
        .world()
        .get::<AgentId>(agent)
        .cloned()
        .unwrap_or_default();
    drop(app);
    // A crash left half a line.
    assert!(store.append(&id.0, b"{\"seq\":").is_ok());
    let before = store.read(&id.0).ok();

    let read = read_session(&store);
    assert!(read.is_ok());
    let SavedSession { agents, .. } = read.unwrap_or_default();
    assert_eq!(store.read(&id.0).ok(), before, "read without writing");
    assert_eq!(agents.len(), 1);
    let Some(saved) = agents.first() else {
        return;
    };
    assert!(saved.id == id.0 && saved.parent.is_none() && !saved.retired);
    assert!(saved.started_ms > 0);
    // The conversation as it is, the summary in place of the first two
    // messages.
    assert_eq!(saved.conversation.messages().len(), 2);
    assert_eq!(saved.conversation.id(0), second_question);
    let counter = saved.saved.get(std::any::type_name::<Counter>());
    assert_eq!(counter, Some(&serde_json::json!(2)), "{:?}", saved.saved);

    // Forked as of the second question: the conversation and the counter
    // as they were then.
    let fork = MemoryStore::default();
    assert!(second_question.is_some(), "the second question was logged");
    let Some(second_question) = second_question else {
        return;
    };
    assert!(fork_session(&store, &fork, Some((&id.0, second_question))).is_ok());
    let forked = read_session(&fork).unwrap_or_default();
    assert_eq!(forked.agents.len(), 1);
    let Some(branch) = forked.agents.first() else {
        return;
    };
    let kept = [Message::user("question 1"), Message::assistant("answer 1")];
    assert_eq!(branch.conversation.messages(), kept);
    let counter = branch.saved.get(std::any::type_name::<Counter>());
    assert_eq!(counter, Some(&serde_json::json!(1)));
}

/// A plugin's mode, saved with the session.
#[derive(Resource, Reflect, Default)]
#[reflect(Resource, Saved)]
struct Mode(String);

#[test]
fn a_session_saved_before_the_log_format_moved_still_restores() {
    // rig-cassette's fixture: this kernel wrote it, with `Counter` and
    // `Mode` of this module, before the format moved there.
    let store = MemoryStore::default();
    let mut log = None;
    for line in include_str!("../../../rig-cassette/src/journal/session/saved-session.txt").lines()
    {
        let written = match line.strip_prefix("=== ") {
            Some(name) => {
                log = Some(name);
                Ok(())
            }
            None => match log.map(|name| (name, name.strip_prefix("blob "))) {
                Some((_, Some(blob))) => store.put_blob(blob, line.as_bytes()),
                Some((agent, None)) => store.append(agent, format!("{line}\n").as_bytes()),
                None => Ok(()),
            },
        };
        assert!(written.is_ok());
    }
    let mut app = App::new();
    app.add_plugins((AgentPlugin, JournalPlugin))
        .register_type::<Mode>()
        .insert_resource(SessionStore::new(store.clone()));
    app.finish();
    app.update();
    let world = app.world_mut();
    let mut restored = world.query::<(Entity, &AgentId, Option<&SpawnedBy>, &Conversation)>();
    let mut restored: Vec<_> = restored
        .iter(world)
        .map(|(entity, id, parent, conversation)| {
            let texts = conversation.messages().len();
            (entity, id.0.clone(), parent.map(|parent| parent.0), texts)
        })
        .collect();
    restored.sort_by(|a, b| a.1.cmp(&b.1));
    let main = restored.iter().find(|(.., parent, _)| parent.is_none());
    let main = main.map(|(entity, ..)| *entity);
    let shape: Vec<_> = restored
        .iter()
        .map(|(_, id, parent, messages)| (id.as_str(), *parent, *messages))
        .collect();
    let main_id = "cd9a3280-91f0-4fb4-aa5c-c1c3629a0f32";
    // The agent and its subagent, not the retired one.
    assert_eq!(shape, [(main_id, None, 5), ("child-agent", main, 1)]);
    let Some(main) = main else {
        return;
    };
    let conversation = world.get::<Conversation>(main);
    let halted = conversation.and_then(|conversation| conversation.halted());
    assert_eq!(halted, Some(Halt::Kept), "the user's last message waits");
    let counter = world.get::<Counter>(main).map(|counter| counter.0);
    assert_eq!(counter, Some(3));
    let mode = world.get_resource::<Mode>().map(|mode| mode.0.as_str());
    assert_eq!(mode, Some("plan"));
    // Nothing was rewritten.
    let main_log = store.read(main_id).ok();
    app.update();
    assert_eq!(store.read(main_id).ok(), main_log);
}
