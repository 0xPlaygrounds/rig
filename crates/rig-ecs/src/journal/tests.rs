use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::system::RunSystemOnce;
use bevy_reflect::Reflect;
use rig_cassette::journal::{JournalStore, MemoryStore};
use rig_core::completion::Message;
use rig_core::message::UserContent;

use super::{Commit, Committed, JournalPlugin, ReflectSaved, SessionStore, fork_session};
use super::{SavedSession, read_session};
use crate::AgentPlugin;
use crate::agent::{Agent, AgentId, Conversation, Entry, Halt, MessageId, STOPPED, SpawnedBy};
use crate::inbox::Origin;

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
fn state(conversation: &Conversation) -> (Vec<Entry>, Option<Halt>) {
    let entries = (0..conversation.messages().len()).filter_map(|at| conversation.entry(at));
    (entries.collect(), conversation.halted())
}

/// The session in `store` as a restore reads it, by agent id.
fn saved_state(store: &MemoryStore, id: &AgentId) -> Option<(Vec<Entry>, Option<Halt>)> {
    let session = read_session(store).ok()?;
    let saved = session.agents.into_iter().find(|saved| saved.id == *id)?;
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
    let live = app.world().get::<Conversation>(agent).map(state);
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
    let live = app.world().get::<Conversation>(agent).map(state);
    assert_eq!(live.as_ref().map(|(_, halted)| *halted), Some(None));
    assert_eq!(saved_state(&store, &id), live);
    let next = app
        .world()
        .get::<Conversation>(agent)
        .map(Conversation::next_id);
    drop(app);
    let mut second = self::app(&store);
    let Some(&agent) = agents(&mut second).first() else {
        return;
    };
    let restored = second.world().get::<Conversation>(agent);
    assert_eq!(restored.map(state), live);
    assert_eq!(restored.map(Conversation::next_id), next, "new ids go on");
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
    assert!(saved.id == id && saved.parent.is_none() && !saved.retired);
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
    assert!(fork_session(&store, &fork, Some((&id, second_question))).is_ok());
    let forked = read_session(&fork).unwrap_or_default();
    assert_eq!(forked.agents.len(), 1);
    let Some(branch) = forked.agents.first() else {
        return;
    };
    let kept = [Message::user("question 1"), Message::assistant("answer 1")];
    assert_eq!(branch.conversation.messages(), kept);
    let counter = branch.saved.get(std::any::type_name::<Counter>());
    assert_eq!(counter, Some(&serde_json::json!(1)));
    // A message the agent never made is no fork point.
    let nowhere = MessageId(999);
    let nowhere = fork_session(&store, &MemoryStore::default(), Some((&id, nowhere)));
    assert!(nowhere.is_err());
}

/// The JSON line of a record, `seq` in its log and `n` in the session, all
/// written in the same millisecond.
fn line(seq: u64, n: u64, record: &serde_json::Value) -> String {
    let mut line = serde_json::json!({ "seq": seq, "n": n, "t": 1_000 });
    if let (Some(line), Some(record)) = (line.as_object_mut(), record.as_object()) {
        line.extend(record.clone());
    }
    format!("{line}\n")
}

fn said(message: Message) -> serde_json::Value {
    let message = serde_json::to_value(message).unwrap_or_default();
    serde_json::json!({ "type": "message", "message": message })
}

#[test]
fn a_fork_keeps_what_other_agents_logged_before_its_cut_in_the_same_millisecond() {
    let header = |agent: &str| serde_json::json!({ "type": "header", "agent": agent });
    let from = MemoryStore::default();
    let a = [
        line(0, 0, &header("a")),
        line(1, 1, &said(Message::user("a first"))),
        line(2, 4, &said(Message::assistant("a second"))),
    ];
    let b = [
        line(0, 2, &header("b")),
        line(1, 3, &said(Message::user("b first"))),
        line(2, 5, &said(Message::assistant("b second"))),
    ];
    assert!(from.append("a", a.concat().as_bytes()).is_ok());
    assert!(from.append("b", b.concat().as_bytes()).is_ok());
    let to = MemoryStore::default();
    let second = MessageId(1);
    let fork = fork_session(&from, &to, Some((&AgentId("a".to_owned()), second)));
    assert!(fork.is_ok(), "{fork:?}");
    let session = read_session(&to);
    let kept = |agent: &str| -> Vec<Message> {
        let session = session.as_ref().ok();
        let agent =
            session.and_then(|session| session.agents.iter().find(|saved| saved.id.0 == agent));
        agent.map_or_else(Vec::new, |agent| agent.conversation.messages().to_vec())
    };
    assert_eq!(kept("a"), [Message::user("a first")]);
    assert_eq!(kept("b"), [Message::user("b first")]);
}
