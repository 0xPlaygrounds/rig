use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::system::RunSystemOnce;
use bevy_reflect::Reflect;
use rig_cassette::journal::{JournalStore, MemoryStore};
use rig_core::completion::Message;

use super::{
    Commit, Committed, JournalPlugin, ReflectSaved, SessionStore, commit_message, fork_session,
};
use super::{SavedSession, read_session};
use crate::AgentPlugin;
use crate::agent::{Agent, AgentId, Condensed, Conversation, SpawnedBy};

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
fn commit(app: &mut App, agent: Entity, edit: fn(Entity, &mut Conversation, &mut Commit)) {
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
        commit.truncate(agent, conversation, 2);
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
            Committed::Truncate { len, .. } => format!("truncate {len}"),
            Committed::Retired { .. } => "retired".to_owned(),
            other => format!("{other:?}"),
        })
        .collect();
    assert_eq!(changes, ["truncate 2", "retired"]);
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

#[test]
fn a_cut_before_a_summary_undoes_it() {
    let store = MemoryStore::default();
    let mut first = app(&store);
    let Some(&agent) = agents(&mut first).first() else {
        return;
    };
    talk(&mut first, agent);
    let summary = Condensed {
        upto: 2,
        summary: "The first question.".to_owned(),
    };
    first.world_mut().entity_mut(agent).insert(summary);
    first.update();
    commit(&mut first, agent, |agent, conversation, commit| {
        commit.truncate(agent, conversation, 1);
    });
    first.update();
    assert!(first.world().get::<Condensed>(agent).is_none());
    drop(first);

    let mut second = app(&store);
    let Some(&agent) = agents(&mut second).first() else {
        return;
    };
    assert_eq!(messages(&second, agent), [Message::user("question 1")]);
    assert!(second.world().get::<Condensed>(agent).is_none());
}

/// A plugin's counter, saved with each agent.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved)]
struct Counter(u32);

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
    let summary = Condensed {
        upto: 2,
        summary: "The first question.".to_owned(),
    };
    app.world_mut().entity_mut(agent).insert(summary);
    app.update();
    let second_question = app
        .world()
        .get::<Conversation>(agent)
        .and_then(|conversation| conversation.seq(2));
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
    // The whole conversation, the summary over its first two messages.
    assert_eq!(saved.conversation.messages().len(), 4);
    assert_eq!(
        saved.condensed.as_ref().map(|condensed| condensed.upto),
        Some(2)
    );
    let counter = saved.saved.get(std::any::type_name::<Counter>());
    assert_eq!(counter, Some(&serde_json::json!(2)), "{:?}", saved.saved);

    // Forked as of the second question: the conversation and the counter
    // as they were then.
    let fork = MemoryStore::default();
    assert!(second_question.is_some(), "the second question was logged");
    let Some(seq) = second_question else {
        return;
    };
    assert!(fork_session(&store, &fork, Some((&id, seq))).is_ok());
    let forked = read_session(&fork).unwrap_or_default();
    assert_eq!(forked.agents.len(), 1);
    let Some(branch) = forked.agents.first() else {
        return;
    };
    let kept = [Message::user("question 1"), Message::assistant("answer 1")];
    assert_eq!(branch.conversation.messages(), kept);
    assert!(branch.condensed.is_none());
    let counter = branch.saved.get(std::any::type_name::<Counter>());
    assert_eq!(counter, Some(&serde_json::json!(1)));
    // A record the agent never logged is no fork point.
    let nowhere = fork_session(&store, &MemoryStore::default(), Some((&id, 999)));
    assert!(nowhere.is_err());
}
