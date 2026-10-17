use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::Reflect;
use bevy_reflect::prelude::{ReflectDeserialize, ReflectSerialize};
use rig_core::completion::Message;
use rig_core::conversation::{
    self, Aside, Author, AuthorKind, Change, Entry, Halt, MessageId, STOPPED,
};
use rig_core::message::UserContent;
use rig_memory::{Files, LogRecord, MemoryStore, Repair, Session, Store, history};
use serde::{Deserialize, Serialize};

use super::{
    ConversationChanged, ReflectSaved, Retire, SessionLogPlugin, SessionRestored, SessionStore,
    open_session,
};
use crate::AgentPlugin;
use crate::agent::{Agent, AgentId, Conversation, Notice, SpawnedBy};
use crate::inbox::{Deliver, DeliveryMode};
use crate::stage::{Admitted, AgentPass, ForAgent, Stage};
use crate::turn::TurnCall;

/// Runs `edit` on the conversation of `agent`, as a plugin's system would.
pub(crate) fn edit(
    world: &mut World,
    agent: Entity,
    edit: impl FnOnce(&mut Conversation) + Send + Sync + 'static,
) {
    let run =
        |In((agent, edit)): In<(Entity, Box<dyn FnOnce(&mut Conversation) + Send + Sync>)>,
         mut agents: Query<&mut Conversation>| {
            if let Ok(mut conversation) = agents.get_mut(agent) {
                edit(&mut conversation);
            }
        };
    let edited = world.run_system_cached_with(run, (agent, Box::new(edit)));
    assert!(edited.is_ok(), "{edited:?}");
}

/// Pushes `message` to the conversation of `agent`.
pub(crate) fn push_message(world: &mut World, agent: Entity, message: Message) {
    edit(world, agent, |conversation| {
        conversation.push(message);
    });
}

/// An app on `store`, after its first frame restored the session.
fn app(store: &MemoryStore) -> App {
    let mut app = App::new();
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((AgentPlugin, SessionLogPlugin))
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
    conversation.map_or_else(Vec::new, |conversation| {
        conversation
            .messages()
            .iter()
            .map(|message| (**message).clone())
            .collect()
    })
}

fn id_of(app: &App, agent: Entity) -> String {
    let id = app.world().get::<AgentId>(agent);
    id.map(|id| id.0.clone()).unwrap_or_default()
}

/// Every change announced so far.
fn committed(app: &App) -> Vec<ConversationChanged> {
    let committed = app.world().resource::<Messages<ConversationChanged>>();
    let mut cursor = committed.get_cursor();
    cursor.read(committed).cloned().collect()
}

/// The kind of each change of the log `id`.
fn logged(store: &MemoryStore, id: &str) -> Vec<&'static str> {
    let records = history(store, id).unwrap_or_default();
    records
        .into_iter()
        .filter_map(|(_, record)| match record {
            LogRecord::Change(recorded) => Some(kind(&recorded.change)),
            _ => None,
        })
        .collect()
}

fn kind(change: &Change) -> &'static str {
    match change {
        Change::Push { .. } => "push",
        Change::Splice { .. } => "splice",
        Change::Halt { .. } => "halt",
        Change::Aside { .. } => "aside",
        Change::Reset { .. } => "reset",
    }
}

/// Four messages: two questions and their answers.
fn talk(app: &mut App, agent: Entity) {
    for n in 1..=2 {
        let question = Message::user(format!("question {n}"));
        push_message(app.world_mut(), agent, question);
        let answer = Message::assistant(format!("answer {n}"));
        push_message(app.world_mut(), agent, answer);
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
        push_message(first.world_mut(), spawned, Message::user("work"));
    }
    edit(first.world_mut(), agent, |conversation| {
        conversation.splice(2.., [], Author::plugin("cut"));
    });
    let child_id = id_of(&first, child);
    first.world_mut().trigger(Retire { entity: child });
    first.update();
    assert_eq!(messages(&first, agent).len(), 2);
    assert!(
        first.world().get_entity(grandchild).is_err(),
        "retired with its parent"
    );
    let cuts: Vec<_> = committed(&first)
        .into_iter()
        .filter(|committed| committed.agent == agent)
        .filter_map(|committed| match committed.change.change {
            Change::Splice { removed, .. } => Some((removed.len(), committed.change.by)),
            _ => None,
        })
        .collect();
    assert_eq!(cuts, [(2, Some(Author::plugin("cut")))]);
    let session = Session::read(&store).unwrap_or_default();
    let child = session.agent(&child_id);
    let child = child.map(|saved| (saved.retired, saved.conversation.messages().len()));
    assert_eq!(child, Some((true, 1)), "the work it did is saved first");
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

/// Every entry of `conversation` and its halt.
fn state(conversation: &conversation::Conversation) -> (Vec<Entry>, Option<Halt>) {
    let entries = (0..conversation.messages().len()).filter_map(|at| conversation.entry(at));
    (entries.collect(), conversation.halted())
}

#[test]
fn edits_keep_ids_and_authors_and_a_restore_makes_the_same_changes() {
    let store = MemoryStore::default();
    let mut app = app(&store);
    let Some(&agent) = agents(&mut app).first() else {
        return;
    };
    edit(app.world_mut(), agent, |conversation| {
        conversation.extend([Message::user("question 1"), Message::assistant("answer 1")]);
        conversation.push(Entry::new(Message::user("question 2")).by(Author::plugin("note")));
        conversation.extend([Message::assistant("answer 2"), Message::user("question 3")]);
        conversation.halt(Halt::Stopped);
    });
    let ids = |app: &App| {
        let conversation = app.world().get::<Conversation>(agent);
        conversation.map_or_else(Vec::new, |conversation| {
            conversation.entries().map(|entry| entry.id).collect()
        })
    };
    let before: Vec<MessageId> = ids(&app);
    assert_eq!(before.len(), 5);
    let [q1, a1, q2, a2, q3] = before.as_slice() else {
        return;
    };
    // An edit in place, then a summary of the first two.
    edit(app.world_mut(), agent, |conversation| {
        let Some(mut edited) = conversation.entry(2) else {
            return;
        };
        edited.message = Message::user("question 2, edited").into();
        conversation.splice(2..3, [edited], Author::plugin("editor"));
        let summary = Entry::new(Message::user("Summary: question 1."));
        let compaction = Author::plugin("compaction");
        conversation.splice(0..2, [summary.by(compaction.clone())], compaction);
    });
    app.update();
    let after = ids(&app);
    assert_eq!(after.get(1..), Some([*q2, *a2, *q3].as_slice()));
    assert!(!after.contains(q1) && !after.contains(a1));
    let live = app.world().get::<Conversation>(agent).map(|c| state(c));
    let halted = live.as_ref().and_then(|(_, halted)| *halted);
    assert_eq!(
        halted,
        Some(Halt::Stopped),
        "a splice before the end keeps it"
    );
    let note = live.as_ref().and_then(|(entries, _)| entries.get(1));
    let note = note.and_then(|entry| entry.author.clone());
    assert_eq!(
        note,
        Some(Author::plugin("note")),
        "the edit kept its author"
    );
    let id = id_of(&app, agent);
    let saved = |store: &MemoryStore| {
        let session = Session::read(store).ok()?;
        session.agent(&id).map(|saved| state(&saved.conversation))
    };
    assert_eq!(saved(&store), live, "saved as made live");

    // A cut of the tail clears the halt, and a restore agrees.
    edit(app.world_mut(), agent, |conversation| {
        let last = conversation.messages().len().saturating_sub(1);
        conversation.splice(last.., [], Author::user());
    });
    app.update();
    let live = app.world().get::<Conversation>(agent).map(|c| state(c));
    assert_eq!(live.as_ref().map(|(_, halted)| *halted), Some(None));
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
    edit(app.world_mut(), agent, |conversation| {
        conversation.push(Message::user("run the tests"));
        conversation.halt(Halt::Stopped);
        // A retry asks again; the user steers before the model answers.
        assert!(conversation.resume());
        conversation.push(Message::user("only the fast ones"));
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
    let id = id_of(&first, agent);
    assert!(
        store.read_log(&id).is_err(),
        "a session nobody used is not kept"
    );
    // A plugin's state, such as a mode the user turned on before any
    // message, is kept for the next start.
    first.world_mut().entity_mut(agent).insert(Counter(1));
    first.update();
    assert!(store.read_log(&id).is_ok());
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

/// A plugin's bookmark on a message, saved with the agent through serde:
/// rig-core's `MessageId` is not `Reflect`.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Clone, Serialize, Deserialize)]
struct Bookmark {
    at: MessageId,
    label: String,
}

#[test]
fn a_saved_component_holding_a_message_id_survives_a_restore() {
    let store = MemoryStore::default();
    let mut first = app(&store);
    let Some(&agent) = agents(&mut first).first() else {
        return;
    };
    talk(&mut first, agent);
    let second_question = first
        .world()
        .get::<Conversation>(agent)
        .and_then(|c| c.id(2));
    assert!(second_question.is_some());
    let Some(at) = second_question else {
        return;
    };
    let mark = Bookmark {
        at,
        label: "the second question".to_owned(),
    };
    first.world_mut().entity_mut(agent).insert(mark.clone());
    first.update();
    drop(first);

    let mut second = app(&store);
    let Some(&agent) = agents(&mut second).first() else {
        return;
    };
    let world = second.world();
    assert_eq!(world.get::<Bookmark>(agent), Some(&mark));
    let index = world
        .get::<Conversation>(agent)
        .and_then(|c| c.index_of(at));
    assert_eq!(index, Some(2), "the id names the same message");
}

/// The pass of the frame, counted in Arrive.
#[derive(Resource, Default)]
struct Pass(u32);

/// The pass a plugin wrote its note in, and the pass a later stage read
/// it in.
#[derive(Resource, Default)]
struct Seen {
    wrote: Option<u32>,
    read: Option<u32>,
}

#[test]
fn a_committed_reader_in_a_later_stage_of_the_pass_sees_the_change() {
    let mut app = App::new();
    let count = |mut pass: ResMut<Pass>| pass.0 += 1;
    // A note on each turn's first call, put in when Admit admits it.
    let write = |calls: Query<&ForAgent, (Added<Admitted>, With<TurnCall>)>,
                 mut agents: Query<&mut Conversation>,
                 pass: Res<Pass>,
                 mut seen: ResMut<Seen>| {
        for &ForAgent(agent) in &calls {
            if let Ok(mut conversation) = agents.get_mut(agent) {
                let note = Entry::new(Message::user("a note")).by(Author::plugin("writer"));
                conversation.push(note);
                seen.wrote.get_or_insert(pass.0);
            }
        }
    };
    let read = |mut committed: MessageReader<ConversationChanged>,
                pass: Res<Pass>,
                mut seen: ResMut<Seen>| {
        let writer = Some(Author::plugin("writer"));
        if committed
            .read()
            .any(|committed| committed.change.by == writer)
        {
            seen.read.get_or_insert(pass.0);
        }
    };
    app.add_plugins(AgentPlugin)
        .init_resource::<Pass>()
        .init_resource::<Seen>()
        .add_systems(
            AgentPass,
            (
                count.in_set(Stage::Arrive),
                write.in_set(Stage::Admit),
                read.in_set(Stage::Prepare),
            ),
        );
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.query_filtered::<Entity, With<Agent>>().single(world);
    assert!(agent.is_ok());
    let Ok(agent) = agent else {
        return;
    };
    world.trigger(Deliver::new(agent, "hello", DeliveryMode::Steer));
    app.update();
    let seen = app.world().resource::<Seen>();
    assert!(seen.wrote.is_some());
    assert_eq!(seen.read, seen.wrote, "read in the pass it was written in");
}

#[test]
fn a_conversation_inserted_over_an_agents_own_is_saved_as_one_reset() {
    let store = MemoryStore::default();
    let mut first = app(&store);
    let Some(&agent) = agents(&mut first).first() else {
        return;
    };
    talk(&mut first, agent);
    // An edit not saved yet when the value is replaced in the same frame.
    push_message(first.world_mut(), agent, Message::user("question 3"));
    let fresh = conversation::Conversation::from_messages([Message::user("fresh")]);
    first
        .world_mut()
        .entity_mut(agent)
        .insert(Conversation(fresh));
    first.update();
    let id = id_of(&first, agent);
    let kinds = ["push", "push", "push", "push", "push", "reset"];
    assert_eq!(logged(&store, &id), kinds);
    let committed = committed(&first);
    let last = committed
        .iter()
        .rev()
        .take(2)
        .map(|c| kind(&c.change.change));
    assert_eq!(last.collect::<Vec<_>>(), ["reset", "push"]);
    // The value goes on from the reset.
    push_message(first.world_mut(), agent, Message::assistant("hi"));
    first.update();
    assert_eq!(logged(&store, &id).last(), Some(&"push"));
    drop(first);

    let mut second = app(&store);
    let Some(&agent) = agents(&mut second).first() else {
        return;
    };
    let kept = [Message::user("fresh"), Message::assistant("hi")];
    assert_eq!(messages(&second, agent), kept);
}

/// How many frames saw an agent's conversation changed.
#[derive(Resource, Default)]
struct ChangedFrames(u32);

#[test]
fn a_saved_conversation_is_not_marked_changed_again() {
    let mut app = app(&MemoryStore::default());
    let count = |changed: Query<(), Changed<Conversation>>, mut frames: ResMut<ChangedFrames>| {
        if !changed.is_empty() {
            frames.0 += 1;
        }
    };
    app.init_resource::<ChangedFrames>()
        .add_systems(Update, count);
    app.update();
    app.world_mut().resource_mut::<ChangedFrames>().0 = 0;
    let Some(&agent) = agents(&mut app).first() else {
        return;
    };
    push_message(app.world_mut(), agent, Message::user("hello"));
    app.update();
    app.update();
    app.update();
    assert_eq!(app.world().resource::<ChangedFrames>().0, 1);
    assert_eq!(committed(&app).len(), 1, "saved once");
}

#[test]
fn unsaved_asides_never_reach_the_log() {
    let store = MemoryStore::default();
    let mut first = app(&store);
    let Some(&agent) = agents(&mut first).first() else {
        return;
    };
    edit(first.world_mut(), agent, |conversation| {
        conversation.push(Message::user("hello"));
        let author = Author::plugin("notes");
        let draft = Aside::new("draft", author.clone(), "for now".into()).unsaved();
        conversation.add_aside(draft);
        conversation.add_aside(Aside::new("note", author, "kept".into()));
    });
    first.update();
    push_message(first.world_mut(), agent, Message::assistant("hi"));
    first.update();
    let id = id_of(&first, agent);
    assert_eq!(logged(&store, &id), ["push", "aside", "push"]);
    drop(first);

    let mut second = app(&store);
    let Some(&agent) = agents(&mut second).first() else {
        return;
    };
    let conversation = second.world().get::<Conversation>(agent);
    let kinds: Vec<String> = conversation
        .map(|conversation| conversation.asides().map(|(_, a)| a.kind.clone()).collect())
        .unwrap_or_default();
    assert_eq!(kinds, ["note"]);
}

/// Every log of `store`, with its bytes.
fn logs(store: &MemoryStore) -> Vec<(String, Vec<u8>)> {
    let mut names = store.logs().unwrap_or_default();
    names.sort();
    let read = |name: String| {
        let bytes = store.read_log(&name).unwrap_or_default();
        (name, bytes)
    };
    names.into_iter().map(read).collect()
}

/// A session the rig before this format saved is refused untouched: the
/// app runs on, saving nothing, and says to start a new session.
#[test]
fn an_old_format_session_is_refused_untouched() {
    let store = MemoryStore::default();
    let old = br#"{"seq":0,"n":0,"t":1791779409181,"type":"header","format":"rig-session/3","log":"a1","lineage":7}"#;
    assert!(
        store
            .append_log("a1", &[old.as_slice(), b"\n"].concat())
            .is_ok()
    );
    let before = logs(&store);
    let mut world = World::new();
    let opened = open_session(&mut world, Arc::new(store.clone()));
    assert!(opened.is_err());
    assert!(!world.contains_resource::<SessionStore>());

    let mut app = app(&store);
    assert!(!app.world().contains_resource::<SessionRestored>());
    let notices = app.world().resource::<Messages<Notice>>();
    let mut cursor = notices.get_cursor();
    let refused = cursor.read(notices).any(|notice| {
        notice.text.starts_with("The session cannot be resumed")
            && notice.text.contains("start a new session")
    });
    assert!(refused);
    // The app runs on, saving nothing.
    if let Some(&agent) = agents(&mut app).first() {
        push_message(app.world_mut(), agent, Message::user("hello"));
    }
    app.update();
    app.update();
    assert_eq!(logs(&store), before);
    assert_eq!(store.files().unwrap_or_default(), Vec::<String>::new());
}

#[test]
fn a_last_line_a_crash_cut_short_is_set_aside_and_reported() {
    let store = MemoryStore::default();
    let mut first = app(&store);
    let Some(&agent) = agents(&mut first).first() else {
        return;
    };
    talk(&mut first, agent);
    let id = id_of(&first, agent);
    drop(first);
    assert!(store.append_log(&id, b"{\"seq\":").is_ok());

    let mut second = app(&store);
    let repairs = second.world().get_resource::<SessionRestored>();
    let repairs = repairs
        .map(|restored| restored.repairs.clone())
        .unwrap_or_default();
    let torn = match repairs.as_slice() {
        [
            Repair::TornTail {
                cut: 7,
                set_aside: Some(file),
                ..
            },
        ] => Some(file),
        _ => None,
    };
    assert!(torn.is_some(), "one torn tail: {repairs:?}");
    let Some(file) = torn else {
        return;
    };
    assert_eq!(store.read_file(file).ok(), Some(b"{\"seq\":".to_vec()));
    let Some(&agent) = agents(&mut second).first() else {
        return;
    };
    assert_eq!(messages(&second, agent).len(), 4);
    let notices = second.world().resource::<Messages<Notice>>();
    let mut cursor = notices.get_cursor();
    let reported = cursor
        .read(notices)
        .any(|notice| notice.text.contains("cut short"));
    assert!(reported);
    // The log goes on whole.
    push_message(second.world_mut(), agent, Message::user("question 3"));
    second.update();
    drop(second);
    let session = Session::read(&store).unwrap_or_default();
    assert_eq!(session.repairs, []);
    let saved = session
        .agent(&id)
        .map(|saved| saved.conversation.messages().len());
    assert_eq!(saved, Some(5));
}

#[test]
fn a_message_from_another_author_is_its_own_entry() {
    let mut app = app(&MemoryStore::default());
    let Some(&agent) = agents(&mut app).first() else {
        return;
    };
    let note =
        Deliver::new(agent, "build passed", DeliveryMode::Inform).with_author(Author::plugin("ci"));
    app.world_mut().trigger(note);
    app.update();
    let world = app.world();
    let conversation = world.get::<Conversation>(agent);
    let authors: Vec<_> = conversation
        .map(|conversation| {
            let entries = conversation.entries();
            entries
                .map(|entry| entry.author.map(|author| author.kind.clone()))
                .collect()
        })
        .unwrap_or_default();
    assert_eq!(authors, [Some(AuthorKind::Plugin("ci".to_owned()))]);
}

/// Prints the environment, as `env` does, with a key in it.
struct Env;

/// The key [`Env`] prints, too short for an OpenAI key's shape: found as
/// the value of `OPENAI_API_KEY`.
const SEEDED_KEY: &str = "sk-test-probe-1234567890abcdef";

/// A key the user types, found by its shape.
const TYPED_KEY: &str = "sk-proj-abcdefghijklmnopqrstuvwxyz0123456789";

impl rig_core::tool::Tool for Env {
    const NAME: &'static str = "env";
    type Args = serde_json::Value;
    type Output = String;
    type Error = rig_core::tool::ToolExecutionError;

    fn description(&self) -> String {
        "Prints the environment.".to_owned()
    }

    fn parameters(&self) -> serde_json::Value {
        serde_json::json!({ "type": "object" })
    }

    async fn call(
        &self,
        _: &mut rig_core::tool::ToolContext,
        _: serde_json::Value,
    ) -> Result<String, rig_core::tool::ToolExecutionError> {
        Ok(format!("HOME=/home/me\nOPENAI_API_KEY={SEEDED_KEY}\n"))
    }
}

/// A key a tool printed is redacted before its result is committed: the
/// model's next request, the conversation and the session log never hold
/// it, nor does the log hold one the user typed.
#[test]
fn a_seeded_key_never_reaches_the_model_the_conversation_or_the_log() {
    use rig_core::catalog::Catalog;
    use rig_core::operation::Completion;
    use rig_core::serve::ErasedHandler;
    use rig_core::serve::adapters::ModelAdapter;
    use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};

    use crate::effects::Handler;
    use crate::model::Connection;
    use crate::tools::AppToolsExt;

    let store = MemoryStore::default();
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call("e1", "env", serde_json::json!({})),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        vec![
            MockStreamEvent::text("done"),
            MockStreamEvent::final_response_with_default_usage(),
        ],
    ]);
    let spec = Catalog::builtin().resolve("deepseek/deepseek-flash");
    let connection = spec.ok().map(|spec| {
        let spec = spec.shared();
        let handler = ModelAdapter::<Completion>::new(spec.reference(), model.clone());
        Connection {
            spec,
            handler: Handler(ErasedHandler::new(handler)),
        }
    });
    let mut app = App::new();
    app.set_error_handler(bevy_ecs::error::warn)
        .add_plugins((
            bevy_app::TaskPoolPlugin::default(),
            AgentPlugin,
            SessionLogPlugin,
        ))
        .insert_resource(SessionStore::new(store.clone()))
        .add_tool(Env)
        .add_observer(move |added: On<Add<Agent>>, mut commands: Commands| {
            if let Some(connection) = connection.clone() {
                commands.entity(added.entity).insert(connection);
            }
        });
    app.finish();
    app.update();
    let agent = agents(&mut app).first().copied();
    assert!(agent.is_some(), "no agent");
    let Some(agent) = agent else { return };
    let typed = format!("my key is {TYPED_KEY}, run env");
    app.world_mut()
        .trigger(Deliver::new(agent, typed, DeliveryMode::Steer));
    let started = std::time::Instant::now();
    while model.request_count() < 2 || app.world().get::<crate::agent::ActiveTurn>(agent).is_some()
    {
        assert!(started.elapsed().as_secs() < 10, "the turn did not end");
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(1));
    }
    app.update();

    let requests = model.requests();
    let sent = serde_json::to_string(&requests.get(1).map(|request| &request.chat_history));
    let sent = sent.unwrap_or_default();
    assert!(sent.contains("OPENAI_API_KEY=[REDACTED]"), "{sent}");
    let said = serde_json::to_string(&messages(&app, agent)).unwrap_or_default();
    assert!(said.contains("OPENAI_API_KEY=[REDACTED]"), "{said}");
    let log = String::from_utf8(store.read_log(&id_of(&app, agent)).unwrap_or_default());
    let log = log.unwrap_or_default();
    assert!(log.contains("OPENAI_API_KEY=[REDACTED]"), "{log}");
    assert!(
        !log.contains(SEEDED_KEY) && !log.contains(TYPED_KEY),
        "{log}"
    );
    // The tool result never held it; the user's own message, sent as typed,
    // is redacted only on disk.
    let results: Vec<&Message> = requests
        .iter()
        .flat_map(|request| request.chat_history.iter().map(|message| &**message))
        .filter(|message| {
            serde_json::to_string(message).is_ok_and(|text| text.contains("toolresult"))
        })
        .collect();
    assert!(!results.is_empty());
    for result in results {
        let text = serde_json::to_string(result).unwrap_or_default();
        assert!(!text.contains(SEEDED_KEY), "{text}");
    }
}
