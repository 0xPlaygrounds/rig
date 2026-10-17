use std::sync::mpsc::Receiver;

use rig_basics::subagents::{SubagentsPlugin, TASK};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::rig_memory::Session as Saved;
use rig_harness_test_support::{
    Collected, MemoryStore, ToolOutputs, app_on, calls, collect, log_tool_outputs, press, reply,
    run_until, scripted_models, tui_view,
};
use rig_tui::{Editor, MainEditor};
use serde_json::json;

use super::*;

/// The main agent's model.
const MAIN: &str = "ct/main";
/// The database subagent's model.
const SUB: &str = "ct/sub";
/// Its sibling's model.
const SIB: &str = "ct/sib";
/// The database subagent's task.
const TASK_TITLE: &str = "find DB config";
/// What the test's `read` tool reads.
const DB_TOML: &str = "[db]\npool_size = 8\ntimeout = 30";

/// The open calls of the test's `hold` tool, which the test answers.
#[derive(Resource, Default)]
struct Held(Vec<Entity>);

/// Keeps its caller's turn running until the test answers it.
fn hold(called: On<ToolCalled<serde_json::Value>>, mut held: ResMut<Held>) {
    held.0.push(called.call);
}

/// The test's `read`: every file holds [`DB_TOML`].
fn read(called: On<ToolCalled<serde_json::Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut decision) = calls.get_mut(called.call) {
        decision.decide(Decider::plugin("read"), ToolResult::success(DB_TOML.into()));
    }
}

/// The models name a subagent by its task, its id unknown to a script:
/// the id they would have read in `task`'s result.
fn name_the_subagent(
    mut calls: Query<&mut ToolRequest, Added<Admitted>>,
    subagents: Query<(&AgentId, &AgentTitle), With<Subtask>>,
) {
    for mut request in &mut calls {
        if request.call.function.name.as_str() != FOLLOW_UP {
            continue;
        }
        let args = &mut request.call.function.arguments;
        let title = args.get("agent_id").and_then(|agent| agent.as_str());
        let named = subagents
            .iter()
            .find(|(_, task)| title == Some(task.0.as_str()));
        if let Some((id, _)) = named {
            args.insert("agent_id".to_owned(), json!(id.short()));
        }
    }
}

/// An app on `store` with the subagents, the terminal view and the plugin,
/// the main agent on [`MAIN`] and shown.
struct Session {
    app: App,
    wakes: Receiver<()>,
    main: Entity,
}

impl Session {
    fn new(store: &MemoryStore, models: &[(&str, &MockCompletionModel)]) -> Self {
        let fresh = store.logs().unwrap_or_default().is_empty();
        let (mut app, wakes) = app_on(store);
        let reads = ToolOptions {
            footprint: Some(Touches::Reads(Scope::Anything)),
        };
        app.insert_resource(scripted_models(models.iter().copied()))
            .init_resource::<ToolOutputs>()
            .add_plugins((
                AgentPlugin,
                SessionLogPlugin,
                SubagentsPlugin,
                ResumeSubagentPlugin,
            ))
            .add_open_tool("read", "Read a file.", reads.clone(), read)
            .add_open_tool("hold", "Hold on.", reads, hold)
            .init_resource::<Held>()
            .add_systems(AgentPass, name_the_subagent.in_set(PrepareSystems::Input))
            .add_observer(log_tool_outputs);
        collect::<Notice>(&mut app);
        tui_view(&mut app);
        if fresh {
            app.world_mut().spawn((Agent, ModelChoice(MAIN.to_owned())));
        }
        app.finish();
        app.update();
        let world = app.world_mut();
        let mut roots = world.query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
        let main = roots.iter(world).next().unwrap_or(Entity::PLACEHOLDER);
        app.world_mut().resource_mut::<ShownAgent>().0 = Some(main);
        app.update();
        Self { app, wakes, main }
    }

    /// Types `text` in the terminal view's input and presses Enter.
    fn type_line(&mut self, text: &str) {
        let world = self.app.world_mut();
        let mut main = world.query_filtered::<&mut Editor, With<MainEditor>>();
        if let Ok(mut input) = main.single_mut(world) {
            input.set(text);
        }
        press(&mut self.app, "enter");
    }

    fn run(&mut self, done: impl Fn(&World) -> bool) {
        run_until(&mut self.app, &self.wakes, done);
    }

    /// Lets the journal write what the last frames changed.
    fn settle(&mut self) {
        for _ in 0..4 {
            self.app.update();
        }
    }

    fn output(&self, id: &str) -> String {
        let outputs = &self.app.world().resource::<ToolOutputs>().0;
        outputs.get(id).cloned().unwrap_or_default()
    }

    /// The subagent of [`TASK_TITLE`].
    fn subagent(&mut self) -> Option<(Entity, AgentId)> {
        let world = self.app.world_mut();
        let mut subagents = world.query::<(Entity, &AgentId, &AgentTitle)>();
        let found = subagents
            .iter(world)
            .find(|(.., title)| title.0 == TASK_TITLE);
        found.map(|(agent, id, _)| (agent, id.clone()))
    }

    /// Answers the `hold` call `at`.
    fn release(&mut self, at: usize) {
        let held = self.app.world().resource::<Held>().0.get(at).copied();
        let decision = held.and_then(|call| self.app.world_mut().get_mut::<Decision>(call));
        if let Some(mut decision) = decision {
            decision.decide(Decider::plugin("test"), ToolResult::success("ok".into()));
        }
    }

    fn shown(&self) -> Option<Entity> {
        self.app.world().resource::<ShownAgent>().0
    }

    fn messages(&self, agent: Entity) -> usize {
        let conversation = self.app.world().get::<Conversation>(agent);
        conversation.map_or(0, |conversation| conversation.messages().len())
    }
}

/// Whether no agent of the world has a turn running.
fn idle(world: &World) -> bool {
    world
        .try_query::<&ActiveTurn>()
        .is_none_or(|mut turns| turns.iter(world).next().is_none())
}

/// What `model` read in its request `n` (from 0).
fn read_by(model: &MockCompletionModel, n: usize) -> String {
    let request = model.requests().into_iter().nth(n);
    request.map_or_else(String::new, |request| {
        json!(request.chat_history).to_string()
    })
}

/// A `follow_up` call `id` to the subagent of [`TASK_TITLE`].
fn follow_up(id: &str, message: &str) -> Vec<MockStreamEvent> {
    calls(
        id,
        FOLLOW_UP,
        json!({ "agent_id": TASK_TITLE, "message": message }),
    )
}

/// The messages of the saved subagent `id` in `store`.
fn saved_messages(store: &MemoryStore, id: &AgentId) -> usize {
    let session = Saved::read(store).ok();
    let agents = session.into_iter().flat_map(|session| session.agents);
    let agent = agents.into_iter().find(|agent| agent.id == id.0);
    agent.map_or(0, |agent| agent.conversation.messages().len())
}

/// The spec's scripted test: list, talk, follow up, a sibling refused, and
/// talking again after a restart.
#[test]
fn a_finished_subagent_is_listed_talked_to_followed_up_and_resumed() {
    let task = |id: &str, title: &str, model: &str| {
        let args = json!({ "description": title, "prompt": title, "model": model });
        MockStreamEvent::tool_call(id, TASK, args)
    };
    let main = MockCompletionModel::from_stream_turns([
        vec![
            task("t1", TASK_TITLE, SUB),
            task("t2", "check the logs", SIB),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Started both."),
        reply("Config is in db.toml."),
        follow_up("f1", "what's the timeout?"),
        reply("The timeout is 30."),
    ]);
    let sub = MockCompletionModel::from_stream_turns([
        calls("r1", "read", json!({ "path": "db.toml" })),
        reply("config in db.toml"),
        reply("Line 2 sets pool_size = 8."),
        reply("timeout = 30"),
    ]);
    let sib = MockCompletionModel::from_stream_turns([
        follow_up("s1", "what did you find?"),
        reply("Nothing from the logs."),
    ]);
    let store = MemoryStore::default();
    let mut session = Session::new(&store, &[(MAIN, &main), (SUB, &sub), (SIB, &sib)]);
    let main_agent = session.main;
    session.type_line("Find the DB config and check the logs.");
    session.run(|world| {
        main.request_count() >= 3
            && sub.request_count() >= 2
            && sib.request_count() >= 2
            && idle(world)
    });

    // A sibling cannot reach it.
    let refused = session.output("s1");
    assert!(refused.contains("is not a subagent of yours"), "{refused}");

    // `/subagents` lists it, with its task.
    session.type_line("/subagents");
    let notices = &session.app.world().resource::<Collected<Notice>>().0;
    let listed = notices
        .iter()
        .filter(|notice| notice.text.contains("Subagents"));
    let listed: Vec<&str> = listed.map(|notice| notice.text.as_str()).collect();
    assert_eq!(listed.len(), 1, "{listed:?}");
    let list = listed.first().copied().unwrap_or_default();
    assert_eq!(list.matches(TASK_TITLE).count(), 1, "{list}");
    assert!(list.contains(SUB) && list.contains("finished"), "{list}");

    // `/talk <id>`: the user's message goes to it, and the main
    // conversation is unchanged.
    let found = session.subagent();
    assert!(found.is_some(), "no subagent");
    let (subagent, id) = found.unwrap_or((Entity::PLACEHOLDER, AgentId::default()));
    let before = session.messages(main_agent);
    session.type_line(&format!("/talk {}", id.short()));
    assert_eq!(session.shown(), Some(subagent));
    session.type_line("which line sets the pool size?");
    session.run(|world| sub.request_count() >= 3 && idle(world));
    let asked = read_by(&sub, 2);
    assert!(asked.contains("which line sets the pool size?"), "{asked}");
    assert!(
        asked.contains("pool_size = 8"),
        "its earlier read result: {asked}"
    );
    assert_eq!(session.messages(main_agent), before);
    assert_eq!(main.request_count(), 3);

    // `/talk main`, then the main agent follows up: the reply is the
    // tool's result.
    session.type_line("/talk main");
    assert_eq!(session.shown(), Some(main_agent));
    session.type_line("Ask it about the timeout.");
    session.run(|world| main.request_count() >= 5 && idle(world));
    assert_eq!(session.output("f1"), "timeout = 30");
    assert!(read_by(&main, 4).contains("timeout = 30"));
    session.settle();
    let kept = session.messages(subagent);
    drop(session);
    assert_eq!(saved_messages(&store, &id), kept);

    // After a restart, `/talk <id>` still works, from what it read.
    let main = MockCompletionModel::from_stream_turns([reply("unused")]);
    let sub = MockCompletionModel::from_stream_turns([reply("Still here.")]);
    let mut resumed = Session::new(&store, &[(MAIN, &main), (SUB, &sub)]);
    let subagent = resumed.subagent().map(|(agent, _)| agent);
    assert!(subagent.is_some(), "no subagent after resume");
    resumed.type_line(&format!("/talk {}", id.short()));
    assert_eq!(resumed.shown(), subagent);
    resumed.type_line("are you still there?");
    resumed.run(|world| sub.request_count() >= 1 && idle(world));
    let asked = read_by(&sub, 0);
    assert!(
        asked.contains("are you still there?") && asked.contains("pool_size = 8"),
        "{asked}"
    );
    assert_eq!(main.request_count(), 0);
}

/// The parent's follow-up waits while the user's message to the subagent
/// is answered, and gets its own reply.
#[test]
fn messages_from_the_user_and_the_parent_are_answered_one_at_a_time() {
    let main = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call(
                "t1",
                TASK,
                json!({ "description": TASK_TITLE, "prompt": TASK_TITLE, "model": SUB }),
            ),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Started."),
        reply("Config is in db.toml."),
        follow_up("f1", "what's the timeout?"),
        reply("The timeout is 30."),
        follow_up("f2", "and the pool size?"),
        reply("The pool is 8."),
    ]);
    let sub = MockCompletionModel::from_stream_turns([
        calls("r1", "read", json!({ "path": "db.toml" })),
        reply("config in db.toml"),
        calls("h1", "hold", json!({})),
        reply("for the user: line 2"),
        reply("for the parent: timeout = 30"),
        calls("h2", "hold", json!({})),
        reply("for the parent: pool 8"),
        reply("for the user: hello"),
    ]);
    let store = MemoryStore::default();
    let mut session = Session::new(&store, &[(MAIN, &main), (SUB, &sub)]);
    session.type_line("Find the DB config.");
    session.run(|world| main.request_count() >= 3 && sub.request_count() >= 2 && idle(world));
    let id = session.subagent().map(|(_, id)| id).unwrap_or_default();

    session.type_line(&format!("/talk {}", id.short()));
    session.type_line("which line sets the pool size?");
    session.run(|world| !world.resource::<Held>().0.is_empty());
    session.type_line("/talk main");
    session.type_line("Ask it about the timeout.");
    session.run(|_| main.request_count() >= 4);
    session.settle();
    // The follow-up waits for the user's message to be answered.
    assert_eq!(sub.request_count(), 3);
    assert!(session.output("f1").is_empty());

    session.release(0);
    session.run(|world| main.request_count() >= 5 && idle(world));
    assert_eq!(session.output("f1"), "for the parent: timeout = 30");
    let user_turn = read_by(&sub, 3);
    assert!(!user_turn.contains("what's the timeout?"), "{user_turn}");
    assert!(read_by(&sub, 4).contains("what's the timeout?"));

    // The other way round: the user's message waits for the follow-up's
    // step, and the follow-up gets its own step's reply.
    session.type_line("Ask it about the pool.");
    session.run(|world| world.resource::<Held>().0.len() >= 2);
    session.type_line(&format!("/talk {}", id.short()));
    session.type_line("hello?");
    session.settle();
    session.release(1);
    session.run(|world| main.request_count() >= 7 && idle(world));
    assert_eq!(session.output("f2"), "for the parent: pool 8");
    assert!(!read_by(&sub, 6).contains("hello?"));
    assert!(read_by(&sub, 7).contains("hello?"));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ResumeSubagentPlugin);
}
