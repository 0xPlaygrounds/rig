use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::clock::ClockPlugin;
use rig_harness_test_support::{
    AnswerAsks, Answers, MemoryStore, ToolOutputs, app_on, calls, log_tool_outputs,
    pass_wall_clock, reply, run_until, scripted_models,
};
use rig_subagents::{SubagentsPlugin, TASK, WAIT};
use serde_json::json;

use super::*;

/// The parent's model.
const PARENT: &str = "oc-ask/parent";
/// The first subagent's model.
const CHILD: &str = "oc-ask/child";
/// The second subagent's model.
const OTHER: &str = "oc-ask/other";

/// Where the test's `write` tool writes.
#[derive(Resource, Clone)]
struct Project(PathBuf);

/// The arguments of the test's `write` tool.
#[derive(Deserialize, JsonSchema)]
struct WriteArgs {
    path: String,
    text: String,
}

/// Writes a file under the [`Project`].
fn write(called: On<ToolCalled<WriteArgs>>, project: Res<Project>, mut commands: Commands) {
    let path = project.0.join(&called.args.path);
    let made = path.parent().map(std::fs::create_dir_all);
    let written =
        made.is_some_and(|made| made.is_ok()) && std::fs::write(&path, &called.args.text).is_ok();
    let output = if written {
        answered("Written.")
    } else {
        refused("Not written.")
    };
    commands.entity(called.call).insert_if_new(output);
}

/// The arguments of the test's `hold` tool.
#[derive(Deserialize, JsonSchema)]
struct HoldArgs {}

/// Keeps a subagent busy until the test decides the call ([`release`]).
fn hold(_: On<ToolCalled<HoldArgs>>) {}

/// Decides the open call `id` of `hold`.
fn release(world: &mut World, id: &str) {
    let mut calls = world.query::<(&ToolRequest, &mut Decision)>();
    for (request, mut decision) in calls.iter_mut(world) {
        if request.call.id.to_string() == id {
            decision.decide(Decider::plugin("test"), answered("Released."));
        }
    }
}

/// The model writes `wait` for a subagent by its task title, its id
/// unknown to a script: the id it would have read in `task`'s result.
fn name_the_subagent(
    mut calls: Query<(&mut ToolRequest, &ForAgent), Added<Admitted>>,
    children: Query<(&AgentId, &AgentTitle, &SpawnedBy)>,
) {
    for (mut request, &ForAgent(agent)) in &mut calls {
        if request.call.function.name.as_str() != WAIT {
            continue;
        }
        let args = &mut request.call.function.arguments;
        let title = args.get("agent").and_then(|agent| agent.as_str());
        let child = children.iter().find(|&(_, task, &SpawnedBy(parent))| {
            parent == agent && title == Some(task.0.as_str())
        });
        if let Some((id, ..)) = child {
            args.insert("agent".to_owned(), json!(id.short()));
        }
    }
}

/// An app on `store` with the subagents and this plugin, questions timing
/// out after `timeout_secs`, and a parent on [`PARENT`] unless the store
/// restores one.
struct Session {
    app: App,
    wakes: Receiver<()>,
    parent: Entity,
}

impl Drop for Session {
    fn drop(&mut self) {
        if let Some(project) = self.app.world().get_resource::<Project>() {
            std::fs::remove_dir_all(&project.0).ok();
        }
    }
}

impl Session {
    fn new(
        store: &MemoryStore,
        models: &[(&str, &MockCompletionModel)],
        timeout_secs: u64,
    ) -> Self {
        let fresh = store.files().unwrap_or_default().is_empty();
        let (mut app, wakes) = app_on(store);
        let project = std::env::current_exe()
            .unwrap_or_default()
            .with_file_name(format!("oc-ask-parent-{}", std::process::id()));
        std::fs::remove_dir_all(&project).ok();
        let settings = format!("[ask_parent]\ntimeout_secs = {timeout_secs}");
        app.insert_resource(Settings::from_toml(&settings))
            .insert_resource(scripted_models(models.iter().copied()))
            .insert_resource(Project(project))
            .init_resource::<ToolOutputs>()
            .add_plugins((AgentPlugin, SessionLogPlugin, ClockPlugin, AnswerAsks))
            .add_plugins((SubagentsPlugin, AskParentPlugin))
            .add_open_tool("write", "Write a file.", ToolOptions::default(), write)
            .add_open_tool("hold", "Hold on.", ToolOptions::default(), hold)
            .add_systems(AgentPass, name_the_subagent.in_set(PrepareSystems::Input))
            .add_observer(log_tool_outputs);
        if fresh {
            app.world_mut()
                .spawn((Agent, ModelChoice(PARENT.to_owned())));
        }
        app.finish();
        app.update();
        let world = app.world_mut();
        let mut roots = world.query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
        let parent = roots.iter(world).next().unwrap_or(Entity::PLACEHOLDER);
        Self { app, wakes, parent }
    }

    /// Asks the parent `text`, and runs until `done`.
    fn ask(&mut self, text: &str, done: impl Fn(&World) -> bool) {
        let ask = Deliver::new(self.parent, text, DeliveryMode::Queue);
        self.app.world_mut().trigger(ask);
        self.run(done);
    }

    /// Runs until `done`.
    fn run(&mut self, done: impl Fn(&World) -> bool) {
        run_until(&mut self.app, &self.wakes, done);
    }

    /// The output of the tool call `id`, as its model read it.
    fn output(&self, id: &str) -> String {
        let outputs = &self.app.world().resource::<ToolOutputs>().0;
        outputs.get(id).cloned().unwrap_or_default()
    }
}

/// How many questions are open.
fn open(world: &World) -> usize {
    let questions = world.try_query::<&Question>();
    questions.map_or(0, |mut questions| questions.iter(world).count())
}

/// What `model` read in its request `n` (from 0).
fn read_by(model: &MockCompletionModel, n: usize) -> String {
    let request = model.requests().into_iter().nth(n);
    request.map_or_else(String::new, |request| {
        json!(request.chat_history).to_string()
    })
}

/// A reply that calls `task` on `model`.
fn task(id: &str, title: &str, model: &str) -> MockStreamEvent {
    let args = json!({ "description": title, "prompt": title, "model": model });
    MockStreamEvent::tool_call(id, TASK, args)
}

/// A reply that asks the parent `question` in call `id`.
fn ask_parent(id: &str, question: &str) -> Vec<MockStreamEvent> {
    calls(id, ASK_PARENT, json!({ "question": question }))
}

#[test]
fn a_waiting_parent_reads_the_question_in_its_turn_and_its_answer_is_the_call_result() {
    let parent = MockCompletionModel::from_stream_turns([
        vec![
            task("t1", "Update the config", CHILD),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        calls("w1", WAIT, json!({ "agent": "Update the config" })),
        calls(
            "c1",
            ANSWER_CHILD,
            json!({ "id": "a1", "answer": "use config/prod.toml" }),
        ),
        calls("w2", WAIT, json!({ "agent": "Update the config" })),
        reply("done"),
    ]);
    let child = MockCompletionModel::from_stream_turns([
        ask_parent("a1", "prod.toml or dev.toml?"),
        calls(
            "e1",
            "write",
            json!({ "path": "config/prod.toml", "text": "logging = \"v2\"\n" }),
        ),
        reply("Updated config/prod.toml."),
    ]);
    let mut session = Session::new(
        &MemoryStore::default(),
        &[(PARENT, &parent), (CHILD, &child)],
        600,
    );
    session.ask("Update the config.", |_| parent.request_count() >= 5);

    // The parent read the question in the turn that started the task, its
    // `wait` for the subagent ended by it, before any report came.
    let extra = read_by(&parent, 2);
    assert!(extra.contains("Question from subagent `"), "{extra}");
    assert!(
        extra.contains("(task: Update the config): prod.toml or dev.toml?"),
        "{extra}"
    );
    assert!(!extra.contains("Updated config/prod.toml."), "{extra}");
    assert_eq!(session.output("a1"), "use config/prod.toml");
    let project = session.app.world().resource::<Project>().0.clone();
    let written = std::fs::read_to_string(project.join("config/prod.toml")).unwrap_or_default();
    assert_eq!(written, "logging = \"v2\"\n");
    // Its last request has the subagent's report.
    let last = read_by(&parent, 4);
    assert!(last.contains("Updated config/prod.toml."), "{last}");
    assert_eq!(open(session.app.world()), 0);
}

#[test]
fn a_question_ends_a_wait_for_another_subagent_and_each_answer_goes_to_its_asker() {
    let answers = vec![
        MockStreamEvent::tool_call("cb", ANSWER_CHILD, json!({ "id": "qb", "answer": "8080" })),
        MockStreamEvent::tool_call("ca", ANSWER_CHILD, json!({ "id": "qa", "ask_user": true })),
        MockStreamEvent::final_response_with_default_usage(),
    ];
    let mut turns = vec![
        vec![
            task("t1", "Check the wrapper", CHILD),
            task("t2", "Check the port", OTHER),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        calls("w1", WAIT, json!({ "agent": "Check the port" })),
        calls("w2", WAIT, json!({ "agent": "Check the port" })),
        answers,
    ];
    turns.extend((0..4).map(|_| reply("Noted.")));
    let parent = MockCompletionModel::from_stream_turns(turns);
    let wrapper = MockCompletionModel::from_stream_turns([
        ask_parent("qa", "Keep the wrapper?"),
        reply("A done."),
    ]);
    let port = MockCompletionModel::from_stream_turns([
        calls("h1", "hold", json!({})),
        ask_parent("qb", "Which port?"),
        reply("B done."),
    ]);
    let models = [(PARENT, &parent), (CHILD, &wrapper), (OTHER, &port)];
    let mut session = Session::new(&MemoryStore::default(), &models, 600);
    session
        .app
        .insert_resource(Answers::new([Answer::text("drop it")]));
    session.ask("Check both.", |_| parent.request_count() >= 3);

    // Waiting for the port's subagent, still busy, the parent read the
    // wrapper's question.
    let read = read_by(&parent, 2);
    assert!(read.contains("Keep the wrapper?"), "{read}");
    assert!(!read.contains("Which port?"), "{read}");
    assert!(session.output("w1").contains("asked you a question"));

    // The port's subagent asks too; the parent answers it first, and passes
    // the wrapper's question on to the user.
    release(session.app.world_mut(), "h1");
    session.run(|_| wrapper.request_count() >= 2 && port.request_count() >= 3);

    assert_eq!(session.output("qb"), "8080");
    assert_eq!(session.output("qa"), "The user answered: drop it");
    let told = session.output("ca");
    assert!(told.contains("drop it"), "{told}");
    assert!(read_by(&port, 2).contains("8080"));
    assert!(!read_by(&wrapper, 1).contains("8080"));
}

#[test]
fn an_unanswered_question_times_out_and_the_subagent_goes_on() {
    let parent = MockCompletionModel::from_stream_turns([
        vec![
            task("t1", "Update the config", CHILD),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Waiting."),
        reply("Hmm."),
        reply("Ok."),
    ]);
    let child = MockCompletionModel::from_stream_turns([
        ask_parent("a1", "prod or dev?"),
        reply("Assumed prod."),
    ]);
    let mut session = Session::new(
        &MemoryStore::default(),
        &[(PARENT, &parent), (CHILD, &child)],
        1,
    );
    session.ask("Update the config.", |world| open(world) == 1);
    pass_wall_clock(session.app.world_mut(), Duration::from_secs(2));
    session.run(|_| child.request_count() >= 2);

    let output = session.output("a1");
    assert!(
        output.starts_with("no answer from parent within 1 s"),
        "{output}"
    );
    assert!(read_by(&child, 1).contains("no answer from parent within 1 s"));
}

#[test]
fn stopping_the_parent_cancels_the_question_and_stops_the_subagent() {
    let parent = MockCompletionModel::from_stream_turns([
        vec![
            task("t1", "Update the config", CHILD),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Waiting."),
        reply("Hmm."),
        reply("Ok."),
        reply("Ok."),
    ]);
    let child =
        MockCompletionModel::from_stream_turns([ask_parent("a1", "prod or dev?"), reply("Never.")]);
    let mut session = Session::new(
        &MemoryStore::default(),
        &[(PARENT, &parent), (CHILD, &child)],
        600,
    );
    session.ask("Update the config.", |world| open(world) == 1);
    let parent_agent = session.parent;
    session.app.world_mut().trigger(StopTurn::new(parent_agent));
    session.run(|world| world.resource::<ToolOutputs>().0.contains_key("a1"));
    for _ in 0..4 {
        session.app.update();
    }

    let output = session.output("a1");
    assert!(
        output.contains("Not run: the user stopped the turn."),
        "{output}"
    );
    assert_eq!(child.request_count(), 1);
    assert_eq!(open(session.app.world()), 0);
    // The subagent's report that it stopped waits in the parent's
    // conversation and starts no parent turn.
    let stopped = "the user stopped the subagent before it answered";
    let requests = parent.requests().into_iter();
    let sent = requests.map(|request| json!(request.chat_history).to_string());
    assert!(!sent.into_iter().any(|sent| sent.contains(stopped)));
    let world = session.app.world();
    let kept = world
        .get::<Conversation>(parent_agent)
        .is_some_and(|conversation| json!(conversation.messages()).to_string().contains(stopped));
    assert!(kept);
}

#[test]
fn a_question_open_when_the_session_stopped_is_asked_again_on_resume() {
    let store = MemoryStore::default();
    let parent = MockCompletionModel::from_stream_turns([
        vec![
            task("t1", "Update the config", CHILD),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Waiting."),
        reply("Let me think."),
    ]);
    let child = MockCompletionModel::from_stream_turns([ask_parent("a1", "prod or dev?")]);
    let mut session = Session::new(&store, &[(PARENT, &parent), (CHILD, &child)], 600);
    session.ask("Update the config.", |world| {
        open(world) == 1 && parent.request_count() >= 3
    });
    for _ in 0..4 {
        session.app.update();
    }
    drop(session);

    let parent = MockCompletionModel::from_stream_turns([
        calls("c1", ANSWER_CHILD, json!({ "id": "a1", "answer": "prod" })),
        reply("Answered."),
        reply("done"),
    ]);
    let child = MockCompletionModel::from_stream_turns([reply("Used prod.")]);
    let mut session = Session::new(&store, &[(PARENT, &parent), (CHILD, &child)], 600);
    session.run(|_| child.request_count() >= 1);

    let asked = read_by(&parent, 0);
    assert_eq!(asked.matches("prod or dev?").count(), 2, "{asked}");
    assert_eq!(session.output("a1"), "prod");
    assert!(read_by(&child, 0).contains("prod"));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(AskParentPlugin);
}
