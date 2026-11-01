use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{ReadTool, ShellTool};
use rig_basics::subagents::{SubagentsPlugin, TASK};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{
    Collected, MemoryStore, ToolOutputs, app_on, calls, collect, log_tool_outputs, reply,
    run_until, scripted_models,
};
use serde_json::{Value, json};

use super::*;

/// The main agent's model.
const MAIN: &str = "sc/main";
/// The subagents' model.
const SUB: &str = "sc/sub";
/// The lookup the main agent keeps asking for.
const FIND: &str = "find   def of\n load_config";
/// A task whose subagent runs a command.
const BUILD: &str = "check that it builds";

/// An app on `store` working in `project`, with the subagents, the real
/// `read`, `search` and `shell` tools and the plugin.
struct Session {
    app: App,
    wakes: Receiver<()>,
    main: Entity,
    main_model: MockCompletionModel,
}

impl Session {
    fn new(
        store: &MemoryStore,
        project: &Path,
        main: MockCompletionModel,
        sub: &MockCompletionModel,
    ) -> Self {
        let fresh = store.logs().unwrap_or_default().is_empty();
        let (mut app, wakes) = app_on(store);
        app.insert_resource(scripted_models([(MAIN, &main), (SUB, sub)]))
            .insert_resource(ProjectRoot(project.to_path_buf()))
            .init_resource::<ToolOutputs>()
            .add_plugins((
                AgentPlugin,
                SessionLogPlugin,
                SubagentsPlugin,
                ReadTool,
                ShellTool,
                SubagentResultCachePlugin,
            ))
            .add_tool_per_call(
                rig_harness::rig_tools::Search::default(),
                ToolOptions::default(),
            )
            .add_observer(log_tool_outputs);
        collect::<Notice>(&mut app);
        if fresh {
            app.world_mut().spawn((Agent, ModelChoice(MAIN.to_owned())));
        }
        app.finish();
        app.update();
        let world = app.world_mut();
        let mut roots = world.query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
        let main_agent = roots.iter(world).next().unwrap_or(Entity::PLACEHOLDER);
        Self {
            app,
            wakes,
            main: main_agent,
            main_model: main,
        }
    }

    /// Sends the main agent `text` and runs until its model was asked
    /// `requests` times in all and no agent works.
    fn ask(&mut self, text: &str, requests: usize) {
        let message = Deliver::new(self.main, text, DeliveryMode::Queue);
        self.app.world_mut().trigger(message);
        let model = self.main_model.clone();
        run_until(&mut self.app, &self.wakes, move |world| {
            model.requests().len() == requests && idle(world)
        });
        for _ in 0..4 {
            self.app.update();
        }
    }

    fn command(&mut self, line: &str) {
        self.app
            .world_mut()
            .trigger(RunCommand::new(self.main, line));
        self.app.update();
    }

    fn output(&self, id: &str) -> String {
        let outputs = &self.app.world().resource::<ToolOutputs>().0;
        outputs.get(id).cloned().unwrap_or_default()
    }

    fn last_notice(&self) -> String {
        let notices = &self.app.world().resource::<Collected<Notice>>().0;
        notices
            .last()
            .map(|notice| notice.text.clone())
            .unwrap_or_default()
    }
}

/// Whether no agent has a turn running and no subagent is about to.
fn idle(world: &World) -> bool {
    let none = |world: &World, busy: fn(&World) -> bool| !busy(world);
    none(world, |world| {
        world
            .try_query::<&ActiveTurn>()
            .is_some_and(|mut turns| turns.iter(world).next().is_some())
    }) && none(world, |world| {
        world
            .try_query::<&Incoming>()
            .is_some_and(|mut sent| sent.iter(world).next().is_some())
    })
}

/// The main agent's script for one `task` call `id` of `task`: the call,
/// what it says after, and, when a subagent runs, what it says to its
/// report.
fn asks(id: &str, task: &str, runs: bool) -> Vec<Vec<MockStreamEvent>> {
    let args = json!({ "description": "lookup", "prompt": task, "model": SUB });
    let mut script = vec![calls(id, TASK, args), reply("Waiting for it.")];
    if runs {
        script.push(reply("Thanks."));
    }
    script
}

/// A subagent's script for [`FIND`]: read, search, answer.
fn finds(n: usize) -> Vec<Vec<MockStreamEvent>> {
    vec![
        calls(&format!("r{n}"), "read", json!({ "path": "src/config.rs" })),
        calls(
            &format!("s{n}"),
            "search",
            json!({ "pattern": "load_config", "path": "src/" }),
        ),
        reply("src/config.rs:12"),
    ]
}

/// A subagent's script for [`BUILD`]: a command, then an answer.
fn builds(n: usize) -> Vec<Vec<MockStreamEvent>> {
    vec![
        calls(&format!("sh{n}"), "shell", json!({ "command": "true" })),
        reply("It builds."),
    ]
}

/// A project of the test `name` beside the test binary, with
/// `src/config.rs`.
fn project(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("subcache-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(dir.join("src")).ok();
    let config = "use std::fs;\n\n// Loading.\n\n\n\n\n\n\n\n\npub fn load_config() {}\n";
    std::fs::write(dir.join("src/config.rs"), config).ok();
    dir
}

/// The spec's scripted test: a hit, misses after a new file and a changed
/// byte, a subagent that ran a command never recorded, `/subcache`, and a
/// hit after a restart until `/subcache off`.
#[test]
fn a_repeated_read_only_task_is_answered_from_its_record_while_its_inputs_hold() {
    let dir = project("hits");
    let store = MemoryStore::default();
    let script = [
        asks("t1", FIND, true),
        asks("t2", FIND, false),
        asks("t3", FIND, true),
        asks("t4", FIND, true),
        asks("t5", BUILD, true),
        asks("t6", BUILD, true),
    ];
    let main = MockCompletionModel::from_stream_turns(script.concat());
    let sub = MockCompletionModel::from_stream_turns(
        [finds(1), finds(3), finds(4), builds(5), builds(6)].concat(),
    );
    let mut session = Session::new(&store, &dir, main, &sub);

    session.ask("Where is the config loaded?", 3);
    assert_eq!(sub.requests().len(), 3);
    session.ask("And again?", 5);
    assert_eq!(
        sub.requests().len(),
        3,
        "a hit asks the subagent's model nothing"
    );
    let reused = session.output("t2");
    assert!(
        reused.starts_with("src/config.rs:12\n(reused result from turn 1; inputs unchanged)"),
        "{reused}"
    );

    std::fs::write(dir.join("src/extra.rs"), "pub fn extra() {}\n").ok();
    session.ask("Once more?", 8);
    assert_eq!(
        sub.requests().len(),
        6,
        "a new file in a searched directory is a miss"
    );
    let cache = session.app.world().get::<SubagentCache>(session.main);
    let records = cache.map(|cache| cache.0.clone()).unwrap_or_default();
    let listed = |record: &Record| {
        record
            .dirs
            .values()
            .flatten()
            .any(|f| f.ends_with("extra.rs"))
    };
    assert!(
        records.len() == 1 && records.iter().all(listed),
        "the miss replaced the record: {records:?}"
    );

    let config = std::fs::read_to_string(dir.join("src/config.rs")).unwrap_or_default();
    std::fs::write(
        dir.join("src/config.rs"),
        config.replacen("Loading", "Loaded!", 1),
    )
    .ok();
    session.ask("And now?", 11);
    assert_eq!(sub.requests().len(), 9, "a changed file is a miss");

    session.ask("Does it build?", 14);
    session.ask("Does it build now?", 17);
    assert_eq!(
        sub.requests().len(),
        13,
        "a subagent that ran a command is never recorded"
    );

    session.command("subcache");
    let listed = session.last_notice();
    assert!(
        listed.contains("1 record(s), 3 model calls saved"),
        "{listed}"
    );
    assert!(
        listed.contains("\"find def of load_config\"") && listed.contains("1 hit(s)"),
        "{listed}"
    );
    drop(session);

    let main = MockCompletionModel::from_stream_turns(
        [asks("t7", FIND, false), asks("t8", FIND, true)].concat(),
    );
    let sub = MockCompletionModel::from_stream_turns(finds(8));
    let mut session = Session::new(&store, &dir, main, &sub);
    session.ask("After the restart?", 2);
    assert_eq!(sub.requests().len(), 0, "the record holds after a restart");
    assert!(
        session.output("t7").contains("inputs unchanged"),
        "{}",
        session.output("t7")
    );
    session.command("subcache off");
    session.ask("With the cache off?", 5);
    assert_eq!(
        sub.requests().len(),
        3,
        "with the cache off the subagent runs"
    );
    std::fs::remove_dir_all(&dir).ok();
}

/// The main agent's script for one `task` call `id` of [`FIND`] with
/// `tools`, run by a subagent: the call, then what it says after.
fn asks_with(id: &str, tools: &Value) -> Vec<Vec<MockStreamEvent>> {
    let args = json!({ "description": "lookup", "prompt": FIND, "model": SUB, "tools": tools });
    vec![
        calls(id, TASK, args),
        reply("Waiting for it."),
        reply("Thanks."),
    ]
}

/// What the main agent's `task` call `id` answers once it sent `text`:
/// a reused answer, a refusal, or the start of a subagent.
fn answer_to(session: &mut Session, text: &str, id: &str) -> String {
    let ask = Deliver::new(session.main, text, DeliveryMode::Queue);
    session.app.world_mut().trigger(ask);
    let call = id.to_owned();
    run_until(&mut session.app, &session.wakes, move |world| {
        world.resource::<ToolOutputs>().0.contains_key(&call)
    });
    session.output(id)
}

#[test]
fn an_answer_from_a_limited_run_is_not_reused_once_the_tools_are_back() {
    for tools in [Value::Null, json!(["read"])] {
        let dir = project("limited");
        let store = MemoryStore::default();
        let main = MockCompletionModel::from_stream_turns(asks_with("t1", &tools));
        let sub = MockCompletionModel::from_stream_turns([reply("I cannot message you.")]);
        let mut session = Session::new(&store, &dir, main, &sub);
        session.app.insert_resource(RunTools::new([TASK, "read"]));
        session.ask("Where is the config loaded?", 3);
        assert_eq!(sub.requests().len(), 1, "{tools}");
        drop(session);

        let main = MockCompletionModel::from_stream_turns(asks_with("t2", &tools));
        let sub = MockCompletionModel::from_stream_turns(finds(2));
        let mut session = Session::new(&store, &dir, main, &sub);
        let answer = answer_to(&mut session, "And with every tool?", "t2");
        assert!(answer.starts_with("Started subagent"), "{tools}: {answer}");
        std::fs::remove_dir_all(&dir).ok();
    }
}

#[test]
fn a_list_task_refuses_is_never_answered_from_a_record() {
    let dir = project("refused");
    let main = MockCompletionModel::from_stream_turns(
        [
            asks_with("t1", &json!(["read"])),
            asks_with("t2", &json!(["read", "shell"])),
        ]
        .concat(),
    );
    let sub = MockCompletionModel::from_stream_turns([reply("src/config.rs:12")]);
    let mut session = Session::new(&MemoryStore::default(), &dir, main, &sub);
    let keeps = ToolAccess::all().narrow([TASK, "read"]);
    session
        .app
        .world_mut()
        .entity_mut(session.main)
        .insert(keeps);
    session.ask("Where is the config loaded?", 3);
    let refused = answer_to(&mut session, "And with the shell?", "t2");
    assert!(
        refused.contains("`shell` is not one of your tools"),
        "{refused}"
    );
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(SubagentResultCachePlugin);
}
