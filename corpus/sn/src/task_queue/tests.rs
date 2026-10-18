use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Collected, app_on, calls, collect, reply, run_until, scripted_models,
};
use serde_json::json;

use super::*;

/// The `shell` calls the test has not answered yet.
#[derive(Resource, Default)]
struct Shells(Vec<Entity>);

/// What happened, in order: `request:<agent>` as each model call starts and
/// `end:<agent>` as each turn ends.
#[derive(Resource, Default)]
struct Log(Vec<String>);

/// What `shell` is called with.
#[derive(serde::Deserialize, schemars::JsonSchema)]
struct ShellArgs {
    /// The command.
    command: String,
}

/// A `shell` whose calls wait for the test to answer them; one without a
/// command is refused at once.
fn shell(
    called: On<ToolCalled<ShellArgs>>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut shells: ResMut<Shells>,
) {
    if !called.args.command.is_empty() {
        shells.0.push(called.call);
    } else if let Ok(mut decision) = calls.get_mut(called.call) {
        decision.decide(Decider::plugin("test"), Outcome::refused("no command"));
    }
}

fn log_requests(
    started: On<Add<Started>>,
    calls: Query<&ForAgent, With<ModelRequest>>,
    names: Query<&Name>,
    mut log: ResMut<Log>,
) {
    if let Ok(&ForAgent(agent)) = calls.get(started.entity) {
        let name = names.get(agent).map(Name::as_str).unwrap_or_default();
        log.0.push(format!("request:{name}"));
    }
}

fn log_turn_ends(
    settled: On<Add<Settled>>,
    turns: Query<&ForAgent, With<Turn>>,
    names: Query<&Name>,
    mut log: ResMut<Log>,
) {
    if let Ok(&ForAgent(agent)) = turns.get(settled.entity) {
        let name = names.get(agent).map(Name::as_str).unwrap_or_default();
        log.0.push(format!("end:{name}"));
    }
}

/// An app on `store` with the queue, the main agent on `test/main` and the
/// tasks' model `test/tasks`, after its first frame; and its main agent.
fn session(
    store: &MemoryStore,
    tasks: &MockCompletionModel,
    main: &MockCompletionModel,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    collect::<Notice>(&mut app);
    app.insert_resource(scripted_models([
        ("test/main", main),
        ("test/tasks", tasks),
    ]))
    .init_resource::<rig_harness::worker::WorkerOutbox>()
    .add_plugins((AgentPlugin, SessionLogPlugin, TaskQueuePlugin))
    .add_open_tool("shell", "Run a command.", ToolOptions::default(), shell)
    .init_resource::<Shells>()
    .init_resource::<Log>()
    .add_observer(log_requests)
    .add_observer(log_turn_ends);
    app.finish();
    app.update();
    let world = app.world_mut();
    let mut roots = world.query_filtered::<Entity, (With<Agent>, Without<SpawnedBy>)>();
    let main = roots.iter(world).next().unwrap_or(Entity::PLACEHOLDER);
    let model = "test/main".to_owned();
    world.trigger(SetModel {
        entity: main,
        model,
    });
    app.update();
    (app, wakes, main)
}

fn send(app: &mut App, agent: Entity, line: &str) {
    let line = Deliver::new(agent, line, DeliveryMode::Queue);
    app.world_mut().trigger(line);
    app.update();
}

fn state(world: &World, number: u32) -> Option<TaskState> {
    let tasks = &world.resource::<TaskQueue>().tasks;
    let task = tasks.iter().find(|task| task.number == number);
    task.map(|task| task.state)
}

fn agent_of(app: &mut App, number: u32) -> Option<Entity> {
    let world = app.world_mut();
    let mut agents = world.query::<(Entity, &QueuedTask)>();
    let found = agents.iter(world).find(|(_, task)| task.0 == number);
    found.map(|(agent, _)| agent)
}

/// The text of `agent`'s conversation.
fn said(app: &App, agent: Entity) -> String {
    let conversation = app.world().get::<Conversation>(agent);
    format!(
        "{:?}",
        conversation.map(|conversation| conversation.messages())
    )
}

/// Answers every `shell` call waiting.
fn release(app: &mut App) {
    let waiting = std::mem::take(&mut app.world_mut().resource_mut::<Shells>().0);
    for call in waiting {
        if let Some(mut decision) = app.world_mut().get_mut::<Decision>(call) {
            decision.decide(Decider::plugin("test"), ToolResult::success("ok".into()));
        }
    }
    app.update();
}

fn shell_waits(world: &World) -> bool {
    !world.resource::<Shells>().0.is_empty()
}

fn at(app: &App, entry: &str) -> Option<usize> {
    app.world()
        .resource::<Log>()
        .0
        .iter()
        .position(|seen| seen == entry)
}

fn notices(app: &App) -> Vec<String> {
    let notices = &app.world().resource::<Collected<Notice>>().0;
    notices.iter().map(|notice| notice.text.clone()).collect()
}

#[test]
fn tasks_run_one_after_another_in_agents_of_their_own() {
    let store = MemoryStore::default();
    let tasks = MockCompletionModel::from_stream_turns([
        calls("t1", "shell", json!({ "command": "ls" })),
        reply("T1 listed the files"),
        calls("t2", "shell", json!({ "command": "date" })),
        reply("T2 told the date"),
        reply("T1 follow-up answered"),
    ]);
    let main = MockCompletionModel::from_stream_turns([reply("main answered")]);
    let (mut app, wakes, agent) = session(&store, &tasks, &main);
    for line in [
        "/queue --model test/tasks T1 list the files",
        "/queue --model test/tasks T2 tell the date",
        "/queue --model test/tasks T3 never runs",
        "/queue cancel 3",
    ] {
        send(&mut app, agent, line);
    }
    assert_eq!(state(app.world(), 3), Some(TaskState::Cancelled));

    // T1 waits in its `shell` call: the main agent's turn runs and ends
    // meanwhile, and T2 waits its turn.
    run_until(&mut app, &wakes, shell_waits);
    send(&mut app, agent, "hello");
    run_until(&mut app, &wakes, |world| {
        world
            .resource::<Log>()
            .0
            .iter()
            .any(|seen| seen == "end:agent")
    });
    assert_eq!(
        state(app.world(), 1),
        Some(TaskState::Running),
        "{:?}",
        notices(&app)
    );
    assert_eq!(state(app.world(), 2), Some(TaskState::Pending));

    release(&mut app);
    run_until(&mut app, &wakes, shell_waits);
    release(&mut app);
    run_until(&mut app, &wakes, |world| {
        state(world, 2) == Some(TaskState::Done)
    });
    let log = app.world().resource::<Log>().0.clone();
    assert_eq!(state(app.world(), 2), Some(TaskState::Done), "{log:?}");
    // T2 starts only once T1 answered; T3 never asks anything.
    let (t1_end, t2_first) = (at(&app, "end:task 1"), at(&app, "request:task 2"));
    assert!(t1_end.is_some() && t1_end < t2_first, "{log:?}");
    assert!(!log.iter().any(|seen| seen.ends_with("task 3")), "{log:?}");
    assert_eq!(agent_of(&mut app, 3), None);
    assert!(
        notices(&app)
            .iter()
            .any(|text| text.starts_with("Task 2 done"))
    );
    // The views were sent the panel's rows as the queue changed.
    let outbox = app.world().resource::<rig_harness::worker::WorkerOutbox>();
    let last = outbox
        .events
        .iter()
        .rev()
        .find(|event| event.kind == <QueueRows as rig_harness::channel::ChannelEvent>::KIND);
    let rows = last.and_then(|event| serde_json::from_value::<QueueRows>(event.data.clone()).ok());
    let rows = rows.map(|rows| rows.rows).unwrap_or_default();
    assert_eq!(rows.len(), 3, "{rows:?}");
    assert!(rows.iter().any(|row| row.contains("done")), "{rows:?}");

    // Each task's conversation holds its prompt and replies alone, and the
    // main agent's none of theirs.
    let (t1, t2) = (agent_of(&mut app, 1), agent_of(&mut app, 2));
    assert!(t1.is_some() && t2.is_some(), "{log:?}");
    let (t1, t2) = (
        t1.unwrap_or(Entity::PLACEHOLDER),
        t2.unwrap_or(Entity::PLACEHOLDER),
    );
    let (t1_said, t2_said, main_said) = (said(&app, t1), said(&app, t2), said(&app, agent));
    assert!(t1_said.contains("T1 list the files") && t1_said.contains("T1 listed"));
    assert!(!t1_said.contains("T2"), "{t1_said}");
    assert!(
        t2_said.contains("T2 tell the date") && !t2_said.contains("T1"),
        "{t2_said}"
    );
    assert!(main_said.contains("main answered"), "{main_said}");
    assert!(
        !main_said.contains("T1") && !main_said.contains("T2"),
        "{main_said}"
    );
    let list = format!("{:?}", app.world().resource::<TaskQueue>().tasks);
    assert!(list.contains("T1 listed the files"), "{list}");

    // `/queue open 1` shows T1, and what the user sends there goes to it.
    send(&mut app, agent, "/queue open 1");
    let shown = app.world().resource::<ShownAgent>().0;
    assert_eq!(shown, Some(t1));
    send(&mut app, t1, "anything else?");
    run_until(&mut app, &wakes, |world| {
        world
            .resource::<Log>()
            .0
            .iter()
            .filter(|seen| *seen == "end:task 1")
            .count()
            == 2
    });
    let t1_said = said(&app, t1);
    assert!(t1_said.contains("T1 follow-up answered"), "{t1_said}");
    assert!(!said(&app, agent).contains("anything else"));
}

#[test]
fn a_restart_interrupts_the_running_task_and_runs_the_pending_ones() {
    let store = MemoryStore::default();
    let main = MockCompletionModel::from_stream_turns([reply("unused")]);
    let tasks = MockCompletionModel::from_stream_turns([calls(
        "t4",
        "shell",
        json!({ "command": "make" }),
    )]);
    let (mut app, wakes, agent) = session(&store, &tasks, &main);
    send(&mut app, agent, "/queue --model test/tasks T4 build it");
    send(&mut app, agent, "/queue --model test/tasks T5 test it");
    run_until(&mut app, &wakes, shell_waits);
    let before = agent_of(&mut app, 1).and_then(|t4| app.world().get::<AgentId>(t4).cloned());
    // Quit while T4 (task 1) runs; T5 (task 2) waits.
    app.world_mut().write_message(AppExit::Success);
    app.update();
    drop(app);

    let tasks = MockCompletionModel::from_stream_turns([
        calls("t5", "shell", json!({ "command": "test" })),
        reply("T5 passed"),
        calls("t4b", "shell", json!({ "command": "make" })),
        reply("T4 built"),
    ]);
    let (mut app, wakes, agent) = session(&store, &tasks, &main);
    assert_eq!(state(app.world(), 1), Some(TaskState::Interrupted));
    run_until(&mut app, &wakes, shell_waits);
    release(&mut app);
    run_until(&mut app, &wakes, |world| {
        state(world, 2) == Some(TaskState::Done)
    });
    assert_eq!(
        state(app.world(), 2),
        Some(TaskState::Done),
        "{:?}",
        notices(&app)
    );
    // T4's restored turn was stopped before it asked the model.
    assert_eq!(tasks.request_count(), 2);
    assert!(
        at(&app, "end:task 1").is_some(),
        "{:?}",
        app.world().resource::<Log>().0
    );
    assert_eq!(state(app.world(), 1), Some(TaskState::Interrupted));

    // `/queue retry 1` runs it again in its own agent.
    send(&mut app, agent, "/queue retry 1");
    run_until(&mut app, &wakes, shell_waits);
    release(&mut app);
    run_until(&mut app, &wakes, |world| {
        state(world, 1) == Some(TaskState::Done)
    });
    assert_eq!(
        state(app.world(), 1),
        Some(TaskState::Done),
        "{:?}",
        notices(&app)
    );
    let t4 = agent_of(&mut app, 1).unwrap_or(Entity::PLACEHOLDER);
    let after = app.world().get::<AgentId>(t4).cloned();
    assert!(before.is_some() && before == after, "{before:?} {after:?}");
    let t4_said = said(&app, t4);
    assert_eq!(t4_said.matches("T4 build it").count(), 2, "{t4_said}");
    assert!(t4_said.contains("T4 built"), "{t4_said}");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(TaskQueuePlugin);
}

/// An idle queue is left alone, so its panel and status do not redraw the
/// terminal on every update.
#[test]
fn an_idle_queue_is_not_marked_changed_each_update() {
    let store = MemoryStore::default();
    let unused = MockCompletionModel::from_stream_turns([reply("unused")]);
    let (tasks, main) = (unused.clone(), unused);
    let (mut app, _wakes, _) = session(&store, &tasks, &main);
    let changed = |app: &App| {
        let ticks = app.world().get_resource_change_ticks::<TaskQueue>();
        ticks.map(|ticks| ticks.changed)
    };
    let before = changed(&app);
    for _ in 0..3 {
        app.update();
    }
    assert!(before.is_some());
    assert_eq!(changed(&app), before);
}
