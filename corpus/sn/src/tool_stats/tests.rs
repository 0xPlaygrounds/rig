use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{ReadTool, ShellTool};
use rig_basics::subagents::{SubagentsPlugin, TASK};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::Home;
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Collected, answer_open, app_on, calls, collect, first_agent, reply, run_until, scratch_home,
    scripted_models,
};
use serde_json::json;

use super::*;

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ToolStatsPlugin);
}

/// Whether a `shell` call started.
#[derive(Resource, Default)]
struct ShellStarted(bool);

fn shell_started(
    started: On<Add<Started>>,
    calls: Query<&ToolRequest>,
    mut seen: ResMut<ShellStarted>,
) {
    if calls
        .get(started.entity)
        .is_ok_and(|request| request.call.function.name.as_str() == "shell")
    {
        seen.0 = true;
    }
}

/// A session of the project at `root`, in `home`, its main agent on
/// `test/main` and subagents free to pick `test/sub`.
fn session(
    home: &Home,
    root: &Path,
    main: &MockCompletionModel,
    sub: &MockCompletionModel,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(RigHome(home.clone()))
        .insert_resource(ProjectRoot(root.to_path_buf()))
        .insert_resource(scripted_models([("test/main", main), ("test/sub", sub)]))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            ReadTool,
            ShellTool,
            SubagentsPlugin,
            ToolStatsPlugin,
        ))
        .init_resource::<ShellStarted>()
        .add_observer(shell_started);
    collect::<ToolStatsShown>(&mut app);
    collect::<Notice>(&mut app);
    app.world_mut().spawn(Answerer::default());
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let model = "test/main".to_owned();
    app.world_mut().trigger(SetModel {
        entity: agent,
        model,
        kind: ChoiceKind::ForThisRun,
    });
    (app, wakes, agent)
}

/// The session's totals of `tool`, every agent's.
fn session_calls(world: &World, tool: &str) -> Totals {
    let table = &world.resource::<SessionStats>().0;
    table
        .summed(0, false)
        .remove(&(tool.to_owned(), None))
        .unwrap_or_default()
}

/// Runs `/toolstats <args>` and returns the table it showed.
fn toolstats(app: &mut App, wakes: &Receiver<()>, agent: Entity, args: &str) -> ToolStatsShown {
    let before = app.world().resource::<Collected<ToolStatsShown>>().0.len();
    let line = format!("toolstats {args}");
    app.world_mut().trigger(RunCommand::new(agent, line));
    run_until(app, wakes, |world| {
        world.resource::<Collected<ToolStatsShown>>().0.len() > before
    });
    let shown = app
        .world()
        .resource::<Collected<ToolStatsShown>>()
        .0
        .last()
        .cloned();
    shown.unwrap_or(ToolStatsShown {
        agent,
        of: StatsOf::Session,
        rows: Vec::new(),
    })
}

/// The row of `tool` (and `agent`, when split by agent).
fn row<'a>(shown: &'a ToolStatsShown, tool: &str, agent: Option<&str>) -> Option<&'a ToolSummary> {
    shown.rows.iter().find(|row| {
        row.tool == tool && agent.is_none_or(|agent| row.agent.as_deref() == Some(agent))
    })
}

fn calls_of(shown: &ToolStatsShown, tool: &str) -> u64 {
    row(shown, tool, None).map_or(0, |row| row.totals.calls)
}

fn read(id: &str, path: &Path) -> MockStreamEvent {
    MockStreamEvent::tool_call(id, "read", json!({ "path": path.display().to_string() }))
}

#[test]
fn calls_are_counted_per_session_per_agent_and_per_project() {
    let home = scratch_home("tool-stats");
    let root: PathBuf = home.root().join("project");
    std::fs::create_dir_all(&root).ok();
    let file = root.join("notes.txt");
    std::fs::write(&file, "some notes\n").ok();
    let missing = |name: &str| root.join(format!("missing-{name}.txt"));

    let first_turn = vec![
        read("r1", &file),
        read("r2", &file),
        read("r3", &file),
        read("m1", &missing("17")),
        read("m2", &missing("4242")),
        MockStreamEvent::tool_call("u1", "no_such_tool", json!({})),
        MockStreamEvent::tool_call("s1", "shell", json!({ "command": "sleep 5" })),
        MockStreamEvent::final_response_with_default_usage(),
    ];
    let task = json!({ "description": "Read twice", "prompt": "Read it.", "model": "test/sub" });
    let main = MockCompletionModel::from_stream_turns([
        first_turn,
        calls("t1", TASK, task),
        reply("Started."),
        reply("Done."),
        reply("Done."),
    ]);
    let sub = MockCompletionModel::from_stream_turns([
        vec![
            read("r6", &file),
            read("r7", &file),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Read it twice."),
    ]);
    let (mut app, wakes, agent) = session(&home, &root, &main, &sub);

    app.world_mut()
        .trigger(Deliver::new(agent, "go", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| world.resource::<ShellStarted>().0);
    std::thread::sleep(Duration::from_secs(1));
    app.world_mut().trigger(StopTurn::new(agent));
    run_until(&mut app, &wakes, |world| {
        session_calls(world, "shell").cancelled == 1
    });

    let shown = toolstats(&mut app, &wakes, agent, "");
    let read_row = row(&shown, "read", None).map(|row| (row.totals.clone(), row.top_errors()));
    let (read_totals, read_errors) = read_row.unwrap_or_default();
    assert_eq!(
        (read_totals.calls, read_totals.errors),
        (5, 2),
        "{shown:#?}"
    );
    assert_eq!(read_errors.len(), 1, "{read_errors:?}");
    assert_eq!(
        read_errors.first().map(|(_, count)| *count),
        Some(2),
        "{read_errors:?}"
    );
    let unknown = row(&shown, "no_such_tool", None).map(|row| row.totals.errors);
    assert_eq!(unknown, Some(1), "{shown:#?}");
    let shell = row(&shown, "shell", None).map(|row| row.totals.clone());
    let shell = shell.unwrap_or_default();
    assert_eq!(shell.cancelled, 1, "{shown:#?}");
    assert!(
        shell.percentile_ms(0.5).is_some_and(|ms| ms >= 800),
        "{shell:?}"
    );

    app.world_mut()
        .trigger(Deliver::new(agent, "delegate", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        session_calls(world, "read").calls == 7
    });
    let shown = toolstats(&mut app, &wakes, agent, "");
    assert_eq!(calls_of(&shown, "read"), 7, "{shown:#?}");
    let by_agent = toolstats(&mut app, &wakes, agent, "--by-agent");
    let main_name = app
        .world()
        .get::<Name>(agent)
        .map(|name| name.as_str().to_owned());
    let under_sub: Vec<u64> = by_agent
        .rows
        .iter()
        .filter(|row| row.tool == "read" && row.agent != main_name)
        .map(|row| row.totals.calls)
        .collect();
    assert_eq!(under_sub, [2], "{by_agent:#?}");
    let all = toolstats(&mut app, &wakes, agent, "--all");
    let notices: Vec<String> = app
        .world()
        .resource::<Collected<Notice>>()
        .0
        .iter()
        .map(|n| n.text.clone())
        .collect();
    assert_eq!(calls_of(&all, "read"), 7, "{notices:#?}");
    drop(app);

    // A new session of the same project.
    let main = MockCompletionModel::from_stream_turns([
        vec![
            read("r8", &file),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        reply("Read it."),
    ]);
    let (mut app, wakes, agent) = session(&home, &root, &main, &sub);
    app.world_mut()
        .trigger(Deliver::new(agent, "read it", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        session_calls(world, "read").calls == 1
    });
    let shown = toolstats(&mut app, &wakes, agent, "");
    assert_eq!(calls_of(&shown, "read"), 1, "{shown:#?}");
    let all = toolstats(&mut app, &wakes, agent, "--all");
    assert_eq!(calls_of(&all, "read"), 8, "{all:#?}");
    let week = toolstats(&mut app, &wakes, agent, "--since 7d");
    assert_eq!(calls_of(&week, "read"), 8, "{week:#?}");

    app.world_mut()
        .trigger(RunCommand::new(agent, "toolstats reset"));
    let yes = Answer::choice("yes");
    run_until(&mut app, &wakes, |world| {
        world
            .try_query_filtered::<&Ask, Without<Outcome>>()
            .is_some_and(|mut asks| asks.iter(world).any(|ask| ask.title == RESET_TITLE))
    });
    assert!(answer_open(app.world_mut(), RESET_TITLE, &yes));
    app.update();
    let all = toolstats(&mut app, &wakes, agent, "--all");
    assert!(all.rows.is_empty(), "{all:#?}");
}
