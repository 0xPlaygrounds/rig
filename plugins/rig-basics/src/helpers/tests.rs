use std::path::PathBuf;

use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    MemoryStore, ToolOutputs, app_on, calls, connect, first_agent, log_tool_outputs, reply,
    run_until, scratch_home,
};
use serde_json::json;

use super::*;

/// A stand-in for `rig -p --json`: it notes its arguments and how many
/// run at once in the directory it is given, and answers its task (the
/// last argument) by what the task says.
const CHILD: &str = r#"d=$1; shift
for a; do task=$a; done
mkdir -p "$d/run"; echo "$@" >> "$d/args"
touch "$d/run/$$"; ls "$d/run" | wc -l >> "$d/counts"
case "$task" in
  fail*) echo '{"v":1,"type":"exit","code":1,"error":"it broke"}'; rm "$d/run/$$"; exit 1;;
  crash*) echo boom >&2; rm "$d/run/$$"; exit 3;;
  slow*) sleep 30;;
  wait*) sleep 0.3;;
esac
rm "$d/run/$$"
printf '{"v":1,"type":"exit","code":0,"answer":"did: %s"}\n' "$task"
"#;

/// An app whose agent's model calls `helper` with each of `args` in turn,
/// then answers, its helpers the [`CHILD`] noting in its own directory.
fn helped(name: &str, args: &[Value]) -> (App, std::sync::mpsc::Receiver<()>, PathBuf) {
    let dir = scratch_home(&format!("rig-basics-helpers-{name}"))
        .root()
        .to_path_buf();
    std::fs::create_dir_all(&dir).ok();
    let mut turns: Vec<_> = args
        .iter()
        .enumerate()
        .map(|(index, args)| calls(&format!("h{index}"), HELPER, args.clone()))
        .collect();
    turns.push(reply("done"));
    let model = MockCompletionModel::from_stream_turns(turns);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let command = HelperCommand {
        program: "sh".into(),
        args: vec![
            "-c".into(),
            CHILD.into(),
            "child".into(),
            dir.clone().into(),
        ],
    };
    app.insert_resource(ProjectRoot(dir.clone()))
        .add_plugins((AgentPlugin, SessionLogPlugin, HelpersPlugin))
        .insert_resource(command)
        .add_open_tool(
            "read",
            "Read.",
            ToolOptions::default(),
            |_: On<ToolCalled<Value>>| {},
        )
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    app.finish();
    app.cleanup();
    // Registered while the app runs, as an MCP server's tools are: a child
    // would not have it.
    app.add_open_tool(
        "mcp__late",
        "Late.",
        ToolOptions::default(),
        |_: On<ToolCalled<Value>>| {},
    );
    app.update();
    let world = app.world_mut();
    if let (Some(agent), Some(connection)) = (first_agent(world), connect(&model)) {
        world.entity_mut(agent).insert(connection);
        world.trigger(Deliver::new(agent, "Go.", DeliveryMode::Steer));
    }
    (app, wakes, dir)
}

/// The output of the call `h{index}`, once the app has it.
fn output(app: &mut App, wakes: &std::sync::mpsc::Receiver<()>, index: usize) -> String {
    let id = format!("h{index}");
    run_until(app, wakes, |world| {
        world.resource::<ToolOutputs>().0.contains_key(&id)
    });
    app.world()
        .resource::<ToolOutputs>()
        .0
        .get(&id)
        .cloned()
        .unwrap_or_default()
}

#[test]
fn a_helper_answers_and_a_chain_hands_each_answer_on() {
    let (mut app, wakes, dir) = helped(
        "chain",
        &[
            json!({ "task": "one" }),
            json!({ "chain": [{ "task": "two" }, { "task": "three after {previous}" }] }),
            json!({ "task": "four", "tools": ["shell"] }),
        ],
    );
    assert_eq!(output(&mut app, &wakes, 0), "did: one");
    assert_eq!(output(&mut app, &wakes, 1), "did: three after did: two");
    let refused = output(&mut app, &wakes, 2);
    assert!(
        refused.contains("`shell` is not one of your tools"),
        "{refused}"
    );
    let args = std::fs::read_to_string(dir.join("args")).unwrap_or_default();
    let first = args.lines().next().unwrap_or_default();
    assert!(
        first.starts_with("-p --json --tools=read -- one"),
        "a one-shot JSON run with the agent's tools but `helper` and the late one: {first}"
    );
}

#[test]
fn side_by_side_helpers_run_four_at_a_time_and_failures_are_reported() {
    let mut tasks: Vec<Value> = (0..5)
        .map(|n| json!({ "task": format!("wait {n}") }))
        .collect();
    tasks.extend([json!({ "task": "fail" }), json!({ "task": "crash" })]);
    let nine: Vec<Value> = (0..9).map(|n| json!({ "task": format!("{n}") })).collect();
    let (mut app, wakes, dir) = helped(
        "side",
        &[
            json!({ "tasks": tasks }),
            json!({ "tasks": nine }),
            json!({ "task": "fail" }),
        ],
    );
    let side = output(&mut app, &wakes, 0);
    assert!(side.starts_with("5 of 7 helpers answered."), "{side}");
    assert!(
        side.contains("did: wait 4") && side.contains("it broke"),
        "{side}"
    );
    assert!(
        side.contains("exit code 3") && side.contains("boom"),
        "{side}"
    );
    let counts = std::fs::read_to_string(dir.join("counts")).unwrap_or_default();
    let most = counts
        .lines()
        .filter_map(|count| count.trim().parse::<usize>().ok())
        .max();
    assert!(most.is_some_and(|most| most <= AT_ONCE), "{counts}");
    let too_many = output(&mut app, &wakes, 1);
    assert!(
        too_many.contains("give at most 8 tasks, not 9"),
        "{too_many}"
    );
    assert!(
        output(&mut app, &wakes, 2).contains("it broke"),
        "a failed child fails the call"
    );
}

#[test]
fn stopping_the_turn_kills_the_helpers() {
    let (mut app, wakes, dir) = helped("stop", &[json!({ "task": "slow" })]);
    let running = dir.join("run");
    let pids = |dir: &Path| -> Vec<String> {
        let entries = std::fs::read_dir(dir).into_iter().flatten().flatten();
        entries
            .map(|entry| entry.file_name().to_string_lossy().into_owned())
            .collect()
    };
    run_until(&mut app, &wakes, |_| !pids(&running).is_empty());
    let world = app.world_mut();
    if let Some(agent) = first_agent(world) {
        world.trigger(StopTurn::new(agent));
    }
    let alive = |pid: &String| Path::new("/proc").join(pid).exists();
    let deadline = Instant::now() + Duration::from_secs(10);
    while pids(&running).iter().any(alive) && Instant::now() < deadline {
        app.update();
        std::thread::sleep(Duration::from_millis(50));
    }
    let left: Vec<String> = pids(&running).into_iter().filter(alive).collect();
    assert!(left.is_empty(), "the helper still runs: {left:?}");
}

/// A helper's directory is joined as the file tools join a path: `~` is
/// the home directory, which is outside the caller's.
#[test]
fn a_helper_in_tilde_is_in_the_home_directory() {
    // Some build sandboxes have no home at all.
    let Some(home) = std::env::home_dir().and_then(|home| home.canonicalize().ok()) else {
        return;
    };
    let lent = Lent {
        model: None,
        access: ToolAccess::All,
        lendable: Vec::new(),
        base: std::env::temp_dir().join(format!("rig-helpers-tilde-{}", std::process::id())),
    };
    let job = Job {
        task: "look around".into(),
        dir: Some("~".into()),
        tools: None,
    };
    assert_eq!(
        lent.child(&job).err(),
        Some(format!("{} is outside your directory", home.display()))
    );
}
