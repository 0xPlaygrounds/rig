use std::path::PathBuf;

use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    MemoryStore, ToolOutputs, app_on, calls, connect, first_agent, log_tool_outputs, reply,
    run_until, scratch_home,
};
use serde_json::json;

use super::*;
use crate::budget::Limit;
use crate::telemetry::usage::{AgentUsage, Spent, UnknownSpend};

/// A stand-in for `rig -p --json`: it notes its arguments, how many run
/// at once and the budget it was left in the directory it is given, and
/// answers its task (the last argument) by what the task says. A `spend`
/// task reports a call of 15 tokens, waits for `go` in the directory, then
/// reports another, and its `exit` line's usage, which nothing records,
/// says something else; an `incomplete` one says its spend is not all
/// known, as one whose own helper's was not does.
const CHILD: &str = r#"d=$1; shift
for a; do task=$a; done
mkdir -p "$d/run"; echo "$@" >> "$d/args"
echo "${RIG_BUDGET_USD-none} ${RIG_BUDGET_TOKENS-none}" >> "$d/budget"
touch "$d/run/$$"; ls "$d/run" | wc -l >> "$d/counts"
usage='{"v":3,"type":"usage","model":"test/model","purpose":"turn","usage":{"input_tokens":10,"output_tokens":5}}'
case "$task" in
  fail*) echo '{"v":3,"type":"exit","code":1,"error":"it broke","usage_complete":true}'; rm "$d/run/$$"; exit 1;;
  crash*) echo boom >&2; rm "$d/run/$$"; exit 3;;
  slow*) sleep 30;;
  wait*) sleep 0.3;;
  old*) echo '{"v":2,"type":"exit","code":0,"answer":"old"}'; rm "$d/run/$$"; exit 0;;
  partial*) echo "$usage"; rm "$d/run/$$"; exit 3;;
  spend*) echo "$usage"; while [ ! -e "$d/go" ]; do sleep 0.05; done; echo "$usage";
    echo '{"v":3,"type":"exit","code":0,"answer":"spent","usage":{"calls":99},"usage_complete":true}'; rm "$d/run/$$"; exit 0;;
  garbled*) echo '{"v":3,"type":"usage","usage":{"input_tokens":"many"}}'; echo '{"v":3,"type":"exit","code":0,"answer":"odd","usage_complete":true}'; rm "$d/run/$$"; exit 0;;
  incomplete*) echo "$usage"; echo '{"v":3,"type":"exit","code":0,"answer":"some","usage_complete":false}'; rm "$d/run/$$"; exit 0;;
esac
rm "$d/run/$$"
printf '{"v":3,"type":"exit","code":0,"answer":"did: %s","usage_complete":true}\n' "$task"
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
    assert_eq!(
        output(&mut app, &wakes, 0),
        "did: one\n[helper no model call]"
    );
    assert_eq!(
        output(&mut app, &wakes, 1),
        "did: three after did: two\n[helper no model call]\n[helper no model call]"
    );
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
fn a_helper_gets_only_the_tools_of_this_run() {
    let (mut app, wakes, dir) = helped(
        "run-tools",
        &[
            json!({ "task": "one", "tools": ["read"] }),
            json!({ "task": "two" }),
        ],
    );
    // The caller keeps `read`; this run leaves it only `helper`.
    app.insert_resource(RunTools::new([HELPER]));
    assert!(output(&mut app, &wakes, 0).starts_with("did: one"));
    assert!(output(&mut app, &wakes, 1).starts_with("did: two"));
    let args = std::fs::read_to_string(dir.join("args")).unwrap_or_default();
    let mut lines = args.lines();
    for task in ["one", "two"] {
        let line = lines.next().unwrap_or_default();
        assert!(
            line.starts_with(&format!("-p --json --tools= -- {task}")),
            "{args}"
        );
    }
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

/// What the agent's calls for its helpers used: its calls of purpose
/// `helper`, and their tokens.
fn helpers_spent(world: &World) -> (u64, u64) {
    let agent = world
        .iter_entities()
        .find(|entity| entity.contains::<Agent>())
        .and_then(|entity| entity.get::<AgentUsage>());
    let helper = agent.and_then(|spent| spent.purposes.get(HELPER));
    helper.map_or((0, 0), |spent| (spent.calls, spent.total_tokens()))
}

#[test]
fn a_helpers_spend_is_the_callers_as_it_reports_it_and_counts_once() {
    let (mut app, wakes, dir) = helped("spend", &[json!({ "task": "spend" })]);
    run_until(&mut app, &wakes, |world| helpers_spent(world).0 == 1);
    let returned = app.world().resource::<ToolOutputs>().0.contains_key("h0");
    assert!(!returned, "recorded while the helper still works");
    std::fs::write(dir.join("go"), "").ok();
    let said = output(&mut app, &wakes, 0);
    assert_eq!(said, "spent\n[helper 30 tokens]");
    assert_eq!(helpers_spent(app.world()), (2, 30), "not its exit line's");
    let session = &app.world().resource::<SessionUsage>().0;
    let helper = session.purposes.get(HELPER).map(|spent| spent.calls);
    assert_eq!(helper, Some(2));
    assert_eq!(
        session.models.get("test/model").map(|spent| spent.calls),
        Some(2)
    );
}

#[test]
fn a_helper_off_this_version_or_cut_short_has_its_spend_unknown() {
    let (mut app, wakes, _) = helped(
        "unknown",
        &[
            json!({ "task": "old" }),
            json!({ "task": "crash" }),
            json!({ "task": "partial" }),
            json!({ "task": "incomplete" }),
            json!({ "task": "garbled" }),
        ],
    );
    let old = output(&mut app, &wakes, 0);
    assert!(old.ends_with("[helper spend unknown]"), "{old}");
    let crashed = output(&mut app, &wakes, 1);
    assert!(crashed.ends_with("[helper spend unknown]"), "{crashed}");
    let partial = output(&mut app, &wakes, 2);
    assert!(
        partial.ends_with("[helper at least 15 tokens, the rest unknown]"),
        "{partial}"
    );
    let incomplete = output(&mut app, &wakes, 3);
    assert!(
        incomplete.ends_with("[helper at least 15 tokens, the rest unknown]"),
        "{incomplete}"
    );
    assert_eq!(helpers_spent(app.world()), (2, 30), "what they did report");
    let garbled = output(&mut app, &wakes, 4);
    assert!(
        garbled.ends_with("[helper spend unknown]"),
        "a usage line that does not read: {garbled}"
    );
    let unknown = app.world().resource::<UnknownSpend>().0;
    assert_eq!(unknown, 5, "each counts as not all known");
}

/// A run that exits while a helper works cannot hear the rest of its
/// spend: that helper counts as not all known, before the session is
/// saved and the `exit` line written.
#[test]
fn a_helper_still_working_at_the_exit_has_its_spend_unknown() {
    let (mut app, wakes, dir) = helped("exit", &[json!({ "task": "slow" })]);
    let running = dir.join("run");
    run_until(&mut app, &wakes, |_| {
        std::fs::read_dir(&running).is_ok_and(|mut entries| entries.next().is_some())
    });
    app.world_mut().write_message(AppExit::Success);
    app.update();
    assert_eq!(app.world().resource::<UnknownSpend>().0, 1);
}

/// A child whose program is not found never ran: it is settled with
/// nothing spent. (An error after its process was made leaves its spend
/// unknown.)
#[test]
fn a_helper_whose_program_is_not_found_spent_nothing() {
    let (mut app, wakes, _) = helped("no-start", &[json!({ "task": "one" })]);
    app.insert_resource(HelperCommand {
        program: "/nonexistent/rig".into(),
        args: Vec::new(),
    });
    let said = output(&mut app, &wakes, 0);
    assert!(said.starts_with("it could not start"), "{said}");
    assert!(!said.contains("spend unknown"), "{said}");
    app.world_mut().write_message(AppExit::Success);
    app.update();
    assert_eq!(app.world().resource::<UnknownSpend>().0, 0);
}

/// What the exit-order test queues, as a child's reports that come on the
/// exit frame, after the frame's other drains.
#[derive(Resource)]
struct Late(Vec<Heard>);

/// Queues the [`Late`] reports.
fn queue_late(mut late: ResMut<Late>, reported: Res<Reported>) {
    for heard in late.0.drain(..) {
        reported.sender.try_send(heard).ok();
    }
}

/// At the exit, a child's end is known only once its report of it is
/// drained, after the spend it reported before: a child whose call and
/// end come on the exit frame counts fully, and one whose end is not in
/// counts as unknown, its call recorded all the same.
#[test]
fn the_exit_counts_a_child_whose_end_is_not_in_as_unknown() {
    for (ended, unknown) in [(true, 0), (false, 1)] {
        let (mut app, _wakes, _) = helped("exit-order", &[]);
        let spent = Spent {
            agent: None,
            turn: None,
            model: None,
            purpose: Some(HELPER.to_owned()),
            usage: Usage::new().input_tokens(10).output_tokens(5),
        };
        let mut late = vec![Heard::Spent(spent)];
        if ended {
            late.push(Heard::Known);
        }
        let unsettled = app.world().resource::<Reported>().unsettled.clone();
        unsettled.fetch_add(1, Ordering::SeqCst);
        app.insert_resource(Late(late)).add_systems(
            Last,
            queue_late
                .after(rig_ecs::StopTurnsOnExit)
                .before(record_at_exit)
                .run_if(on_message::<AppExit>),
        );
        app.world_mut().write_message(AppExit::Success);
        app.update();
        let world = app.world();
        assert_eq!(
            world.resource::<UnknownSpend>().0,
            unknown,
            "ended: {ended}"
        );
        let session = world.resource::<SessionUsage>().0.total;
        assert_eq!(session.total_tokens(), 15, "the call it reported is in");
    }
}

/// A stopped call starts no more children: the four running are stopped,
/// their spend unknown, and the fifth, refused its start, spent nothing.
#[test]
fn a_helper_a_stopped_call_never_started_spent_nothing() {
    let tasks: Vec<Value> = (0..5)
        .map(|n| json!({ "task": format!("slow {n}") }))
        .collect();
    let (mut app, wakes, dir) = helped("stopped-start", &[json!({ "tasks": tasks })]);
    let running = dir.join("run");
    run_until(&mut app, &wakes, |_| {
        std::fs::read_dir(&running).is_ok_and(|entries| entries.count() == AT_ONCE)
    });
    if let Some(agent) = first_agent(app.world_mut()) {
        app.world_mut().trigger(StopTurn::new(agent));
    }
    let output = output(&mut app, &wakes, 0);
    assert!(
        output.contains("stopped") || output.contains("Stopped"),
        "{output}"
    );
    // The call's thread settles each child as it ends.
    let deadline = Instant::now() + Duration::from_secs(10);
    let settled = |app: &App| {
        app.world()
            .resource::<Reported>()
            .unsettled
            .load(Ordering::SeqCst)
            == 0
    };
    while !settled(&app) && Instant::now() < deadline {
        app.update();
        std::thread::sleep(Duration::from_millis(50));
    }
    // The fifth's start comes a moment after the fourth settles: let it.
    for _ in 0..5 {
        std::thread::sleep(Duration::from_millis(100));
        app.update();
    }
    assert_eq!(
        app.world().resource::<UnknownSpend>().0,
        AT_ONCE as u64,
        "the fifth spent nothing"
    );
}

/// A task too long to start a child with never ran: nothing spent.
#[cfg(target_os = "linux")]
#[test]
fn a_helper_whose_task_is_too_long_to_start_spent_nothing() {
    let long = "x".repeat(200_000);
    let (mut app, wakes, _) = helped("too-long", &[json!({ "task": long })]);
    let said = output(&mut app, &wakes, 0);
    assert!(said.starts_with("it could not start"), "{said}");
    assert!(!said.contains("spend unknown"), "{said}");
    app.world_mut().write_message(AppExit::Success);
    app.update();
    assert_eq!(app.world().resource::<UnknownSpend>().0, 0);
}

#[test]
fn a_helper_is_left_what_is_left_of_the_budget() {
    let (mut app, wakes, dir) = helped("budget", &[json!({ "task": "one" })]);
    let limit = Limit {
        usd: Some(0.5),
        tokens: Some(1000),
    };
    app.insert_resource(Budget(limit));
    // The session spent 100 tokens with no price before the call.
    app.world_mut().trigger(Spent {
        agent: None,
        turn: None,
        model: None,
        purpose: None,
        usage: Usage::new().input_tokens(60).output_tokens(40),
    });
    output(&mut app, &wakes, 0);
    let left = std::fs::read_to_string(dir.join("budget")).unwrap_or_default();
    assert_eq!(left.trim(), "0.5 900");
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
        access: ToolAccess::all(),
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
