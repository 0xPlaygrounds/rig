use std::collections::BTreeMap;
use std::path::{Path, PathBuf};

use rig_basics::subagents::{SubagentsPlugin, TASK};
use rig_basics::telemetry::effect_log::EffectLogPlugin;
use rig_core::catalog::{Catalog, Connector, ModelSpec, Pricing};
use rig_core::completion::{Cost, Usage};
use rig_core::providers::registry::ProviderId;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::{Invocation, SessionDir, SessionId};
use rig_harness::rig_ecs::{StopTurnsOnExit, WriteLogs};
use rig_harness_test_support::{
    Ended, app_in, first_agent, log_ended, run_until, scratch_home, scripted_models,
};
use serde_json::{Value, json};

use super::*;

/// Moves every record's times by this many milliseconds before it is
/// written: the kernel stamps effects with the wall clock, which a test
/// cannot set, so this puts a session on the day the test says.
#[derive(Resource)]
struct Shift(i64);

fn shift(shift: Res<Shift>, mut records: MessageMutator<EffectRecorded>) {
    let moved = |ms: u64| u64::try_from(i64::try_from(ms).unwrap_or(0) + shift.0).unwrap_or(0);
    for EffectRecorded(record) in records.read() {
        record.started_ms = record.started_ms.map(moved);
        record.finished_ms = record.finished_ms.map(moved);
    }
}

/// Usage of 1 000 tokens in (100 of them cached) and 200 out, with the
/// price the provider reported, if any.
fn used(cost: Option<f64>) -> Usage {
    Usage::new()
        .input_tokens(1000)
        .cached_input_tokens(100)
        .output_tokens(200)
        .total_tokens(1200)
        .cost(cost.map(Cost::from_total))
}

fn answer(text: &str, cost: Option<f64>) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::text(text),
        MockStreamEvent::final_response(used(cost)),
    ]
}

fn call(id: &str, tool: &str, args: Value, cost: Option<f64>) -> Vec<MockStreamEvent> {
    vec![
        MockStreamEvent::tool_call(id, tool, args),
        MockStreamEvent::final_response(used(cost)),
    ]
}

/// A `noop` tool that only says it ran.
fn noop(called: On<ToolCalled<Value>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(Decider::plugin("test"), ToolResult::success("ok".into()));
    }
}

/// A saved session of `home` started in `project` on `day` (at noon, UTC),
/// whose agent talks to the scripted `model` until `turns` turns ended.
/// The app stays open.
fn session(
    home: &Home,
    project: &Path,
    day: &str,
    model: (&str, &MockCompletionModel),
    turns: usize,
) -> (App, SessionDir) {
    let Ok((id, dir)) = home.new_session() else {
        return (App::new(), home.session(&SessionId::generate()));
    };
    std::fs::write(dir.directory(), project.to_string_lossy().as_bytes()).ok();
    let noon = chrono::NaiveDate::parse_from_str(day, "%Y-%m-%d")
        .ok()
        .and_then(|day| day.and_hms_opt(12, 0, 0))
        .map_or(0, |at| at.and_utc().timestamp_millis());
    let now = chrono::Utc::now().timestamp_millis();
    let (mut app, wakes) = app_in(home, &id);
    app.insert_resource(scripted_models([model]))
        .insert_resource(Shift(noon - now))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            EffectLogPlugin,
            SubagentsPlugin,
        ))
        .add_open_tool("noop", "Do nothing.", ToolOptions::default(), noop)
        .add_systems(Last, shift.after(StopTurnsOnExit).before(WriteLogs))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let world = app.world_mut();
    world.trigger(SetModel {
        entity: agent,
        model: model.0.to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    world.trigger(Deliver::new(agent, "Go.", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
    app.update();
    (app, dir)
}

/// What `rig usage` with `args` makes in `home`, run from `cwd`, as the
/// launcher runs it, with `test/m-a` now in the price table.
fn usage(home: &Home, cwd: &Path, args: &[&str]) -> UsageRun {
    let mut connector = Connector::new(Catalog::builtin().clone());
    if let Ok(vendor) = ProviderId::declare("test") {
        let priced = ModelSpec::new(vendor, "m-a").with_pricing(Pricing::new(1.0, 2.0));
        connector.catalog_mut().insert(priced);
    }
    let invocation = Invocation {
        command: Some(COMMAND.to_owned()),
        arguments: args.iter().map(|arg| (*arg).to_owned()).collect(),
        ..Invocation::default()
    };
    let mut app = App::new();
    app.insert_resource(RigHome(home.clone()))
        .insert_resource(ProjectRoot(cwd.to_owned()))
        .insert_resource(Models(connector))
        .insert_resource(Invoked {
            args: invocation,
            terminal: false,
        })
        .add_plugins(UsageReportPlugin);
    app.finish();
    app.update();
    app.world_mut()
        .remove_resource::<UsageRun>()
        .unwrap_or_default()
}

/// Every file under `dir`, with its content.
fn files(dir: &Path) -> BTreeMap<PathBuf, Vec<u8>> {
    let mut files = BTreeMap::new();
    let mut dirs = vec![dir.to_owned()];
    while let Some(dir) = dirs.pop() {
        for entry in std::fs::read_dir(&dir).into_iter().flatten().flatten() {
            let path = entry.path();
            if path.is_dir() {
                dirs.push(path);
            } else {
                let bytes = std::fs::read(&path).unwrap_or_default();
                files.insert(path, bytes);
            }
        }
    }
    files
}

fn row<'a>(report: &'a Value, key: &str) -> &'a Value {
    let rows = report
        .get("rows")
        .and_then(Value::as_array)
        .map_or(&[][..], Vec::as_slice);
    rows.iter()
        .find(|row| row.get("key").and_then(Value::as_str) == Some(key))
        .unwrap_or(&Value::Null)
}

#[test]
fn reports_saved_sessions_by_model_project_and_day() {
    let home = scratch_home("sn-usage-report-cli");
    let (p1, p2) = (home.root().join("p1"), home.root().join("p2"));
    std::fs::create_dir_all(&p1).ok();
    std::fs::create_dir_all(&p2).ok();

    // S1: P1, m-a, two requests with a tool call between, priced.
    let s1 = MockCompletionModel::from_stream_turns([
        call("c1", "noop", json!({}), Some(0.01)),
        answer("Done.", Some(0.01)),
    ]);
    let (s1, _) = session(&home, &p1, "2026-10-01", ("test/m-a", &s1), 1);
    drop(s1);
    // S2: P1, m-b, a `task` subagent: the parent asks three times, the
    // subagent once; whichever gets which text reply, the sums are these.
    let task = json!({ "description": "Look", "prompt": "Look around." });
    let s2 = MockCompletionModel::from_stream_turns([
        call("t1", TASK, task, Some(0.02)),
        answer("Started.", Some(0.02)),
        answer("Nothing here.", Some(0.02)),
        answer("Done.", Some(0.02)),
    ]);
    let (open, s2_dir) = session(&home, &p1, "2026-10-09", ("test/m-b", &s2), 3);
    // S3: P2, m-a, one request saved without a price.
    let s3 = MockCompletionModel::from_stream_turns([answer("Hi.", None)]);
    let (s3, s3_dir) = session(&home, &p2, "2026-10-10", ("test/m-a", &s3), 1);
    drop(s3);
    let before = files(s2_dir.path());

    let all = ["--all-projects", "--since", "2026-09-01"];
    let run = usage(
        &home,
        &p2,
        &[&all[..], &["--by", "model", "--format", "json"]].concat(),
    );
    assert_eq!(run.status, 0, "{run:?}");
    let report: Value = serde_json::from_str(&run.stdout).unwrap_or_default();
    let (a, b) = (row(&report, "test/m-a"), row(&report, "test/m-b"));
    assert_eq!(
        a.get("requests").cloned().unwrap_or_default(),
        3,
        "{report:#}"
    );
    assert_eq!(a.get("input_tokens").cloned().unwrap_or_default(), 3000);
    assert_eq!(
        a.get("cached_input_tokens").cloned().unwrap_or_default(),
        300
    );
    assert_eq!(a.get("output_tokens").cloned().unwrap_or_default(), 600);
    assert_eq!(a.get("tool_calls").cloned().unwrap_or_default(), 1);
    assert_eq!(
        a.get("cost_approximate").cloned().unwrap_or_default(),
        true,
        "S3 was priced now"
    );
    // 2 × $0.01 recorded, and S3 at $1 / $2 per million: 900 + 100 cached
    // (no cache price) in, 200 out.
    let cost = a
        .get("cost_usd")
        .cloned()
        .unwrap_or_default()
        .as_f64()
        .unwrap_or_default();
    assert!((cost - (0.02 + 0.0009 + 0.0004)).abs() < 1e-9, "{cost}");
    assert_eq!(
        b.get("requests").cloned().unwrap_or_default(),
        4,
        "the subagent's request counts: {report:#}"
    );
    assert_eq!(b.get("input_tokens").cloned().unwrap_or_default(), 4000);
    assert_eq!(b.get("output_tokens").cloned().unwrap_or_default(), 800);
    assert_eq!(b.get("tool_calls").cloned().unwrap_or_default(), 1);
    assert_eq!(
        b.get("cost_approximate").cloned().unwrap_or_default(),
        false
    );
    assert_eq!(
        report
            .pointer("/totals/requests")
            .cloned()
            .unwrap_or_default(),
        7
    );

    // P1 since 2026-10-05, as CSV: only S2's day, then the totals.
    let run = usage(
        &home,
        &p2,
        &[
            "--project",
            "../p1",
            "--since",
            "2026-10-05",
            "--format",
            "csv",
        ],
    );
    let lines: Vec<&str> = run.stdout.lines().collect();
    assert_eq!(lines.len(), 3, "{}", run.stdout);
    assert!(
        lines
            .first()
            .is_some_and(|header| header.starts_with("day,requests,input_tokens"))
    );
    assert!(
        lines
            .get(1)
            .is_some_and(|row| row.starts_with("2026-10-09,4,4000,400,800,0.080000,false,1,")),
        "{lines:?}"
    );
    assert!(
        lines
            .get(2)
            .is_some_and(|row| row.starts_with("total,4,4000,")),
        "{lines:?}"
    );

    // A damaged session is skipped, said on stderr only.
    let log = rig_cassette::effect_log::jsonl::SESSION_EFFECT_LOG;
    let store = StoreOpener::default().open(&s3_dir);
    let mut damaged = store
        .as_ref()
        .ok()
        .and_then(|store| store.read_file(log).ok())
        .unwrap_or_default();
    damaged.splice(0..0, b"{not json}\n".iter().copied());
    if let Ok(store) = &store {
        store.write_file(log, &damaged).ok();
    }
    let run = usage(&home, &p2, &[&all[..], &["--format", "json"]].concat());
    assert_eq!(run.status, 0, "{run:?}");
    let s3_id = s3_dir.id().map(|id| id.to_string()).unwrap_or_default();
    assert!(run.stderr.contains(&s3_id), "{}", run.stderr);
    let report: Value = serde_json::from_str(&run.stdout).unwrap_or_default();
    assert_eq!(
        report
            .pointer("/totals/requests")
            .cloned()
            .unwrap_or_default(),
        6,
        "{}",
        run.stdout
    );

    let run = usage(
        &home,
        &p2,
        &[&all[..], &["--fail-over", "0.000001"]].concat(),
    );
    assert_eq!(run.status, 3, "{run:?}");
    assert!(
        run.stdout.contains("total"),
        "the report is printed: {}",
        run.stdout
    );
    let run = usage(&home, &p2, &["--by", "nonsense"]);
    assert_eq!((run.status, run.stdout.is_empty()), (2, true), "{run:?}");
    assert!(
        run.stderr.contains("--fail-over"),
        "with the help: {}",
        run.stderr
    );

    // S2's app is open all along: its files are as it left them.
    assert_eq!(files(s2_dir.path()), before);
    drop(open);
}
