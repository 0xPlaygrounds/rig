use std::fs;
use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_core::test_utils::MockCompletionModel;
use rig_harness_test_support::{
    Ended, MemoryStore, ToolOutputs, app_on, calls, first_agent, log_ended, log_tool_outputs,
    reply, run_until, scripted_models,
};
use serde_json::json;

use super::*;

/// The fixture's "test command": logs its arguments, then prints canned
/// output in the format its `mode` file names.
const RUN_SH: &str = r#"echo "$*" >> args.log
case "$(cat mode)" in
cargo)
  echo "     Running unittests src/lib.rs (target/debug/deps/fixture-0123)"
  echo
  if [ "$#" -gt 0 ]; then
    echo "running 2 tests"
  else
    echo "running 10 tests"
    for t in add sub mul neg parse fmt cmp max; do echo "test tests::$t ... ok"; done
  fi
  cat <<'OUT'
test tests::div_by_zero ... FAILED
test tests::parse_neg ... FAILED

failures:

---- tests::div_by_zero stdout ----

thread 'tests::div_by_zero' panicked at src/lib.rs:12:9:
attempt to divide by zero
note: run with `RUST_BACKTRACE=1` environment variable to display a backtrace

---- tests::parse_neg stdout ----

thread 'tests::parse_neg' panicked at src/lib.rs:30:5:
assertion `left == right` failed
  left: Ok(5)
 right: Ok(-5)


failures:
    tests::div_by_zero
    tests::parse_neg

OUT
  if [ "$#" -gt 0 ]; then p=0; else p=8; fi
  echo "test result: FAILED. $p passed; 2 failed; 0 ignored; 0 measured; 0 filtered out; finished in 0.01s"
  exit 101 ;;
pytest)
  cat <<'OUT'
============================= test session starts ==============================
collected 10 items

tests/test_math.py::test_add PASSED                                      [ 10%]
tests/test_math.py::test_div FAILED                                      [ 20%]
tests/test_math.py::test_mul PASSED                                      [ 30%]
tests/test_math.py::test_sub PASSED                                      [ 40%]
tests/test_math.py::test_neg PASSED                                      [ 50%]
tests/test_text.py::test_lower PASSED                                    [ 60%]
tests/test_text.py::test_upper FAILED                                    [ 70%]
tests/test_text.py::test_strip PASSED                                    [ 80%]
tests/test_text.py::test_split PASSED                                    [ 90%]
tests/test_text.py::test_join PASSED                                     [100%]

=================================== FAILURES ===================================
___________________________________ test_div ___________________________________

    def test_div():
>       assert 1 / 1 == 2
E       assert 1.0 == 2

tests/test_math.py:7: AssertionError
__________________________________ test_upper __________________________________

    def test_upper():
>       assert "a".upper() == "B"
E       AssertionError: assert 'A' == 'B'

tests/test_text.py:12: AssertionError
=========================== short test summary info ============================
FAILED tests/test_math.py::test_div - assert 1.0 == 2
FAILED tests/test_text.py::test_upper - AssertionError: assert 'A' == 'B'
========================= 2 failed, 8 passed in 0.03s ==========================
OUT
  exit 1 ;;
garbage)
  echo "~~ the runner fell over ~~"
  echo "core dumped somewhere"
  exit 3 ;;
slow)
  echo $$ > pid
  echo "tests/test_math.py::test_add PASSED"
  echo "tests/test_math.py::test_mul PASSED"
  echo "tests/test_math.py::test_sub PASSED"
  echo "tests/test_math.py::test_div FAILED"
  sleep 30 ;;
esac
"#;

/// A fixture project named `name` beside the test binary, its output in
/// `mode`.
fn fixture(name: &str, mode: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(name);
    fs::remove_dir_all(&dir).ok();
    assert!(fs::create_dir_all(&dir).is_ok());
    assert!(fs::write(dir.join("run.sh"), RUN_SH).is_ok());
    set_mode(&dir, mode);
    dir
}

fn set_mode(dir: &Path, mode: &str) {
    assert!(fs::write(dir.join("mode"), mode).is_ok());
}

/// The arguments of each run of the fixture.
fn runs(dir: &Path) -> Vec<String> {
    let log = fs::read_to_string(dir.join("args.log")).unwrap_or_default();
    log.lines().map(str::to_owned).collect()
}

/// The plugin on `store` with `[tests]` running the fixture in `dir` read
/// as `kind`, its agent on `model`, after the first frame restored the
/// session.
fn start(
    store: &MemoryStore,
    dir: &Path,
    kind: &str,
    model: &MockCompletionModel,
) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(store);
    let config = format!("[tests]\ncommand = \"sh run.sh\"\nkind = \"{kind}\"");
    app.insert_resource(Settings::from_toml(&config))
        .insert_resource(ProjectRoot(dir.to_owned()))
        .insert_resource(scripted_models([("test/model", model)]))
        .add_plugins((AgentPlugin, SessionLogPlugin, TestRunnerPlugin))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/model".to_owned(),
        kind: ChoiceKind::ForThisRun,
    });
    (app, wakes, agent)
}

/// Sends `text` and runs until the turn ended, or `until` holds.
fn turn(app: &mut App, wakes: &Receiver<()>, agent: Entity, until: impl Fn(&World) -> bool) {
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut().trigger(Deliver::new(
        agent,
        "Make the tests pass.",
        DeliveryMode::Steer,
    ));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended || until(world)
    });
}

fn output(app: &App, id: &str) -> String {
    let outputs = &app.world().resource::<ToolOutputs>().0;
    outputs.get(id).cloned().unwrap_or_default()
}

/// A cargo-style run reports 8 passed and the two failures without naming
/// a passing test; `only_failed` reruns exactly the two, also after the
/// session is resumed in a new app.
#[test]
fn failures_are_reported_and_rerun_exactly_across_a_resume() {
    let dir = fixture("ct-test-runner-cargo", "cargo");
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", RUN_TESTS, json!({})),
        calls("c2", RUN_TESTS, json!({ "only_failed": true })),
        reply("Two still fail."),
        calls("c3", RUN_TESTS, json!({ "only_failed": true })),
        reply("Still two."),
    ]);
    let store = MemoryStore::default();
    let (mut app, wakes, agent) = start(&store, &dir, "cargo", &model);
    turn(&mut app, &wakes, agent, |_| false);

    let full = output(&app, "c1");
    assert!(full.contains("8 passed, 2 failed, 0 skipped"), "{full}");
    assert!(
        full.contains("FAILED tests::div_by_zero (src/lib.rs:12)"),
        "{full}"
    );
    assert!(full.contains("attempt to divide by zero"), "{full}");
    assert!(
        full.contains("FAILED tests::parse_neg (src/lib.rs:30)"),
        "{full}"
    );
    assert!(full.contains("assertion `left == right` failed"), "{full}");
    for passing in ["tests::add", "tests::fmt", "tests::max"] {
        assert!(!full.contains(passing), "{passing} listed: {full}");
    }
    let rerun = output(&app, "c2");
    assert!(rerun.contains("0 passed, 2 failed"), "{rerun}");
    let exactly = "-- --exact tests::div_by_zero tests::parse_neg";
    assert_eq!(runs(&dir), ["", exactly]);
    drop(app);

    let (mut resumed, wakes, agent) = start(&store, &dir, "cargo", &model);
    let failed = &resumed.world().resource::<LastRun>().failed;
    assert_eq!(failed, &["tests::div_by_zero", "tests::parse_neg"]);
    turn(&mut resumed, &wakes, agent, |_| false);
    assert!(output(&resumed, "c3").contains("0 passed, 2 failed"));
    assert_eq!(runs(&dir), ["", exactly, exactly]);
}

/// A pytest-style run is read too; output no parser reads falls back to
/// the exit code and its tail; a cancelled run answers with what finished.
#[test]
fn pytest_output_garbage_and_a_cancelled_run() {
    let dir = fixture("ct-test-runner-pytest", "pytest");
    let model = MockCompletionModel::from_stream_turns([
        calls("p1", RUN_TESTS, json!({})),
        reply("Two fail."),
        calls("p2", RUN_TESTS, json!({})),
        reply("It fell over."),
        calls("p3", RUN_TESTS, json!({ "timeout_secs": 60 })),
        reply("unused"),
    ]);
    let (mut app, wakes, agent) = start(&MemoryStore::default(), &dir, "pytest", &model);
    turn(&mut app, &wakes, agent, |_| false);
    let parsed = output(&app, "p1");
    assert!(parsed.contains("8 passed, 2 failed, 0 skipped"), "{parsed}");
    let div = "FAILED tests/test_math.py::test_div (tests/test_math.py:7)\n  assert 1.0 == 2";
    assert!(parsed.contains(div), "{parsed}");
    assert!(parsed.contains("tests/test_text.py::test_upper (tests/test_text.py:12)"));
    assert!(!parsed.contains("test_add"), "{parsed}");

    set_mode(&dir, "garbage");
    turn(&mut app, &wakes, agent, |_| false);
    let fallback = output(&app, "p2");
    assert!(fallback.contains("Fell back"), "{fallback}");
    assert!(fallback.contains("Exit code: 3"), "{fallback}");
    assert!(fallback.contains("core dumped somewhere"), "{fallback}");

    set_mode(&dir, "slow");
    let failed_line = |world: &World| {
        let progress = world.try_query::<&ToolProgress>();
        progress.is_some_and(|mut progress| {
            progress
                .iter(world)
                .any(|ToolProgress(line)| line.ends_with("test_div FAILED"))
        })
    };
    turn(&mut app, &wakes, agent, failed_line);
    app.world_mut().trigger(StopTurn::new(agent));
    let pid = fs::read_to_string(dir.join("pid")).unwrap_or_default();
    let running = PathBuf::from(format!("/proc/{}", pid.trim()));
    assert!(
        !pid.trim().is_empty() && running.exists(),
        "the slow run runs"
    );
    run_until(&mut app, &wakes, |world| {
        world.resource::<Ended>().0.len() == 3 && !running.exists()
    });
    assert!(!running.exists(), "the cancelled run's command was killed");
    let cancelled = output(&app, "p3");
    assert!(
        cancelled.contains("Cancelled before the run finished"),
        "{cancelled}"
    );
    assert!(
        cancelled.contains("3 passed, 1 failed, 0 skipped"),
        "{cancelled}"
    );
    assert!(
        cancelled.contains("FAILED tests/test_math.py::test_div"),
        "{cancelled}"
    );
    assert_eq!(
        model.request_count(),
        5,
        "the interrupted turn asked no more"
    );
}
