use std::path::{Path, PathBuf};
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{EditTool, WriteTool};
use rig_core::message::{Message, UserContent as Content};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::clock::ClockPlugin;
use rig_harness_test_support::{
    Collected, Ended, MemoryStore, app_on, calls, collect, connect, first_agent, log_ended, reply,
    run_until, scratch_home,
};

use super::*;

/// `src/calc.rs` before the agent's first edit.
const CALC: &str = "//! A tiny calculator.

/// Adds.
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

/// Divides, or says why not.
pub fn div(a: i32, b: i32) -> Result<i32, String> {
    let ok = b != 0;
    if ok {
        return Ok(a.wrapping_div(b));
    }
    match a {
        0 => Err(\"zero by zero\".to_owned()),
    }
    // the end
}
";

/// The stand-in coverage command: with `slow`, sleeps a minute; with
/// `fail`, a test fails; else an lcov report where lines 10-15 ran, 16-18
/// ran once `tests/calc_test.rs` exists, and 5 and 20 never did.
const COV: &str = r#"out="$1"
if [ -f slow ]; then sleep 60 & echo $! > pid; wait; fi
if [ -f fail ]; then echo "running 3 tests"; echo "test calc::div failed"; exit 1; fi
hit=0; [ -f tests/calc_test.rs ] && hit=1
{
  echo "SF:src/calc.rs"
  echo "DA:5,0"
  for n in 10 11 12 13 14 15; do echo "DA:$n,1"; done
  for n in 16 17 18; do echo "DA:$n,$hit"; done
  echo "DA:20,0"
  echo "end_of_record"
} > "$out"
"#;

/// The three arms the agent adds after line 15, as lines 16-18.
const ARMS: &str = "        1 => Err(\"one by zero\".to_owned()),\n        \
                    -1 => Err(\"minus one by zero\".to_owned()),\n        \
                    _ => Err(format!(\"{a} by zero\")),\n";

/// A scratch project `name` with `src/calc.rs`, the stand-in command and
/// `coverage.toml`.
fn project(name: &str) -> Option<PathBuf> {
    let dir = scratch_home(name).root().to_path_buf();
    std::fs::create_dir_all(dir.join("src")).ok()?;
    std::fs::write(dir.join("src/calc.rs"), CALC).ok()?;
    std::fs::write(dir.join("cov.sh"), COV).ok()?;
    let config = "command = \"sh cov.sh {out}\"\nformat = \"lcov\"\nsources = [\"src/**/*.rs\"]\n";
    std::fs::write(dir.join("coverage.toml"), config).ok()?;
    Some(dir)
}

/// A reply whose `edit` changes line 12 and adds lines 16-18.
fn handle_zero(dir: &Path) -> Vec<MockStreamEvent> {
    let path = dir.join("src/calc.rs").display().to_string();
    let zero = "        0 => Err(\"zero by zero\".to_owned()),\n";
    let edits = [
        serde_json::json!({ "old_text": "a.wrapping_div(b)", "new_text": "a / b" }),
        serde_json::json!({ "old_text": zero, "new_text": format!("{zero}{ARMS}") }),
    ];
    calls(
        "e1",
        "edit",
        serde_json::json!({ "path": path, "edits": edits }),
    )
}

/// An app on `dir` with the file tools and the plugin, its wakes, and its
/// agent on `model`.
fn app(dir: &Path, model: &MockCompletionModel) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(ProjectRoot(dir.to_path_buf()))
        .add_plugins((AgentPlugin, SessionLogPlugin, ClockPlugin))
        .add_plugins((EditTool, WriteTool, CoverageGapNudgePlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    (app, wakes, agent)
}

fn say(app: &mut App, agent: Entity, text: &str) {
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
}

/// Whether `count` turns ended and `runs` coverage runs ended.
fn after(agent: Entity, count: usize, runs: u64) -> impl Fn(&World) -> bool {
    move |world| {
        let state = world.get::<Coverage>(agent);
        world.resource::<Ended>().0.len() >= count && state.is_some_and(|it| it.runs >= runs)
    }
}

/// The text of each message to the model that is no tool result.
fn user_texts(app: &App, agent: Entity) -> Vec<String> {
    let Some(conversation) = app.world().get::<Conversation>(agent) else {
        return Vec::new();
    };
    let text = |content: &Vec<Content>| {
        let parts = content.iter().filter_map(|part| match part {
            Content::Text(text) => Some(text.text.clone()),
            _ => None,
        });
        parts.collect::<Vec<_>>().join("\n")
    };
    let messages = conversation.messages().iter();
    let texts = messages.filter_map(|message| match &**message {
        Message::User { content } => Some(text(content)),
        _ => None,
    });
    texts.filter(|text| !text.is_empty()).collect()
}

/// The follow-ups the plugin sent, in order.
fn follow_ups(app: &App, agent: Entity) -> Vec<String> {
    let texts = user_texts(app, agent).into_iter();
    texts
        .filter(|text| text.contains("Coverage after") || text.contains("tests failed"))
        .collect()
}

#[test]
fn uncovered_changed_lines_get_one_follow_up_and_the_users_lines_never_count() {
    let dir = project("corpus-oc-coverage-gap-nudge-main");
    assert!(dir.is_some(), "a scratch project");
    let Some(dir) = dir else { return };
    let test_file = dir.join("tests/calc_test.rs").display().to_string();
    let calc = dir.join("src/calc.rs").display().to_string();
    let wrap =
        [serde_json::json!({ "old_text": "    a + b\n", "new_text": "    a.wrapping_add(b)\n" })];
    let model = MockCompletionModel::from_stream_turns([
        // Turn 1: line 12 changed, 16-18 added.
        handle_zero(&dir),
        reply("Done."),
        // The follow-up: a test that runs 16-18.
        calls(
            "w1",
            "write",
            serde_json::json!({ "path": test_file, "content": "#[test] fn div() {}\n" }),
        ),
        reply("Added a test for the other arms."),
        // Turn 3, after the user changed line 20: line 5 changed.
        calls(
            "e2",
            "edit",
            serde_json::json!({ "path": calc, "edits": wrap }),
        ),
        reply("Done."),
        // Its follow-up, answered without a test.
        reply("Line 5 is too simple to need a test."),
    ]);
    let (mut app, wakes, agent) = app(&dir, &model);

    say(&mut app, agent, "Handle division by zero.");
    run_until(&mut app, &wakes, after(agent, 2, 1));
    let sent = follow_ups(&app, agent);
    let first = sent.first().cloned().unwrap_or_default();
    assert_eq!(sent.len(), 1, "{sent:?}");
    assert!(first.contains("src/calc.rs, lines 16-18"), "{first}");
    assert!(
        first.contains("minus one by zero") && first.contains("{a} by zero"),
        "{first}"
    );
    assert!(
        !first.contains("a / b") && !first.contains("line 12"),
        "{first}"
    );

    // The test runs 16-18: no further follow-up.
    run_until(&mut app, &wakes, after(agent, 2, 2));
    for _ in 0..10 {
        app.update();
    }
    let state = app.world().get::<Coverage>(agent);
    assert_eq!(state.and_then(|it| it.last.clone()), Some(Checked::Covered));
    assert_eq!(model.requests().len(), 4, "no follow-up once covered");

    // The user changes line 20, which no test runs either.
    let now = std::fs::read_to_string(dir.join("src/calc.rs")).unwrap_or_default();
    let changed = now.replace("// the end", "let user_line = 20;");
    std::fs::write(dir.join("src/calc.rs"), &changed).ok();
    assert!(
        changed
            .lines()
            .nth(19)
            .is_some_and(|line| line.contains("user_line"))
    );

    say(&mut app, agent, "Make add wrap.");
    run_until(&mut app, &wakes, after(agent, 4, 4));
    let sent = follow_ups(&app, agent);
    let second = sent.get(1).cloned().unwrap_or_default();
    assert_eq!(sent.len(), 2, "{sent:?}");
    assert!(
        second.contains("src/calc.rs, line 5:") && second.contains("a.wrapping_add(b)"),
        "{second}"
    );
    assert!(
        sent.iter()
            .all(|text| !text.contains("user_line") && !text.contains("line 20")),
        "{sent:?}"
    );

    // One round only: the rest is a summary.
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let summary = notices
        .iter()
        .any(|notice| notice.text == "1 changed line in calc.rs remains uncovered");
    assert!(summary, "{notices:?}");
    assert_eq!(model.requests().len(), 7);
}

#[test]
fn failing_tests_are_listed_instead_of_lines() {
    let dir = project("corpus-oc-coverage-gap-nudge-fail");
    assert!(dir.is_some(), "a scratch project");
    let Some(dir) = dir else { return };
    std::fs::write(dir.join("fail"), "").ok();
    let model = MockCompletionModel::from_stream_turns([
        handle_zero(&dir),
        reply("Done."),
        reply("I will look at calc::div."),
    ]);
    let (mut app, wakes, agent) = app(&dir, &model);

    say(&mut app, agent, "Handle division by zero.");
    run_until(&mut app, &wakes, after(agent, 2, 1));
    let sent = follow_ups(&app, agent);
    let first = sent.first().cloned().unwrap_or_default();
    assert!(first.contains("test calc::div failed"), "{sent:?}");
    assert!(!first.contains("lines 16-18"), "{first}");
}

#[test]
fn a_new_message_kills_the_running_command() {
    let dir = project("corpus-oc-coverage-gap-nudge-slow");
    assert!(dir.is_some(), "a scratch project");
    let Some(dir) = dir else { return };
    std::fs::write(dir.join("slow"), "").ok();
    let model =
        MockCompletionModel::from_stream_turns([handle_zero(&dir), reply("Done."), reply("Sure.")]);
    let (mut app, wakes, agent) = app(&dir, &model);

    say(&mut app, agent, "Handle division by zero.");
    let pid_file = dir.join("pid");
    run_until(&mut app, &wakes, |_| {
        std::fs::read_to_string(&pid_file).is_ok_and(|pid| pid.ends_with('\n'))
    });
    let pid = std::fs::read_to_string(&pid_file).unwrap_or_default();
    let stat = PathBuf::from(format!("/proc/{}/stat", pid.trim()));
    let alive = move || {
        std::fs::read_to_string(&stat).is_ok_and(|stat| {
            let state = stat
                .rsplit_once(") ")
                .and_then(|(_, rest)| rest.chars().next());
            state.is_some_and(|state| state != 'Z')
        })
    };
    assert!(!pid.trim().is_empty() && alive(), "the command runs");
    assert!(
        app.world()
            .get::<Coverage>(agent)
            .is_some_and(|it| it.running)
    );
    std::fs::remove_file(dir.join("slow")).ok();

    say(&mut app, agent, "Something else, please.");
    run_until(&mut app, &wakes, |_| !alive());
    assert!(!alive(), "a new turn kills the coverage command");
}

#[test]
fn tracked_lines_move_through_changes() {
    let before: String = (1..=14).map(|n| format!("{n}\n")).collect();
    let after = before
        .replacen("2\n", "TWO\n", 1)
        .replacen("10\n", "new\n10\n", 1);
    let change = file_change("a.rs", Some(&before), Some(&after));
    assert_eq!(change.hunks.len(), 2, "{}", change.unified());
    let earlier = Changed {
        text: Some(before),
        lines: BTreeSet::from([1, 2, 3, 9, 10, 14]),
    };
    // 2 was rewritten; 10 and 14 move down one past the added line.
    assert_eq!(
        earlier.now("a.rs", Some(&after)),
        BTreeSet::from([1, 3, 9, 11, 15])
    );
    assert_eq!(change.changed_lines().collect::<Vec<_>>(), [2, 10]);
    assert!(runnable("*ptr = x;") && runnable("--n;"));
    assert!(!runnable(" * a doc line") && !runnable("-- a SQL comment") && !runnable("};"));
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(CoverageGapNudgePlugin);
}
