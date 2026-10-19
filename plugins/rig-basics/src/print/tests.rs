use std::sync::{Arc, Mutex};

use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::Invocation;
use rig_harness_test_support::{
    MemoryStore, app_on, calls, connect, first_agent, reply, run_until,
};

use super::*;
use crate::approval::{ApprovalPlugin, NeedsApproval};

/// A print run of `prompt` with `--answer=answers` when given, after its
/// first frame.
fn print_run(answers: Option<&str>) -> App {
    let mut app = App::new();
    let flags = answers.map(|ids| (ANSWER.to_owned(), Some(ids.to_owned())));
    app.insert_resource(Invoked {
        args: Invocation {
            print: Some("hi".to_owned()),
            flags: flags.into_iter().collect(),
            ..Invocation::default()
        },
        terminal: false,
    })
    .add_plugins((TaskPoolPlugin::default(), AgentPlugin, PrintPlugin));
    app.finish();
    app.cleanup();
    app.update();
    app
}

/// How each of `asks` was answered after a frame: `Some(None)` when
/// refused, `None` when not at all.
fn answered(app: &mut App, asks: impl IntoIterator<Item = Ask>) -> Vec<Option<Option<Answer>>> {
    let asks: Vec<Entity> = asks
        .into_iter()
        .map(|ask| app.world_mut().spawn(ask).id())
        .collect();
    app.update();
    let world = app.world();
    let outcome = |ask: &Entity| world.get::<Outcome>(*ask).map(Answer::of);
    asks.iter().map(outcome).collect()
}

#[test]
fn questions_are_answered_from_answer_or_refused() {
    let approval = || {
        Ask::new("Run rm -rf build?")
            .choice("deny", "Deny")
            .choice("allow", "Allow")
    };
    let mut app = print_run(Some("allow,yes"));
    let answerers = app
        .world_mut()
        .query::<&Answerer>()
        .iter(app.world())
        .count();
    assert_eq!(answerers, 1, "plugins may ask");
    let several = Ask::yes_no("Which?").choice("allow", "Allow").multiple();
    let answers = answered(
        &mut app,
        [approval(), Ask::new("Name?").free_text(), several],
    );
    let allowed = Some(Answer::choice("allow"));
    let both = Some(Answer::choices(["yes", "allow"]));
    assert_eq!(answers, [Some(allowed), Some(None), Some(both)]);

    let mut app = print_run(None);
    let answerers = app
        .world_mut()
        .query::<&Answerer>()
        .iter(app.world())
        .count();
    assert_eq!(answerers, 0, "plugins apply their own policy");
    assert_eq!(answered(&mut app, [approval()]), [Some(None)]);
}

#[test]
fn sections_follow_the_answer_by_tag() {
    let sections = [
        PrintSection::new("todo", "todo 1/2\n[x] read\n[ ] fix\n"),
        PrintSection::new("cost", "$0.02"),
        PrintSection::new("empty", ""),
    ];
    let printed = output(Some("Fixed.\n"), &sections);
    assert_eq!(
        printed.as_deref(),
        Some("Fixed.\n\n$0.02\n\ntodo 1/2\n[x] read\n[ ] fix")
    );
    assert_eq!(output(None, &[]), None);
}

#[test]
fn a_prompt_takes_what_is_piped_in_and_never_waits_on_a_silent_stdin() {
    let wait = Duration::from_millis(50);
    assert_eq!(with_piped("Review", &b"diff"[..], wait), "Review\n\ndiff");
    assert_eq!(with_piped("", &b"diff"[..], wait), "diff");
    // A pipe whose writer stays open and silent, as an agent's shell may
    // leave stdin.
    let Ok((silent, writer)) = std::io::pipe() else {
        return;
    };
    assert_eq!(with_piped("Review", silent, wait), "Review");
    drop(writer);
}

/// The lines written, shared with the test.
#[derive(Clone, Default)]
struct Lines(Arc<Mutex<Vec<u8>>>);

impl Write for Lines {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if let Ok(mut lines) = self.0.lock() {
            lines.extend_from_slice(bytes);
        }
        Ok(bytes.len())
    }

    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Lines {
    /// Each line, parsed; `Null` for one that is not JSON.
    fn parsed(&self) -> Vec<Value> {
        let bytes = self.0.lock().map(|lines| lines.clone()).unwrap_or_default();
        String::from_utf8_lossy(&bytes)
            .lines()
            .map(|line| serde_json::from_str(line).unwrap_or(Value::Null))
            .collect()
    }

    /// The lines of type `kind`.
    fn of(&self, kind: &str) -> Vec<Value> {
        let lines = self.parsed().into_iter();
        lines.filter(|line| at(line, "type") == kind).collect()
    }
}

/// On a call of `wait`: the test answers it.
#[derive(Component)]
struct Waiting;

/// Marks each `wait` call for the test, which answers it; with
/// `approve`, every call needs the user's approval first.
fn wait_tool(app: &mut App, approve: bool) {
    app.add_open_tool(
        "wait",
        "Wait.",
        ToolOptions::default(),
        |called: On<ToolCalled<Value>>, mut commands: Commands| {
            commands.entity(called.call).insert(Waiting);
        },
    );
    if approve {
        let ask = |calls: Query<Entity, (ReadyToStart, ToolCalls)>, mut commands: Commands| {
            for call in &calls {
                commands
                    .entity(call)
                    .queue(NeedsApproval::add("test", "It waits."));
            }
        };
        app.add_plugins(ApprovalPlugin).add_systems(
            AgentPass,
            ask.in_set(Stage::Gate)
                .before(crate::approval::ApprovalSystems),
        );
    }
}

/// A long-lived `--json` run on `model`, its lines, and the sender of its
/// request lines.
fn json_run(
    model: &MockCompletionModel,
    approve: bool,
) -> (App, std::sync::mpsc::Receiver<()>, Lines, Sender<String>) {
    let store = MemoryStore::default();
    let (mut app, wakes) = app_on(&store);
    let lines = Lines::default();
    let (input, requests) = JsonIn::channel();
    app.insert_resource(Invoked {
        args: Invocation {
            flags: vec![(JSON.to_owned(), None)],
            ..Invocation::default()
        },
        terminal: false,
    })
    .insert_resource(JsonOut::new(lines.clone()))
    .insert_resource(requests)
    .add_plugins((
        AgentPlugin,
        SessionLogPlugin,
        rig_harness::host::exit::ExitPlugin,
        PrintPlugin,
    ));
    wait_tool(&mut app, approve);
    app.finish();
    app.update();
    let world = app.world_mut();
    if let (Some(agent), Some(connection)) = (first_agent(world), connect(model)) {
        world.entity_mut(agent).insert(connection);
    }
    (app, wakes, lines, input)
}

/// Sends `line` as a request.
fn send(input: &Sender<String>, line: Value) {
    assert!(input.try_send(line.to_string()).is_ok(), "the run reads");
}

#[test]
fn a_driver_sends_a_message_mid_turn_and_reads_the_answer() {
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", "wait", json!({})),
        reply("Both done."),
    ]);
    let (mut app, wakes, lines, input) = json_run(&model, false);
    send(
        &input,
        json!({ "v": 1, "type": "send", "id": 1, "text": "Fix the tests." }),
    );
    run_until(&mut app, &wakes, |world| {
        world
            .iter_entities()
            .any(|entity| entity.contains::<Waiting>())
    });
    // While the tool call runs: it reaches the turn's next model call.
    send(
        &input,
        json!({ "v": 1, "type": "send", "id": 2, "text": "Also the docs." }),
    );
    run_until(&mut app, &wakes, |_| lines.of("reply").len() == 2);
    let world = app.world_mut();
    let mut waiting = world.query_filtered::<&mut Decision, With<Waiting>>();
    for mut decision in waiting.iter_mut(world) {
        decision.decide(
            Decider::plugin("test"),
            Outcome::from(rig_core::tool::ToolResult::success("Waited.".into())),
        );
    }
    run_until(&mut app, &wakes, |_| !lines.of("turn_end").is_empty());

    let parsed = lines.parsed();
    assert!(
        parsed.iter().all(|line| at(line, "v") == VERSION),
        "{parsed:?}"
    );
    let kinds: Vec<&str> = parsed
        .iter()
        .filter_map(|line| at(line, "type").as_str())
        .collect();
    let mut rest = kinds.iter();
    for kind in [
        "ready",
        "turn_start",
        "tool_call",
        "tool_result",
        "text",
        "turn_end",
    ] {
        assert!(
            rest.any(|seen| *seen == kind),
            "{kind} in order in {kinds:?}"
        );
    }
    let end = lines.of("turn_end");
    assert_eq!(end.len(), 1, "one turn: {end:?}");
    assert_eq!(
        end.first().map(|end| at(end, "answer").clone()),
        Some("Both done.".into())
    );
    let second = model
        .requests()
        .get(1)
        .map(|request| format!("{:?}", request.chat_history));
    assert!(
        second.is_some_and(|history| history.contains("Also the docs.")),
        "the next call reads it"
    );
}

#[test]
fn an_approval_is_answered_over_the_pipe_and_eof_ends_the_run() {
    let model =
        MockCompletionModel::from_stream_turns([calls("c1", "wait", json!({})), reply("ok")]);
    let (mut app, wakes, lines, input) = json_run(&model, true);
    send(&input, json!({ "v": 1, "type": "send", "text": "Wait." }));
    run_until(&mut app, &wakes, |_| !lines.of("ask").is_empty());
    let ask = lines.of("ask").first().cloned().unwrap_or_default();
    let ids: Vec<&str> = at(&ask, "choices")
        .as_array()
        .into_iter()
        .flatten()
        .filter_map(|c| at(c, "id").as_str())
        .collect();
    assert_eq!(ids, ["deny", "allow"], "{ask}");
    send(&input, json!({ "v": 1, "type": "state", "id": "s" }));
    send(
        &input,
        json!({ "v": 1, "type": "answer", "id": "a", "ask": at(&ask, "ask"), "choices": ["allow"] }),
    );
    run_until(&mut app, &wakes, |world| {
        world
            .iter_entities()
            .any(|entity| entity.contains::<Waiting>())
    });
    let replies = lines.of("reply");
    let state = replies
        .iter()
        .find(|reply| at(reply, "id") == "s")
        .map(|reply| at(reply, "state").clone());
    assert_eq!(
        state.map(|state| state.pointer("/asks/0/ask").cloned().unwrap_or_default()),
        Some(at(&ask, "ask").clone())
    );
    assert!(
        replies
            .iter()
            .any(|reply| at(reply, "id") == "a" && at(reply, "ok") == true),
        "{replies:?}"
    );

    // The driver goes away mid-turn: the agent exits at once.
    drop(input);
    run_until(&mut app, &wakes, |world| {
        world.contains_resource::<ExitReason>()
    });
    assert_eq!(
        app.world().get_resource::<ExitReason>(),
        Some(&ExitReason::Quit)
    );
}

#[test]
fn a_line_of_another_version_is_refused_naming_the_side_to_update() {
    let model = MockCompletionModel::from_stream_turns([reply("ok")]);
    let (mut app, wakes, lines, input) = json_run(&model, false);
    send(&input, json!({ "v": 2, "type": "state", "id": 1 }));
    send(&input, json!({ "type": "state", "id": 2 }));
    send(&input, json!({ "v": 1, "type": "sned", "id": 3 }));
    run_until(&mut app, &wakes, |_| lines.of("reply").len() == 3);
    let errors: Vec<String> = lines
        .of("reply")
        .iter()
        .map(|reply| format!("{} {}", at(reply, "ok"), at(reply, "error")))
        .collect();
    let [newer, older, unknown] = errors.as_slice() else {
        return assert_eq!(errors.len(), 3, "{errors:?}");
    };
    assert!(
        newer.starts_with("false") && newer.contains("update rig"),
        "{newer}"
    );
    assert!(
        older.starts_with("false") && older.contains("update the driver"),
        "{older}"
    );
    assert!(unknown.contains("sned"), "{unknown}");
}

/// The field `key` of `line`, `Null` without one.
fn at<'a>(line: &'a Value, key: &str) -> &'a Value {
    line.get(key).unwrap_or(&Value::Null)
}
