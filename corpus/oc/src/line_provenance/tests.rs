use std::path::Path;
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::{EditTool, ShellTool, WriteTool};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::Invocation;
use rig_harness_test_support::{
    Ended, app_in, calls, connect, first_agent, log_ended, press, reply, run_until, scratch_home,
    tui_view,
};
use rig_tui::TuiTestScreen;
use serde_json::{Value, json};

use super::*;

/// `net.rs` before the first session.
const NET: &str = "net 1\nnet 2\nnet 3\nnet 4\nnet 5\nnet 6\nnet 7\nnet 8\nnet 9\nend\n";

/// An `edit` of `path` replacing `old` with `new`.
fn edit(id: &str, path: &Path, old: &str, new: &str) -> Vec<MockStreamEvent> {
    let edits = json!([{ "old_text": old, "new_text": new }]);
    calls(
        id,
        "edit",
        json!({ "path": path.display().to_string(), "edits": edits }),
    )
}

/// A session of `home` on the project `dir`, its agent on `model`, with
/// the terminal view on a test screen.
fn session(home: &Home, dir: &Path, model: &MockCompletionModel) -> (App, Receiver<()>, Entity) {
    let (mut app, wakes) = app_in(home, &SessionId::generate());
    app.insert_resource(ProjectRoot(dir.to_owned()))
        .insert_resource(TuiTestScreen::new(100, 30))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            EditTool,
            WriteTool,
            ShellTool,
            LineProvenancePlugin,
        ))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    tui_view(&mut app);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    (app, wakes, agent)
}

/// Sends `text` from the user and runs until its turn ended.
fn prompt(app: &mut App, wakes: &Receiver<()>, agent: Entity, text: &str) {
    let ended = app.world().resource::<Ended>().0.len();
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() > ended
    });
    for _ in 0..5 {
        app.update();
    }
}

/// The session id of an app.
fn id_of(app: &App) -> String {
    let paths = app.world().get_resource::<SessionPaths>();
    paths
        .and_then(|paths| paths.id())
        .map(|id| id.to_string())
        .unwrap_or_default()
}

/// What `rig provenance <args>` prints as JSON, run from `dir` with the
/// project data a session had.
fn cli(home: &Home, data: &PluginData, dir: &Path, args: &[&str]) -> Value {
    let invocation = Invocation {
        command: Some(NAME.to_owned()),
        arguments: args.iter().map(|arg| (*arg).to_owned()).collect(),
        ..Invocation::default()
    };
    let mut app = App::new();
    app.insert_resource(ProjectRoot(dir.to_owned()))
        .insert_resource(data.clone())
        .insert_resource(RigHome(home.clone()))
        .insert_resource(Invoked {
            args: invocation,
            terminal: false,
        })
        .add_plugins(LineProvenancePlugin);
    app.finish();
    app.update();
    let run = app.world_mut().remove_resource::<ProvenanceRun>();
    let stdout = run.map(|run| run.stdout).unwrap_or_default();
    serde_json::from_str(&stdout).unwrap_or(Value::Null)
}

/// Each span of an answer as `(from, to, session, prompt, tool, changed by
/// you)`.
fn spans(answer: &Value) -> Vec<(u64, u64, String, String, String, bool)> {
    let spans = answer["spans"].as_array().cloned().unwrap_or_default();
    let text = |value: &Value| value.as_str().unwrap_or_default().to_owned();
    spans
        .iter()
        .map(|span| {
            let record = &span["record"];
            (
                span["from"].as_u64().unwrap_or_default(),
                span["to"].as_u64().unwrap_or_default(),
                text(&record["session"]),
                text(&record["prompt"]),
                text(&record["tool"]),
                span["changed_by_you"].as_bool().unwrap_or_default(),
            )
        })
        .collect()
}

#[test]
fn lines_are_attributed_to_the_prompt_that_wrote_them_after_moving() {
    let home = scratch_home("corpus-oc-line-provenance");
    let dir = home.root().join("project");
    assert!(std::fs::create_dir_all(&dir).is_ok(), "a scratch project");
    let net = dir.join("net.rs");
    assert!(std::fs::write(&net, NET).is_ok(), "net.rs is written");
    assert!(std::fs::write(dir.join("other.rs"), "untouched\n").is_ok());
    let git = std::process::Command::new("git")
        .args(["init", "-q"])
        .current_dir(&dir)
        .status();
    assert!(git.is_ok_and(|status| status.success()), "git init");

    // Session 1: "add retry" inserts lines 10-12.
    let model = MockCompletionModel::from_stream_turns([
        edit(
            "s1-1",
            &net,
            "net 9\n",
            "net 9\nretry 1\nretry 2\nretry 3\n",
        ),
        reply("Added."),
    ]);
    let (mut app, wakes, agent) = session(&home, &dir, &model);
    prompt(&mut app, &wakes, agent, "add retry");
    let first = id_of(&app);
    drop(app);
    let named = first
        .parse()
        .map(|id| home.session(&id).set_name(Some("retry")));
    assert!(named.is_ok_and(|named| named.is_ok()), "session 1 is named");

    // Between sessions the user changes line 11.
    let text = std::fs::read_to_string(&net).unwrap_or_default();
    assert!(std::fs::write(&net, text.replace("retry 2\n", "retry 2 // mine\n")).is_ok());

    // Session 2: "add logging" writes log.rs and inserts 3 lines at the
    // top of net.rs; then "rename" runs `sed -i` on it.
    let log = dir.join("log.rs").display().to_string();
    let model = MockCompletionModel::from_stream_turns([
        calls(
            "s2-1",
            "write",
            json!({ "path": log, "content": "log 1\nlog 2\n" }),
        ),
        edit("s2-2", &net, "net 1\n", "log a\nlog b\nlog c\nnet 1\n"),
        reply("Logged."),
        calls(
            "s2-3",
            "shell",
            json!({ "command": "sed -i 's/^net 5$/NET 5/' net.rs" }),
        ),
        reply("Renamed."),
    ]);
    let (mut app, wakes, agent) = session(&home, &dir, &model);
    prompt(&mut app, &wakes, agent, "add logging");
    let second = id_of(&app);
    let data = app.world().resource::<PluginData>().clone();

    let moved = cli(&home, &data, &dir, &["net.rs:13-15", "--json"]);
    let by = |from, to, session: &str, prompt: &str, tool: &str, mine| {
        (
            from,
            to,
            session.to_owned(),
            prompt.to_owned(),
            tool.to_owned(),
            mine,
        )
    };
    assert_eq!(
        spans(&moved),
        vec![
            by(13, 13, &first, "add retry", "edit", false),
            by(14, 14, &first, "add retry", "edit", true),
            by(15, 15, &first, "add retry", "edit", false),
        ],
        "the moved lines 10-12, the middle one changed by the user"
    );
    assert_eq!(
        moved.pointer("/spans/0/session_name"),
        Some(&json!("retry"))
    );
    let top = spans(&cli(&home, &data, &dir, &["net.rs:1", "--json"]));
    assert_eq!(top, vec![by(1, 1, &second, "add logging", "edit", false)]);
    let created = spans(&cli(&home, &data, &dir, &["log.rs:1", "--json"]));
    assert_eq!(
        created,
        vec![by(1, 1, &second, "add logging", "write", false)]
    );
    let untouched = cli(&home, &data, &dir, &["other.rs", "--json"]);
    assert_eq!(
        untouched.pointer("/spans/0/by_agent"),
        Some(&json!(false)),
        "{untouched}"
    );

    prompt(&mut app, &wakes, agent, "rename");
    let renamed = cli(&home, &data, &dir, &["net.rs:8", "--json"]);
    let call = renamed.pointer("/spans/0/record/call_id");
    assert_eq!(call, Some(&json!("s2-3")), "{renamed}");
    assert_eq!(
        spans(&renamed),
        vec![by(8, 8, &second, "rename", "shell", false)]
    );

    // `/provenance` offers the turn that wrote a line; choosing it opens
    // session 1, read from its store, in a read-only panel at that turn.
    app.world_mut()
        .trigger(RunCommand::new(agent, format!("{NAME} net.rs:13")));
    app.update();
    let mut pickers = app.world_mut().query::<&Picker>();
    let command = pickers
        .iter(app.world())
        .flat_map(|picker| picker.items.iter())
        .map(|item| item.command.clone())
        .next()
        .unwrap_or_default();
    assert!(command.starts_with("provenance --open "), "{command}");
    press(&mut app, "esc");
    app.world_mut().trigger(RunCommand::new(agent, command));
    app.update();
    app.update();
    let screen = app.world().resource::<TuiTestScreen>().rows().join("\n");
    assert!(
        screen.contains("Session `retry`, turn 1: add retry")
            && screen.contains("add retry")
            && screen.contains("Added."),
        "{screen}"
    );
    assert!(!screen.contains("add logging"), "{screen}");
    press(&mut app, "esc");
    let mut panels = app
        .world_mut()
        .query_filtered::<(), (With<rig_tui::TranscriptPane>, Without<rig_tui::MainPane>)>();
    assert_eq!(panels.iter(app.world()).count(), 0, "Esc closes the panel");

    app.world_mut()
        .trigger(RunCommand::new(agent, format!("{NAME} --forget retry")));
    for _ in 0..5 {
        app.update();
    }
    let forgotten = cli(&home, &data, &dir, &["net.rs:13-15", "--json"]);
    let agents: Vec<_> = forgotten
        .pointer("/spans")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .map(|span| span["by_agent"].clone())
        .collect();
    assert_eq!(agents, vec![json!(false)], "{forgotten}");
    assert_eq!(
        spans(&cli(&home, &data, &dir, &["net.rs:1", "--json"])).len(),
        1,
        "session 2 stays"
    );
    drop(app);
    std::fs::remove_dir_all(home.root()).ok();
}
