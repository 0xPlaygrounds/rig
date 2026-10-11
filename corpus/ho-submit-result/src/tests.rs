use std::path::PathBuf;
use std::sync::mpsc::Receiver;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::ReadTool;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::Invocation;
use rig_harness::host::exit::ExitPlugin;
use rig_harness_test_support::{
    Ended, ToolOutputs, app_on, calls, connect, first_agent, log_ended, log_tool_outputs, reply,
    run_until, scratch_home,
};

use super::*;

/// A project named `name` with a schema of a required `verdict` string and
/// `a.txt`.
fn project(name: &str) -> Option<PathBuf> {
    let dir = scratch_home(&format!("corpus-ho-submit-result-{name}"))
        .root()
        .to_path_buf();
    std::fs::create_dir_all(&dir).ok()?;
    let schema = serde_json::json!({
        "type": "object",
        "properties": { "verdict": { "type": "string" } },
        "required": ["verdict"],
    });
    std::fs::write(dir.join("out.json"), schema.to_string()).ok()?;
    std::fs::write(dir.join("a.txt"), "the a file").ok()?;
    Some(dir)
}

/// `rig -p --schema=out.json …` in `dir`, answered by `model`, run until
/// `done`.
fn run(dir: &Path, model: &MockCompletionModel, done: impl Fn(&World) -> bool) -> Option<App> {
    let (mut app, wakes): (App, Receiver<()>) = app_on(&MemoryStore::default());
    let args = Invocation {
        print: Some("classify these failing tests".to_owned()),
        flags: vec![(SCHEMA.to_owned(), Some("out.json".to_owned()))],
        ..Invocation::default()
    };
    app.insert_resource(Invoked {
        args,
        terminal: false,
    })
    .insert_resource(ProjectRoot(dir.to_path_buf()))
    .add_plugins((
        AgentPlugin,
        JournalPlugin,
        ExitPlugin,
        ReadTool,
        SubmitResultPlugin,
    ))
    .init_resource::<Ended>()
    .add_observer(log_ended)
    .init_resource::<ToolOutputs>()
    .add_observer(log_tool_outputs);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world)?;
    world.entity_mut(agent).insert(connect(model)?);
    world.trigger(Deliver::new(
        agent,
        "Classify these failing tests.",
        DeliveryMode::Steer,
    ));
    run_until(&mut app, &wakes, done);
    for _ in 0..10 {
        app.update();
    }
    Some(app)
}

#[test]
fn a_valid_result_ends_the_turn_after_the_other_calls() {
    let dir = project("valid");
    assert!(dir.is_some(), "a scratch project");
    let Some(dir) = dir else { return };
    let path = dir.join("a.txt").display().to_string();
    let both = vec![
        MockStreamEvent::tool_call("c2", TOOL, serde_json::json!({ "verdict": "flaky" })),
        MockStreamEvent::tool_call("c3", "read", serde_json::json!({ "path": path })),
        MockStreamEvent::final_response_with_default_usage(),
    ];
    let model = MockCompletionModel::from_stream_turns([
        calls("c1", TOOL, serde_json::json!({})),
        both,
        reply("never asked for"),
    ]);
    let app = run(&dir, &model, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    assert!(app.is_some(), "the app starts");
    let Some(app) = app else { return };

    assert_eq!(model.request_count(), 2, "no request after the result");
    let ended = &app.world().resource::<Ended>().0;
    assert!(
        matches!(ended.as_slice(), [TurnOutcome::Stopped(Some(why))] if why == SUBMITTED),
        "{ended:?}"
    );
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let output = |id: &str| outputs.get(id).cloned().unwrap_or_default();
    assert!(
        output("c1").contains("`verdict` is required"),
        "{}",
        output("c1")
    );
    assert!(output("c3").contains("the a file"), "{}", output("c3"));
    let printed = app.world().resource::<SubmitResult>().printed();
    assert_eq!(printed.as_deref(), Some(r#"{"verdict":"flaky"}"#));
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn two_text_answers_get_one_reminder_then_fail_the_run() {
    let dir = project("text");
    assert!(dir.is_some(), "a scratch project");
    let Some(dir) = dir else { return };
    let model = MockCompletionModel::from_stream_turns([
        reply("They look flaky."),
        reply("Still flaky."),
        reply("never asked for"),
    ]);
    let app = run(&dir, &model, |world| {
        world.contains_resource::<ExitReason>()
    });
    assert!(app.is_some(), "the app starts");
    let Some(app) = app else { return };

    assert_eq!(
        app.world().get_resource::<ExitReason>(),
        Some(&ExitReason::Failed(NO_RESULT_EXIT))
    );
    let requests = model.requests();
    assert_eq!(requests.len(), 2, "the reminder's request only");
    let last = requests
        .last()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .unwrap_or_default();
    assert_eq!(last.matches(REMINDER).count(), 1, "one reminder: {last}");
    std::fs::remove_dir_all(&dir).ok();
}
