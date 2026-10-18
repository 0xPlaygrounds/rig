use std::path::PathBuf;

use rig_basics::coding_tools::ReadTool;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::settings::Settings;
use rig_harness::host::exit::ExitPlugin;
use rig_harness_test_support::MemoryStore;
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

/// `rig -p 'classify these failing tests'` in `dir`, with
/// `[submit_result] schema = "out.json"`, answered by `model`, until `done`
/// holds of its turns' outcomes.
fn run(
    dir: &Path,
    model: &MockCompletionModel,
    done: impl Fn(&[TurnOutcome]) -> bool,
) -> Option<App> {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Settings::from_toml("[submit_result]\nschema = \"out.json\""))
        .insert_resource(Unattended(true))
        .insert_resource(ProjectRoot(dir.to_path_buf()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
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
        "classify these failing tests",
        DeliveryMode::Queue,
    ));
    run_until(&mut app, &wakes, |world| done(&world.resource::<Ended>().0));
    Some(app)
}

/// Whether a turn failed.
fn failed(outcomes: &[TurnOutcome]) -> bool {
    outcomes
        .iter()
        .any(|outcome| matches!(outcome, TurnOutcome::Failed(_)))
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
    let app = run(&dir, &model, |outcomes| !outcomes.is_empty());
    assert!(app.is_some(), "the app starts");
    let Some(app) = app else { return };

    assert_eq!(model.request_count(), 2, "no request after the result");
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let output = |id: &str| outputs.get(id).cloned().unwrap_or_default();
    let invalid = output("c1");
    // rig-core refuses arguments that do not fit the tool's schema before
    // the plugin sees them.
    assert!(
        invalid.contains("do not fit its schema") && invalid.contains("verdict"),
        "{invalid}"
    );
    assert!(output("c3").contains("the a file"), "{}", output("c3"));
    let result = app.world().resource::<SubmitResult>();
    assert_eq!(result.0, Some(serde_json::json!({ "verdict": "flaky" })));
    let answers: Vec<Option<String>> = app
        .world()
        .resource::<Ended>()
        .0
        .iter()
        .map(|outcome| match outcome {
            TurnOutcome::Answered(message) => message.final_answer(),
            _ => None,
        })
        .collect();
    assert_eq!(
        answers,
        [Some(r#"{"verdict":"flaky"}"#.to_owned())],
        "the turn's answer, which `rig -p` prints"
    );
    assert!(!app.world().contains_resource::<ExitReason>());
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn two_text_answers_get_one_reminder_then_fail_the_turn() {
    let dir = project("text");
    assert!(dir.is_some(), "a scratch project");
    let Some(dir) = dir else { return };
    let model = MockCompletionModel::from_stream_turns([
        reply("They look flaky."),
        reply("Still flaky."),
        reply("never asked for"),
    ]);
    let app = run(&dir, &model, failed);
    assert!(app.is_some(), "the app starts");
    let Some(app) = app else { return };

    assert!(!app.world().contains_resource::<ExitReason>(), "the worker goes on");
    let requests = model.requests();
    assert_eq!(requests.len(), 2, "the reminder's request only");
    let last = requests
        .last()
        .map(|request| serde_json::to_string(request).unwrap_or_default())
        .unwrap_or_default();
    assert_eq!(last.matches(REMINDER).count(), 1, "one reminder: {last}");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn three_refused_results_fail_the_turn() {
    let dir = project("invalid");
    assert!(dir.is_some(), "a scratch project");
    let Some(dir) = dir else { return };
    let empty = || calls("c", TOOL, serde_json::json!({}));
    let model = MockCompletionModel::from_stream_turns([
        empty(),
        empty(),
        empty(),
        reply("never asked for"),
    ]);
    let app = run(&dir, &model, failed);
    assert!(app.is_some(), "the app starts");
    let Some(app) = app else { return };

    assert_eq!(model.request_count(), 3, "no request after the third");
    assert!(!app.world().contains_resource::<ExitReason>(), "the worker goes on");
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(SubmitResultPlugin);
}
