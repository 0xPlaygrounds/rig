use std::path::Path;
use std::sync::mpsc::Receiver;

use rig_basics::coding_tools::ShellTool;
use rig_basics::subagents::SubagentsPlugin;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::commands::CommandPlugin;
use rig_harness::harness_protocol::SessionId;
use rig_harness::pick::PickPlugin;
use rig_harness_test_support::{
    Collected, Ended, ToolOutputs, app_in, calls, collect, first_agent, log_ended,
    log_tool_outputs, reply, run_until, scratch_home, scripted_models, tui_view,
};
use rig_tui::TuiTestScreen;
use serde_json::json;

use super::*;

/// The project's policy of the spec's test.
const POLICY: &str = "allow = [\"KEEP_ME\"]\ndeny = [\"*_SECRET\"]\nset = { CI = \"1\" }\n";

/// A `shell` call `id` running `command`.
fn shell(id: &str, command: &str) -> Vec<rig_core::test_utils::MockStreamEvent> {
    calls(id, "shell", json!({ "command": command }))
}

/// The output of the tool call `id`, as the model read it.
fn output(app: &App, id: &str) -> String {
    let outputs = &app.world().resource::<ToolOutputs>().0;
    outputs.get(id).cloned().unwrap_or_default()
}

/// Whether `output` has the variable line `line` (`NAME=value`) or any
/// line of `NAME=` when `line` ends with `=`.
fn has_line(output: &str, line: &str) -> bool {
    output.lines().any(|it| it.starts_with(line))
}

/// What the test screen shows.
fn screen(app: &App) -> Vec<String> {
    app.world().resource::<TuiTestScreen>().rows()
}

/// The warnings written about `file`.
fn warnings(app: &App, file: &str) -> usize {
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let warns = notices
        .iter()
        .filter(|notice| notice.level == NoticeLevel::Warn);
    warns.filter(|notice| notice.text.contains(file)).count()
}

/// Sends `text` to `agent` and runs until `main` was asked `requests`
/// times in all and no turn runs.
fn send(
    app: &mut App,
    wakes: &Receiver<()>,
    (agent, text): (Entity, &str),
    main: &MockCompletionModel,
    requests: usize,
) {
    app.world_mut()
        .trigger(Deliver::new(agent, text, DeliveryMode::Steer));
    run_until(app, wakes, |world| {
        let idle = world
            .try_query::<&ActiveTurn>()
            .is_some_and(|mut turns| turns.iter(world).next().is_none());
        main.requests().len() >= requests && idle
    });
}

/// The project's policy file.
fn write_policy(root: &Path, text: &str) {
    let written = std::fs::write(root.join(PROJECT_FILE), text);
    assert!(written.is_ok(), "{written:?}");
}

#[test]
fn commands_see_only_what_the_policy_allows() {
    // SAFETY: nextest runs each test in a process of its own, and these are
    // set before the app or any thread of the test starts.
    unsafe {
        std::env::set_var("KEEP_ME", "y");
        std::env::set_var("FOO_SECRET", "s");
        std::env::set_var("OTHER", "o");
    }
    let home = scratch_home("corpus-sp-shell-env-policy");
    let root = home.root().join("project");
    assert!(std::fs::create_dir_all(root.join(".rig")).is_ok());
    write_policy(&root, POLICY);

    let main = MockCompletionModel::from_stream_turns([
        shell("e1", "env"),
        reply("Printed."),
        shell("e2", "echo $FOO_SECRET; touch ran.txt"),
        reply("Refused."),
        calls(
            "t1",
            "task",
            json!({ "description": "helper", "prompt": "Print the environment.", "model": "test/sub" }),
        ),
        reply("Started."),
        reply("Thanks."),
        shell("e3", "env"),
        reply("Printed again."),
        shell("e4", "env"),
        reply("Still filtered."),
        shell("e5", "env"),
        reply("Still filtered."),
    ]);
    let sub = MockCompletionModel::from_stream_turns([shell("s1", "env"), reply("Done.")]);

    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.set_error_handler(rig_harness::error::warn)
        .insert_resource(scripted_models([("test/main", &main), ("test/sub", &sub)]))
        .insert_resource(ProjectRoot(root.clone()))
        .insert_resource(UserPolicyFile(None))
        .insert_resource(TuiTestScreen::new(120, 200))
        .add_plugins((AgentPlugin, SessionLogPlugin, CommandPlugin, ShellTool))
        .add_plugins((PickPlugin, SubagentsPlugin, ShellEnvPolicyPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    collect::<Notice>(&mut app);
    tui_view(&mut app);
    app.finish();
    app.cleanup();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    app.world_mut().trigger(SetModel {
        entity: agent,
        model: "test/main".to_owned(),
        kind: ChoiceKind::ForThisRun,
    });

    // The allowed, the set and the default variables; nothing else.
    send(
        &mut app,
        &wakes,
        (agent, "Print the environment."),
        &main,
        2,
    );
    let env = output(&app, "e1");
    assert!(
        has_line(&env, "KEEP_ME=y") && has_line(&env, "CI=1"),
        "{env}"
    );
    assert!(has_line(&env, "PATH="), "{env}");
    assert!(
        !env.contains("FOO_SECRET") && !has_line(&env, "OTHER="),
        "{env}"
    );

    // A command naming a denied variable never starts.
    send(&mut app, &wakes, (agent, "Echo the secret."), &main, 4);
    let refused = output(&app, "e2");
    assert!(refused.contains("FOO_SECRET"), "{refused}");
    assert!(refused.contains(PROJECT_FILE), "{refused}");
    assert!(!root.join("ran.txt").exists(), "the refused command ran");

    // An agent's own section reaches that agent alone, from the next call.
    write_policy(
        &root,
        &format!("{POLICY}\n[agent.helper]\nallow = [\"OTHER\"]\n"),
    );
    send(&mut app, &wakes, (agent, "Ask a helper."), &main, 7);
    let helper = output(&app, "s1");
    assert!(
        has_line(&helper, "OTHER=o") && has_line(&helper, "KEEP_ME=y"),
        "{helper}"
    );
    send(&mut app, &wakes, (agent, "Print it yourself."), &main, 9);
    let own = output(&app, "e3");
    assert!(
        has_line(&own, "KEEP_ME=y") && !has_line(&own, "OTHER="),
        "{own}"
    );

    // A malformed file keeps the last good policy, warned about once.
    write_policy(&root, "allow = [\"OTHER\"\n");
    send(&mut app, &wakes, (agent, "Print again."), &main, 11);
    let kept = output(&app, "e4");
    assert!(
        has_line(&kept, "KEEP_ME=y") && has_line(&kept, "CI=1"),
        "{kept}"
    );
    assert!(
        !kept.contains("FOO_SECRET") && !has_line(&kept, "OTHER="),
        "{kept}"
    );
    let warned = |app: &App| {
        let rows = screen(app);
        rows.iter()
            .filter(|row| row.contains("is not a valid environment policy"))
            .count()
    };
    assert_eq!(warned(&app), 1, "{}", screen(&app).join("\n"));
    send(&mut app, &wakes, (agent, "And again."), &main, 13);
    assert!(has_line(&output(&app, "e5"), "KEEP_ME=y"));
    assert_eq!(warnings(&app, PROJECT_FILE), 1);
    assert_eq!(warned(&app), 1, "{}", screen(&app).join("\n"));

    // `/env` names what the focused agent's commands see, values hidden
    // but for those the policy sets.
    app.world_mut().trigger(RunCommand::new(agent, "env"));
    app.update();
    app.update();
    let rows = screen(&app);
    let shown = |text: &str| rows.iter().any(|row| row.trim() == text);
    assert!(shown("KEEP_ME") && shown("CI=1"), "{}", rows.join("\n"));
    assert!(
        !rows.iter().any(|row| row.trim() == "OTHER"),
        "{}",
        rows.join("\n")
    );
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ShellEnvPolicyPlugin);
}

#[test]
fn commands_name_the_variables_they_read() {
    let names = referenced("echo $A ${B} ${C:-x} \"$D_1\" $1; printenv E F | grep x; ${#G}");
    assert_eq!(names, ["A", "B", "C", "D_1", "G", "E", "F"]);
}
