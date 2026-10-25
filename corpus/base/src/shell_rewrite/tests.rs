use std::path::Path;

use corpus_shared::confirm_destructive::ConfirmDestructivePlugin;
use rig_core::effect::EffectKind;
use rig_core::message::Message;
use rig_core::test_utils::MockCompletionModel;
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    AnswerAsks, Answers, Collected, Ended, ToolOutputs, app_in, calls, collect, connect,
    first_agent, log_ended, log_tool_outputs, reply, run_until, scratch_home,
};

use super::*;

/// `rm -rf` and `rm` go to the trash, when there is one.
const RULES: &str = r#"
[[rule]]
command = "rm -rf"
replace = "trash"
needs_program = "trash"

[[rule]]
command = "rm"
replace = "trash"
needs_program = "trash"
"#;

/// A `trash` in `dir/bin` that moves files to `./.trash`.
fn fake_trash(dir: &Path) {
    let trash = dir.join("bin/trash");
    std::fs::create_dir_all(dir.join("bin")).ok();
    std::fs::write(&trash, "#!/bin/sh\nmkdir -p .trash && mv \"$@\" .trash/\n").ok();
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        std::fs::set_permissions(&trash, std::fs::Permissions::from_mode(0o755)).ok();
    }
}

/// The model calls `shell` with `rm a.txt`, then with `rm -rf b`, in a
/// project whose `PATH` has a `trash`, under the rules of the test's
/// `RIG_HOME/shell-rewrite.toml`, with spec 01's guard asking about
/// destructive commands.
#[test]
fn rm_runs_as_trash_the_guard_judges_it_and_the_conversation_keeps_the_models_call() {
    let home = scratch_home("corpus-shell-rewrite");
    let dir = home.root().join("project");
    fake_trash(&dir);
    std::fs::write(home.root().join(FILE), RULES).ok();
    std::fs::write(dir.join("a.txt"), "a").ok();
    std::fs::create_dir_all(dir.join("b")).ok();
    let path = std::env::var_os("PATH").unwrap_or_default();
    let dirs = std::iter::once(dir.join("bin")).chain(std::env::split_paths(&path));
    let path = std::env::join_paths(dirs).unwrap_or_default();
    let processes = Processes::default();
    processes.set(processes.launcher().in_dir(&dir).with_env("PATH", path));

    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    app.insert_resource(processes)
        .insert_resource(ProjectRoot(dir.clone()))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            rig_basics::coding_tools::ShellTool,
            ConfirmDestructivePlugin,
            ShellRewritePlugin,
            AnswerAsks,
        ));
    collect::<EffectRecorded>(&mut app);
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([
        calls(
            "c1",
            Shell::NAME,
            serde_json::json!({ "command": "rm a.txt" }),
        ),
        calls(
            "c2",
            Shell::NAME,
            serde_json::json!({ "command": "rm -rf b" }),
        ),
        reply("Deleted."),
    ]);
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        world.entity_mut(agent).insert(connection);
    }
    let deliver = Deliver::new(agent, "Delete a.txt and b", DeliveryMode::Steer);
    world.trigger(deliver);
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let world = app.world();

    // What ran is the rewrite: both went to the trash.
    assert!(dir.join(".trash/a.txt").exists());
    assert!(dir.join(".trash/b").exists());
    assert!(!dir.join("a.txt").exists());
    // The guard judged `trash b`, not `rm -rf b`: nobody was asked.
    assert_eq!(world.resource::<Answers>().asked, Vec::<String>::new());
    // The conversation keeps the model's own calls.
    let messages = world
        .get::<Conversation>(agent)
        .map(|conversation| conversation.messages());
    let called: Vec<String> = messages
        .unwrap_or_default()
        .iter()
        .filter_map(|message| match &**message {
            Message::Assistant(assistant) => Some(assistant.tool_calls()),
            _ => None,
        })
        .flatten()
        .map(|call| call.function.raw_arguments())
        .collect();
    assert_eq!(
        called,
        [r#"{"command":"rm a.txt"}"#, r#"{"command":"rm -rf b"}"#]
    );
    // The effect log records what ran.
    let logged: Vec<&str> = world
        .resource::<Collected<EffectRecorded>>()
        .0
        .iter()
        .filter_map(|EffectRecorded(record)| match &record.kind {
            EffectKind::ToolCall { args, .. } => Some(args.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(
        logged,
        [r#"{"command":"trash a.txt"}"#, r#"{"command":"trash b"}"#]
    );
    // Each result says what ran.
    let outputs = &world.resource::<ToolOutputs>().0;
    for (id, wanted) in [("c1", "trash a.txt"), ("c2", "trash b")] {
        let output = outputs.get(id).map(String::as_str).unwrap_or_default();
        let wanted = format!("(ran as: `{wanted}`)\n");
        assert!(output.starts_with(&wanted), "{output:?}");
    }
    std::fs::remove_dir_all(home.root()).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ShellRewritePlugin);
}

/// The tool guards, loaded together.
#[test]
fn stages_are_ordered_with_the_other_tool_guards() {
    rig_defaults_test_support::assert_stages_ordered((
        ShellRewritePlugin,
        ConfirmDestructivePlugin,
        corpus_shared::permission_modes::PermissionModesPlugin,
        corpus_shared::protected_paths::ProtectedPathsPlugin,
        corpus_shared::plan_mode::PlanModePlugin,
    ));
}
