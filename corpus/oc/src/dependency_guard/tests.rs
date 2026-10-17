use std::os::unix::fs::PermissionsExt;
use std::path::{Path, PathBuf};

use rig_basics::coding_tools::{EditTool, ShellTool};
use rig_core::test_utils::MockCompletionModel;
use rig_harness::rig_tools::process::Launcher;
use rig_harness_test_support::{
    AnswerAsks, Answers, Ended, MemoryStore, ToolOutputs, app_on, calls, connect, first_agent,
    log_ended, log_tool_outputs, reply, run_until, scratch_home,
};
use serde_json::json;

use super::*;

/// A project with `Cargo.toml` and `.rig/deps.toml`, and a `bin/` of fake
/// `cargo` and `npm` that append their arguments to `bin/calls.log`,
/// under the test binary's directory.
fn project() -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("dependency-guard-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(dir.join(".rig")).ok();
    std::fs::create_dir_all(dir.join("bin")).ok();
    let manifest =
        "[package]\nname = \"demo\"\nversion = \"0.1.0\"\n\n[dependencies]\nserde = \"1\"\n";
    std::fs::write(dir.join("Cargo.toml"), manifest).ok();
    let policy = "allow = [\"serde*\"]\ndeny = [\"openssl\"]\n";
    std::fs::write(dir.join(".rig").join(POLICY_FILE), policy).ok();
    for tool in ["cargo", "npm"] {
        let script = format!("#!/bin/sh\necho \"{tool} $*\" >> \"$(dirname \"$0\")/calls.log\"\n");
        let path = dir.join("bin").join(tool);
        std::fs::write(&path, script).ok();
        std::fs::set_permissions(&path, std::fs::Permissions::from_mode(0o755)).ok();
    }
    dir
}

fn shell(id: &str, command: &str) -> Vec<rig_core::test_utils::MockStreamEvent> {
    calls(id, "shell", json!({ "command": command }))
}

fn edit(id: &str, old: &str, new: &str) -> Vec<rig_core::test_utils::MockStreamEvent> {
    let edits = json!([{ "old_text": old, "new_text": new }]);
    calls(id, "edit", json!({ "path": "Cargo.toml", "edits": edits }))
}

fn output<'a>(outputs: &'a ToolOutputs, id: &str) -> &'a str {
    outputs.0.get(id).map(String::as_str).unwrap_or_default()
}

fn log(dir: &Path) -> String {
    std::fs::read_to_string(dir.join("bin").join("calls.log")).unwrap_or_default()
}

#[test]
fn new_dependencies_wait_for_the_user_unless_the_policy_decides() {
    let dir = project();
    let model = MockCompletionModel::from_stream_turns([
        shell("c1", "cargo add serde_json"),
        edit("c2", "serde = \"1\"\n", "serde = \"1\"\nrand = \"0.8\"\n"),
        edit("c3", "serde = \"1\"", "serde = \"1.0.200\""),
        shell("c4", "cargo add openssl"),
        shell("c5", "npm install"),
        shell("c6", "cargo add rand"),
        edit(
            "c7",
            "serde = \"1.0.200\"\n",
            "serde = \"1.0.200\"\nrand = \"0.8\"\n",
        ),
        reply("Done."),
    ]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let path = format!(
        "{}:{}",
        dir.join("bin").display(),
        std::env::var("PATH").unwrap_or_default()
    );
    let processes = Processes::default();
    processes.set(Launcher::default().with_env("PATH", path));
    // The user denies the first question and allows the second for the
    // session.
    let answers = Answers::new([Answer::choice(DENY), Answer::choice(SESSION)]);
    app.insert_resource(ProjectRoot(dir.clone()))
        .insert_resource(RigHome(scratch_home("dependency-guard-home")))
        .insert_resource(processes)
        .insert_resource(answers)
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            AnswerAsks,
            ShellTool,
            EditTool,
            DependencyGuardPlugin,
        ))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs)
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let agent = first_agent(app.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(&model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    app.world_mut()
        .trigger(Deliver::new(agent, "add retry logic", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    // Two questions: the denied `rand` edit and the `cargo add rand`.
    let asked = &app.world().resource::<Answers>().asked;
    assert_eq!(
        asked,
        &[
            "Add dependency `rand@0.8` to Cargo.toml?".to_owned(),
            "Add dependency `rand` with `cargo add rand`?".to_owned(),
        ]
    );
    let outputs = app.world().resource::<ToolOutputs>();
    let denied = output(outputs, "c2");
    assert!(denied.contains("denied adding `rand@0.8`"), "{denied}");
    let refused = output(outputs, "c4");
    assert!(
        refused.contains("`openssl` refused by deps.toml"),
        "{refused}"
    );
    // The allowed `cargo add` calls and the lockfile restore ran; the
    // refused one did not.
    assert_eq!(
        log(&dir),
        "cargo add serde_json\nnpm install\ncargo add rand\n"
    );
    // The denied edit left the file alone, the version bump and the
    // `rand` allowed for the session were applied.
    let manifest = std::fs::read_to_string(dir.join("Cargo.toml")).unwrap_or_default();
    assert!(
        manifest.ends_with("[dependencies]\nserde = \"1.0.200\"\nrand = \"0.8\"\n"),
        "{manifest}"
    );
    let deps = app.world().resource::<SessionDeps>();
    assert_eq!(deps.for_session, ["rand"]);
    assert_eq!(deps.denied, ["rand@0.8", "openssl"]);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(DependencyGuardPlugin);
}
