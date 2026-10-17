use std::io::ErrorKind;
use std::net::TcpListener;

use rig_basics::coding_tools::ShellTool;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::MemoryStore;
use rig_harness_test_support::{
    Collected, Ended, ToolOutputs, app_on, calls, collect, connect, first_agent, log_ended,
    log_tool_outputs, reply, run_until,
};
use serde_json::json;

use super::*;

/// A fresh `project`, `home` and `tmp` under the test binary's directory,
/// in the target directory.
fn scratch(name: &str) -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let base = exe
        .parent()?
        .join(format!("shell-sandbox-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&base).ok();
    for dir in ["project", "home", "tmp"] {
        std::fs::create_dir_all(base.join(dir)).ok()?;
    }
    Some(base)
}

/// The kernel, the default shell and the sandbox in `base`, looking for
/// its program in `search_path`; the first agent, answered by `model`, is
/// asked once and runs until its turn ended. Every notice, and the calls'
/// outputs.
fn run(
    base: &Path,
    search_path: OsString,
    model: &MockCompletionModel,
) -> (Vec<String>, ToolOutputs) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let config = SandboxConfig {
        home: Some(base.join("home")),
        tmpdir: base.join("tmp"),
        search_path,
        ..SandboxConfig::default()
    };
    app.insert_resource(ProjectRoot(base.join("project")))
        .insert_resource(config)
        .add_plugins((AgentPlugin, SessionLogPlugin, ShellTool, ShellSandboxPlugin))
        .init_resource::<ToolOutputs>()
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .add_observer(log_tool_outputs);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let world = app.world_mut();
    if let (Some(agent), Some(connection)) = (first_agent(world), connect(model)) {
        world.entity_mut(agent).insert(connection);
        world.trigger(Deliver::new(agent, "Go.", DeliveryMode::Queue));
        run_until(&mut app, &wakes, |world| {
            !world.resource::<Ended>().0.is_empty()
        });
    }
    let world = app.world_mut();
    let notices = world.remove_resource::<Collected<Notice>>();
    let notices = notices.map(|kept| kept.0.into_iter().map(|notice| notice.text).collect());
    let outputs = world.remove_resource::<ToolOutputs>();
    (notices.unwrap_or_default(), outputs.unwrap_or_default())
}

/// What call `id` returned to the model.
fn output<'a>(outputs: &'a ToolOutputs, id: &str) -> &'a str {
    outputs.0.get(id).map(String::as_str).unwrap_or_default()
}

fn shell(id: &str, command: &str) -> Vec<MockStreamEvent> {
    calls(id, SHELL, json!({ "command": command }))
}

#[test]
fn commands_write_only_in_the_project_and_reach_no_network() {
    let path = std::env::var_os("PATH").unwrap_or_default();
    if lookup("bwrap", &path).is_none() {
        eprintln!("skipped: no bwrap on PATH");
        return;
    }
    let (base, listener) = (scratch("sandboxed"), TcpListener::bind("127.0.0.1:0"));
    assert!(
        base.is_some() && listener.is_ok(),
        "a scratch directory and a listener"
    );
    let (Some(base), Ok(listener)) = (base, listener) else {
        return;
    };
    assert!(listener.set_nonblocking(true).is_ok());
    let port = listener.local_addr().map(|addr| addr.port()).unwrap_or(0);
    let model = MockCompletionModel::from_stream_turns([
        shell("escape", "touch $HOME/escape && echo done"),
        shell("inside", "echo hi > inside.txt"),
        shell(
            "net",
            &format!("curl -sS --max-time 5 http://127.0.0.1:{port}"),
        ),
        reply("Done."),
    ]);
    let (notices, outputs) = run(&base, path, &model);
    assert!(notices.is_empty(), "{notices:?}");

    assert!(!base.join("home/escape").exists());
    let escape = output(&outputs, "escape");
    assert!(escape.contains("blocked by the sandbox"), "{escape}");
    assert!(!escape.contains("done\n"), "{escape}");

    let inside = std::fs::read_to_string(base.join("project/inside.txt"));
    assert_eq!(
        inside.ok().as_deref(),
        Some("hi\n"),
        "{}",
        output(&outputs, "inside")
    );

    let net = output(&outputs, "net");
    assert!(net.contains("`/sandbox net on`"), "{net}");
    let accepted = listener.accept().map(|_| ());
    assert_eq!(
        accepted.map_err(|error| error.kind()),
        Err(ErrorKind::WouldBlock)
    );
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn without_a_sandbox_program_calls_are_refused() {
    let base = scratch("unavailable");
    assert!(base.is_some(), "a scratch directory");
    let Some(base) = base else {
        return;
    };
    let model = MockCompletionModel::from_stream_turns([
        shell("inside", "echo hi > inside.txt"),
        reply("Done."),
    ]);
    let (notices, outputs) = run(&base, OsString::new(), &model);
    let said = notices.iter().filter(|notice| notice.contains(UNAVAILABLE));
    assert_eq!(said.count(), 1, "said once: {notices:?}");
    let refused = output(&outputs, "inside");
    assert!(refused.contains(UNAVAILABLE), "{refused}");
    assert!(!base.join("project/inside.txt").exists());
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ShellSandboxPlugin);
}
