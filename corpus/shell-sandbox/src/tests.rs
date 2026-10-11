use std::io::ErrorKind;
use std::net::TcpListener;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::ShellTool;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{Ended, app_on, calls, connect, log_ended, reply, run_until};
use serde_json::json;

use super::*;

/// Each settled `shell` call's id and what it returned to the model.
#[derive(Resource, Default)]
struct Outputs(Vec<(String, String)>);

impl Outputs {
    fn of(&self, id: &str) -> &str {
        let found = self.0.iter().find(|(call, _)| call == id);
        found.map(|(_, output)| output.as_str()).unwrap_or_default()
    }
}

fn keep_outputs(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &Outcome)>,
    mut outputs: ResMut<Outputs>,
) {
    if let Ok((request, outcome)) = calls.get(settled.entity) {
        let output = outcome.tool_result().output().render();
        outputs.0.push((request.call.id.to_string(), output));
    }
}

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
/// asked once and runs until its turn ended. The first frame's notices,
/// and the shell calls' outputs.
fn run(base: &Path, search_path: OsString, model: &MockCompletionModel) -> (Vec<String>, Outputs) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let config = SandboxConfig {
        home: Some(base.join("home")),
        tmpdir: base.join("tmp"),
        search_path,
        ..SandboxConfig::default()
    };
    app.insert_resource(ProjectRoot(base.join("project")))
        .insert_resource(config)
        .add_plugins((AgentPlugin, JournalPlugin, ShellTool, ShellSandboxPlugin))
        .init_resource::<Outputs>()
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .add_observer(keep_outputs);
    app.finish();
    app.update();
    let notices = app.world().resource::<Messages<Notice>>();
    let notices = notices.iter_current_update_messages();
    let notices = notices.map(|notice| notice.text.clone()).collect();
    let world = app.world_mut();
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let (Some(agent), Some(connection)) = (agent, connect(model)) else {
        return (notices, Outputs::default());
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(agent, "Go.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let outputs = app.world_mut().remove_resource::<Outputs>();
    (notices, outputs.unwrap_or_default())
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
    let escape = outputs.of("escape");
    assert!(escape.contains("blocked by the sandbox"), "{escape}");
    assert!(!escape.contains("done\n"), "{escape}");

    let inside = std::fs::read_to_string(base.join("project/inside.txt"));
    assert_eq!(
        inside.ok().as_deref(),
        Some("hi\n"),
        "{}",
        outputs.of("inside")
    );

    let net = outputs.of("net");
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
    assert_eq!(notices.len(), 1, "said once at startup: {notices:?}");
    assert!(notices.iter().all(|notice| notice.contains(UNAVAILABLE)));
    let refused = outputs.of("inside");
    assert!(refused.contains(UNAVAILABLE), "{refused}");
    assert!(!base.join("project/inside.txt").exists());
    std::fs::remove_dir_all(&base).ok();
}
