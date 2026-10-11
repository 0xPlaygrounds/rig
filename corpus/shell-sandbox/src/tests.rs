use std::io::ErrorKind;
use std::net::TcpListener;

use rig_cassette::journal::MemoryStore;
use rig_coding_tools::ShellTool;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness_test_support::{app_on, calls, connect, reply, run_until};
use serde_json::json;

use super::*;

/// Each `shell` output, by call id, and whether the turn ended.
#[derive(Resource, Default)]
struct Seen {
    outputs: Vec<(String, String)>,
    ended: bool,
}

impl Seen {
    fn output(&self, id: &str) -> &str {
        let found = self.outputs.iter().find(|(call, _)| call == id);
        found.map(|(_, output)| output.as_str()).unwrap_or_default()
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

fn config(base: &Path, search_path: OsString) -> SandboxConfig {
    SandboxConfig {
        project: base.join("project"),
        home: Some(base.join("home")),
        tmpdir: base.join("tmp"),
        search_path,
        ..SandboxConfig::default()
    }
}

/// The kernel, the default shell and the sandbox on `config`; the first
/// agent, answered by `model`, is asked once and runs until its turn ended.
/// The app's first frame's notices, and what was seen.
fn run(config: SandboxConfig, model: &MockCompletionModel) -> (Vec<String>, Seen) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(config)
        .add_plugins((AgentPlugin, JournalPlugin, ShellTool, ShellSandboxPlugin))
        .init_resource::<Seen>()
        .add_observer(
            |done: On<Insert<ToolOutput>>,
             calls: Query<(&ToolCallRun, &ToolOutput)>,
             mut seen: ResMut<Seen>| {
                if let Ok((run, output)) = calls.get(done.entity) {
                    let id = run.call.id.to_string();
                    seen.outputs.push((id, output.0.output().render()));
                }
            },
        )
        .add_observer(|_: On<TurnEnded>, mut seen: ResMut<Seen>| seen.ended = true);
    app.finish();
    app.update();
    let notices = app.world().resource::<Messages<Notice>>();
    let notices = notices.iter_current_update_messages();
    let notices = notices.map(|notice| notice.text.clone()).collect();
    let world = app.world_mut();
    // The default shell is off and the sandboxed one took its name, so no
    // command can reach the real home directory.
    let mut enabled = world.query_filtered::<&ToolDef, Without<Disabled>>();
    let shells = enabled
        .iter(world)
        .filter(|def| def.0.name.as_str() == SHELL);
    assert_eq!(shells.count(), 1);
    let mut off = world.query_filtered::<&ToolDef, With<Disabled>>();
    assert_eq!(off.iter(world).count(), 1, "the default shell is disabled");
    let agent = world
        .query_filtered::<Entity, With<Agent>>()
        .iter(world)
        .next();
    let (Some(agent), Some(connection)) = (agent, connect(model)) else {
        return (notices, Seen::default());
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(agent, "Go.", DeliveryMode::Queue));
    run_until(&mut app, &wakes, |world| world.resource::<Seen>().ended);
    let seen = app.world_mut().remove_resource::<Seen>();
    (notices, seen.unwrap_or_default())
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
    let (notices, seen) = run(config(&base, path), &model);
    assert!(notices.is_empty(), "{notices:?}");
    assert!(seen.ended);

    assert!(!base.join("home/escape").exists());
    let escape = seen.output("escape");
    assert!(escape.contains("blocked by the sandbox"), "{escape}");
    assert!(!escape.contains("done\n"), "{escape}");

    let inside = std::fs::read_to_string(base.join("project/inside.txt"));
    assert_eq!(
        inside.ok().as_deref(),
        Some("hi\n"),
        "{}",
        seen.output("inside")
    );

    let net = seen.output("net");
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
    let (notices, seen) = run(config(&base, OsString::new()), &model);
    assert_eq!(notices.len(), 1, "said once at startup: {notices:?}");
    assert!(notices.iter().all(|notice| notice.contains(UNAVAILABLE)));
    let refused = seen.output("inside");
    assert!(refused.contains(UNAVAILABLE), "{refused}");
    assert!(!base.join("project/inside.txt").exists());
    std::fs::remove_dir_all(&base).ok();
}
