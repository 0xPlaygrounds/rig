use std::path::PathBuf;
use std::time::Instant;

use rig_basics::coding_tools::{ShellTool, WriteTool};
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::harness_protocol::SessionId;
use rig_harness_test_support::{
    Collected, Ended, ToolOutputs, app_in, calls, collect, connect, first_agent, log_ended,
    log_tool_outputs, reply, run_until, scratch_home,
};
use serde_json::json;

use super::*;

/// A fresh scratch directory under the test binary's: `local` (the
/// project), `remote` (the fake remote project), `remote-home`, and the
/// transport stand-in, which runs its last argument with `sh` in
/// `remote-home`. A command saying `FAILONCE` fails the transport once
/// (while `fail-once` exists).
fn scratch() -> Option<PathBuf> {
    let exe = std::env::current_exe().ok()?;
    let base = exe
        .parent()?
        .join(format!("remote-workspace-{}", std::process::id()));
    std::fs::remove_dir_all(&base).ok();
    for dir in ["local", "remote", "remote-home"] {
        std::fs::create_dir_all(base.join(dir)).ok()?;
    }
    let b = base.display();
    let script = format!(
        "#!/bin/sh\nfor last; do :; done\n\
         case \"$last\" in *FAILONCE*) if [ -e '{b}/fail-once' ]; then rm -f '{b}/fail-once'; \
         echo 'connection reset' >&2; exit 255; fi;; esac\n\
         export HOME='{b}/remote-home'\ncd \"$HOME\" || exit 255\nexec sh -c \"$last\"\n"
    );
    let transport = base.join("fake-transport");
    std::fs::write(&transport, script).ok()?;
    std::process::Command::new("chmod")
        .arg("+x")
        .arg(&transport)
        .status()
        .ok()?;
    std::fs::write(base.join("fail-once"), "").ok()?;
    Some(base)
}

fn shell(id: &str, command: &str) -> Vec<MockStreamEvent> {
    calls(id, "shell", json!({ "command": command }))
}

fn output<'a>(outputs: &'a ToolOutputs, id: &str) -> &'a str {
    outputs.0.get(id).map(String::as_str).unwrap_or_default()
}

#[test]
fn tools_act_on_the_remote_and_a_failed_transport_fails_one_call() {
    let base = scratch();
    assert!(base.is_some(), "a scratch directory");
    let Some(base) = base else {
        return;
    };
    let remote_dir = base.join("remote");
    let model = MockCompletionModel::from_stream_turns([
        vec![
            MockStreamEvent::tool_call(
                "c1",
                "write",
                json!({ "path": "hello.txt", "content": "hi" }),
            ),
            MockStreamEvent::tool_call("c2", "shell", json!({ "command": "cat hello.txt && pwd" })),
            MockStreamEvent::final_response_with_default_usage(),
        ],
        shell("c3", "echo FAILONCE"),
        shell("c4", "echo back"),
        shell("c5", "sleep 100 & echo $! > sleep.pid; wait"),
        reply("Done."),
    ]);

    let home = scratch_home("remote-workspace");
    let (mut app, wakes) = app_in(&home, &SessionId::generate());
    let settings = format!(
        "[remote.hosts]\nfake = [\"{}\"]\n",
        base.join("fake-transport").display()
    );
    app.insert_resource(Settings::from_toml(&settings))
        .insert_resource(ProjectRoot(base.join("local")))
        .add_plugins((AgentPlugin, SessionLogPlugin))
        .add_plugins((WriteTool, ShellTool, RemoteWorkspacePlugin))
        .insert_resource(RemoteTarget {
            target: format!("fake:{}", remote_dir.display()),
        })
        .init_resource::<ToolOutputs>()
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .add_observer(log_tool_outputs);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world);
    assert!(agent.is_some(), "an agent");
    let (Some(agent), Some(connection)) = (agent, connect(&model)) else {
        return;
    };
    world.entity_mut(agent).insert(connection);
    world.trigger(Deliver::new(agent, "Go.", DeliveryMode::Queue));

    // The shell creates the file before it writes the pid: wait for the pid.
    let pid_file = remote_dir.join("sleep.pid");
    let pid_of = || std::fs::read_to_string(&pid_file).unwrap_or_default();
    run_until(&mut app, &wakes, |_| !pid_of().trim().is_empty());
    let pid = pid_of();
    assert!(!pid.trim().is_empty(), "the remote sleep started");
    app.world_mut().trigger(StopTurn::new(agent));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });

    // The write and the shell acted on the remote, not the local project.
    let written = std::fs::read_to_string(remote_dir.join("hello.txt"));
    assert_eq!(written.ok().as_deref(), Some("hi"));
    assert!(!base.join("local/hello.txt").exists());
    let outputs = app.world().resource::<ToolOutputs>();
    let catted = output(outputs, "c2");
    assert!(catted.contains("hi"), "{catted}");
    assert!(
        catted.contains(&remote_dir.display().to_string()),
        "{catted}"
    );

    // The failed transport failed one call; the next ran.
    let dropped = output(outputs, "c3");
    assert!(
        dropped.contains("the transport to fake failed"),
        "{dropped}"
    );
    assert!(dropped.contains("connection reset"), "{dropped}");
    assert!(!dropped.contains("[exit code"), "{dropped}");
    let back = output(outputs, "c4");
    assert!(
        back.contains("back") && back.contains("[exit code 0]"),
        "{back}"
    );
    let notices = &app.world().resource::<Collected<Notice>>().0;
    assert!(
        notices
            .iter()
            .any(|notice| notice.text.contains("Working in fake:")),
        "{notices:?}"
    );

    // Cancelling the shell call killed the remote process.
    let proc_dir = PathBuf::from(format!("/proc/{}", pid.trim()));
    let started = Instant::now();
    while proc_dir.exists() && started.elapsed() < Duration::from_secs(5) {
        std::thread::sleep(Duration::from_millis(50));
    }
    assert!(
        !proc_dir.exists(),
        "the remote sleep {} still runs",
        pid.trim()
    );
    std::fs::remove_dir_all(&base).ok();
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(RemoteWorkspacePlugin);
}
