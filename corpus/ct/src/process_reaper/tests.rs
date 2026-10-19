use std::net::TcpListener;
use std::path::{Path, PathBuf};
use std::process::Command;

use rig_basics::coding_tools::ShellTool;
use rig_core::test_utils::{MockCompletionModel, MockStreamEvent};
use rig_harness::commands::CommandPlugin;
use rig_harness::harness_protocol::{Home, SessionId};
use rig_harness::host::exit::ExitPlugin;
use rig_harness_test_support::{
    Collected, Ended, ToolOutputs, app_in, calls, collect, connect, first_agent, log_ended,
    log_tool_outputs, reply, run_until, scratch_home,
};
use serde_json::json;

use super::*;

/// What the fixture writes once it listens: its port, its pid and its
/// child's pid.
#[derive(Clone, Copy, Debug)]
struct Fixture {
    port: u16,
    pid: u32,
    child: u32,
}

impl Fixture {
    fn read(file: &Path) -> Option<Self> {
        let text = std::fs::read_to_string(file).ok()?;
        let mut words = text.split_whitespace();
        Some(Self {
            port: words.next()?.parse().ok()?,
            pid: words.next()?.parse().ok()?,
            child: words.next()?.parse().ok()?,
        })
    }

    /// Whether the server or its child still runs.
    fn runs(self) -> bool {
        procfs::runs(self.pid) || procfs::runs(self.child)
    }

    /// Whether its port is free again.
    fn port_free(self) -> bool {
        TcpListener::bind(("127.0.0.1", self.port)).is_ok()
    }
}

/// A server for the tests to start through `shell`: listens on a free
/// port, starts a child, writes both to the file `REAPER_FIXTURE` names and
/// sleeps. Run by its test binary, `--exact <module>::tests::fixture_server
/// --ignored`.
#[test]
#[ignore = "a server the other tests start through `shell`"]
fn fixture_server() {
    let Ok(file) = std::env::var("REAPER_FIXTURE") else {
        return;
    };
    let Ok(listener) = TcpListener::bind("127.0.0.1:0") else {
        return;
    };
    let port = listener
        .local_addr()
        .map(|at| at.port())
        .unwrap_or_default();
    let Ok(mut child) = Command::new("sleep").arg("120").spawn() else {
        return;
    };
    let text = format!("{port} {} {}", std::process::id(), child.id());
    let partial = format!("{file}.partial");
    if std::fs::write(&partial, text).is_ok() {
        std::fs::rename(&partial, &file).ok();
    }
    std::thread::sleep(Duration::from_secs(120));
    child.kill().ok();
    child.wait().ok();
    drop(listener);
}

/// A `shell` call that starts the fixture in the background, writing to
/// `file`, and returns once it listens.
fn start_fixture(id: &str, file: &Path) -> Vec<MockStreamEvent> {
    let exe = std::env::current_exe().unwrap_or_default();
    let (exe, file) = (exe.display(), file.display());
    let server = test_path("fixture_server");
    let command = format!(
        "REAPER_FIXTURE='{file}' '{exe}' --exact {server} --ignored >/dev/null 2>&1 & \
         while [ ! -s '{file}' ]; do sleep 0.05; done"
    );
    calls(id, "shell", json!({ "command": command }))
}

/// A directory of the test's own beside its binary, emptied.
fn scratch(name: &str) -> PathBuf {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("process-reaper-{name}-{}", std::process::id()));
    std::fs::remove_dir_all(&dir).ok();
    std::fs::create_dir_all(&dir).ok();
    dir
}

/// The kernel, the default shell, the reaper and the exit on the session
/// `id` of `home`, the first agent answered by `model`.
fn app(
    home: &Home,
    id: &SessionId,
    dir: &Path,
    model: &MockCompletionModel,
) -> (App, std::sync::mpsc::Receiver<()>, Entity) {
    let (mut app, wakes) = app_in(home, id);
    app.insert_resource(ProjectRoot(dir.to_path_buf()))
        .add_plugins((
            AgentPlugin,
            SessionLogPlugin,
            CommandPlugin,
            ExitPlugin,
            ShellTool,
        ))
        .add_plugins(ProcessReaperPlugin)
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = first_agent(world).unwrap_or(Entity::PLACEHOLDER);
    if let Some(connection) = connect(model) {
        world.entity_mut(agent).insert(connection);
    }
    (app, wakes, agent)
}

/// Asks `agent` once and runs until `turns` turns ended in all.
fn ask(app: &mut App, wakes: &std::sync::mpsc::Receiver<()>, agent: Entity, turns: usize) {
    app.world_mut()
        .trigger(Deliver::new(agent, "Go on.", DeliveryMode::Queue));
    run_until(app, wakes, |world| {
        world.resource::<Ended>().0.len() >= turns
    });
}

fn notices(app: &App) -> Vec<String> {
    let notices = &app.world().resource::<Collected<Notice>>().0;
    notices.iter().map(|notice| notice.text.clone()).collect()
}

#[test]
fn processes_are_reported_stopped_kept_and_reaped_on_quit() {
    let dir = scratch("session");
    let [a, b, c] = ["a", "b", "c"].map(|name| dir.join(format!("{name}.port")));
    let model = MockCompletionModel::from_stream_turns([
        start_fixture("c1", &a),
        reply("Started it."),
        calls("c2", "processes", json!({})),
        calls("c3", "stop_process", json!({ "id": 1 })),
        reply("Stopped it."),
        start_fixture("c4", &b),
        start_fixture("c5", &c),
        reply("Started two."),
        reply("Fine."),
    ]);
    let home = scratch_home("process-reaper-session");
    let (mut app, wakes, agent) = app(&home, &SessionId::generate(), &dir, &model);

    // The end of the turn tells the user, and the model, with the port.
    ask(&mut app, &wakes, agent, 1);
    let first = Fixture::read(&a);
    assert!(first.is_some(), "the fixture started: {:?}", notices(&app));
    let Some(first) = first else { return };
    let told = "1 process still running: `";
    let port = format!("(port {})", first.port);
    let notice = notices(&app)
        .into_iter()
        .find(|text| text.starts_with(told));
    assert!(
        notice.as_ref().is_some_and(|text| text.contains(&port)),
        "{:?}",
        notices(&app)
    );
    let conversation = app
        .world()
        .get::<Conversation>(agent)
        .map(|c| format!("{:?}", c.messages()));
    let noted = conversation.unwrap_or_default();
    assert!(
        noted.contains("keep_process") && noted.contains(&port),
        "{noted}"
    );

    // `processes` lists it; `stop_process` stops it and its child.
    ask(&mut app, &wakes, agent, 2);
    let outputs = &app.world().resource::<ToolOutputs>().0;
    let output = |id: &str| outputs.get(id).cloned().unwrap_or_default();
    let listed = output("c2");
    assert!(
        listed.starts_with("1 `") && listed.contains(&format!("ports {}", first.port)),
        "{listed}"
    );
    assert!(listed.contains("1 child process"), "{listed}");
    let stopped = output("c3");
    assert!(stopped.starts_with("Stopped process 1 `"), "{stopped}");
    assert!(!first.runs() && first.port_free(), "{first:?} still runs");

    // Two more; the user keeps one. The next turn's end stops the other.
    ask(&mut app, &wakes, agent, 3);
    let (kept, other) = (Fixture::read(&b), Fixture::read(&c));
    assert!(
        kept.is_some() && other.is_some(),
        "both started: {:?}",
        notices(&app)
    );
    let (Some(kept), Some(other)) = (kept, other) else {
        return;
    };
    assert!(
        notices(&app)
            .iter()
            .any(|text| text.starts_with("2 processes still running"))
    );
    app.world_mut()
        .trigger(RunCommand::new(agent, "procs keep 2"));
    app.update();
    ask(&mut app, &wakes, agent, 4);
    run_until(&mut app, &wakes, |_| !other.runs());
    assert!(!other.runs(), "{other:?} was not kept, and still runs");
    assert!(kept.runs(), "{kept:?} was kept, and stopped");

    // Quitting stops the kept one before the app exits.
    app.world_mut().trigger(Exit(ExitReason::Quit));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Messages<AppExit>>().is_empty()
    });
    assert!(
        !app.world().resource::<Messages<AppExit>>().is_empty(),
        "the app exits"
    );
    assert!(
        !kept.runs() && kept.port_free(),
        "{kept:?} outlived the session"
    );
}

#[test]
fn a_crashed_runs_processes_are_reaped_on_the_next_start_and_nothing_else() {
    let dir = scratch("crash");
    let file = dir.join("d.port");
    let model =
        MockCompletionModel::from_stream_turns([start_fixture("c1", &file), reply("Started.")]);
    let (home, id) = (scratch_home("process-reaper-crash"), SessionId::generate());
    let (mut app, wakes, agent) = self::app(&home, &id, &dir, &model);
    ask(&mut app, &wakes, agent, 1);
    let orphan = Fixture::read(&file);
    assert!(orphan.is_some(), "the fixture started");
    let Some(orphan) = orphan else { return };
    // A crash: the app goes without exiting.
    drop(app);
    assert!(orphan.runs());

    // A process of the user's, under a pid record the saved list has with
    // another start time.
    let unrelated = Command::new("sleep").arg("30").spawn();
    assert!(unrelated.is_ok(), "sleep starts");
    let Ok(mut unrelated) = unrelated else { return };
    let saved_at = home.data().join(format!("{FILES}{}.json", id.as_str()));
    let saved = std::fs::read(&saved_at).ok();
    let mut saved: Saved = saved
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default();
    assert!(!saved.entries.is_empty(), "the run saved its processes");
    let start = procfs::member(unrelated.id()).map_or(0, |member| member.start);
    saved.entries.push(Entry {
        id: 99,
        command: "sleep 30".to_owned(),
        agent: None,
        turn: None,
        kept: false,
        members: vec![Member {
            pid: unrelated.id(),
            start: start + 7,
        }],
        stopping: false,
    });
    std::fs::write(&saved_at, serde_json::to_vec(&saved).unwrap_or_default()).ok();

    let model = MockCompletionModel::from_stream_turns([reply("Back.")]);
    let (mut app, wakes, _) = self::app(&home, &id, &dir, &model);
    run_until(&mut app, &wakes, |_| !orphan.runs());
    assert!(
        !orphan.runs() && orphan.port_free(),
        "{orphan:?} outlived the crash"
    );
    let untouched = matches!(unrelated.try_wait(), Ok(None));
    unrelated.kill().ok();
    unrelated.wait().ok();
    assert!(untouched, "the user's process was stopped");
}

#[test]
fn stages_are_ordered_with_the_defaults() {
    rig_defaults_test_support::assert_stages_ordered(ProcessReaperPlugin);
}

/// The test `name` of this module as its test binary names it: the module's
/// path without the crate's name.
fn test_path(name: &str) -> String {
    let module = module_path!().split_once("::").map_or("", |(_, path)| path);
    format!("{module}::{name}")
}
