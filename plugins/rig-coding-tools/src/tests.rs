use std::ffi::{OsStr, OsString};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_core::serde_json::json;
use rig_harness::rig_core::tool::ToolContext;
use rig_harness_test_support::{
    Collected, MemoryStore, ToolOutputs, app_on, collect, log_tool_outputs, run_until, start_call,
};
use rig_tools::process::{Launcher, Process, Spawn};
use rig_tools::workspace::{Contents, Listing, Stat, Workspace};
use rig_tools::{Shell, ShellArgs};

use super::{AttachPlugin, AttachSystems, ReadTool, SharedShell, ShellOutput, ShellTool};

/// The shell tool on `processes`, cutting as `output` says.
fn shell(processes: &Processes, output: &ShellOutput) -> SharedShell {
    SharedShell {
        shell: Shell::default(),
        processes: processes.clone(),
        output: output.clone(),
        dir: None,
    }
}

#[test]
fn an_agent_with_a_work_dir_reads_and_runs_commands_there() {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("work-dir-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(dir.join("note.txt"), "in the work dir").ok();
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.add_plugins((AgentPlugin, ReadTool, ShellTool))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.spawn((Agent, WorkDir(dir.clone()))).id();
    let calls = [
        ("read", json!({ "path": "note.txt" })),
        ("read", json!({ "path": "../note.txt" })),
        ("shell", json!({ "command": "pwd" })),
    ];
    for (n, (tool, args)) in calls.into_iter().enumerate() {
        start_call(world, agent, (tool, &format!("c{n}")), args);
    }
    let output = |world: &World, call: &str| world.resource::<ToolOutputs>().0.get(call).cloned();
    run_until(&mut app, &wakes, |world| output(world, "c2").is_some());
    let [read, outside, pwd] =
        ["c0", "c1", "c2"].map(|call| output(app.world(), call).unwrap_or_default());
    assert!(read.contains("in the work dir"), "{read}");
    assert!(
        !outside.is_empty() && !outside.contains("in the work dir"),
        "{outside}"
    );
    let canonical = dir.canonicalize().unwrap_or_default();
    assert!(pwd.contains(&*canonical.to_string_lossy()), "{pwd}");
    std::fs::remove_dir_all(&dir).ok();
}

/// A workspace that is not this machine's disk: it has one file,
/// `/served/note.txt`, and runs nothing.
#[derive(Debug)]
struct Served;

impl Served {
    const NOTE: &str = "/served/note.txt";
}

impl Workspace for Served {
    fn host(&self) -> Option<&str> {
        Some("test://served")
    }

    fn stat(&self, path: &Path) -> io::Result<Stat> {
        Ok(match path == Path::new(Self::NOTE) {
            true => Stat::File {
                size: 11,
                read_only: false,
                modified: None,
            },
            false => Stat::Missing,
        })
    }

    fn read(&self, path: &Path, _: u64) -> io::Result<Contents> {
        match path == Path::new(Self::NOTE) {
            true => Ok(Contents {
                bytes: b"served text".to_vec(),
                size: 11,
            }),
            false => Err(io::ErrorKind::NotFound.into()),
        }
    }

    fn write(&self, _: &Path, _: &[u8]) -> io::Result<()> {
        Err(io::ErrorKind::Unsupported.into())
    }

    fn remove(&self, _: &Path) -> io::Result<()> {
        Err(io::ErrorKind::Unsupported.into())
    }

    fn rename(&self, _: &Path, _: &Path) -> io::Result<()> {
        Err(io::ErrorKind::Unsupported.into())
    }

    fn canonical(&self, path: &Path) -> PathBuf {
        path.to_path_buf()
    }

    fn files(&self, _: &Path, _: Option<&str>, _: Listing) -> io::Result<Vec<PathBuf>> {
        Ok(Vec::new())
    }

    fn spawn(
        &self,
        _: &Launcher,
        _: Option<&Path>,
        _: &OsStr,
        _: &[OsString],
        _: Spawn,
    ) -> io::Result<Process> {
        Err(io::ErrorKind::Unsupported.into())
    }
}

#[test]
fn a_call_without_a_work_dir_reads_from_the_project_root_in_the_apps_workspace() {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(ProjectWorkspace(Arc::new(Served)))
        .insert_resource(ProjectRoot("/served".into()))
        .add_plugins((AgentPlugin, ReadTool))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    start_call(world, agent, ("read", "r1"), json!({ "path": "note.txt" }));
    let read = |world: &World| world.resource::<ToolOutputs>().0.get("r1").cloned();
    run_until(&mut app, &wakes, |world| read(world).is_some());
    let read = read(app.world()).unwrap_or_default();
    assert!(read.contains("served text"), "{read}");
}

/// The labels of the files attached to each message, as a plugin after
/// [`AttachSystems`] sees them.
#[derive(Resource, Default)]
struct Labels(Vec<String>);

fn read_labels(messages: Query<&Incoming, Added<Admitted>>, mut labels: ResMut<Labels>) {
    let attached = messages.iter().flat_map(|message| &message.parts);
    labels.0.extend(attached.filter_map(|part| match part {
        MessagePart::Attachment(attached) => Some(attached.label.clone()),
        MessagePart::Text(_) => None,
    }));
}

#[test]
fn an_agent_attaches_from_its_work_dir_and_is_warned_of_a_missing_path() {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("attach-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(dir.join("a.txt"), "attached text").ok();
    let mut app = App::new();
    app.add_plugins((TaskPoolPlugin::default(), AgentPlugin, AttachPlugin))
        .init_resource::<Labels>()
        .add_systems(
            AgentPass,
            read_labels
                .in_set(PrepareSystems::Compose)
                .after(AttachSystems),
        );
    collect::<Notice>(&mut app);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.spawn((Agent, WorkDir(dir.clone()))).id();
    let text = "compare @a.txt with @gone/x.rs, @alice";
    world.spawn((Incoming::new(text, DeliveryMode::Steer), ForAgent(agent)));
    app.update();
    assert_eq!(app.world().resource::<Labels>().0, ["a.txt"]);
    let notices = &app.world().resource::<Collected<Notice>>().0;
    let warned: Vec<&str> = notices
        .iter()
        .filter(|notice| notice.level == NoticeLevel::Warn)
        .map(|notice| notice.text.as_str())
        .collect();
    assert_eq!(warned, ["@gone/x.rs: no such file"]);
    std::fs::remove_dir_all(&dir).ok();
}

#[test]
fn in_another_workspace_a_message_waits_for_its_files() {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(ProjectWorkspace(Arc::new(Served)))
        .insert_resource(ProjectRoot("/served".into()))
        .add_plugins((AgentPlugin, AttachPlugin));
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    let message = world
        .spawn((
            Incoming::new("read @note.txt", DeliveryMode::Steer),
            ForAgent(agent),
        ))
        .id();
    app.update();
    let held = app.world().get::<Holds>(message).is_some();
    assert!(held, "the message waits for its files");
    let said = |world: &World| {
        let conversation = world.get::<Conversation>(agent);
        format!("{conversation:?}")
    };
    run_until(&mut app, &wakes, |world| {
        said(world).contains("served text")
    });
    assert!(
        said(app.world()).contains("served text"),
        "{}",
        said(app.world())
    );
}

/// What `shell` answers `command`.
fn run(shell: &SharedShell, command: &str) -> String {
    let args = ShellArgs::new(command);
    bevy_tasks::block_on(shell.call(&mut ToolContext::new(), args)).unwrap_or_default()
}

#[test]
fn each_shell_call_starts_as_the_processes_say_at_the_time() {
    let processes = Processes::default();
    let shell = shell(&processes, &ShellOutput::default());
    let greeting = "echo \"greeting: $GREETING\"";
    assert!(run(&shell, greeting).contains("greeting: \n"));
    processes.set(processes.launcher().with_env("GREETING", "hello"));
    let output = run(&shell, greeting);
    assert!(output.contains("greeting: hello"), "{output}");
}

#[test]
fn without_a_cut_the_shell_returns_every_line() {
    let output = ShellOutput::default();
    let shell = shell(&Processes::default(), &output);
    let lines = rig_tools::MAX_LINES + 500;
    let count = format!("seq 1 {lines}");
    let numbers = |text: &str| {
        text.lines()
            .filter(|line| line.parse::<usize>().is_ok())
            .count()
    };
    assert!(numbers(&run(&shell, &count)) <= rig_tools::MAX_LINES);
    output.set(None);
    let whole = run(&shell, &count);
    assert_eq!(numbers(&whole), lines, "{whole:.200}");
}

#[test]
fn the_longest_shell_timeout_is_the_users_setting() {
    let mut app = App::new();
    app.insert_resource(Settings::from_toml("[shell]\nmax_timeout = 5"))
        .add_plugins((AgentPlugin, ShellTool));
    app.finish();
    let mut tools = app.world_mut().query::<&ToolDef>();
    let described: Vec<String> = tools
        .iter(app.world())
        .map(|tool| tool.0.description.clone())
        .collect();
    assert!(
        described.iter().any(|it| it.contains("at most 5)")),
        "{described:?}"
    );
}
