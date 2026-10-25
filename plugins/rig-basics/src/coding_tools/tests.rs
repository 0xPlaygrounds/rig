use std::ffi::{OsStr, OsString};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use rig_harness::prelude::*;
use rig_harness::rig_core::serde_json::json;
use rig_harness::rig_core::test_utils::MockCompletionModel;
use rig_harness::rig_core::tool::ToolContext;
use rig_harness_test_support::{
    Collected, MemoryStore, ToolOutputs, app_on, collect, connect, log_tool_outputs, reply,
    run_until, start_call,
};
use rig_tools::process::{Launcher, Process, Spawn};
use rig_tools::workspace::{Contents, Listing, Stat, Workspace};
use rig_tools::{Shell, ShellArgs};

use super::remote::RemoteHostsPlugin;
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
    let ids = ["c0", "c1", "c2"];
    // The three calls run at once and may end in any order.
    run_until(&mut app, &wakes, |world| {
        ids.iter().all(|call| output(world, call).is_some())
    });
    let [read, outside, pwd] = ids.map(|call| output(app.world(), call).unwrap_or_default());
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
    // A model answers it, so the message stays in the conversation.
    let model = MockCompletionModel::from_stream_turns([reply("read")]);
    let agent = world.spawn(Agent).id();
    if let Some(connection) = connect(&model) {
        world.entity_mut(agent).insert(connection);
    }
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

/// The texts of the system prompt's `remote_hosts` sections.
fn host_sections(app: &mut App) -> Vec<String> {
    let mut sections = app.world_mut().query::<&PromptSection>();
    sections
        .iter(app.world())
        .filter(|section| section.tag == "remote_hosts")
        .map(|section| section.text.clone())
        .collect()
}

/// With `[remote.hosts]`, `read` and `shell` reach a `HOST:PATH` through
/// the host's transport, plain paths stay here, and the system prompt names
/// the hosts; without hosts there is no such section.
#[test]
fn a_host_path_reaches_the_host_the_settings_name() {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("remote-hosts-{}", std::process::id()));
    let home = dir.join("home");
    std::fs::create_dir_all(&home).ok();
    std::fs::write(home.join("note.txt"), "on the host").ok();
    std::fs::write(dir.join("note.txt"), "on this machine").ok();
    // Runs its last argument with `sh` in `home`, as ssh runs it there.
    let settings = format!(
        "[remote.hosts]\nbench = ['sh', '-c', 'cd \"$0\" && HOME=\"$0\" exec sh -c \"$1\"', '{}']\n",
        home.display()
    );
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.insert_resource(Settings::from_toml(&settings))
        .insert_resource(ProjectRoot(dir.clone()))
        .add_plugins((AgentPlugin, ReadTool, ShellTool, RemoteHostsPlugin))
        .init_resource::<ToolOutputs>()
        .add_observer(log_tool_outputs);
    app.finish();
    app.update();
    let world = app.world_mut();
    let agent = world.spawn(Agent).id();
    let calls = [
        ("read", json!({ "path": "bench:~/note.txt" })),
        ("read", json!({ "path": "note.txt" })),
        ("shell", json!({ "command": "pwd", "cwd": "bench:~" })),
    ];
    for (n, (tool, args)) in calls.into_iter().enumerate() {
        start_call(world, agent, (tool, &format!("c{n}")), args);
    }
    let output = |world: &World, call: &str| world.resource::<ToolOutputs>().0.get(call).cloned();
    run_until(&mut app, &wakes, |world| {
        ["c0", "c1", "c2"]
            .iter()
            .all(|call| output(world, call).is_some())
    });
    let [there, here, pwd] =
        ["c0", "c1", "c2"].map(|call| output(app.world(), call).unwrap_or_default());
    assert!(there.contains("on the host"), "{there}");
    assert!(here.contains("on this machine"), "{here}");
    assert!(pwd.contains(&*home.to_string_lossy()), "{pwd}");
    let sections = host_sections(&mut app);
    assert!(
        sections.len() == 1 && sections.iter().all(|text| text.contains("`bench`")),
        "{sections:?}"
    );
    // The arguments that take a path say they take one there too.
    let mut tools = app.world_mut().query::<&ToolDef>();
    let told: Vec<(String, bool)> = tools
        .iter(app.world())
        .map(|def| {
            let argument = if def.0.name.as_str() == "shell" {
                "cwd"
            } else {
                "path"
            };
            let pointer = format!("/properties/{argument}/description");
            let text = def
                .0
                .parameters
                .pointer(&pointer)
                .and_then(|text| text.as_str());
            (
                def.0.name.to_string(),
                text.is_some_and(|text| text.contains("`bench:~/app")),
            )
        })
        .collect();
    assert!(
        told.len() == 2 && told.iter().all(|(_, told)| *told),
        "{told:?}"
    );
    std::fs::remove_dir_all(&dir).ok();

    let mut local = App::new();
    local
        .insert_resource(Settings::from_toml(""))
        .add_plugins((AgentPlugin, RemoteHostsPlugin));
    local.finish();
    assert!(host_sections(&mut local).is_empty());
    assert!(
        local
            .world()
            .resource::<ProjectWorkspace>()
            .0
            .host()
            .is_none()
    );
}
