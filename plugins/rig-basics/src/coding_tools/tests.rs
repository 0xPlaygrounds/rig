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
use rig_tools::workspace::{Contents, Listed, Listing, Stat, Workspace};
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

    fn files(&self, _: &Path, _: Option<&str>, _: Listing) -> io::Result<Listed> {
        Ok(Listed::default())
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

/// An app with [`AttachPlugin`] in `workspace` at `root`, its loop's
/// wakes, and an agent whose model answers every message, so each stays in
/// the conversation, with the notices kept.
fn attaching(
    workspace: Option<Arc<dyn Workspace>>,
    root: &Path,
    settings: &str,
) -> (App, std::sync::mpsc::Receiver<()>, Entity) {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    if let Some(workspace) = workspace {
        app.insert_resource(ProjectWorkspace(workspace));
    }
    app.insert_resource(Settings::from_toml(settings))
        .insert_resource(ProjectRoot(root.to_path_buf()))
        .add_plugins((AgentPlugin, AttachPlugin, RemoteHostsPlugin));
    collect::<Notice>(&mut app);
    collect::<Recalled>(&mut app);
    app.finish();
    app.update();
    let model = MockCompletionModel::from_stream_turns([reply("read"), reply("read")]);
    let agent = app.world_mut().spawn(Agent).id();
    if let Some(connection) = connect(&model) {
        app.world_mut().entity_mut(agent).insert(connection);
    }
    (app, wakes, agent)
}

/// What `agent`'s conversation holds, as text.
fn said(world: &World, agent: Entity) -> String {
    format!("{:?}", world.get::<Conversation>(agent))
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
    let served: Arc<dyn Workspace> = Arc::new(Served);
    let (mut app, wakes, agent) = attaching(Some(served), Path::new("/served"), "");
    let message = app
        .world_mut()
        .spawn((
            Incoming::new("read @note.txt", DeliveryMode::Steer),
            ForAgent(agent),
        ))
        .id();
    app.update();
    let held = app.world().get::<Holds>(message).is_some();
    assert!(held, "the message waits for its files");
    run_until(&mut app, &wakes, |world| {
        said(world, agent).contains("served text")
    });
    assert!(
        said(app.world(), agent).contains("served text"),
        "{}",
        said(app.world(), agent)
    );
}

/// `@HOST:PATH` attaches that host's file through its transport, even with
/// a file at the same path here.
#[cfg(unix)]
#[test]
fn a_host_reference_attaches_the_hosts_file() {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("attach-host-{}", std::process::id()));
    let home = dir.join("home");
    std::fs::create_dir_all(&home).ok();
    std::fs::write(home.join("note.txt"), "ON_THE_HOST").ok();
    std::fs::create_dir_all(dir.join("bench:~")).ok();
    std::fs::write(dir.join("bench:~").join("note.txt"), "WRONG_LOCAL_SHADOW").ok();
    let settings = format!(
        "[remote.hosts]\nbench = ['sh', '-c', 'cd \"$0\" && HOME=\"$0\" exec sh -c \"$1\"', '{}']\n",
        home.display()
    );
    let (mut app, wakes, agent) = attaching(None, &dir, &settings);
    let text = "quote @bench:~/note.txt";
    let world = app.world_mut();
    world.spawn((Incoming::new(text, DeliveryMode::Steer), ForAgent(agent)));
    run_until(&mut app, &wakes, |world| {
        said(world, agent).contains("ON_THE_HOST")
    });
    let first = said(app.world(), agent);
    assert!(
        first.contains("ON_THE_HOST") && !first.contains("WRONG_LOCAL_SHADOW"),
        "{first}"
    );
    // From a work directory on the host, `~` is the host's home, and the
    // file goes by its place there.
    std::fs::write(home.join("other.txt"), "OTHER_ON_THE_HOST").ok();
    let world = app.world_mut();
    world
        .entity_mut(agent)
        .insert(WorkDir("bench:~/app".into()));
    let text = "quote @~/other.txt";
    world.spawn((Incoming::new(text, DeliveryMode::Steer), ForAgent(agent)));
    run_until(&mut app, &wakes, |world| {
        said(world, agent).contains("OTHER_ON_THE_HOST")
    });
    let said = said(app.world(), agent);
    assert!(said.contains("bench:~/other.txt"), "{said}");
    std::fs::remove_dir_all(&dir).ok();
}

/// Whether the file name of `path` holds `part`.
fn named(path: &Path, part: &str) -> bool {
    path.file_name()
        .is_some_and(|name| name.to_string_lossy().contains(part))
}

/// This machine's disk behind a slow transport: every path but a file
/// named `…local…` is said to be on a host, and every look at one but a
/// file named `…fast…` waits until `released` (at most 5 s).
#[derive(Debug, Default)]
struct Slow {
    released: Arc<std::sync::atomic::AtomicBool>,
}

impl Workspace for Slow {
    fn host_for(&self, path: &Path) -> Option<&str> {
        (!named(path, "local")).then_some("test://slow")
    }

    fn stat(&self, path: &Path) -> io::Result<Stat> {
        let started = std::time::Instant::now();
        let fast = named(path, "fast");
        while !fast
            && !self.released.load(std::sync::atomic::Ordering::Relaxed)
            && started.elapsed() < std::time::Duration::from_secs(5)
        {
            std::thread::sleep(std::time::Duration::from_millis(10));
        }
        rig_tools::workspace::Local.stat(path)
    }

    fn read(&self, path: &Path, limit: u64) -> io::Result<Contents> {
        rig_tools::workspace::Local.read(path, limit)
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

/// While a slow workspace reads a message's files the frame goes on, and
/// stopping the turn puts the message back in the input instead of sending
/// it.
#[test]
fn a_slow_read_holds_the_message_and_a_stop_recalls_it() {
    let slow = Slow::default();
    let released = slow.released.clone();
    let (mut app, _wakes, agent) = attaching(Some(Arc::new(slow)), Path::new("/nowhere"), "");
    let text = "read @notes.txt";
    let message = app
        .world_mut()
        .spawn((Incoming::new(text, DeliveryMode::Steer), ForAgent(agent)))
        .id();
    let started = std::time::Instant::now();
    app.update();
    let waited = started.elapsed();
    let held = app.world().get::<Holds>(message).is_some();
    assert!(held, "the message waits for its files");
    assert!(waited < std::time::Duration::from_secs(2), "{waited:?}");
    app.world_mut().trigger(StopTurn::new(agent));
    app.update();
    app.update();
    // The read it left, once it ends, sends nothing.
    released.store(true, std::sync::atomic::Ordering::Relaxed);
    let after = std::time::Instant::now();
    while after.elapsed() < std::time::Duration::from_millis(300) {
        app.update();
        std::thread::sleep(std::time::Duration::from_millis(10));
    }
    let recalled = &app.world().resource::<Collected<Recalled>>().0;
    let back: Vec<&str> = recalled.iter().map(|back| back.text.as_str()).collect();
    assert_eq!(back, [text]);
    let conversation = app.world().get::<Conversation>(agent);
    assert!(
        conversation.is_none_or(|conversation| conversation.messages().is_empty()),
        "{conversation:?}"
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

/// The lines of `agent`'s transcript as the terminal view lays them out
/// at 120 columns.
#[cfg(feature = "tui")]
fn transcript(app: &mut App, agent: Entity) -> Vec<String> {
    let view = rig_tui::TranscriptView::default();
    let laid = rig_tui::TranscriptLayout::lay_out(app.world_mut(), &view, agent, 120);
    let lines = laid.map(|laid| laid.lines().map(|line| line.text).collect());
    lines.unwrap_or_default()
}

/// A resumed session draws the rows of `read`, `edit` and a failing
/// `shell` as the live one did, from what each call published: the
/// session keeps it with the call's result, so no effect log is needed.
#[cfg(feature = "tui")]
#[test]
fn a_resumed_session_draws_its_tool_rows_as_the_live_one_did() {
    use rig_harness::settings::Settings;
    use rig_harness_test_support::{
        Ended, calls, first_agent, log_ended, scripted_models, tui_view,
    };

    use super::EditTool;

    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("resumed-rows-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(dir.join("notes.txt"), "one\ntwo\nthree\n").ok();
    let edits = json!([{ "old_text": "two", "new_text": "TWO" }]);
    let model = MockCompletionModel::from_stream_turns([
        calls("r1", "read", json!({ "path": "notes.txt" })),
        calls("e1", "edit", json!({ "path": "notes.txt", "edits": edits })),
        calls("s1", "shell", json!({ "command": "echo out; exit 3" })),
        reply("done"),
    ]);
    let store = MemoryStore::default();
    let open = |model: &MockCompletionModel| {
        let (mut app, wakes) = app_on(&store);
        app.insert_resource(Settings::default())
            .insert_resource(scripted_models([("test/model", model)]))
            .add_plugins((AgentPlugin, SessionLogPlugin, ReadTool, EditTool, ShellTool))
            .init_resource::<Ended>()
            .add_observer(log_ended);
        tui_view(&mut app);
        app.finish();
        app.cleanup();
        app.update();
        (app, wakes)
    };
    let (mut live, wakes) = open(&model);
    let world = live.world_mut();
    let agent = (
        Agent,
        ModelChoice("test/model".to_owned()),
        WorkDir(dir.clone()),
    );
    let agent = world.spawn(agent).id();
    world.trigger(Deliver::new(agent, "go", DeliveryMode::Queue));
    run_until(&mut live, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    live.update();
    let rows = transcript(&mut live, agent);
    drop(live);
    let (mut resumed, _) = open(&MockCompletionModel::from_stream_turns([reply("unused")]));
    let agent = first_agent(resumed.world_mut()).unwrap_or(Entity::PLACEHOLDER);
    let resumed_rows = transcript(&mut resumed, agent);
    for rows in [rows, resumed_rows] {
        let has = |text: &str| rows.iter().any(|row| row.contains(text));
        // The read's count and numbered lines, the edit's diff and how the
        // command ended.
        assert!(
            has("read notes.txt 3 lines") && has("     2  two"),
            "{rows:#?}"
        );
        assert!(has("@@ -1,3 +1,3 @@"), "{rows:#?}");
        let shell = rows.iter().find(|row| row.contains("$ echo out; exit 3"));
        let ended = shell.is_some_and(|row| row.trim_end().ends_with("· exit 3"));
        assert!(ended, "{rows:#?}");
    }
    std::fs::remove_dir_all(&dir).ok();
}

/// A message sent while one before it still waits for its files waits
/// too, so the model reads them in the order they were sent.
#[test]
fn a_message_does_not_pass_one_whose_files_are_read() {
    let slow = Slow::default();
    let released = slow.released.clone();
    let (mut app, wakes, agent) = attaching(Some(Arc::new(slow)), Path::new("/nowhere"), "");
    let first = "FIRST_MESSAGE @notes.txt";
    app.world_mut()
        .spawn((Incoming::new(first, DeliveryMode::Steer), ForAgent(agent)));
    app.update();
    let second = app
        .world_mut()
        .spawn((
            Incoming::new("SECOND_MESSAGE", DeliveryMode::Steer),
            ForAgent(agent),
        ))
        .id();
    app.update();
    let held = app.world().get::<Holds>(second).is_some();
    assert!(held, "the second message waits for the first");
    released.store(true, std::sync::atomic::Ordering::Relaxed);
    run_until(&mut app, &wakes, |world| {
        said(world, agent).contains("SECOND_MESSAGE")
    });
    let said = said(app.world(), agent);
    let at = |text: &str| said.find(text);
    assert!(
        at("FIRST_MESSAGE").is_some() && at("FIRST_MESSAGE") < at("SECOND_MESSAGE"),
        "{said}"
    );
}

/// Stopping the turn while a message waits for its files, and another
/// behind it, puts both back in the input, in the order they were sent.
#[test]
fn a_stop_recalls_every_held_message_in_order() {
    let slow = Slow::default();
    let released = slow.released.clone();
    let (mut app, _wakes, agent) = attaching(Some(Arc::new(slow)), Path::new("/nowhere"), "");
    for text in ["FIRST_MESSAGE @notes.txt", "SECOND_MESSAGE"] {
        let message = (Incoming::new(text, DeliveryMode::Steer), ForAgent(agent));
        app.world_mut().spawn(message);
        app.update();
    }
    app.world_mut().trigger(StopTurn::new(agent));
    app.update();
    released.store(true, std::sync::atomic::Ordering::Relaxed);
    let recalled = &app.world().resource::<Collected<Recalled>>().0;
    let back: Vec<&str> = recalled.iter().map(|back| back.text.as_str()).collect();
    assert_eq!(back, ["FIRST_MESSAGE @notes.txt\n\nSECOND_MESSAGE"]);
}

/// A message that is not the user's own, such as a skill's, does not wait
/// behind one whose files are read, so a stop that puts the user's back
/// in the input leaves none of it waiting.
#[test]
fn a_skills_message_is_not_left_waiting_by_a_stop() {
    let slow = Slow::default();
    let released = slow.released.clone();
    let (mut app, wakes, agent) = attaching(Some(Arc::new(slow)), Path::new("/nowhere"), "");
    let first = (
        Incoming::new("FIRST_MESSAGE @notes.txt", DeliveryMode::Steer),
        ForAgent(agent),
    );
    app.world_mut().spawn(first);
    app.update();
    let mut skill = Incoming::new("SKILL_BODY", DeliveryMode::Steer);
    skill.author = Author::user().titled("skill");
    let skill = app.world_mut().spawn((skill, ForAgent(agent))).id();
    app.update();
    app.world_mut().trigger(StopTurn::new(agent));
    app.update();
    released.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(app.world().get::<Holds>(skill).is_none());
    run_until(&mut app, &wakes, |world| {
        said(world, agent).contains("SKILL_BODY")
    });
    assert!(said(app.world(), agent).contains("SKILL_BODY"));
}

/// A skill's message whose own files were read while it waited behind a
/// user's message goes on once a stop puts the user's back in the input.
#[test]
fn a_read_skills_message_goes_on_after_a_stop() {
    let slow = Slow::default();
    let released = slow.released.clone();
    let (mut app, wakes, agent) = attaching(Some(Arc::new(slow)), Path::new("/nowhere"), "");
    let first = (
        Incoming::new("FIRST_MESSAGE @notes.txt", DeliveryMode::Steer),
        ForAgent(agent),
    );
    app.world_mut().spawn(first);
    app.update();
    let mut skill = Incoming::new("SKILL_BODY @fast.txt", DeliveryMode::Steer);
    skill.author = Author::user().titled("skill");
    let skill = app.world_mut().spawn((skill, ForAgent(agent))).id();
    // Its own files are read; it waits only for the first.
    let read = |world: &World| {
        let holds = world.try_query::<(&HoldOf, &super::ReadFiles)>();
        holds.is_some_and(|mut holds| holds.iter(world).any(|(&HoldOf(on), _)| on == skill))
    };
    run_until(&mut app, &wakes, read);
    assert!(app.world().get::<Holds>(skill).is_some(), "it waits");
    app.world_mut().trigger(StopTurn::new(agent));
    run_until(&mut app, &wakes, |world| {
        said(world, agent).contains("SKILL_BODY")
    });
    released.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(said(app.world(), agent).contains("SKILL_BODY"));
}

/// A message held behind one whose files are on a host has its own files
/// on this machine attached at once, for the plugins after
/// [`AttachSystems`], such as one that redacts secrets; only its delivery
/// waits.
#[test]
fn a_held_messages_local_files_are_attached_at_once() {
    let exe = std::env::current_exe().unwrap_or_default();
    let dir = exe.with_file_name(format!("attach-local-{}", std::process::id()));
    std::fs::create_dir_all(&dir).ok();
    std::fs::write(dir.join("fast_local.txt"), "LOCAL_TEXT").ok();
    let slow = Slow::default();
    let released = slow.released.clone();
    let (mut app, wakes, agent) = attaching(Some(Arc::new(slow)), &dir, "");
    app.world_mut().spawn((
        Incoming::new("FIRST_MESSAGE @notes.txt", DeliveryMode::Steer),
        ForAgent(agent),
    ));
    app.update();
    let second = (
        Incoming::new("SECOND_MESSAGE @fast_local.txt", DeliveryMode::Steer),
        ForAgent(agent),
    );
    let second = app.world_mut().spawn(second).id();
    app.update();
    let held = app.world().get::<Holds>(second).is_some();
    let attached = app.world().get::<Incoming>(second).is_some_and(|message| {
        (message.parts.iter()).any(|part| matches!(part, MessagePart::Attachment(_)))
    });
    released.store(true, std::sync::atomic::Ordering::Relaxed);
    assert!(held && attached, "held {held}, attached {attached}");
    run_until(&mut app, &wakes, |world| {
        said(world, agent).contains("SECOND_MESSAGE")
    });
    std::fs::remove_dir_all(&dir).ok();
}
