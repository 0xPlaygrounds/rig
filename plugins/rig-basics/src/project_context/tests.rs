use std::ffi::{OsStr, OsString};
use std::io;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{Receiver, Sender, channel};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use rig_core::test_utils::MockCompletionModel;
use rig_ecs::prelude::*;
use rig_harness_test_support::{Ended, MemoryStore, app_on, connect, log_ended, reply, run_until};
use rig_tools::process::{Launcher, Process, Spawn};
use rig_tools::workspace::{Contents, Stat, Workspace};

use rig_harness::commands::RunCommand;
use rig_harness::harness_protocol::Home;
use rig_harness::services::WorkDir;

use super::{
    ContextEntry, DEFAULT_SYSTEM_PROMPT, PROJECT, ProjectContextPlugin, ProjectRoot,
    ProjectWorkspace, Reading, RefreshProjectContext, RigHome, default_prompt,
};

#[test]
fn an_agent_without_a_prompt_of_its_own_is_told_it_is_rig() {
    let model = MockCompletionModel::from_stream_turns([reply("hi"), reply("hello")]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    app.add_plugins((AgentPlugin, JournalPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended)
        .add_systems(AgentPass, default_prompt.in_set(Stage::Prepare));
    app.finish();
    app.update();
    let Some(connection) = connect(&model) else {
        return;
    };
    let reviewer = SystemPrompt("You review code.".to_owned());
    for prompt in [SystemPrompt::default(), reviewer] {
        let world = app.world_mut();
        let agent = world.spawn((Agent, prompt, connection.clone())).id();
        world.trigger(Deliver::new(agent, "hello", DeliveryMode::Steer));
        let ended = world.resource::<Ended>().0.len() + 1;
        run_until(&mut app, &wakes, |world| {
            world.resource::<Ended>().0.len() == ended
        });
    }
    let prompts: Vec<Option<String>> = model
        .requests()
        .iter()
        .map(|request| request.system_instructions().map(str::to_owned))
        .collect();
    let told = [DEFAULT_SYSTEM_PROMPT, "You review code."].map(|prompt| Some(prompt.to_owned()));
    assert_eq!(prompts, told);
}

#[test]
fn context_lists_what_other_plugins_put_in_the_context() {
    let root = std::env::temp_dir().join(format!("rig-basics-{}", std::process::id()));
    let mut app = App::new();
    app.insert_resource(RigHome(Home::at(root.join("home"))))
        .insert_resource(ProjectRoot(root.clone()))
        .add_plugins((AgentPlugin, ProjectContextPlugin));
    let world = app.world_mut();
    world.spawn(ContextEntry::new("nested rules", "src/AGENTS.md"));
    world.spawn(ContextEntry::new("unused", ""));
    let agent = world.spawn(Agent).id();
    world.trigger(RunCommand::new(agent, "context".to_owned()));
    world.flush();
    let mut notices = world.resource_mut::<Messages<Notice>>();
    let shown: Vec<String> = notices.drain().map(|notice| notice.text).collect();
    let shown = shown.join("\n");
    assert!(shown.ends_with("\nnested rules: src/AGENTS.md"), "{shown}");
    assert!(!shown.contains("unused"), "{shown}");
}

#[test]
fn each_agent_is_told_its_own_working_directory() {
    let root = std::env::temp_dir().join(format!("rig-basics-env-{}", std::process::id()));
    let mut app = App::new();
    app.insert_resource(RigHome(Home::at(root.join("home"))))
        .insert_resource(ProjectRoot(root.clone()))
        .add_plugins((
            AgentPlugin,
            rig_harness::services::ServicesPlugin,
            ProjectContextPlugin,
        ));
    let world = app.world_mut();
    let plain = world.spawn(Agent).id();
    let worktree = root.join("worktree");
    let working = world.spawn((Agent, WorkDir(worktree.clone()))).id();
    let child = world.spawn((Agent, SpawnedBy(working))).id();
    world.flush();
    let mut sections = world.query::<(&SectionOf, &PromptSection)>();
    let told: Vec<(Entity, String)> = sections
        .iter(world)
        .filter(|(_, section)| section.tag == "environment")
        .filter_map(|(of, section)| Some((of.0, section.text.lines().next()?.to_owned())))
        .collect();
    let told = |agent: Entity| -> Vec<&str> {
        let lines = told.iter().filter(|(of, _)| *of == agent);
        lines.map(|(_, line)| line.as_str()).collect()
    };
    let line = |dir: &std::path::Path| format!("Working directory: {}", dir.display());
    assert_eq!(told(plain), [line(&root)]);
    assert_eq!(told(working), [line(&worktree)]);
    assert_eq!(told(child), [line(&worktree)]);
}

/// The project section's text.
fn project_text(world: &World) -> String {
    let mut sections = world.try_query::<&PromptSection>();
    let mut sections = sections
        .iter_mut()
        .flat_map(|sections| sections.iter(world));
    let project = sections.find(|section| section.tag == PROJECT);
    project
        .map(|section| section.text.clone())
        .unwrap_or_default()
}

#[test]
fn a_refresh_puts_an_edited_agents_md_in_the_prompt() {
    let exe = std::env::current_exe().unwrap_or_default();
    let root = exe.with_file_name(format!("refresh-{}", std::process::id()));
    std::fs::create_dir_all(&root).ok();
    std::fs::write(root.join("AGENTS.md"), "Use tabs.").ok();
    let mut app = App::new();
    app.insert_resource(RigHome(Home::at(root.join("home"))))
        .insert_resource(ProjectRoot(root.clone()))
        .add_plugins((AgentPlugin, ProjectContextPlugin));
    app.finish();
    let world = app.world_mut();
    assert!(project_text(world).contains("Use tabs."));
    std::fs::write(root.join("AGENTS.md"), "Use spaces.").ok();
    world.trigger(RefreshProjectContext);
    world.flush();
    let text = project_text(world);
    assert!(
        text.contains("Use spaces.") && !text.contains("Use tabs."),
        "{text}"
    );
    std::fs::remove_dir_all(&root).ok();
}

/// A workspace that is not this machine: it has one file,
/// `/remote/AGENTS.md`, and runs nothing.
#[derive(Debug, Default)]
struct Remote {
    /// `AGENTS.md`'s text.
    text: Mutex<String>,
    /// When set, the next read says it read, then waits to be released.
    gate: Mutex<Option<(Sender<()>, Receiver<()>)>>,
}

impl Remote {
    const RULES: &str = "/remote/AGENTS.md";

    /// A workspace whose `AGENTS.md` says `text`.
    fn saying(text: &str) -> Self {
        let remote = Self::default();
        remote.set(text);
        remote
    }

    /// Makes `AGENTS.md` say `text`.
    fn set(&self, text: &str) {
        if let Ok(mut current) = self.text.lock() {
            *current = text.to_owned();
        }
    }

    /// `AGENTS.md`'s bytes.
    fn bytes(&self) -> Vec<u8> {
        let text = self.text.lock().map(|text| text.clone());
        text.unwrap_or_default().into_bytes()
    }
}

impl Workspace for Remote {
    fn host(&self) -> Option<&str> {
        Some("ssh://dev@box")
    }

    fn stat(&self, path: &Path) -> io::Result<Stat> {
        if path == Path::new(Self::RULES) {
            return Ok(Stat::File {
                size: self.bytes().len() as u64,
                read_only: false,
                modified: None,
            });
        }
        Ok(Stat::Missing)
    }

    fn read(&self, path: &Path, _: u64) -> io::Result<Contents> {
        if path != Path::new(Self::RULES) {
            return Err(io::ErrorKind::NotFound.into());
        }
        let bytes = self.bytes();
        let gate = self.gate.lock().ok().and_then(|mut gate| gate.take());
        if let Some((read, release)) = gate {
            read.send(()).ok();
            release.recv().ok();
        }
        let size = bytes.len() as u64;
        Ok(Contents { bytes, size })
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

    fn files(&self, _: &Path, _: Option<&str>) -> io::Result<Vec<PathBuf>> {
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
fn in_another_workspace_the_first_request_waits_for_its_instruction_files() {
    let model = MockCompletionModel::from_stream_turns([reply("ok")]);
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let home = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name("no-home");
    app.insert_resource(RigHome(Home::at(home)))
        .insert_resource(ProjectRoot("/remote".into()))
        .insert_resource(ProjectWorkspace(Arc::new(Remote::saying(
            "Run the remote's tests.",
        ))))
        .add_plugins((AgentPlugin, JournalPlugin, ProjectContextPlugin))
        .init_resource::<Ended>()
        .add_observer(log_ended);
    app.finish();
    app.update();
    let Some(connection) = connect(&model) else {
        return;
    };
    let world = app.world_mut();
    let agent = world.spawn((Agent, connection)).id();
    world.trigger(Deliver::new(agent, "hello", DeliveryMode::Steer));
    run_until(&mut app, &wakes, |world| {
        !world.resource::<Ended>().0.is_empty()
    });
    let sent: Vec<String> = model
        .requests()
        .iter()
        .map(|request| request.system_instructions().unwrap_or_default().to_owned())
        .collect();
    assert_eq!(sent.len(), 1, "{sent:?}");
    let prompt = sent.first().cloned().unwrap_or_default();
    assert!(prompt.contains("Run the remote's tests."), "{prompt}");
    assert!(prompt.contains("Host: ssh://dev@box"), "{prompt}");
}

#[test]
fn a_refresh_asked_for_during_a_remote_read_is_read() {
    let (mut app, wakes) = app_on(&MemoryStore::default());
    let remote = Arc::new(Remote::saying("Use tabs."));
    let (read, was_read) = channel();
    let (release, released) = channel();
    if let Ok(mut gate) = remote.gate.lock() {
        *gate = Some((read, released));
    }
    let home = std::env::current_exe()
        .unwrap_or_default()
        .with_file_name("no-home");
    app.insert_resource(RigHome(Home::at(home)))
        .insert_resource(ProjectRoot("/remote".into()))
        .insert_resource(ProjectWorkspace(remote.clone()))
        .add_plugins((AgentPlugin, ProjectContextPlugin));
    app.finish();
    let world = app.world_mut();
    world.trigger(RefreshProjectContext);
    world.flush();
    was_read.recv_timeout(Duration::from_secs(10)).ok();
    remote.set("Use spaces.");
    world.trigger(RefreshProjectContext);
    world.flush();
    release.send(()).ok();
    run_until(&mut app, &wakes, |world| {
        let mut reading = world.try_query::<&Reading>();
        let reading = reading.iter_mut().flat_map(|reading| reading.iter(world));
        reading.count() == 0 && !project_text(world).is_empty()
    });
    let text = project_text(app.world());
    assert!(text.contains("Use spaces."), "{text}");
}
