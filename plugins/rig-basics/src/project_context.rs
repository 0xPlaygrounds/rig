//! The project context in every agent's system prompt: the instruction
//! files (`AGENTS.md`, or `CLAUDE.md`) of `RIG_HOME` and of the project's
//! directory and each directory above it, in the app's
//! [`ProjectWorkspace`], and each agent's environment (its working
//! directory, its [`WorkDir`] or else the project's, the host when the
//! workspace is not this machine, the platform and the date). Both are
//! [`PromptSection`]s, the environment one per agent, re-read when a turn
//! starts, by `/context` and by any plugin that triggers
//! [`RefreshProjectContext`], so an edited `AGENTS.md` counts from the next
//! message without a rebuild. How much of the files is read is the user's
//! `[context]` setting ([`ContextLimits`]); a file cut short says so in the
//! prompt and in a notice. A section only changes when its text does,
//! so the prompt stays cached. An agent with no
//! [`SystemPrompt`](rig_ecs::agent::SystemPrompt) of its own is told it is
//! rig, a coding agent ([`DEFAULT_SYSTEM_PROMPT`]), at the head of its
//! system prompt. `/context` lists the prompt's sections and what other
//! plugins put in the context besides, as [`ContextEntry`]s.

use std::path::Path;
use std::sync::Arc;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_core::tool::ToolExecutionError;
use rig_tools::blocking;
use rig_tools::context::{Instructions, Limits};
use rig_tools::workspace::Workspace;

use rig_ecs::agent::{Agent, Notice, TurnOf};
use rig_ecs::prompt::{AgentSections, PromptSection, SectionOf};
use rig_ecs::stage::{Admitted, AgentPass, HoldOf, PrepareSystems};
use rig_ecs::tasks::{Running, Wake};
use rig_ecs::turn::{ModelRequest, SideCall, TurnCall};
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::services::{ProjectRoot, RigHome, WorkDir};
use rig_harness::settings::AppSettingsExt;
use rig_harness::workspace::ProjectWorkspace;
use serde::Deserialize;

/// What an agent with no [`SystemPrompt`](rig_ecs::agent::SystemPrompt) of its own is told it is.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are rig, a coding agent. You work in the \
    user's project from their terminal: \
    you read and search code, edit files and run commands with the tools you are \
    given, and answer questions about the code.\n\
    \n\
    - Work in the working directory named below, unless the user says otherwise.\n\
    - Read a file before you edit it. Change what was asked, in the style of the \
    code around it, and nothing else.\n\
    - After a change, check it when you can: build it, run the tests, or run the \
    code.\n\
    - When something fails, read the error and fix the cause. Ask the user when the \
    request is unclear or you are blocked, rather than guess.\n\
    - Do not undo changes you did not make, and do not run commands that delete \
    work, rewrite history or reach outside the project unless the user asked.\n\
    - Keep answers short. Say what you changed and what is left, and name files by \
    their path.\n\
    - Give that answer once, at the end of the turn: no running commentary or \
    interim summaries between tool calls.";

/// Re-reads the project's instruction files and every agent's environment
/// into their prompt sections, saying so when the files changed. rig-basics
/// triggers it as each turn starts and on `/context`; a plugin that changes
/// an instruction file triggers it too:
///
/// ```
/// use rig_basics::RefreshProjectContext;
/// use rig_harness::prelude::*;
///
/// fn after_saving_a_rule(mut commands: Commands) {
///     // ... the rule was appended to AGENTS.md
///     commands.trigger(RefreshProjectContext);
/// }
/// ```
///
/// In this machine's workspace the files are read at once. In another,
/// they are read off the main thread, and every model call built from an
/// agent meanwhile waits for them (a [`HoldOf`] it).
#[derive(Event, Clone, Copy, Debug, Default)]
pub struct RefreshProjectContext;

/// How much of the instruction files goes in the prompt: `[context]` in
/// `settings.toml`, each limit rig-tools' default when left out.
///
/// ```toml
/// [context]
/// file_bytes = 65536   # of one file; 32 KiB by default
/// total_bytes = 131072 # of all of them; 64 KiB by default
/// ```
#[derive(Resource, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[serde(transparent)]
pub struct ContextLimits(pub Limits);

/// Spawns the project [`PromptSection`] and each agent's environment one,
/// re-reads them when a turn starts and on [`RefreshProjectContext`], adds
/// `/context`, and gives an agent with no prompt of its own rig's
/// ([`DEFAULT_SYSTEM_PROMPT`]).
#[derive(Default)]
pub struct ProjectContextPlugin;

impl Plugin for ProjectContextPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .add_setting::<ContextLimits>("context");
        let section = PromptSection::new(PromptSection::ORDER_PROJECT, PROJECT, "");
        let project = (Name::new("prompt:project"), Project::default(), section);
        app.world_mut().spawn(project);
        app.add_command(
            "context",
            "Re-read AGENTS.md and the environment, and list what the prompt holds",
            show_context,
        )
        .add_observer(refresh_on_turn)
        .add_observer(refresh)
        .add_observer(read_off_thread)
        .add_observer(environment_of_agent)
        .add_systems(
            AgentPass,
            (default_prompt, wait_for_instructions).in_set(PrepareSystems::Compose),
        );
    }

    /// Reads the instruction files once every plugin is in, so the
    /// workspace is the one the app runs in: at once in this machine's,
    /// else as the first turn starts.
    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        let (Some(home), Some(root), Some(workspace), Some(limits)) = (
            world.get_resource::<RigHome>(),
            world.get_resource::<ProjectRoot>(),
            world.get_resource::<ProjectWorkspace>(),
            world.get_resource::<ContextLimits>(),
        ) else {
            return;
        };
        if workspace.0.host().is_some() {
            return;
        }
        let read = discover(home.0.root(), &root.0, &*workspace.0, *limits);
        let mut projects = world.query::<(&mut PromptSection, &mut Project)>();
        let mut cut = Vec::new();
        for (mut section, mut project) in projects.iter_mut(world) {
            project.put(&mut section, &read);
            cut.extend(project.cut_notices());
        }
        world.write_message_batch(cut);
    }
}

/// Makes [`DEFAULT_SYSTEM_PROMPT`] the role of the system prompt of each
/// model call the kernel built from an agent with no prompt of its own, in
/// [`PrepareSystems::Compose`], so every plugin that edits the prompt sees
/// it whole.
fn default_prompt(
    mut calls: Query<&mut ModelRequest, (Added<Admitted>, Or<(With<TurnCall>, With<SideCall>)>)>,
) {
    for mut call in &mut calls {
        if let Some(prompt) = &mut call.prompt
            && prompt.role.is_empty()
        {
            prompt.role = DEFAULT_SYSTEM_PROMPT.to_owned();
        }
    }
}

/// What a plugin puts in the context besides the system prompt's
/// sections, for `/context` to list, such as the rules it adds to tool
/// results: spawn one and keep its `text` current. Entries are listed
/// after the sections, by `tag`; an empty `text`, or Bevy's `Disabled` on
/// the entity, leaves one out.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Clone, Debug, Default)]
pub struct ContextEntry {
    /// What it is, such as `nested rules`.
    pub tag: String,
    /// What `/context` says about it; it may take several lines.
    pub text: String,
}

impl ContextEntry {
    /// `text` about `tag`.
    pub fn new(tag: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            text: text.into(),
        }
    }
}

/// The tag of the instruction files' [`PromptSection`].
const PROJECT: &str = "project_instructions";

/// The tag of an agent's environment [`PromptSection`].
const ENVIRONMENT: &str = "environment";

/// On the instruction files' [`PromptSection`]: what was read last.
#[derive(Component, Default)]
struct Project {
    /// The files' paths, as [`Instructions::paths`] says.
    paths: String,
    /// Files that exist but could not be read, with why.
    unreadable: Vec<String>,
    /// Files the limits cut: each one's path, the bytes kept and its size.
    cut: Vec<(String, usize, u64)>,
}

impl Project {
    /// Puts `read` in the project's `section`; whether its text changed.
    fn put(&mut self, section: &mut Mut<PromptSection>, read: &Instructions) -> bool {
        self.paths = read.paths();
        self.unreadable.clone_from(&read.unreadable);
        let cut = read.files.iter().filter(|file| file.contents.cut());
        self.cut = cut
            .map(|file| {
                let contents = &file.contents;
                let path = file.path.display().to_string();
                (path, contents.text.len(), contents.size)
            })
            .collect();
        let text = read.prompt_text();
        section.set_if_neq(PromptSection::new(
            PromptSection::ORDER_PROJECT,
            PROJECT,
            text,
        ))
    }

    /// The notices of a read that changed the files: which were read, and
    /// which could not be.
    fn notices(&self) -> Vec<Notice> {
        let mut notices = vec![Notice::info(
            None,
            format!("Instruction files re-read: {}.", self.paths),
        )];
        let unreadable = self.unreadable.iter();
        notices.extend(
            unreadable.map(|problem| Notice::error(None, format!("Cannot read {problem}."))),
        );
        notices.extend(self.cut_notices());
        notices
    }

    /// A warning for each file the limits cut.
    fn cut_notices(&self) -> impl Iterator<Item = Notice> + '_ {
        self.cut.iter().map(|(path, kept, size)| {
            let text = format!(
                "{path} is cut to {kept} of its {size} bytes in the prompt; `file_bytes` and \
                 `total_bytes` under [context] in settings.toml set how much is read."
            );
            Notice::warn(None, text)
        })
    }
}

/// On an agent's environment [`PromptSection`] ([`SectionOf`] it).
#[derive(Component)]
struct Environment;

/// The instruction files read off the main thread, for a workspace that
/// is not this machine; every model call admitted meanwhile is held by a
/// child of it.
#[derive(Component)]
struct Reading;

/// On a [`Reading`] asked for again while it ran: what it reads may be
/// out of date, so it reads again when it lands.
#[derive(Component)]
struct Stale;

/// What [`Reading`] read.
#[derive(Component)]
struct Read(Result<Instructions, ToolExecutionError>);

impl From<Result<Instructions, ToolExecutionError>> for Read {
    fn from(read: Result<Instructions, ToolExecutionError>) -> Self {
        Self(read)
    }
}

/// The instruction files of `home` on this machine and of `root` and
/// above in `workspace`, read now within `limits`.
fn discover(
    home: &Path,
    root: &Path,
    workspace: &dyn Workspace,
    limits: ContextLimits,
) -> Instructions {
    Instructions::discover_with([home.to_path_buf()], root, workspace, limits.0)
}

/// The environment section of an agent working in `dir`, in the
/// workspace on `host` (`None`: this machine).
fn environment(dir: &Path, host: Option<&str>) -> PromptSection {
    let mut lines = vec![format!("Working directory: {}", dir.display())];
    match host {
        Some(host) => lines.push(format!("Host: {host}, not this machine")),
        None => lines.push(format!(
            "Platform: {} ({})",
            std::env::consts::OS,
            std::env::consts::ARCH
        )),
    }
    lines.push(format!(
        "Today's date: {}",
        chrono::Local::now().format("%Y-%m-%d")
    ));
    PromptSection::new(
        PromptSection::ORDER_ENVIRONMENT,
        ENVIRONMENT,
        lines.join("\n"),
    )
}

/// Gives a new agent its environment section, and rewrites it when its
/// [`WorkDir`] changes. Applied as a command, so a section spawned for an
/// agent's [`Agent`] is found when its inherited `WorkDir` follows.
fn environment_of_agent(changed: On<Insert<(Agent, WorkDir)>>, mut commands: Commands) {
    let agent = changed.entity;
    commands.queue(move |world: &mut World| {
        let (Some(root), Some(workspace)) = (
            world.get_resource::<ProjectRoot>(),
            world.get_resource::<ProjectWorkspace>(),
        ) else {
            return;
        };
        let dir = world.get::<WorkDir>(agent).map_or(&root.0, |dir| &dir.0);
        let section = environment(dir, workspace.0.host());
        let sections = world.get::<AgentSections>(agent);
        let mut sections = sections.into_iter().flat_map(|sections| sections.iter());
        let mine = sections.find(|&of| world.get::<Environment>(of).is_some());
        match mine {
            Some(mine) => {
                if let Some(mut current) = world.get_mut::<PromptSection>(mine) {
                    current.set_if_neq(section);
                }
            }
            None if world.get::<Agent>(agent).is_some() => {
                let name = Name::new("prompt:environment");
                world.spawn((name, Environment, section, SectionOf(agent)));
            }
            None => {}
        }
    });
}

/// Re-reads the context when a turn starts, before its first model call.
fn refresh_on_turn(_: On<Add<TurnOf>>, mut commands: Commands) {
    commands.trigger(RefreshProjectContext);
}

/// Re-reads the context ([`RefreshProjectContext`]): the environments at
/// once, and the instruction files at once in this machine's workspace,
/// else off the main thread; a read that runs then reads again when it
/// lands ([`Stale`]).
fn refresh(
    _: On<RefreshProjectContext>,
    mut projects: Query<(&mut PromptSection, &mut Project), Without<Environment>>,
    mut environments: Query<(&SectionOf, &mut PromptSection), With<Environment>>,
    agents: Query<Option<&WorkDir>, With<Agent>>,
    reading: Query<Entity, With<Reading>>,
    (home, root, workspace, wake, limits): (
        Res<RigHome>,
        Res<ProjectRoot>,
        Res<ProjectWorkspace>,
        Res<Wake>,
        Res<ContextLimits>,
    ),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let host = workspace.0.host();
    for (&SectionOf(agent), mut section) in &mut environments {
        if let Ok(dir) = agents.get(agent) {
            let dir = dir.map_or(&root.0, |dir| &dir.0);
            section.set_if_neq(environment(dir, host));
        }
    }
    if host.is_none() {
        let read = discover(home.0.root(), &root.0, &*workspace.0, *limits);
        for (mut section, mut project) in &mut projects {
            if project.put(&mut section, &read) {
                notices.write_batch(project.notices());
            }
        }
        return;
    }
    if let Some(reading) = reading.iter().next() {
        commands.entity(reading).insert(Stale);
        return;
    }
    commands.spawn((
        Name::new("project instructions"),
        Reading,
        read_files((&home, &root, &workspace, &wake), *limits),
    ));
}

/// Reads the instruction files off the main thread within `limits`,
/// ending as a [`Read`].
fn read_files(
    (home, root, workspace, wake): (&RigHome, &ProjectRoot, &ProjectWorkspace, &Wake),
    limits: ContextLimits,
) -> Running {
    let (home, root) = (home.0.root().to_path_buf(), root.0.clone());
    let workspace: Arc<dyn Workspace> = workspace.0.clone();
    let read = blocking(move || Ok(discover(&home, &root, &*workspace, limits)));
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    Running::spawn_into::<Read, _>(pool, wake, read)
}

/// Holds each model call built from an agent while the instruction files
/// are read off the main thread, in [`PrepareSystems::Compose`].
fn wait_for_instructions(
    calls: Query<Entity, (Added<Admitted>, Or<(With<TurnCall>, With<SideCall>)>)>,
    reading: Query<Entity, With<Reading>>,
    mut commands: Commands,
) {
    let Some(reading) = reading.iter().next() else {
        return;
    };
    for call in &calls {
        let hold = (Name::new("hold: project instructions"), HoldOf(call));
        commands.entity(reading).with_child(hold);
    }
}

/// Puts the instruction files read off the main thread in the project
/// section and in the requests of the calls that waited for them, then
/// releases those calls; a [`Stale`] read reads again, the calls still
/// waiting.
fn read_off_thread(
    done: On<Insert<Read>>,
    reads: Query<(&Read, Option<&Children>, Has<Stale>)>,
    holds: Query<&HoldOf>,
    mut projects: Query<(&mut PromptSection, &mut Project)>,
    mut requests: Query<&mut ModelRequest>,
    (home, root, workspace, wake, limits): (
        Res<RigHome>,
        Res<ProjectRoot>,
        Res<ProjectWorkspace>,
        Res<Wake>,
        Res<ContextLimits>,
    ),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((Read(read), held, stale)) = reads.get(done.entity) else {
        return;
    };
    if stale {
        let again = read_files((&home, &root, &workspace, &wake), *limits);
        commands
            .entity(done.entity)
            .remove::<(Read, Stale)>()
            .insert(again);
        return;
    }
    match read {
        Ok(read) => {
            for (mut section, mut project) in &mut projects {
                if project.put(&mut section, read) {
                    notices.write_batch(project.notices());
                }
            }
            let text = read.prompt_text();
            let calls = held.into_iter().flatten();
            let calls = calls.filter_map(|&hold| holds.get(hold).ok());
            for &HoldOf(call) in calls {
                let Ok(mut request) = requests.get_mut(call) else {
                    continue;
                };
                let prompt = request.prompt.as_mut().into_iter();
                let sections = prompt.flat_map(|prompt| &mut prompt.sections);
                for section in sections.filter(|section| section.tag == PROJECT) {
                    section.text.clone_from(&text);
                }
            }
        }
        Err(failure) => {
            let why = format!("Cannot read the instruction files: {failure}");
            notices.write(Notice::error(None, why));
        }
    }
    commands.entity(done.entity).despawn();
}

/// `/context`: re-reads the context now ([`RefreshProjectContext`]), then
/// lists the instruction files, every section of the system prompt of the
/// agent that asked, its environment and the [`ContextEntry`]s.
fn show_context(In(args): In<CommandArgs>, mut commands: Commands) {
    commands.trigger(RefreshProjectContext);
    let agent = args.agent;
    commands.queue(move |world: &mut World| {
        let _ = world.run_system_cached_with(list_context, agent);
    });
}

/// Lists what `/context` shows to `agent`.
fn list_context(
    In(agent): In<Entity>,
    sections: Query<(&PromptSection, Option<&SectionOf>)>,
    projects: Query<&Project>,
    entries: Query<&ContextEntry>,
    agents: Query<Option<&WorkDir>, With<Agent>>,
    root: Res<ProjectRoot>,
    workspace: Res<ProjectWorkspace>,
    mut notices: MessageWriter<Notice>,
) {
    let mine = |of: Option<&SectionOf>| of.is_none_or(|of| of.0 == agent);
    let mut listed: Vec<&PromptSection> = sections
        .iter()
        .filter(|(_, of)| mine(*of))
        .map(|(section, _)| section)
        .collect();
    listed.sort_by(|a, b| a.order.cmp(&b.order).then_with(|| a.tag.cmp(&b.tag)));
    let mut lines = Vec::new();
    for project in &projects {
        lines.push(format!(
            "Instruction files, most general first: {}.",
            project.paths
        ));
        let unreadable = project.unreadable.iter();
        lines.extend(unreadable.map(|problem| format!("Cannot read {problem}.")));
    }
    lines.push("The system prompt's sections:".to_owned());
    for section in listed {
        lines.push(format!("  {} ({} bytes)", section.tag, section.text.len()));
    }
    let dir = agents.get(agent).ok().flatten();
    let dir = dir.map_or(&root.0, |dir| &dir.0);
    lines.push(environment(dir, workspace.0.host()).text);
    let mut entries: Vec<&ContextEntry> = entries
        .iter()
        .filter(|entry| !entry.text.is_empty())
        .collect();
    entries.sort_by(|a, b| a.tag.cmp(&b.tag));
    for entry in entries {
        lines.push(format!("{}: {}", entry.tag, entry.text));
    }
    notices.write(Notice::info(agent, lines.join("\n")));
}

#[cfg(test)]
mod tests;
