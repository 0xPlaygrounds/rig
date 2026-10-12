//! The project context in every agent's system prompt: the instruction
//! files (`AGENTS.md`, or `CLAUDE.md`) of `RIG_HOME` and of the project's
//! directory and each directory above it, and each agent's environment
//! (its working directory, its [`WorkDir`] or else the project's, the
//! platform and the date). Both are [`PromptSection`]s, the environment
//! one per agent, re-read when a turn starts and by `/context`, so an
//! edited `AGENTS.md` counts from the next message without a rebuild.
//! A section only changes when its text does, so the prompt stays cached.
//! An agent with no [`SystemPrompt`] of its own is told it is rig, a coding
//! agent ([`DEFAULT_SYSTEM_PROMPT`]), at the head of its system prompt.
//! `/context` lists the prompt's sections and what other plugins put in
//! the context besides, as [`ContextEntry`]s.

use std::path::{Path, PathBuf};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_tools::context::Instructions;
use rig_tools::workspace::Local;

use rig_ecs::agent::{Agent, Notice, SystemPrompt, TurnOf};
use rig_ecs::prompt::{AgentSections, PromptSection, SectionOf};
use rig_ecs::stage::{Admitted, AgentPass, ForAgent, PrepareSystems};
use rig_ecs::turn::{BuildFrom, ModelRequest, TurnCall};
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::services::{ProjectRoot, RigHome, WorkDir};

/// What an agent with no [`SystemPrompt`] of its own is told it is.
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

/// Spawns the project [`PromptSection`] and each agent's environment one,
/// re-reads them when a turn starts, adds `/context`, and gives an agent
/// with no prompt of its own rig's ([`DEFAULT_SYSTEM_PROMPT`]).
#[derive(Default)]
pub struct ProjectContextPlugin;

impl Plugin for ProjectContextPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RigHome>()
            .init_resource::<ProjectRoot>();
        let world = app.world();
        let context = Context::read(world.resource(), world.resource());
        let project = (Name::new("prompt:project"), Project, context.project());
        app.world_mut().spawn(project);
        app.add_command(
            "context",
            "Re-read AGENTS.md and the environment, and list what the prompt holds",
            show_context,
        )
        .add_observer(refresh_on_turn)
        .add_observer(environment_of_agent)
        .add_systems(AgentPass, default_prompt.in_set(PrepareSystems::Compose));
    }
}

/// Puts [`DEFAULT_SYSTEM_PROMPT`] at the head of the system prompt of each
/// model call the kernel built from an agent with no prompt of its own,
/// as its own part of it, in [`PrepareSystems::Compose`], so every plugin
/// that edits the prompt sees it whole.
fn default_prompt(
    mut calls: Query<
        (&mut ModelRequest, &ForAgent),
        (Added<Admitted>, Or<(With<TurnCall>, With<BuildFrom>)>),
    >,
    agents: Query<&SystemPrompt>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        if !agents.get(agent).is_ok_and(|own| own.0.is_empty()) {
            continue;
        }
        if let Some(prompt) = call.system_prompt_mut() {
            *prompt = match prompt.is_empty() {
                true => DEFAULT_SYSTEM_PROMPT.to_owned(),
                false => format!("{DEFAULT_SYSTEM_PROMPT}\n\n{prompt}"),
            };
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

/// On the instruction files' [`PromptSection`].
#[derive(Component)]
struct Project;

/// On an agent's environment [`PromptSection`] ([`SectionOf`] it).
#[derive(Component)]
struct Environment;

/// The instruction files of `RIG_HOME` and of the project's directory and
/// each directory above it.
struct Context {
    instructions: Instructions,
    root: PathBuf,
}

impl Context {
    /// Reads the instruction files of `home` and of `root` and above now.
    fn read(home: &RigHome, root: &ProjectRoot) -> Self {
        Self {
            instructions: Instructions::discover([home.0.root().to_path_buf()], &root.0, &Local),
            root: root.0.clone(),
        }
    }

    /// The instruction files' section, from the most general to the most
    /// specific.
    fn project(&self) -> PromptSection {
        PromptSection::new(
            PromptSection::ORDER_PROJECT,
            "project_instructions",
            self.instructions.prompt_text(),
        )
    }

    /// The environment section of an agent working in `dir`, else in the
    /// project's directory.
    fn environment(&self, dir: Option<&WorkDir>) -> PromptSection {
        environment(dir.map_or(&self.root, |dir| &dir.0))
    }

    /// Puts what was read in the project section and the environment
    /// sections of `agents`, by their [`WorkDir`]s; whether the
    /// instruction files changed.
    fn apply(
        &self,
        sections: &mut Sections,
        agents: &Query<Option<&WorkDir>, With<Agent>>,
    ) -> bool {
        let mut changed = false;
        for (of, mut section) in sections {
            let Some(SectionOf(agent)) = of else {
                changed |= section.set_if_neq(self.project());
                continue;
            };
            if let Ok(dir) = agents.get(*agent) {
                section.set_if_neq(self.environment(dir));
            }
        }
        changed
    }
}

/// The context's sections: the project one, without a [`SectionOf`], and
/// the agents' environments.
type Sections<'w, 's> = Query<
    'w,
    's,
    (Option<&'static SectionOf>, &'static mut PromptSection),
    Or<(With<Project>, With<Environment>)>,
>;

/// The environment section of an agent working in `dir`.
fn environment(dir: &Path) -> PromptSection {
    let lines = [
        format!("Working directory: {}", dir.display()),
        format!(
            "Platform: {} ({})",
            std::env::consts::OS,
            std::env::consts::ARCH
        ),
        format!("Today's date: {}", chrono::Local::now().format("%Y-%m-%d")),
    ];
    PromptSection::new(
        PromptSection::ORDER_ENVIRONMENT,
        "environment",
        lines.join("\n"),
    )
}

/// Gives a new agent its environment section, and rewrites it when its
/// [`WorkDir`] changes. Applied as a command, so a section spawned for an
/// agent's [`Agent`] is found when its inherited `WorkDir` follows.
fn environment_of_agent(changed: On<Insert<(Agent, WorkDir)>>, mut commands: Commands) {
    let agent = changed.entity;
    commands.queue(move |world: &mut World| {
        let Some(root) = world.get_resource::<ProjectRoot>() else {
            return;
        };
        let dir = world.get::<WorkDir>(agent).map_or(&root.0, |dir| &dir.0);
        let section = environment(dir);
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

/// Re-reads the sections when a turn starts, before its first model call,
/// and says so when the instruction files changed.
fn refresh_on_turn(
    started: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    mut sections: Sections,
    agents: Query<Option<&WorkDir>, With<Agent>>,
    home: Res<RigHome>,
    root: Res<ProjectRoot>,
    mut notices: MessageWriter<Notice>,
) {
    let context = Context::read(&home, &root);
    if !context.apply(&mut sections, &agents) {
        return;
    }
    let agent = turns.get(started.entity).ok().map(|of| of.0);
    notices.write(Notice::info(
        agent,
        format!(
            "Instruction files re-read: {}.",
            context.instructions.paths()
        ),
    ));
    for problem in context.instructions.unreadable {
        notices.write(Notice::error(agent, format!("Cannot read {problem}.")));
    }
}

/// `/context`: re-reads the sections now and lists the instruction files,
/// every section of the system prompt of the agent that asked, its
/// environment and the [`ContextEntry`]s.
fn show_context(
    In(args): In<CommandArgs>,
    mut sections: Sections,
    agents: Query<Option<&WorkDir>, With<Agent>>,
    others: Query<(&PromptSection, Option<&SectionOf>), (Without<Project>, Without<Environment>)>,
    entries: Query<&ContextEntry>,
    home: Res<RigHome>,
    root: Res<ProjectRoot>,
    mut notices: MessageWriter<Notice>,
) {
    let context = Context::read(&home, &root);
    context.apply(&mut sections, &agents);
    let mine = |of: Option<&SectionOf>| of.is_none_or(|of| of.0 == args.agent);
    let ours = sections.iter().filter(|(of, _)| mine(*of));
    let mut listed: Vec<&PromptSection> = ours.map(|(_, section)| section).collect();
    listed.extend(
        others
            .iter()
            .filter(|(_, of)| mine(*of))
            .map(|(section, _)| section),
    );
    listed.sort_by(|a, b| a.order.cmp(&b.order).then_with(|| a.tag.cmp(&b.tag)));
    let mut lines = vec![format!(
        "Instruction files, most general first: {}.",
        context.instructions.paths()
    )];
    let unreadable = context.instructions.unreadable.iter();
    lines.extend(unreadable.map(|problem| format!("Cannot read {problem}.")));
    lines.push("The system prompt's sections:".to_owned());
    for section in listed {
        lines.push(format!("  {} ({} bytes)", section.tag, section.text.len()));
    }
    let dir = agents.get(args.agent).ok().flatten();
    lines.push(context.environment(dir).text);
    let mut entries: Vec<&ContextEntry> = entries
        .iter()
        .filter(|entry| !entry.text.is_empty())
        .collect();
    entries.sort_by(|a, b| a.tag.cmp(&b.tag));
    for entry in entries {
        lines.push(format!("{}: {}", entry.tag, entry.text));
    }
    notices.write(Notice::info(args.agent, lines.join("\n")));
}

#[cfg(test)]
mod tests;
