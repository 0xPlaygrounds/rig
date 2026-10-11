//! The project context in every agent's system prompt: the instruction
//! files (`AGENTS.md`, or `CLAUDE.md`) of `RIG_HOME` and of the working
//! directory and each directory above it, and the environment (working
//! directory, platform, date). Both are
//! [`PromptSection`]s, re-read when a turn starts and by `/context`, so an
//! edited `AGENTS.md` counts from the next message without a rebuild.
//! A section only changes when its text does, so the prompt stays cached.
//! An agent with no [`SystemPrompt`] of its own is told it is rig, a coding
//! agent ([`DEFAULT_SYSTEM_PROMPT`]), at the head of its system prompt.
//! `/context` lists the prompt's sections and what other plugins put in
//! the context besides, as [`ContextEntry`]s.

use std::path::PathBuf;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_tools::context::Instructions;

use rig_ecs::agent::{Notice, SystemPrompt, TurnOf};
use rig_ecs::commands::{AppCommandsExt, CommandArgs};
use rig_ecs::prompt::PromptSection;
use rig_ecs::stage::{Admitted, AgentPass, ForAgent, Stage};
use rig_ecs::turn::{BuildFrom, ModelRequest, TurnCall};
use rig_harness::services::{ProjectRoot, RigHome};

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

/// Spawns the project and environment [`PromptSection`]s, re-reads them
/// when a turn starts, adds `/context`, and gives an agent with no prompt
/// of its own rig's ([`DEFAULT_SYSTEM_PROMPT`]).
#[derive(Default)]
pub struct ProjectContextPlugin;

impl Plugin for ProjectContextPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RigHome>()
            .init_resource::<ProjectRoot>();
        let world = app.world();
        let context = Context::read(world.resource(), world.resource());
        for (name, which) in [
            ("prompt:project", Section::Project),
            ("prompt:environment", Section::Environment),
        ] {
            let section = context.section(which);
            app.world_mut().spawn((Name::new(name), which, section));
        }
        app.add_command(
            "context",
            "Re-read AGENTS.md and the environment, and list what the prompt holds",
            show_context,
        )
        .add_observer(refresh_on_turn)
        .add_systems(AgentPass, default_prompt.in_set(Stage::Prepare));
    }
}

/// Puts [`DEFAULT_SYSTEM_PROMPT`] at the head of the system prompt of each
/// model call the kernel built from an agent with no prompt of its own,
/// as its own part of it.
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

/// Which of the context's [`PromptSection`]s this is.
#[derive(Component, Clone, Copy, PartialEq, Eq)]
enum Section {
    /// The instruction files.
    Project,
    /// The working directory, platform and date.
    Environment,
}

/// What the context sections are read from.
struct Context {
    /// The instruction files of `RIG_HOME` and of the working directory and
    /// each directory above it.
    instructions: Instructions,
    cwd: PathBuf,
}

impl Context {
    /// Reads the instruction files of `home` and of `root` and above, and
    /// the environment, now.
    fn read(home: &RigHome, root: &ProjectRoot) -> Self {
        Self {
            instructions: Instructions::discover([home.0.root().to_path_buf()], &root.0),
            cwd: root.0.clone(),
        }
    }

    /// The section `which`; the instruction files go from the most general
    /// to the most specific.
    fn section(&self, which: Section) -> PromptSection {
        if which == Section::Project {
            return PromptSection::new(
                PromptSection::ORDER_PROJECT,
                "project_instructions",
                self.instructions.prompt_text(),
            );
        }
        let lines = [
            format!("Working directory: {}", self.cwd.display()),
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

    /// Puts what was read in `sections`; whether the instruction files
    /// changed.
    fn apply(&self, sections: &mut Query<(&Section, &mut PromptSection)>) -> bool {
        let mut changed = false;
        for (which, mut section) in sections {
            changed |= section.set_if_neq(self.section(*which)) && *which == Section::Project;
        }
        changed
    }
}

/// Re-reads the sections when a turn starts, before its first model call,
/// and says so when the instruction files changed.
fn refresh_on_turn(
    started: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    mut sections: Query<(&Section, &mut PromptSection)>,
    home: Res<RigHome>,
    root: Res<ProjectRoot>,
    mut notices: MessageWriter<Notice>,
) {
    let context = Context::read(&home, &root);
    if !context.apply(&mut sections) {
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
/// every section of the system prompt and the [`ContextEntry`]s.
fn show_context(
    In(args): In<CommandArgs>,
    mut sections: Query<(&Section, &mut PromptSection)>,
    others: Query<&PromptSection, Without<Section>>,
    entries: Query<&ContextEntry>,
    home: Res<RigHome>,
    root: Res<ProjectRoot>,
    mut notices: MessageWriter<Notice>,
) {
    let context = Context::read(&home, &root);
    context.apply(&mut sections);
    let mut listed: Vec<&PromptSection> = sections.iter().map(|(_, section)| section).collect();
    listed.extend(&others);
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
    lines.push(context.section(Section::Environment).text);
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
