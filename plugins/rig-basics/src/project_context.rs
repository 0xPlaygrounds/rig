//! The project context in every agent's system prompt: the instruction
//! files (`AGENTS.md`, or `CLAUDE.md`) of `RIG_HOME` and of the working
//! directory and each directory above it, and the environment (working
//! directory, platform, date). Both are
//! [`PromptSection`]s, re-read when a turn starts and by `/context`, so an
//! edited `AGENTS.md` counts from the next message without a rebuild.
//! A section only changes when its text does, so the prompt stays cached.

use std::path::PathBuf;

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig_harness::harness_protocol::Home;
use rig_tools::context::Instructions;

use rig_ecs::agent::{Notice, TurnOf};
use rig_ecs::commands::{AppCommandsExt, CommandArgs};
use rig_ecs::prompt::PromptSection;

/// Spawns the project and environment [`PromptSection`]s, re-reads them
/// when a turn starts, and adds `/context`.
#[derive(Default)]
pub struct ProjectContextPlugin;

impl Plugin for ProjectContextPlugin {
    fn build(&self, app: &mut App) {
        let context = Context::read();
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
        .add_observer(refresh_on_turn);
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
    /// Reads the instruction files and the environment now.
    fn read() -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let home = Home::from_env().root().to_path_buf();
        Self {
            instructions: Instructions::discover([home], &cwd),
            cwd,
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
    mut notices: MessageWriter<Notice>,
) {
    let context = Context::read();
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

/// `/context`: re-reads the sections now and lists the instruction files
/// and every section of the system prompt.
fn show_context(
    In(args): In<CommandArgs>,
    mut sections: Query<(&Section, &mut PromptSection)>,
    others: Query<&PromptSection, Without<Section>>,
    mut notices: MessageWriter<Notice>,
) {
    let context = Context::read();
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
    notices.write(Notice::info(args.agent, lines.join("\n")));
}
