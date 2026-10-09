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
use rig::harness_protocol::Home;
use rig_tools::context::{FILE_NAMES, Instructions};

use crate::core::agent::{Notice, TurnOf};
use crate::core::commands::{AppCommandsExt, CommandArgs};
use crate::core::prompt::PromptSection;

/// Spawns the project and environment [`PromptSection`]s, re-reads them
/// when a turn starts, and adds `/context`.
pub struct ProjectContextPlugin;

impl Plugin for ProjectContextPlugin {
    fn build(&self, app: &mut App) {
        let context = Context::read();
        app.world_mut().spawn((
            Name::new("prompt:project"),
            ProjectSection,
            context.project_section(),
        ));
        app.world_mut().spawn((
            Name::new("prompt:environment"),
            EnvironmentSection,
            context.environment_section(),
        ));
        app.add_command(
            "context",
            "Re-read AGENTS.md and the environment, and list what the prompt holds",
            show_context,
        )
        .add_observer(refresh_on_turn);
    }
}

/// Marks the [`PromptSection`] of the instruction files.
#[derive(Component)]
struct ProjectSection;

/// Marks the [`PromptSection`] of the environment.
#[derive(Component)]
struct EnvironmentSection;

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

    /// The instruction files, from the most general to the most specific.
    fn project_section(&self) -> PromptSection {
        PromptSection::new(
            PromptSection::ORDER_PROJECT,
            "project_instructions",
            self.instructions.prompt_text(),
        )
    }

    /// The working directory, platform and date.
    fn environment_section(&self) -> PromptSection {
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
}

/// Re-reads the sections when a turn starts, before its first model call,
/// and says so when the instruction files changed.
fn refresh_on_turn(
    started: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    mut project: Query<&mut PromptSection, (With<ProjectSection>, Without<EnvironmentSection>)>,
    mut environment: Query<&mut PromptSection, (With<EnvironmentSection>, Without<ProjectSection>)>,
    mut notices: MessageWriter<Notice>,
) {
    let context = Context::read();
    if let Ok(mut section) = environment.single_mut() {
        section.set_if_neq(context.environment_section());
    }
    let Ok(mut section) = project.single_mut() else {
        return;
    };
    if !section.set_if_neq(context.project_section()) {
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

/// `/context`: re-reads the sections now and lists what they hold.
fn show_context(
    In(args): In<CommandArgs>,
    mut project: Query<&mut PromptSection, (With<ProjectSection>, Without<EnvironmentSection>)>,
    mut environment: Query<&mut PromptSection, (With<EnvironmentSection>, Without<ProjectSection>)>,
    mut notices: MessageWriter<Notice>,
) {
    let context = Context::read();
    let environment_section = context.environment_section();
    let environment_text = environment_section.text.clone();
    if let Ok(mut section) = environment.single_mut() {
        section.set_if_neq(environment_section);
    }
    if let Ok(mut section) = project.single_mut() {
        section.set_if_neq(context.project_section());
    }
    let mut lines = vec!["Instruction files, most general first:".to_owned()];
    if context.instructions.files.is_empty() {
        lines.push(format!(
            "  none; put an {} in the project or in {}",
            FILE_NAMES.join(" or "),
            Home::from_env().root().display()
        ));
    }
    for file in &context.instructions.files {
        let cut = if file.contents.cut() {
            format!(", cut to {} bytes", file.contents.text.len())
        } else {
            String::new()
        };
        lines.push(format!(
            "  {} ({} bytes{cut})",
            file.path.display(),
            file.contents.size
        ));
    }
    for problem in &context.instructions.unreadable {
        lines.push(format!("  cannot read {problem}"));
    }
    lines.push(environment_text);
    notices.write(Notice::info(args.agent, lines.join("\n")));
}
