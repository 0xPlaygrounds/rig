//! The project context in every agent's system prompt: the instruction
//! files (`AGENTS.md`, or `CLAUDE.md`) of `RIG_HOME` and of the working
//! directory and each directory above it, and the environment (working
//! directory, platform, date, git branch). Both are
//! [`PromptSection`]s, re-read when a turn starts and by `/context`, so an
//! edited `AGENTS.md` counts from the next message without a rebuild.
//! A section only changes when its text does, so the prompt stays cached.

use std::fs::{self, File};
use std::io::Read as _;
use std::path::{Path, PathBuf};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use rig::harness_protocol::{Home, Mode};

use crate::core::agent::{Notice, TurnOf};
use crate::core::commands::{AppCommandsExt, CommandArgs};
use crate::core::prompt::PromptSection;
use crate::host::headless::RunMode;

/// The names of an instruction file, in order of preference: one per
/// directory is read.
const FILE_NAMES: [&str; 2] = ["AGENTS.md", "CLAUDE.md"];
/// The most of one instruction file sent.
const MAX_FILE_BYTES: usize = 32 * 1024;
/// The most of all instruction files together. The most specific files,
/// nearest the working directory, are kept first.
const MAX_TOTAL_BYTES: usize = 64 * 1024;

/// Spawns the project and environment [`PromptSection`]s, re-reads them
/// when a turn starts, and adds `/context`.
pub struct ProjectContextPlugin;

impl Plugin for ProjectContextPlugin {
    fn build(&self, app: &mut App) {
        // An eval's trials work in directories of their own, not this one.
        if app
            .world()
            .get_resource::<RunMode>()
            .is_some_and(|mode| matches!(mode.mode(), Mode::Eval { .. }))
        {
            return;
        }
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

/// An instruction file as sent.
struct ContextFile {
    path: PathBuf,
    text: String,
    /// The file's size, more than `text` holds when it was cut.
    size: u64,
}

impl ContextFile {
    fn cut(&self) -> bool {
        self.size > self.text.len() as u64
    }
}

/// What the context sections are read from.
struct Context {
    files: Vec<ContextFile>,
    /// Files that exist but could not be read, with why.
    unreadable: Vec<String>,
    cwd: PathBuf,
    git: Option<Git>,
}

/// The git repository the working directory is in.
struct Git {
    root: PathBuf,
    /// The branch, or `detached at <commit>`.
    head: String,
}

impl Context {
    /// Reads the instruction files and the environment now.
    fn read() -> Self {
        let cwd = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let mut dirs = vec![Home::from_env().root().to_path_buf()];
        let mut above: Vec<PathBuf> = cwd.ancestors().map(Path::to_path_buf).collect();
        above.reverse();
        dirs.extend(above);

        let mut found: Vec<PathBuf> = Vec::new();
        let mut seen: Vec<PathBuf> = Vec::new();
        for dir in dirs {
            let Some(path) = instruction_file(&dir) else {
                continue;
            };
            let canonical = fs::canonicalize(&path).unwrap_or_else(|_| path.clone());
            if !seen.contains(&canonical) {
                seen.push(canonical);
                found.push(path);
            }
        }

        // The budget goes to the most specific files first.
        let mut budget = MAX_TOTAL_BYTES;
        let mut files = Vec::new();
        let mut unreadable = Vec::new();
        for path in found.into_iter().rev() {
            let cap = budget.min(MAX_FILE_BYTES);
            match read_capped(&path, cap) {
                Ok(file) => {
                    budget = budget.saturating_sub(file.text.len());
                    if !file.text.trim().is_empty() {
                        files.push(file);
                    }
                }
                Err(error) => unreadable.push(format!("{}: {error}", path.display())),
            }
        }
        files.reverse();
        let git = git_head(&cwd);
        Self {
            files,
            unreadable,
            cwd,
            git,
        }
    }

    /// The instruction files, from the most general to the most specific.
    fn project_section(&self) -> PromptSection {
        let mut text = String::new();
        if !self.files.is_empty() {
            text.push_str(
                "Instructions from the user and the project, from the most general to the most \
                 specific. Follow them; where two disagree, the later one wins.",
            );
        }
        for file in &self.files {
            text.push_str(&format!(
                "\n\n<file path=\"{}\">\n{}",
                file.path.display(),
                file.text.trim_end()
            ));
            if file.cut() {
                text.push_str(&format!(
                    "\n[Cut after {} of its {} bytes; read the file for the rest.]",
                    file.text.len(),
                    file.size
                ));
            }
            text.push_str("\n</file>");
        }
        PromptSection::new(PromptSection::ORDER_PROJECT, "project_instructions", text)
    }

    /// The working directory, platform, date and git branch.
    fn environment_section(&self) -> PromptSection {
        let mut lines = vec![
            format!("Working directory: {}", self.cwd.display()),
            format!(
                "Platform: {} ({})",
                std::env::consts::OS,
                std::env::consts::ARCH
            ),
            format!("Today's date: {}", chrono::Local::now().format("%Y-%m-%d")),
        ];
        lines.push(match &self.git {
            Some(git) => format!("Git repository: {}, {}", git.root.display(), git.head),
            None => "Git repository: none".to_owned(),
        });
        PromptSection::new(
            PromptSection::ORDER_ENVIRONMENT,
            "environment",
            lines.join("\n"),
        )
    }
}

/// The instruction file of `dir`, if it has one.
fn instruction_file(dir: &Path) -> Option<PathBuf> {
    FILE_NAMES
        .iter()
        .map(|name| dir.join(name))
        .find(|path| fs::metadata(path).is_ok_and(|meta| meta.is_file()))
}

/// Reads at most `cap` bytes of the regular file at `path` as text, without
/// a byte-order mark. A cut through a character drops that character.
fn read_capped(path: &Path, cap: usize) -> std::io::Result<ContextFile> {
    let file = File::open(path)?;
    let size = file.metadata()?.len();
    let mut bytes = Vec::new();
    file.take(cap as u64).read_to_end(&mut bytes)?;
    let mut text = match String::from_utf8(bytes) {
        Ok(text) => text,
        Err(error) => {
            let valid = error.utf8_error().valid_up_to();
            let mut bytes = error.into_bytes();
            // A character cut at the end is dropped; bad bytes inside the
            // file are replaced.
            if size > cap as u64 && bytes.len().saturating_sub(valid) < 4 {
                bytes.truncate(valid);
            }
            String::from_utf8_lossy(&bytes).into_owned()
        }
    };
    if let Some(rest) = text.strip_prefix('\u{feff}') {
        text = rest.to_owned();
    }
    Ok(ContextFile {
        path: path.to_path_buf(),
        text,
        size,
    })
}

/// The git repository `cwd` is in and its checked-out branch, read from
/// `.git/HEAD` (or the `gitdir:` a worktree's `.git` file names) without
/// running git.
fn git_head(cwd: &Path) -> Option<Git> {
    let (root, dot_git) = cwd
        .ancestors()
        .map(|dir| (dir, dir.join(".git")))
        .find(|(_, dot_git)| dot_git.exists())?;
    let git_dir = if dot_git.is_file() {
        let link = fs::read_to_string(&dot_git).ok()?;
        let target = link.trim().strip_prefix("gitdir:")?.trim();
        root.join(target)
    } else {
        dot_git
    };
    let head = fs::read_to_string(git_dir.join("HEAD")).ok()?;
    let head = head.trim();
    let head = match head.strip_prefix("ref:") {
        Some(reference) => {
            let reference = reference.trim();
            let branch = reference.strip_prefix("refs/heads/").unwrap_or(reference);
            format!("branch {branch}")
        }
        None => format!("detached at {}", head.get(..12).unwrap_or(head)),
    };
    Some(Git {
        root: root.to_path_buf(),
        head,
    })
}

/// The paths of the files in the project section, for notices.
fn file_list(context: &Context) -> String {
    let names: Vec<String> = context
        .files
        .iter()
        .map(|file| file.path.display().to_string())
        .collect();
    if names.is_empty() {
        "none".to_owned()
    } else {
        names.join(", ")
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
        format!("Instruction files re-read: {}.", file_list(&context)),
    ));
    for problem in context.unreadable {
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
    if context.files.is_empty() {
        lines.push(format!(
            "  none; put an {} in the project or in {}",
            FILE_NAMES.join(" or "),
            Home::from_env().root().display()
        ));
    }
    for file in &context.files {
        let cut = if file.cut() {
            format!(", cut to {} bytes", file.text.len())
        } else {
            String::new()
        };
        lines.push(format!(
            "  {} ({} bytes{cut})",
            file.path.display(),
            file.size
        ));
    }
    for problem in &context.unreadable {
        lines.push(format!("  cannot read {problem}"));
    }
    lines.push(environment_text);
    notices.write(Notice::info(args.agent, lines.join("\n")));
}
