//! rig's own system prompt ([`DefaultPromptPlugin`]) for every agent with
//! no [`SystemPrompt`](rig_ecs::agent::SystemPrompt) of its own:
//! [`DEFAULT_SYSTEM_PROMPT`], or [`UNATTENDED_SYSTEM_PROMPT`] when nobody
//! attends the run ([`Unattended`]), followed by the project's
//! instructions: `AGENTS.md`, or `CLAUDE.md` where a directory has no
//! `AGENTS.md`, in the project directory and each one above it up to the
//! repository root (the directory holding `.git`), the root's first.
//! They are read once, at the first request, so the prompt is the same on
//! every request of the run and the provider's prompt cache keeps it: no
//! date and no tool rules in it, and an edit to `AGENTS.md` reaches the
//! next run. A plugin that wants a prompt that changes adds a
//! [`PromptSection`](rig_ecs::prompt::PromptSection) and pays the cache for
//! each change; one that has news for the model sends it as a message
//! (`DeliveryMode::Inform`) instead.

use std::path::{Path, PathBuf};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use rig_ecs::stage::{Admitted, AgentPass, PrepareSystems};
use rig_ecs::turn::{ModelRequest, SideCall, TurnCall};
use rig_harness::services::ProjectRoot;
use rig_harness::worker::Unattended;

/// The rules every run shares, after the line saying who answers.
macro_rules! rules {
    () => {
        "\n- Work in the current directory unless the user says otherwise.\n\
         - Read a file before you edit it. Change what was asked, in the style of the code \
         around it, and nothing else.\n\
         - After a change, check it when you can: build it, run the tests, or run the code.\n\
         - Do not undo changes you did not make, and do not delete work, rewrite history or \
         reach outside the project unless the user asked.\n\
         - Keep answers short: once, at the end of the turn, say what you changed and what is \
         left, naming files by path. No commentary between tool calls."
    };
}

/// What an agent with no [`SystemPrompt`](rig_ecs::agent::SystemPrompt)
/// of its own is told it is when someone attends the run, before the
/// project's instructions.
pub const DEFAULT_SYSTEM_PROMPT: &str = concat!(
    "You are rig, a coding agent working in the user's project. You read and search code, \
     edit files and run commands with your tools, and answer questions about the code.\n\
     \n\
     - When something fails, read the error and fix the cause. Ask when the request is \
     unclear or you are blocked, rather than guess.",
    rules!()
);

/// The same when nobody attends the run, such as `rig -p`'s: nobody will
/// reply before it ends.
pub const UNATTENDED_SYSTEM_PROMPT: &str = concat!(
    "You are rig, a coding agent working in the user's project. You read and search code, \
     edit files and run commands with your tools, and answer questions about the code.\n\
     \n\
     - Nobody will reply during this run. When something fails, read the error and fix the \
     cause. When the request is unclear, make the reasonable choice, say which choice you \
     made, and carry on; end with a question only when no choice is safe.",
    rules!()
);

/// The longest instruction file read, in bytes; the rest is cut.
const MOST: usize = 64 * 1024;

/// Gives each agent with no prompt of its own rig's, with the project's
/// instructions.
#[derive(Default)]
pub struct DefaultPromptPlugin;

impl Plugin for DefaultPromptPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(AgentPass, default_prompt.in_set(PrepareSystems::Compose));
    }
}

/// The project's instructions under `dir`: each directory from the
/// repository root down to `dir`, its `AGENTS.md` or else its `CLAUDE.md`,
/// headed by its path; only `dir` outside a repository. Empty without any.
pub fn project_instructions(dir: &Path) -> String {
    let mut dirs: Vec<PathBuf> = Vec::new();
    let mut at = Some(dir);
    while let Some(here) = at {
        dirs.push(here.to_owned());
        if here.join(".git").exists() {
            break;
        }
        at = here.parent();
    }
    if !dirs.last().is_some_and(|top| top.join(".git").exists()) {
        dirs.truncate(1);
    }
    let mut out = String::new();
    for dir in dirs.iter().rev() {
        let file = ["AGENTS.md", "CLAUDE.md"]
            .iter()
            .map(|name| dir.join(name))
            .find(|path| path.is_file());
        let Some(path) = file else {
            continue;
        };
        let Ok(text) = std::fs::read_to_string(&path) else {
            continue;
        };
        let mut text = text.trim().to_owned();
        if text.len() > MOST {
            let cut = (0..=MOST)
                .rev()
                .find(|at| text.is_char_boundary(*at))
                .unwrap_or(0);
            text.truncate(cut);
            text.push_str("\n[cut]");
        }
        if text.is_empty() {
            continue;
        }
        out.push_str(&format!("\n\n## {}\n\n{text}", path.display()));
    }
    if out.is_empty() {
        return out;
    }
    format!("\n\n# The project's instructions{out}")
}

/// Makes rig's prompt the role of the system prompt of each model call the
/// kernel built from an agent with no prompt of its own, in
/// [`PrepareSystems::Compose`], so every plugin that edits the prompt sees
/// it whole. The prompt is made at the first call and kept.
fn default_prompt(
    mut calls: Query<&mut ModelRequest, (Added<Admitted>, Or<(With<TurnCall>, With<SideCall>)>)>,
    root: Option<Res<ProjectRoot>>,
    unattended: Option<Res<Unattended>>,
    mut made: Local<Option<String>>,
) {
    for mut call in &mut calls {
        if let Some(prompt) = &mut call.prompt
            && prompt.role.is_empty()
        {
            let role = made.get_or_insert_with(|| {
                let base = match unattended.as_deref() {
                    Some(Unattended(true)) => UNATTENDED_SYSTEM_PROMPT,
                    _ => DEFAULT_SYSTEM_PROMPT,
                };
                let root = root.as_deref().cloned().unwrap_or_default();
                format!("{base}{}", project_instructions(&root.0))
            });
            prompt.role = role.clone();
        }
    }
}

#[cfg(test)]
mod tests;
