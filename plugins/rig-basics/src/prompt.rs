//! rig's own system prompt ([`DefaultPromptPlugin`]): one fixed text for
//! every agent with no [`SystemPrompt`](rig_ecs::agent::SystemPrompt) of
//! its own. It never changes, so the provider's prompt cache keeps it from
//! the first request of a session to the last: no date, no working
//! directory and no instruction files in it. The model reads `AGENTS.md`
//! itself; a plugin that wants a prompt that changes adds a
//! [`PromptSection`](rig_ecs::prompt::PromptSection).

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use rig_ecs::stage::{Admitted, AgentPass, PrepareSystems};
use rig_ecs::turn::{ModelRequest, SideCall, TurnCall};

/// What an agent with no [`SystemPrompt`](rig_ecs::agent::SystemPrompt) of its own is told it is.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are rig, a coding agent. You work in the \
    user's project from their terminal: \
    you read and search code, edit files and run commands with the tools you are \
    given, and answer questions about the code.\n\
    \n\
    - Work in the current directory, unless the user says otherwise. Before your \
    first change, read the AGENTS.md (or CLAUDE.md) files in it and in the \
    directories above it, if there are any: they are the project's instructions.\n\
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

/// Gives an agent with no prompt of its own rig's ([`DEFAULT_SYSTEM_PROMPT`]).
#[derive(Default)]
pub struct DefaultPromptPlugin;

impl Plugin for DefaultPromptPlugin {
    fn build(&self, app: &mut App) {
        app.add_systems(AgentPass, default_prompt.in_set(PrepareSystems::Compose));
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

#[cfg(test)]
mod tests;
