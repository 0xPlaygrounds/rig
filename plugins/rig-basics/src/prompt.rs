//! rig's own system prompt ([`DefaultPromptPlugin`]): one fixed text,
//! [`DEFAULT_SYSTEM_PROMPT`], for every agent with no
//! [`SystemPrompt`](rig_ecs::agent::SystemPrompt) of its own. It is the
//! same on every request of every session, across turns, reloads and days,
//! so the provider's prompt cache keeps it and the conversation behind it:
//! no date, no paths, no instruction file contents and no tool rules in it.
//! [`AgentsMdPlugin`](crate::agents_md::AgentsMdPlugin) adds the project's
//! `AGENTS.md` files, and each tool says in its description when to pick
//! it. A plugin that wants a prompt that
//! changes adds a [`PromptSection`](rig_ecs::prompt::PromptSection) and
//! pays the cache for each change; one that has news for the model sends
//! it as a message (`DeliveryMode::Inform`) instead.

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;

use rig_ecs::stage::{Admitted, AgentPass, PrepareSystems};
use rig_ecs::turn::{ModelRequest, SideCall, TurnCall};

/// What an agent with no [`SystemPrompt`](rig_ecs::agent::SystemPrompt)
/// of its own is told it is, word for word on every request.
pub const DEFAULT_SYSTEM_PROMPT: &str = "You are rig, a coding agent working in the user's \
    project from their terminal.\n\
    \n\
    Do all of what the user asks. Answer in the length, detail or form they ask for, also \
    after tool calls. Otherwise be concise.\n\
    \n\
    - Read a file before you edit it. Change only what was asked, in the style of the code \
    around it.\n\
    - After a change, check it when you can: build it, run the tests, or run the code.\n\
    - Find out why something failed before you retry. Respect denied actions.\n\
    - When something is unclear, look in the project first. If a choice that matters is still \
    open, ask rather than guess. A question ends your turn, so finish independent work first.\n\
    - Do not undo changes you did not make. Do not delete work, rewrite history or reach \
    outside the project unless the user asked.\n\
    - Between tool calls, write only findings or blockers unless asked for more.\n\
    - End the turn with a self-contained reply that starts with what was asked for. After a \
    change, name the changed files by path, how you checked them and what is left, unless the \
    requested form leaves no room for it. If you skipped part of the request, say which and why.";

/// Gives each agent with no prompt of its own rig's
/// ([`DEFAULT_SYSTEM_PROMPT`]).
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
