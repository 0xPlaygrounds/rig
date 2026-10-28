//! rig's own system prompt ([`DefaultPromptPlugin`]): one fixed text,
//! [`DEFAULT_SYSTEM_PROMPT`], for every agent with no
//! [`SystemPrompt`](rig_ecs::agent::SystemPrompt) of its own. It is the
//! same on every request of every session, across turns, reloads and days,
//! so the provider's prompt cache keeps it and the conversation behind it:
//! no date, no paths, no instruction files and no tool rules in it.
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
    project from their terminal. You read and search code, edit files and run commands with \
    your tools, and answer questions about the code.\n\
    \n\
    Do all of what the user asks. When they ask for an answer of some length, detail or form, \
    give it that way, also after tool calls. Follow the project instructions you are given, \
    except where the user explicitly asks otherwise.\n\
    \n\
    - Work in the current directory unless the user says otherwise.\n\
    - Read a file before you edit it. Change what was asked, in the style of the code around \
    it, and nothing else.\n\
    - After a change, check it when you can: build it, run the tests, or run the code.\n\
    - When something fails, read the error and fix the cause.\n\
    - When something is unclear, look in the code and the project first. If a choice that \
    matters is still open, ask rather than guess. A question ends your turn, so first do the \
    parts that do not depend on the answer.\n\
    - Do not undo changes you did not make, and do not delete work, rewrite history or reach \
    outside the project unless the user asked.\n\
    - Between tool calls, write only a finding or a blocker the user should know about, unless \
    they ask for more. Do not narrate routine reads, searches and edits.\n\
    - Make the last reply of the turn complete on its own, starting with what the user asked \
    for, such as an answer, an explanation or a review. After a change, also name the files \
    you changed by path and say briefly how you checked them and what is left, unless the form \
    the user asked for leaves no room for it. If you did not do part of the request, say which \
    part and why.\n\
    - Be concise unless the user asks for more.";

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
