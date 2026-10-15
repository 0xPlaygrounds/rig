//! The system prompt, put together for each model call from three parts:
//! the agent's own [`SystemPrompt`](super::agent::SystemPrompt), the
//! [`ToolRules`] of the tools it is offered, and the [`PromptSection`]s,
//! the app's, such as the project's instructions and the environment, and
//! the agent's own ([`SectionOf`]). Nothing in it depends on the turn, so it stays the same
//! from call to call and the provider's prompt cache keeps it; a section
//! that changes rarely sorts before one that changes more often. A request
//! carries it in parts, rig-core's `SystemPrompt`
//! ([`ModelRequest::prompt`](super::turn::ModelRequest::prompt)), until
//! Start sends it as one system message.

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;

/// A part of every agent's system prompt, on an entity of its own, or of one
/// agent's with a [`SectionOf`] that agent. A plugin spawns one and changes its
/// `text` when what it describes changes (with `set_if_neq`, so an unchanged
/// section is not marked changed); an empty `text`, or Bevy's `Disabled` on the
/// entity, leaves the section out. Sections are sent in `order`, then `tag`,
/// each between `<tag>` and `</tag>`.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Clone, Debug, Default, PartialEq)]
pub struct PromptSection {
    /// Where the section goes: lower first. See the `ORDER_*` constants.
    pub order: i32,
    /// The tag around the text, such as `project_instructions`.
    pub tag: String,
    /// The text.
    pub text: String,
}

impl PromptSection {
    /// The order of a role an agent plays besides its own prompt, such as a
    /// subagent's.
    pub const ORDER_ROLE: i32 = 100;
    /// The order of the project's instructions (`AGENTS.md`).
    pub const ORDER_PROJECT: i32 = 200;
    /// The order of facts about the machine and the day, which change most
    /// often and so come last.
    pub const ORDER_ENVIRONMENT: i32 = 300;

    /// A section of `text` between `<tag>` and `</tag>`, at `order`.
    pub fn new(order: i32, tag: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            order,
            tag: tag.into(),
            text: text.into(),
        }
    }
}

/// On a [`PromptSection`]: it is part of this agent's system prompt only.
/// Despawning the agent despawns it. A restored agent is a new entity, so
/// a plugin spawns the sections of its agents again, such as when the
/// saved component they come from is added.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship(relationship_target = AgentSections)]
pub struct SectionOf(pub Entity);

/// The prompt sections of this agent alone.
#[derive(Component, Reflect, Debug)]
#[reflect(Component)]
#[relationship_target(relationship = SectionOf, linked_spawn)]
pub struct AgentSections(Vec<Entity>);

/// How a tool should be used, on the tool's entity: lines added to the
/// system prompt of every agent the tool is offered to. Registered with
/// [`AppToolsExt::add_tool_with`](super::tools::AppToolsExt::add_tool_with).
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Clone, Debug, Default)]
pub struct ToolRules(pub Vec<String>);

/// The tag of the section the kernel writes the tool rules in, which
/// Start fills ([`ModelStarts::prompt`](super::turn::ModelStarts::prompt)).
pub const TOOL_RULES: &str = "tool_rules";

/// The text of the `tool_rules` section of tools with `rules` (in the
/// order the tools are sent): a line per rule, saying a rule two tools
/// share once; empty without rules.
pub(crate) fn tool_rules<'a>(rules: impl IntoIterator<Item = &'a ToolRules>) -> String {
    let mut lines: Vec<&str> = Vec::new();
    for rule in rules.into_iter().flat_map(|rules| rules.0.iter()) {
        let rule = rule.trim();
        if !rule.is_empty() && !lines.contains(&rule) {
            lines.push(rule);
        }
    }
    let lines: Vec<String> = lines.iter().map(|line| format!("- {line}")).collect();
    lines.join("\n")
}
