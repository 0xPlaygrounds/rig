//! The system prompt, put together for each model call from two parts:
//! the agent's own [`SystemPrompt`](super::agent::SystemPrompt) and the
//! [`PromptSection`]s, the app's and the agent's own ([`SectionOf`]).
//! Nothing in it depends on the turn, so it stays the same from call to
//! call and the provider's prompt cache keeps it. A section whose text
//! changes costs the whole conversation's cache once per change, so what
//! changes goes in a message instead (`DeliveryMode::Inform`); a section
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
    /// The tag around the text, such as `skills`.
    pub tag: String,
    /// The text.
    pub text: String,
}

impl PromptSection {
    /// The order of a role an agent plays besides its own prompt, such as a
    /// subagent's.
    pub const ORDER_ROLE: i32 = 100;
    /// The order of what a project or the harness adds.
    pub const ORDER_PROJECT: i32 = 200;
    /// The order of facts that change most often, which so come last.
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
