//! The agent entity and its components.

use std::sync::Arc;

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::catalog::ModelSpec;
use rig_core::completion::Message;
use rig_core::completion::options::Reasoning;
use rig_core::effect::HandlerKey;
use rig_core::serve::ErasedHandler;
use serde::{Deserialize, Serialize};

use super::models;
use super::registry::Notice;
use super::save::ReflectSaved;

/// Marks an agent. Spawning it alone gives a complete agent: every other
/// agent component is required, with its default.
#[derive(Component, Default, Debug)]
#[require(AgentId, Conversation, Effort, SystemPrompt, ToolAccess, Status)]
pub struct Agent;

/// The agent's stable id, used in logs, saved state and effect records.
/// Unlike `Entity`, it stays the same across restarts. It never changes
/// after spawn.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[component(immutable)]
#[reflect(opaque)]
#[reflect(Component, Clone, Serialize, Deserialize, Saved)]
pub struct AgentId(pub Arc<str>);

impl Default for AgentId {
    /// A new random id, `a-` and 16 hex digits.
    fn default() -> Self {
        Self(format!("a-{:016x}", super::session::random()).into())
    }
}

/// The messages of the agent's conversation, oldest first. The system
/// prompt is not part of it.
#[derive(Component, Reflect, Clone, Default, Debug, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Component, Clone, Default, Serialize, Deserialize, Saved)]
pub struct Conversation(pub Vec<Message>);

/// The catalog model the agent talks to, as `vendor/model`. Inserting it
/// connects the agent ([`Connection`]); it never changes in place.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq)]
#[component(immutable)]
#[reflect(Component, Saved)]
pub struct Model(pub String);

/// How much the model reasons. `None` leaves it to the model's default.
#[derive(Component, Reflect, Clone, Copy, Default, Debug, PartialEq, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Component, Clone, Default, Serialize, Deserialize, Saved)]
pub struct Effort(pub Option<Reasoning>);

/// The system prompt sent ahead of the conversation.
#[derive(Component, Reflect, Clone, Debug)]
#[reflect(Component, Default, Saved)]
pub struct SystemPrompt(pub String);

impl Default for SystemPrompt {
    fn default() -> Self {
        let directory = std::env::current_dir().unwrap_or_default();
        Self(format!(
            "You are a coding agent working in {}. Use the tools to read, search, edit and \
             write files and to run shell commands. Read a file before you edit it. Keep \
             answers short.",
            directory.display()
        ))
    }
}

/// Which registered tools the agent may call.
#[derive(Component, Reflect, Clone, Default, Debug)]
#[reflect(Component, Default, Saved)]
pub enum ToolAccess {
    /// Every registered tool.
    #[default]
    All,
    /// Only the tools named.
    Only(Vec<String>),
}

impl ToolAccess {
    /// Whether the agent may call the tool `name`.
    pub fn allows(&self, name: &str) -> bool {
        match self {
            Self::All => true,
            Self::Only(names) => names.iter().any(|allowed| allowed == name),
        }
    }
}

/// Where the agent is in its loop.
#[derive(Component, Clone, Copy, Default, Debug, PartialEq, Eq)]
pub enum Status {
    /// Waiting for input.
    #[default]
    Idle,
    /// A model call starts next frame.
    Queued,
    /// A model call is streaming.
    Streaming,
    /// The tool calls of the last reply are running, one at a time.
    Tools,
}

/// The connected model of an agent with a [`Model`]: its catalog entry and
/// the handler every model call is dispatched to.
#[derive(Component, Clone, Debug)]
pub struct Connection {
    /// The model's catalog entry.
    pub spec: ModelSpec,
    /// The model as an effect handler.
    pub handler: ErasedHandler,
    /// The handler's key in effect records.
    pub key: HandlerKey,
}

/// Work in flight for an agent: a model call or a tool call entity.
#[derive(Component, Debug)]
#[relationship(relationship_target = Calls)]
pub struct CallOf(pub Entity);

/// The agent's work in flight. Despawning the agent despawns it, which
/// cancels the work.
#[derive(Component, Debug)]
#[relationship_target(relationship = CallOf, linked_spawn)]
pub struct Calls(Vec<Entity>);

/// Connects an agent whose [`Model`] was inserted. A model the agent's
/// [`Effort`] does not fit resets the effort to the model's default.
pub(crate) fn connect(
    insert: On<Insert<Model>>,
    agents: Query<(&Model, &Effort)>,
    mut commands: Commands,
) -> Result {
    let agent = insert.entity;
    let (model, effort) = agents.get(agent)?;
    match models::connect(&model.0) {
        Ok(connection) => {
            commands.trigger(Notice::info(
                agent,
                format!("Model: {} ({}).", connection.spec.display_name, model.0),
            ));
            if let Some(reasoning) = effort.0
                && let Some(reason) = connection.spec.reasoning.refusal(&reasoning)
            {
                commands.entity(agent).insert(Effort::default());
                commands.trigger(Notice::info(
                    agent,
                    format!("Effort reset to the model's default: {reason}."),
                ));
            }
            commands.entity(agent).insert(connection);
        }
        Err(error) => {
            commands.entity(agent).remove::<Connection>();
            commands.trigger(Notice::error(agent, error.to_string()));
        }
    }
    Ok(())
}
