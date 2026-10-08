//! Agent entities: the components that make an agent, its stable id, and
//! the relationship that ties work in flight to it.

use std::hash::{BuildHasher, Hasher};
use std::sync::atomic::{AtomicU64, Ordering};

use bevy::ecs::reflect::ReflectComponent;
use bevy::prelude::*;
use bevy::reflect::{ReflectDeserialize, ReflectSerialize};
use rig_core::completion::{Message, Reasoning};
use serde::{Deserialize, Serialize};

/// Marks an entity as an agent. Spawning it brings every agent component
/// with its default.
#[derive(Component, Reflect, Default, Clone, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Component, Default, Serialize, Deserialize)]
#[type_path = "rig_code"]
#[require(
    AgentId::mint(),
    Conversation,
    ModelChoice,
    EffortChoice,
    SystemPrompt,
    ToolAccess,
    AgentStatus,
    Name::new("main")
)]
pub struct Agent;

/// The agent's stable id, used in logs, saved state and effect scopes.
/// Entity ids change across restarts; this does not.
#[derive(Component, Reflect, Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Component, Serialize, Deserialize)]
#[type_path = "rig_code"]
pub struct AgentId(pub String);

impl AgentId {
    /// A fresh id: `agent-` and 8 hex digits.
    pub fn mint() -> Self {
        Self(format!("agent-{:08x}", random_u64() as u32))
    }
}

/// The agent's conversation, in rig-core's own message format.
#[derive(Component, Reflect, Default, Clone, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Component, Default, Serialize, Deserialize)]
#[type_path = "rig_code"]
pub struct Conversation(pub Vec<Message>);

/// The catalog reference (`vendor/model`) the agent talks to, if one was
/// picked.
#[derive(Component, Reflect, Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Component, Default, Serialize, Deserialize)]
#[type_path = "rig_code"]
pub struct ModelChoice(pub Option<String>);

/// The reasoning setting sent with each request, if one was picked.
#[derive(Component, Reflect, Default, Clone, Debug, PartialEq, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Component, Default, Serialize, Deserialize)]
#[type_path = "rig_code"]
pub struct EffortChoice(pub Option<Reasoning>);

/// The system prompt sent ahead of the conversation.
#[derive(Component, Reflect, Clone, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Component, Default, Serialize, Deserialize)]
#[type_path = "rig_code"]
pub struct SystemPrompt(pub String);

impl Default for SystemPrompt {
    fn default() -> Self {
        let directory = std::env::current_dir()
            .map(|path| path.display().to_string())
            .unwrap_or_default();
        Self(format!(
            "You are a coding agent working in the directory {directory}. Use the tools to read, \
             search, edit and write files and to run shell commands. Read a file before you edit \
             it. Keep answers short."
        ))
    }
}

/// The tools the agent may call, by name. `None` allows every registered
/// tool.
#[derive(Component, Reflect, Default, Clone, Serialize, Deserialize)]
#[reflect(opaque)]
#[reflect(Component, Default, Serialize, Deserialize)]
#[type_path = "rig_code"]
pub struct ToolAccess(pub Option<Vec<String>>);

impl ToolAccess {
    /// Whether the agent may call the tool `name`.
    pub fn allows(&self, name: &str) -> bool {
        self.0
            .as_ref()
            .is_none_or(|names| names.iter().any(|allowed| allowed == name))
    }
}

/// What the agent is doing. Not saved: no work survives a restart, so a
/// restored agent starts idle.
#[derive(Component, Default, Clone, Debug, PartialEq)]
pub enum AgentStatus {
    /// Waiting for input.
    #[default]
    Idle,
    /// A model call is in flight.
    Streaming,
    /// Tool calls are in flight.
    RunningTools,
    /// The last turn failed; the reason is shown to the user.
    Failed(String),
}

impl AgentStatus {
    /// Whether a turn is in flight.
    pub fn is_busy(&self) -> bool {
        matches!(self, Self::Streaming | Self::RunningTools)
    }
}

/// Points a work entity (a model call, a tool call) at the agent it runs
/// for.
#[derive(Component, Debug)]
#[relationship(relationship_target = Work)]
pub struct WorkOf(pub Entity);

/// The work entities in flight for an agent. Despawning the agent, or the
/// work, drops the tasks and so cancels them.
#[derive(Component, Debug, Default)]
#[relationship_target(relationship = WorkOf, linked_spawn)]
pub struct Work(Vec<Entity>);

/// A random 64-bit value from std's per-process hasher keys, a clock and a
/// counter. Good enough for ids that only need to differ.
pub(crate) fn random_u64() -> u64 {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let mut hasher = std::collections::hash_map::RandomState::new().build_hasher();
    hasher.write_u64(COUNTER.fetch_add(1, Ordering::Relaxed));
    hasher.write_u32(std::process::id());
    if let Ok(elapsed) = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        hasher.write_u128(elapsed.as_nanos());
    }
    hasher.finish()
}
