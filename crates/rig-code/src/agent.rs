//! Agents, their in-flight calls, and the messages views exchange with them.
//!
//! An agent is an entity with the [`Agent`] marker; every per-agent setting
//! and the conversation are components on it. A model call or a tool call is
//! its own entity, tied to the agent by [`CallOf`]. Despawning the agent
//! despawns its calls, and dropping a call's [`EffectTask`] cancels the work.
//! Components that derive `Reflect` with `#[reflect(Component)]` are saved
//! with the session; runtime state such as [`AgentStatus`] is not.

use std::sync::mpsc::Receiver;

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use bevy_tasks::Task;
use rig_core::{
    ErrorReport,
    completion::Reasoning,
    effect::Outcome,
    message::{self, Message},
};
use serde::{Deserialize, Serialize};

/// Marks an agent entity and brings every per-agent component with it.
#[derive(Component, Reflect, Debug, Default)]
#[reflect(Component, Default)]
#[require(
    AgentId,
    Conversation,
    ModelChoice,
    Effort,
    SystemPrompt,
    ToolAccess,
    AgentStatus
)]
pub struct Agent;

/// The agent's stable id, used in logs, effects and saved state. An
/// `Entity` is not stable across a restart; this is.
#[derive(Component, Reflect, Debug, Clone, PartialEq, Eq)]
#[reflect(Component, Default)]
pub struct AgentId(pub String);

impl Default for AgentId {
    fn default() -> Self {
        Self(rig_core::id::generate())
    }
}

/// The agent's conversation, oldest message first.
#[derive(Component, Reflect, Debug, Clone, Default, Serialize, Deserialize)]
#[serde(transparent)]
#[reflect(opaque)]
#[reflect(Component, Default, Serialize, Deserialize)]
pub struct Conversation(pub Vec<Message>);

/// The catalog reference (`vendor/model`) the agent talks to, if chosen.
#[derive(Component, Reflect, Debug, Clone, Default)]
#[reflect(Component, Default)]
pub struct ModelChoice(pub Option<String>);

/// The reasoning the agent asks for. `None` uses the model's default.
#[derive(Component, Reflect, Debug, Clone, Copy, Default, Serialize, Deserialize)]
#[serde(transparent)]
#[reflect(opaque)]
#[reflect(Component, Default, Serialize, Deserialize)]
pub struct Effort(pub Option<Reasoning>);

/// The system prompt sent ahead of the conversation.
#[derive(Component, Reflect, Debug, Clone)]
#[reflect(Component, Default)]
pub struct SystemPrompt(pub String);

impl Default for SystemPrompt {
    fn default() -> Self {
        let cwd = std::env::current_dir()
            .map(|dir| dir.display().to_string())
            .unwrap_or_default();
        Self(format!(
            "You are rig-code, a coding agent working in the user's terminal. \
             The working directory is {cwd}. Use the tools to read, search, edit and \
             write files and to run shell commands. Paths are relative to the working \
             directory. Read a file before editing it. Keep answers short."
        ))
    }
}

/// The tools the agent may call by name. `None` allows every tool.
#[derive(Component, Reflect, Debug, Clone, Default)]
#[reflect(Component, Default)]
pub struct ToolAccess(pub Option<Vec<String>>);

/// What the agent is doing now.
#[derive(Component, Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum AgentStatus {
    /// Waiting for input.
    #[default]
    Idle,
    /// A model call is streaming.
    Thinking,
    /// This many tool calls are running.
    Tools(usize),
}

/// Whether an agent has a turn running: it is not idle, owes the model a
/// reply, or has calls in flight. Prompts and `/reload` wait for this to
/// be false.
pub fn turn_running(status: AgentStatus, needs_reply: bool, has_calls: bool) -> bool {
    status != AgentStatus::Idle || needs_reply || has_calls
}

/// The agent owes the model a request: set after a prompt or after tool
/// results, cleared once the model call starts.
#[derive(Component, Debug, Default)]
#[component(storage = "SparseSet")]
pub struct NeedsReply;

/// Ties a call entity to the agent it works for.
#[derive(Component, Debug)]
#[relationship(relationship_target = AgentCalls)]
pub struct CallOf(pub Entity);

/// The calls in flight for an agent. Despawning the agent despawns them.
#[derive(Component, Debug, Default)]
#[relationship_target(relationship = CallOf, linked_spawn)]
pub struct AgentCalls(Vec<Entity>);

/// The task serving one dispatched effect. Dropping it cancels the effect.
#[derive(Component)]
pub struct EffectTask(pub Task<Result<Outcome, ErrorReport>>);

/// A piece of a streaming reply, sent from the task to its call entity.
#[derive(Debug)]
pub enum Feed {
    /// Who the reply is from.
    Origin(message::Origin),
    /// Answer text.
    Text(String),
    /// Reasoning text.
    Reasoning(String),
    /// The reply started a call of this tool.
    ToolStart(String),
}

/// A model call in flight and what it has streamed so far.
#[derive(Component)]
pub struct ModelCall {
    feed: std::sync::Mutex<Receiver<Feed>>,
    /// Who the reply is from, once known.
    pub origin: Option<message::Origin>,
    /// The answer text so far.
    pub text: String,
    /// The reasoning text so far.
    pub reasoning: String,
    /// The tools the reply called so far, by name.
    pub tools: Vec<String>,
}

impl ModelCall {
    /// A call reading its stream from `feed`.
    pub fn new(feed: Receiver<Feed>) -> Self {
        Self {
            feed: std::sync::Mutex::new(feed),
            origin: None,
            text: String::new(),
            reasoning: String::new(),
            tools: Vec::new(),
        }
    }

    /// Move everything the task has sent into this call. Returns whether
    /// anything arrived.
    pub fn drain(&mut self) -> bool {
        let feed = self
            .feed
            .get_mut()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let mut changed = false;
        while let Ok(item) = feed.try_recv() {
            changed = true;
            match item {
                Feed::Origin(origin) => self.origin = Some(origin),
                Feed::Text(text) => self.text.push_str(&text),
                Feed::Reasoning(text) => self.reasoning.push_str(&text),
                Feed::ToolStart(name) => self.tools.push(name),
            }
        }
        changed
    }
}

/// One tool call of a turn, and its result once it finished.
#[derive(Component, Debug)]
pub struct ToolCallSlot {
    /// Position of the call in the assistant turn.
    pub index: usize,
    /// The call as the model made it.
    pub call: message::ToolCall,
    /// The result the model will read.
    pub result: Option<message::ToolResult>,
}

/// The agent finished a turn: the model answered without calling tools,
/// the turn failed, or it was stopped. Autosave observes it.
#[derive(EntityEvent, Debug, Clone, Copy)]
pub struct TurnEnded {
    /// The agent.
    pub entity: Entity,
}

/// Text typed into a view for an agent: a prompt, or a `/command`.
#[derive(Message, Debug, Clone)]
pub struct Submit {
    /// The agent addressed.
    pub agent: Entity,
    /// The text as typed.
    pub text: String,
}

/// Stop the agent's running turn.
#[derive(Message, Debug, Clone, Copy)]
pub struct Interrupt {
    /// The agent addressed.
    pub agent: Entity,
}

/// Leave the app.
#[derive(Message, Debug, Clone, Copy, Default)]
pub struct Quit;

/// Something a view should show the user about an agent.
#[derive(Message, Debug, Clone)]
pub struct Notice {
    /// The agent concerned.
    pub agent: Entity,
    /// The text, possibly several lines.
    pub text: String,
}

/// A request for the user to pick one of several command lines.
#[derive(Message, Debug, Clone)]
pub struct Choose {
    /// The agent the choice is for.
    pub agent: Entity,
    /// What is being chosen.
    pub title: String,
    /// The options, in display order.
    pub options: Vec<Choice>,
}

/// One option of a [`Choose`]: picking it submits `command`.
#[derive(Debug, Clone)]
pub struct Choice {
    /// What the user reads.
    pub label: String,
    /// The command line submitted when picked.
    pub command: String,
}

/// The agent loop's stages in `Update`, in order.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum RigSet {
    /// Views' messages become commands, prompts and interrupts.
    Input,
    /// Agents owing a reply start a model call.
    Start,
    /// Finished calls are collected.
    Poll,
    /// Completed tool batches are answered.
    Finish,
}
