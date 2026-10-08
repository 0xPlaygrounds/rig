//! Tools and slash commands as entities, the messages every view speaks, and
//! routing of submitted input.

use bevy::ecs::observer::IntoEntityObserver;
use bevy::prelude::*;
use rig_core::completion::{Message as ChatMessage, ToolDefinition};
use rig_core::message::ToolResultContent;
use rig_core::serve::ErasedHandler;
use rig_core::serve::adapters::ToolAdapter;
use rig_core::tool::{Tool, tool_definition};

use super::agent::{AgentStatus, Conversation, Work};
use super::turn::{ToolCallDone, ToolCallRun};

/// A tool the agents can call: its definition and the handler that runs
/// it. Spawned by [`AgentAppExt::add_tool`].
#[derive(Component)]
pub struct ToolEntry {
    /// What the model is told about the tool.
    pub definition: ToolDefinition,
    /// What runs a call.
    pub handler: ErasedHandler,
}

/// A slash command: `/name args`. Its handler is an observer of
/// [`RunCommand`] on the command's entity. Spawned by
/// [`AgentAppExt::add_command`].
#[derive(Component, Clone, Debug)]
pub struct SlashCommand {
    /// The name typed after `/`.
    pub name: String,
    /// One line for `/help`.
    pub description: String,
}

impl SlashCommand {
    /// A command named `name` described by `description`.
    pub fn new(name: impl Into<String>, description: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
        }
    }
}

/// Runs a slash command: triggered on the command's entity.
#[derive(EntityEvent, Clone, Debug)]
pub struct RunCommand {
    /// The command's entity.
    #[event_target]
    pub command: Entity,
    /// The agent the command was typed to.
    pub agent: Entity,
    /// Everything after the command name, trimmed.
    pub args: String,
}

/// Input from a view: a prompt, or a slash command line.
#[derive(Message, Clone, Debug)]
pub struct Submit {
    /// The agent the input is for.
    pub agent: Entity,
    /// What was typed.
    pub text: String,
}

/// Stop the agent's turn: its work in flight is dropped.
#[derive(Message, Clone, Debug)]
pub struct Interrupt {
    /// The agent to stop.
    pub agent: Entity,
}

/// How a notice is shown.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum NoticeLevel {
    /// Information.
    Info,
    /// Something went wrong.
    Error,
}

/// A line of text for the user, from the core or a plugin.
#[derive(Message, Clone, Debug)]
pub struct Notice {
    /// The agent it concerns, if any.
    pub agent: Option<Entity>,
    /// The text, possibly several lines.
    pub text: String,
    /// How it is shown.
    pub level: NoticeLevel,
}

impl Notice {
    /// An information notice.
    pub fn info(agent: impl Into<Option<Entity>>, text: impl Into<String>) -> Self {
        Self {
            agent: agent.into(),
            text: text.into(),
            level: NoticeLevel::Info,
        }
    }

    /// An error notice.
    pub fn error(agent: impl Into<Option<Entity>>, text: impl Into<String>) -> Self {
        Self {
            agent: agent.into(),
            text: text.into(),
            level: NoticeLevel::Error,
        }
    }
}

/// One entry a view offers to pick.
#[derive(Clone, Debug)]
pub struct Choice {
    /// What is sent back as the command argument.
    pub value: String,
    /// What is shown.
    pub label: String,
}

/// Asks a view to let the user pick one of `choices`. The pick comes back
/// as a [`Submit`] of `/<command> <value>`.
#[derive(Message, Clone, Debug)]
pub struct OfferChoices {
    /// The agent the pick is for.
    pub agent: Entity,
    /// The picker's title.
    pub title: String,
    /// The command the pick is sent to.
    pub command: String,
    /// The entries.
    pub choices: Vec<Choice>,
}

/// An agent's turn ended: it answered, failed or was stopped.
#[derive(Message, Clone, Copy, Debug)]
pub struct TurnFinished {
    /// The agent.
    pub agent: Entity,
}

/// Inserted on an agent whose conversation needs the model's next reply.
#[derive(Component, Default)]
#[component(storage = "SparseSet")]
pub struct NeedsModelCall;

/// Registers tools and slash commands. Built-in and third-party plugins use
/// the same calls.
pub trait AgentAppExt {
    /// Make `tool` available to agents.
    fn add_tool<T: Tool + 'static>(&mut self, tool: T) -> &mut Self;

    /// Add the slash command `command`, run by the observer `handler` of
    /// [`RunCommand`].
    fn add_command<M>(
        &mut self,
        command: SlashCommand,
        handler: impl IntoEntityObserver<M>,
    ) -> &mut Self;
}

impl AgentAppExt for App {
    fn add_tool<T: Tool + 'static>(&mut self, tool: T) -> &mut Self {
        let definition = tool_definition(&tool);
        self.world_mut().spawn((
            Name::new(T::NAME),
            ToolEntry {
                definition,
                handler: ErasedHandler::new(ToolAdapter::new(tool)),
            },
        ));
        self
    }

    fn add_command<M>(
        &mut self,
        command: SlashCommand,
        handler: impl IntoEntityObserver<M>,
    ) -> &mut Self {
        self.world_mut()
            .spawn((Name::new(format!("/{}", command.name)), command))
            .observe(handler);
        self
    }
}

/// Turns submitted input into commands and prompts, and stops turns on
/// interrupt.
pub(crate) fn route_input(
    mut commands: Commands,
    mut submits: MessageReader<Submit>,
    mut interrupts: MessageReader<Interrupt>,
    slash: Query<(Entity, &SlashCommand)>,
    mut agents: Query<(&mut Conversation, &mut AgentStatus, Option<&Work>)>,
    calls: Query<(&ToolCallRun, Option<&ToolCallDone>)>,
    mut notices: MessageWriter<Notice>,
    mut finished: MessageWriter<TurnFinished>,
) {
    for submit in submits.read() {
        let text = submit.text.trim();
        if let Some(line) = text.strip_prefix('/') {
            let (name, args) = line.split_once(' ').unwrap_or((line, ""));
            match slash.iter().find(|(_, command)| command.name == name) {
                Some((command, _)) => commands.trigger(RunCommand {
                    command,
                    agent: submit.agent,
                    args: args.trim().to_owned(),
                }),
                None => {
                    notices.write(Notice::error(
                        submit.agent,
                        format!("unknown command /{name}; /help lists the commands"),
                    ));
                }
            }
            continue;
        }
        if text.is_empty() {
            continue;
        }
        let Ok((mut conversation, mut status, _)) = agents.get_mut(submit.agent) else {
            continue;
        };
        if status.is_busy() {
            notices.write(Notice::error(
                submit.agent,
                "the agent is busy; press Esc to stop the turn",
            ));
            continue;
        }
        conversation.0.push(ChatMessage::user(text));
        *status = AgentStatus::Streaming;
        commands.entity(submit.agent).insert(NeedsModelCall);
    }

    for interrupt in interrupts.read() {
        let Ok((mut conversation, mut status, work)) = agents.get_mut(interrupt.agent) else {
            continue;
        };
        if !status.is_busy() {
            continue;
        }
        // Every call of the last reply needs a result, or the next request
        // is malformed.
        let mut results: Vec<_> = work
            .into_iter()
            .flat_map(|work| work.iter())
            .filter_map(|entity| calls.get(entity).ok())
            .map(|(run, done)| match done {
                Some(done) => (run.index, done.0.clone()),
                None => (
                    run.index,
                    run.call
                        .error_result(vec![ToolResultContent::text("interrupted by the user")]),
                ),
            })
            .collect();
        if !results.is_empty() {
            results.sort_by_key(|(index, _)| *index);
            conversation.0.push(ChatMessage::tool_results(
                results.into_iter().map(|(_, result)| result).collect(),
            ));
        }
        commands
            .entity(interrupt.agent)
            .despawn_related::<Work>()
            .remove::<NeedsModelCall>();
        *status = AgentStatus::Idle;
        notices.write(Notice::info(interrupt.agent, "turn stopped"));
        finished.write(TurnFinished {
            agent: interrupt.agent,
        });
    }
}
