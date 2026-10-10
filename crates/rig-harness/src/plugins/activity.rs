//! What the agents are doing, kept for views: every agent's [`Activity`]
//! (its status, its open tool calls and a preview of what it writes) and
//! the [`MessageFeed`] of recent deliveries. A view reads them instead of
//! deriving them from turns and calls; the agent tree is the agents'
//! [`SpawnedBy`] and [`Spawned`] relationship.
//!
//! [`Activity`] is updated in `PostUpdate`, in [`ActivitySystems`], and
//! changes only when what it says changes, so `Changed<Activity>` tells a
//! view to redraw.

use std::collections::VecDeque;
use std::fmt;

use rig_core::completion::{AssistantContent, Message};
use rig_ecs::agent::{Calls, Partial, Queued};
use rig_ecs::prelude::*;
use rig_ecs::turn::ModelCall;

/// The most characters a [`Preview`] keeps, from the end of the text.
pub const PREVIEW_CHARS: usize = 2000;
/// The most deliveries the [`MessageFeed`] keeps.
pub const FEED_LEN: usize = 64;
/// The most characters a [`FedMessage`] keeps, from the start of the text.
pub const FEED_CHARS: usize = 1000;

/// Keeps every agent's [`Activity`] and the [`MessageFeed`].
#[derive(Default)]
pub struct ActivityPlugin;

impl Plugin for ActivityPlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components::<Agent, Activity>()
            .init_resource::<MessageFeed>()
            .add_systems(PostUpdate, update_activity.in_set(ActivitySystems))
            .add_observer(feed_deliveries);
    }
}

/// The system in `PostUpdate` that updates each agent's [`Activity`]. A
/// view that reads it in `PostUpdate` runs after it.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ActivitySystems;

/// What an agent is doing; on every agent.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component)]
pub struct Activity {
    /// Its status.
    pub status: Status,
    /// Its turn's tool calls without an output yet, in order.
    pub tools: Vec<ToolActivity>,
    /// The end of what it streams now, or of its last reply when idle.
    pub preview: Option<Preview>,
}

impl Activity {
    /// Whether a turn runs.
    pub fn is_busy(&self) -> bool {
        self.status != Status::Idle
    }
}

/// An agent's status.
#[derive(Reflect, Clone, Debug, Default, PartialEq, Eq)]
pub enum Status {
    /// No turn runs.
    #[default]
    Idle,
    /// Waiting for its model.
    Thinking,
    /// Its tools run.
    RunningTools,
    /// A plugin's call of the turn runs, such as a summary of the older
    /// conversation; the call's [`Name`], such as `compacting`.
    Busy(String),
    /// Waiting `seconds` before retry `attempt` of a failed model call.
    Retrying {
        /// The retry's number, from 1.
        attempt: u32,
        /// Whole seconds left, rounded up.
        seconds: u64,
    },
}

impl fmt::Display for Status {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Idle => f.write_str("idle"),
            Self::Thinking => f.write_str("thinking"),
            Self::RunningTools => f.write_str("running tools"),
            Self::Busy(what) => f.write_str(what),
            Self::Retrying { attempt, seconds } => write!(f, "retry {attempt} in {seconds}s"),
        }
    }
}

/// A tool call of the turn without an output yet.
#[derive(Reflect, Clone, Debug, PartialEq, Eq)]
pub struct ToolActivity {
    /// The tool's name.
    pub name: String,
    /// Whether it waits for earlier calls of its reply.
    pub queued: bool,
}

/// What a [`Preview`] shows.
#[derive(Reflect, Clone, Copy, Debug, PartialEq, Eq)]
pub enum PreviewKind {
    /// The answer streaming in.
    Text,
    /// The reasoning streaming in, before any answer text.
    Reasoning,
    /// The text of the idle agent's last reply.
    LastReply,
}

/// The last [`PREVIEW_CHARS`] characters of an agent's text.
#[derive(Reflect, Clone, Debug, PartialEq, Eq)]
pub struct Preview {
    /// Whose text it is.
    pub kind: PreviewKind,
    /// The text.
    pub text: String,
}

impl Preview {
    fn new(kind: PreviewKind, text: &str) -> Self {
        let start = text
            .char_indices()
            .rev()
            .nth(PREVIEW_CHARS - 1)
            .map_or(0, |(index, _)| index);
        Self {
            kind,
            text: text.get(start..).unwrap_or_default().to_owned(),
        }
    }
}

/// Recent deliveries to agents, oldest first: the user's messages,
/// messages between agents and plugins' messages, at most [`FEED_LEN`].
/// A delivery is recorded when it is sent, whether the agent reads it now
/// or after its turn.
#[derive(Resource, Reflect, Default, Debug)]
#[reflect(Resource)]
pub struct MessageFeed {
    entries: VecDeque<FedMessage>,
}

impl MessageFeed {
    /// The deliveries, oldest first.
    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &FedMessage> + ExactSizeIterator {
        self.entries.iter()
    }
}

/// One delivery of the [`MessageFeed`].
#[derive(Reflect, Clone, Debug)]
pub struct FedMessage {
    /// The agent it went to.
    pub to: Entity,
    /// The agent it came from, when an agent sent it.
    pub from: Option<Entity>,
    /// Where it came from.
    pub origin: Origin,
    /// Its first [`FEED_CHARS`] characters.
    pub text: String,
}

fn feed_deliveries(
    delivery: On<Deliver>,
    agents: Query<(Entity, &AgentId), With<Agent>>,
    mut feed: ResMut<MessageFeed>,
) {
    let from = delivery.origin.from.as_ref().and_then(|from| {
        agents
            .iter()
            .find_map(|(entity, id)| (id == from).then_some(entity))
    });
    if feed.entries.len() >= FEED_LEN {
        feed.entries.pop_front();
    }
    feed.entries.push_back(FedMessage {
        to: delivery.entity,
        from,
        origin: delivery.origin.clone(),
        text: delivery.text.chars().take(FEED_CHARS).collect(),
    });
}

/// The agents and their turns.
type Agents<'w, 's> = Query<
    'w,
    's,
    (
        &'static mut Activity,
        Option<&'static ActiveTurn>,
        Ref<'static, Conversation>,
    ),
    With<Agent>,
>;

/// Each agent's activity, from its turn's calls. An idle agent's preview
/// is read again only when its conversation changed.
fn update_activity(
    mut agents: Agents,
    turns: Query<&Calls>,
    partials: Query<&Partial>,
    tools: Query<(&ToolCallRun, Has<Queued>, Has<ToolOutput>)>,
    waits: Query<&Backoff>,
    others: Query<&Name, (Without<ModelCall>, Without<ToolCallRun>, Without<Backoff>)>,
) {
    for (mut activity, turn, conversation) in &mut agents {
        let Some(turn) = turn else {
            if activity.is_busy() || conversation.is_changed() {
                activity.set_if_neq(Activity {
                    status: Status::Idle,
                    tools: Vec::new(),
                    preview: last_reply(conversation.messages()),
                });
            }
            continue;
        };
        let calls: Vec<Entity> = turns
            .get(turn.turn())
            .map(|calls| calls.iter().collect())
            .unwrap_or_default();
        let status = if let Some(wait) = calls.iter().find_map(|call| waits.get(*call).ok()) {
            Status::Retrying {
                attempt: wait.attempt,
                seconds: wait.seconds_left(),
            }
        } else if let Some(name) = calls.iter().find_map(|call| others.get(*call).ok()) {
            Status::Busy(name.as_str().to_owned())
        } else if calls.iter().any(|call| tools.contains(*call)) {
            Status::RunningTools
        } else {
            Status::Thinking
        };
        let open_tools = calls
            .iter()
            .filter_map(|call| tools.get(*call).ok())
            .filter(|(_, _, done)| !done)
            .map(|(run, queued, _)| ToolActivity {
                name: run.call.function.name.as_str().to_owned(),
                queued,
            })
            .collect();
        let partial = calls.iter().find_map(|call| partials.get(*call).ok());
        let preview = partial.and_then(|partial| {
            if !partial.text.is_empty() {
                Some(Preview::new(PreviewKind::Text, &partial.text))
            } else if !partial.reasoning.is_empty() {
                Some(Preview::new(PreviewKind::Reasoning, &partial.reasoning))
            } else {
                None
            }
        });
        activity.set_if_neq(Activity {
            status,
            tools: open_tools,
            preview,
        });
    }
}

/// The text of the last assistant message that has some.
fn last_reply(messages: &[Message]) -> Option<Preview> {
    messages.iter().rev().find_map(|message| {
        let Message::Assistant(assistant) = message else {
            return None;
        };
        let text = assistant
            .content
            .iter()
            .filter_map(|content| match content {
                AssistantContent::Text(text) => Some(text.text.as_str()),
                _ => None,
            })
            .collect::<Vec<_>>()
            .join("\n");
        (!text.is_empty()).then(|| Preview::new(PreviewKind::LastReply, &text))
    })
}
