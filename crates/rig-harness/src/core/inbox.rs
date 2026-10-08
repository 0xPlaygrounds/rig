//! Typing while the agent works. A message sent to a busy agent waits in
//! its [`Inbox`]: one sent with [`Submit`] steers the
//! running turn and goes to the model with its next call, after the tool
//! results it waits for; one sent with [`FollowUp`] goes once the turn
//! would end, as the next step of the same turn. A subagent's answer,
//! sent with [`Report`], goes like a follow-up. A turn that ends some
//! other way (stopped, failed) hands what was typed and not delivered back
//! to the views as [`Recalled`], so nothing typed is lost or sent unasked,
//! and puts the subagents' answers in the conversation for the next turn.
//! A delivered answer is logged with the subagent message it carries, so a
//! restore knows which answers arrived.

use std::collections::VecDeque;

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use rig_core::catalog::ModelSpec;

use super::agent::{ActiveTurn, Agent, AgentId, Connection, Conversation, Notice, Submit, TurnOf};
use super::attach;
use super::journal::{MessageRef, SessionLog};
use super::turn::CallModel;

/// What was typed to the agent while it worked, not yet sent.
#[derive(Component, Clone, Debug, Default)]
pub struct Inbox {
    /// Messages for the running turn, sent with its next model call.
    pub steering: Vec<String>,
    /// Messages for after the turn, sent one at a time when it would end.
    pub follow_ups: VecDeque<String>,
    /// Answers of the agent's subagents, sent like follow-ups after the
    /// user's.
    pub reports: VecDeque<Answer>,
}

impl Inbox {
    /// Whether nothing waits.
    pub fn is_empty(&self) -> bool {
        self.steering.is_empty() && self.follow_ups.is_empty() && self.reports.is_empty()
    }

    /// Takes every message the user typed, steering first, joined by blank
    /// lines.
    pub fn take_all(&mut self) -> String {
        let all: Vec<String> = self
            .steering
            .drain(..)
            .chain(self.follow_ups.drain(..))
            .collect();
        all.join("\n\n")
    }
}

/// A subagent's answer on its way to the agent that started it.
#[derive(Clone, Debug)]
pub struct Answer {
    /// The answer, headed by the subagent's id and task.
    pub text: String,
    /// The subagent's message it answers with.
    pub origin: MessageRef,
}

/// Send `text` to the agent once its running turn would end, or now when
/// it is idle. A slash command runs now either way.
#[derive(EntityEvent, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct FollowUp {
    /// The agent.
    pub entity: Entity,
    /// The text typed.
    pub text: String,
}

/// A subagent's answer for the agent that started it: sent like a
/// follow-up, so it starts a turn of an idle agent and waits for the end of
/// a busy one's.
#[derive(EntityEvent, Clone, Debug)]
pub(crate) struct Report {
    /// The agent that started the subagent.
    pub(crate) entity: Entity,
    /// The answer.
    pub(crate) answer: Answer,
}

/// Messages typed to the agent that its turn ended without sending, joined
/// in the order they were typed. A view puts them back in its input.
#[derive(Message, Clone, Debug)]
pub struct Recalled {
    /// The agent.
    pub agent: Entity,
    /// The messages.
    pub text: String,
}

/// The agent a message goes to, and what adding it needs.
pub(crate) struct Delivery<'a> {
    /// The agent.
    pub(crate) agent: Entity,
    /// Its id, which names its log.
    pub(crate) id: &'a AgentId,
    /// Its model, which decides whether images go with the message.
    pub(crate) spec: Option<&'static ModelSpec>,
    /// The log every message goes through.
    pub(crate) log: &'a SessionLog,
}

/// Queues a follow-up for a busy agent, or submits it to an idle one.
pub(crate) fn on_follow_up(
    follow_up: On<FollowUp>,
    mut agents: Query<(&mut Inbox, Has<ActiveTurn>), With<Agent>>,
    mut commands: Commands,
) {
    let agent = follow_up.entity;
    let text = follow_up.text.trim();
    let Ok((mut inbox, busy)) = agents.get_mut(agent) else {
        return;
    };
    if text.is_empty() {
        return;
    }
    if busy && !text.starts_with('/') {
        inbox.follow_ups.push_back(text.to_owned());
    } else {
        commands.trigger(Submit {
            entity: agent,
            text: text.to_owned(),
        });
    }
}

/// Queues a subagent's answer for a busy agent, or sends it to an idle one
/// in a new turn.
pub(crate) fn on_report(
    report: On<Report>,
    mut agents: Query<
        (
            &AgentId,
            &mut Inbox,
            &mut Conversation,
            Option<&Connection>,
            Has<ActiveTurn>,
        ),
        With<Agent>,
    >,
    log: Res<SessionLog>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let agent = report.entity;
    let Ok((id, mut inbox, mut conversation, connection, busy)) = agents.get_mut(agent) else {
        return;
    };
    if busy {
        inbox.reports.push_back(report.answer.clone());
        return;
    }
    let to = Delivery {
        agent,
        id,
        spec: connection.map(|connection| connection.spec),
        log: &log,
    };
    let answer = report.answer.clone();
    append_user(
        &to,
        &answer.text,
        Some(answer.origin),
        &mut conversation,
        &mut notices,
    );
    let turn = commands.spawn((Name::new("turn"), TurnOf(agent))).id();
    commands.trigger(CallModel { entity: turn });
}

/// Hands back what the user typed that a turn that just ended did not
/// send, and puts the subagents' answers it did not send in the
/// conversation, where the next turn reads them. A conversation left
/// ending in the user's message is logged as halted, so a restore does not
/// answer it.
pub(crate) fn recall_on_turn_end(
    end: On<Remove<ActiveTurn>>,
    mut agents: Query<(&AgentId, &mut Inbox, &mut Conversation, Option<&Connection>)>,
    log: Res<SessionLog>,
    mut recalled: MessageWriter<Recalled>,
    mut notices: MessageWriter<Notice>,
) {
    let agent = end.entity;
    let Ok((id, mut inbox, mut conversation, connection)) = agents.get_mut(agent) else {
        return;
    };
    if !inbox.reports.is_empty() {
        let to = Delivery {
            agent,
            id,
            spec: connection.map(|connection| connection.spec),
            log: &log,
        };
        let reports: Vec<Answer> = inbox.reports.drain(..).collect();
        for answer in reports {
            append_user(
                &to,
                &answer.text,
                Some(answer.origin),
                &mut conversation,
                &mut notices,
            );
        }
    }
    log.halt(id, &conversation);
    if !inbox.steering.is_empty() || !inbox.follow_ups.is_empty() {
        recalled.write(Recalled {
            agent,
            text: inbox.take_all(),
        });
    }
}

/// Moves the steering messages into the conversation, with their images:
/// into its last message when that is the user's, such as the tool results
/// the model waits for, so user and model keep taking turns. Whether there
/// were any.
pub(crate) fn deliver_steering(
    to: &Delivery<'_>,
    inbox: &mut Inbox,
    conversation: &mut Conversation,
    notices: &mut MessageWriter<Notice>,
) -> bool {
    if inbox.steering.is_empty() {
        return false;
    }
    for text in inbox.steering.drain(..) {
        append_user(to, &text, None, conversation, notices);
    }
    true
}

/// Moves the oldest follow-up into the conversation, the user's before
/// the subagents' answers. Whether there was one.
pub(crate) fn deliver_follow_up(
    to: &Delivery<'_>,
    inbox: &mut Inbox,
    conversation: &mut Conversation,
    notices: &mut MessageWriter<Notice>,
) -> bool {
    let (text, origin) = match inbox.follow_ups.pop_front() {
        Some(text) => (text, None),
        None => match inbox.reports.pop_front() {
            Some(answer) => (answer.text, Some(answer.origin)),
            None => return false,
        },
    };
    append_user(to, &text, origin, conversation, notices);
    true
}

/// Commits `text` as the user's, with the images it names; it goes into
/// the last message when that is the user's. `origin` names the subagent
/// answer it delivers.
fn append_user(
    to: &Delivery<'_>,
    text: &str,
    origin: Option<MessageRef>,
    conversation: &mut Conversation,
    notices: &mut MessageWriter<Notice>,
) {
    let (message, notes) = attach::user_message(text, to.spec);
    for note in notes {
        notices.write(Notice::info(to.agent, note));
    }
    to.log.commit(to.id, conversation, message, origin);
}
