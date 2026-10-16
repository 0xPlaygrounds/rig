//! CT-16: talk to a finished subagent.
//!
//! rig-subagents already keeps a `task` subagent after it reports (its
//! conversation, model, tools and saved state live on as an agent of the
//! session, restored on resume). This plugin
//! makes that usable:
//!
//! - `/subagents` lists the subagents: short id, task, model, turns,
//!   tokens, and when each last finished.
//! - `/talk <id>` asks the terminal view to show that subagent and send
//!   what the user types to it ([`rig_tui::ShowAgent`]); `/talk main` goes
//!   back. Its replies stay in its own conversation: a message from the
//!   user opens no request, so rig-subagents reports nothing to the parent.
//! - `follow_up {agent_id, message}` sends one of the caller's own
//!   subagents a queued message, marked with the call as its request, and
//!   answers the call with the subagent's reply to it: the final text of
//!   the step that message started, read from the subagent's conversation.
//! - The user's messages to a subagent are queued rather than steered, so a
//!   subagent answers the user and its parent one step at a time, in the
//!   order the messages arrived.
//! - At most [`KeptSubagents::keep`] finished subagents are kept; the ones
//!   that finished longest ago are retired.

use bevy_reflect::Reflect;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::Message;
use rig_harness::rig_core::now_ms;
use rig_subagents::{Lifecycle, OpenRequests, Subtask};
use rig_telemetry::AgentUsage;
use schemars::JsonSchema;
use serde::Deserialize;

/// The tool that asks one of the caller's finished subagents a follow-up.
pub const FOLLOW_UP: &str = "follow_up";

const FOLLOW_UP_DESCRIPTION: &str = "Ask one of your own subagents, by its id, a follow-up \
    question once it has reported. It answers from its own conversation, with everything it \
    already read, using its original tools; its reply is this call's result. When a subagent \
    already investigated something and you need one more detail from it, use this rather \
    than a new `task`.";

/// How many finished subagents a session keeps: the `[kept_subagents]`
/// table of the settings, `keep = 16` when not set.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct KeptSubagents {
    /// Beyond this many finished subagents, the ones that finished longest
    /// ago are retired, with the agents they spawned.
    pub keep: usize,
}

impl Default for KeptSubagents {
    fn default() -> Self {
        Self { keep: 16 }
    }
}

/// On a subagent: when its last turn finished, in milliseconds since the
/// Unix epoch. Saved with the session.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct FinishedAt(pub u64);

/// On a message `follow_up` sent: the call its reply answers.
#[derive(Component)]
struct FollowUpOf(Entity);

/// On a turn: the `follow_up` calls delivered into it and not answered
/// yet, each with its request, which marks its message.
#[derive(Component, Default)]
struct AnswersFollowUps(Vec<(Entity, RequestId)>);

/// Lists the subagents, switches the input to one with `/talk`, adds
/// `follow_up`, queues the user's messages to a subagent, and retires the
/// finished subagents beyond [`KeptSubagents`].
#[derive(Default)]
pub struct ResumeSubagentPlugin;

impl Plugin for ResumeSubagentPlugin {
    fn build(&self, app: &mut App) {
        let options = ToolOptions {
            footprint: Some(Touches::Independent),
        };
        app.add_setting::<KeptSubagents>("kept_subagents")
            .keeps_saved::<FinishedAt>()
            .add_open_tool(FOLLOW_UP, FOLLOW_UP_DESCRIPTION, options, on_follow_up)
            .add_command("subagents", "List the subagents of this session", list)
            .add_command(
                "talk",
                "Talk to a subagent: /talk <id>, or /talk main to go back",
                talk,
            )
            .add_observer(stamp_finished)
            .add_observer(mark_turn)
            .add_observer(step_ended)
            .add_observer(answer_follow_ups)
            .add_observer(undelivered)
            .add_systems(AgentPass, queue_user_lines.in_set(PrepareSystems::Input))
            .add_systems(Update, drop_oldest.after(AgentLoop));
    }
}

/// The arguments of a `follow_up` call.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct FollowUpArgs {
    /// The id of one of your subagents, as `task` returned it.
    agent_id: String,
    /// The follow-up, in full.
    message: String,
}

/// Whether `id` names `wanted`, by its id or short id.
fn names(id: &AgentId, wanted: &str) -> bool {
    let wanted = wanted.trim();
    id.0 == wanted || id.short() == wanted
}

/// Sends the follow-up to the caller's own subagent, or refuses the call.
fn on_follow_up(
    called: On<ToolCalled<FollowUpArgs>>,
    subagents: Query<(Entity, &AgentId, &SpawnedBy, Option<&OpenRequests>), With<Subtask>>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let own: Vec<_> = subagents
        .iter()
        .filter(|(_, _, by, _)| by.0 == called.agent)
        .collect();
    let target = own
        .iter()
        .find(|(_, id, ..)| names(id, &called.args.agent_id));
    let refusal = match target {
        None => {
            let ids: Vec<String> = own
                .iter()
                .map(|(_, id, ..)| format!("`{}`", id.short()))
                .collect();
            let ids = if ids.is_empty() {
                "none".to_owned()
            } else {
                ids.join(", ")
            };
            Some(format!(
                "`{}` is not a subagent of yours. Yours: {ids}. Nothing was sent.",
                called.args.agent_id.trim()
            ))
        }
        Some(_) if called.args.message.trim().is_empty() => {
            Some("`message` must not be empty. Nothing was sent.".to_owned())
        }
        Some((_, id, _, Some(open))) if !open.0.is_empty() => Some(format!(
            "`{}` is still working on its task; its report will come as a message. Nothing \
             was sent.",
            id.short()
        )),
        Some(&(agent, ..)) => {
            let request = RequestId(called.tool_call.id.to_string());
            let author = Author::agent(called.agent_id.clone(), Some(request));
            let message = Incoming::new(called.args.message.trim(), DeliveryMode::Queue);
            commands.spawn((
                message.with_author(author),
                ForAgent(agent),
                Effect::with_parent_id(Some(called.effect)),
                FollowUpOf(called.call),
            ));
            None
        }
    };
    if let (Some(why), Ok(mut decision)) = (refusal, calls.get_mut(called.call)) {
        let refused = ToolResult::failed(ToolExecutionError::other(why));
        decision.decide(Decider::plugin(FOLLOW_UP), refused);
    }
}

/// Marks the turn a follow-up went into with the call it answers.
fn mark_turn(
    delivered: On<Add<DeliveredIn>>,
    messages: Query<(&DeliveredIn, &FollowUpOf, &Incoming)>,
    mut turns: Query<&mut AnswersFollowUps>,
    mut commands: Commands,
) {
    let Ok((&DeliveredIn(turn), &FollowUpOf(call), message)) = messages.get(delivered.entity)
    else {
        return;
    };
    let Some(request) = message.author.request.clone() else {
        return;
    };
    match turns.get_mut(turn) {
        Ok(mut answers) => answers.0.push((call, request)),
        Err(_) => {
            commands
                .entity(turn)
                .try_insert(AnswersFollowUps(vec![(call, request)]));
        }
    }
}

/// The reply in `conversation` to the message marked with `request`: the
/// final text of the step it started, once that step was answered and
/// another message followed; `None` before.
fn reply_to(conversation: &Conversation, request: &RequestId) -> Option<String> {
    let marked = |at: &usize| {
        conversation
            .author(*at)
            .and_then(|author| author.request.as_ref())
            == Some(request)
    };
    let at = (0..conversation.messages().len()).find(marked)?;
    let mut reply = None;
    for message in conversation.messages().iter().skip(at + 1) {
        match &**message {
            Message::Assistant(_) => reply = message.final_answer().or(reply),
            Message::User { content } => {
                let results = content
                    .iter()
                    .all(|part| matches!(part, UserContent::ToolResult(_)));
                // Messages delivered with it come before its step's reply.
                if !results && reply.is_some() {
                    return reply;
                }
            }
            _ => {}
        }
    }
    None
}

/// Answers the follow-ups of a turn whose step was answered: one that
/// another queued message carried on.
fn step_ended(
    delivered: On<Add<DeliveredIn>>,
    messages: Query<(&DeliveredIn, &Incoming, &ForAgent), Without<FollowUpOf>>,
    mut turns: Query<&mut AnswersFollowUps>,
    agents: Query<&Conversation>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let Ok((&DeliveredIn(turn), message, &ForAgent(agent))) = messages.get(delivered.entity) else {
        return;
    };
    if message.mode != DeliveryMode::Queue {
        return;
    }
    let (Ok(mut answers), Ok(conversation)) = (turns.get_mut(turn), agents.get(agent)) else {
        return;
    };
    answers.0.retain(|(call, request)| {
        let Some(reply) = reply_to(conversation, request) else {
            return true;
        };
        if let Ok(mut decision) = calls.get_mut(*call) {
            decision.decide(
                Decider::plugin(FOLLOW_UP),
                ToolResult::success(reply.into()),
            );
        }
        false
    });
}

/// Answers each follow-up of a turn that ended still unanswered: the last
/// is answered by the turn's final answer, or why there is none.
fn answer_follow_ups(
    settled: On<Add<Settled>>,
    turns: Query<(&AnswersFollowUps, &ForAgent, Option<&TurnOutcome>)>,
    agents: Query<&Conversation>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let Ok((answers, &ForAgent(agent), outcome)) = turns.get(settled.entity) else {
        return;
    };
    let final_answer = match outcome {
        Some(TurnOutcome::Answered(message)) => message
            .final_answer()
            .ok_or_else(|| "The subagent ended without a final message.".to_owned()),
        Some(TurnOutcome::Failed(why)) => Err(format!("The subagent failed: {why}")),
        Some(TurnOutcome::Stopped(why)) => Err(format!(
            "The subagent was stopped: {}",
            why.as_deref().unwrap_or("by the user")
        )),
        None => Err("The subagent's turn ended without an outcome.".to_owned()),
    };
    let conversation = agents.get(agent).ok();
    for (call, request) in &answers.0 {
        let reply = conversation.and_then(|conversation| reply_to(conversation, request));
        let result = match (reply.as_ref(), &final_answer) {
            (Some(reply), _) | (None, Ok(reply)) => ToolResult::success(reply.clone().into()),
            (None, Err(why)) => ToolResult::failed(ToolExecutionError::other(why.clone())),
        };
        if let Ok(mut decision) = calls.get_mut(*call) {
            decision.decide(Decider::plugin(FOLLOW_UP), result);
        }
    }
}

/// Refuses the call of a follow-up that settled without going into a turn,
/// such as one a plugin took over.
fn undelivered(
    settled: On<Add<Settled>>,
    messages: Query<&FollowUpOf, Without<DeliveredIn>>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    if let Ok(&FollowUpOf(call)) = messages.get(settled.entity)
        && let Ok(mut decision) = calls.get_mut(call)
    {
        let why = "The follow-up never reached the subagent.";
        decision.decide(
            Decider::plugin(FOLLOW_UP),
            ToolResult::failed(ToolExecutionError::other(why.to_owned())),
        );
    }
}

/// Keeps when each subagent last finished.
fn stamp_finished(
    inserted: On<Insert<Lifecycle>>,
    agents: Query<&Lifecycle>,
    mut commands: Commands,
) {
    if let Ok(Lifecycle::Finished(_)) = agents.get(inserted.entity) {
        commands
            .entity(inserted.entity)
            .try_insert(FinishedAt(now_ms()));
    }
}

/// Retires the finished subagents beyond [`KeptSubagents::keep`], those
/// that finished longest ago first; never one the user looks at.
fn drop_oldest(
    changed: Query<(), Changed<FinishedAt>>,
    kept: Query<(Entity, &FinishedAt, Has<ActiveTurn>, Option<&OpenRequests>), With<Subtask>>,
    (settings, shown): (Res<KeptSubagents>, Option<Res<ShownAgent>>),
    mut commands: Commands,
) {
    if changed.is_empty() {
        return;
    }
    let mut finished: Vec<(u64, Entity)> = kept
        .iter()
        .filter(|(_, _, busy, open)| !busy && open.is_none_or(|open| open.0.is_empty()))
        .map(|(agent, at, ..)| (at.0, agent))
        .collect();
    let excess = finished.len().saturating_sub(settings.keep);
    finished.sort_unstable();
    let looked_at = shown.and_then(|shown| shown.0);
    let oldest = finished.into_iter().take(excess).map(|(_, agent)| agent);
    for agent in oldest.filter(|&agent| Some(agent) != looked_at) {
        commands.trigger(Retire { entity: agent });
    }
}

/// Queues what the user sends a subagent rather than steering its running
/// step with it, so the subagent answers each message, the user's or its
/// parent's follow-up, in a step of its own, in the order they came.
fn queue_user_lines(
    mut messages: Query<(&mut Incoming, &ForAgent), (Added<Admitted>, IncomingMessages)>,
    subagents: Query<(), With<Subtask>>,
) {
    for (mut message, &ForAgent(agent)) in &mut messages {
        let typed = message.author.kind == AuthorKind::User && message.command().is_none();
        if typed && message.mode == DeliveryMode::Steer && subagents.contains(agent) {
            message.mode = DeliveryMode::Queue;
        }
    }
}

/// The subagent `wanted` names, by id or short id.
fn find<'a>(
    subagents: impl IntoIterator<Item = (Entity, &'a AgentId, Option<&'a AgentTitle>)>,
    wanted: &str,
) -> Option<(Entity, &'a AgentId, Option<&'a AgentTitle>)> {
    subagents.into_iter().find(|(_, id, _)| names(id, wanted))
}

/// `/talk <id>`: the terminal view shows the subagent and sends what the
/// user types to it; `/talk main` goes back to the main agent.
fn talk(
    In(args): In<CommandArgs>,
    subagents: Query<(Entity, &AgentId, Option<&AgentTitle>), With<Subtask>>,
    roots: MainAgentQuery,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let wanted = args.args.trim();
    if wanted.is_empty() {
        notices.write(Notice::error(
            args.agent,
            "/talk needs a subagent's id, or main",
        ));
        return;
    }
    if wanted == "main" {
        if let Some(main) = main_agent(&roots) {
            commands.trigger(rig_tui::ShowAgent { entity: main });
            notices.write(Notice::info(main, "Talking to the main agent."));
        }
        return;
    }
    let Some((agent, id, title)) = find(subagents, wanted) else {
        let why = format!("/talk: no subagent `{wanted}`; /subagents lists them");
        notices.write(Notice::error(args.agent, why));
        return;
    };
    commands.trigger(rig_tui::ShowAgent { entity: agent });
    let task = title.map(|title| format!(" (\"{}\")", title.0));
    let text = format!(
        "Talking to subagent `{}`{}. /talk main goes back.",
        id.short(),
        task.unwrap_or_default()
    );
    notices.write(Notice::info(agent, text));
}

/// A subagent as `/subagents` shows it.
type Listed<'a> = (
    &'a AgentId,
    Option<&'a AgentTitle>,
    Option<&'a ModelChoice>,
    &'a Conversation,
    Option<&'a AgentUsage>,
    Option<&'a FinishedAt>,
    Has<ActiveTurn>,
);

/// How long ago `then`, in milliseconds since the Unix epoch, was.
fn ago(then: u64) -> String {
    let secs = now_ms().saturating_sub(then) / 1000;
    match secs {
        0..60 => format!("{secs}s ago"),
        60..3600 => format!("{}m ago", secs / 60),
        _ => format!("{}h ago", secs / 3600),
    }
}

/// `/subagents`: each subagent's short id, task, model, turns, tokens and
/// when it finished.
fn list(
    In(args): In<CommandArgs>,
    subagents: Query<Listed, With<Subtask>>,
    mut notices: MessageWriter<Notice>,
) {
    let mut rows: Vec<(u64, String)> = subagents
        .iter()
        .map(
            |(id, title, model, conversation, spending, finished, busy)| {
                let task = title.map_or("", |title| title.0.as_str());
                let model = model.map_or("?", |model| model.0.as_str());
                let turns = conversation.turns().count();
                let tokens = spending.and_then(|spent| spent.total.tokens.total_tokens);
                let tokens = tokens.map_or_else(|| "?".to_owned(), |tokens| tokens.to_string());
                let when = match (busy, finished) {
                    (true, _) => "working".to_owned(),
                    (false, Some(at)) => format!("finished {}", ago(at.0)),
                    (false, None) => "idle".to_owned(),
                };
                let row = format!(
                    "`{}` \"{task}\": {model}, {turns} turns, {tokens} tokens, {when}",
                    id.short()
                );
                (finished.map_or(u64::MAX, |at| at.0), row)
            },
        )
        .collect();
    rows.sort();
    let text = if rows.is_empty() {
        "No subagents.".to_owned()
    } else {
        let rows: Vec<String> = rows.into_iter().map(|(_, row)| row).collect();
        format!(
            "Subagents (/talk <id> to talk to one):\n{}",
            rows.join("\n")
        )
    };
    notices.write(Notice::info(args.agent, text));
}

#[cfg(test)]
mod tests;
