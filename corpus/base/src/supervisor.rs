//! A supervisor agent: `/supervise <goal>` gives the agent it is typed for
//! (the worker) a supervisor of its own, a `rig_basics::subagents` helper
//! ([`SpawnSubagent::helper`]) with its own conversation, model and one
//! tool, `verdict`. Every `calls` tool calls of the worker (10 by default),
//! and every `minutes` when that is set, the supervisor is sent the goal and
//! a digest of what the worker did since the last check: what it was asked,
//! the tools it called with short results, and what it said. It answers
//! with `verdict`: `continue`, `steer` with a message the worker's running
//! turn reads with its next model call, or `stop` with a reason, which
//! interrupts the worker's turn and tells the user why. The worker never
//! waits for it.
//!
//! The [`Supervisor`] component on the supervisor is saved with the session,
//! and so is its conversation, so it remembers its earlier checks.
//! `/supervise off` retires it.

use rig_basics::subagents::{SpawnSubagent, SpawnedAgent, SubagentsPlugin};
use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The supervisor's tool.
pub const VERDICT: &str = "verdict";

/// The supervisor's name, and who its checks and steering come from.
const NAME: &str = "supervisor";

/// Tool calls between checks unless `/supervise` says `calls=N`.
pub const EVERY_CALLS: u32 = 10;

/// Lines of activity a check holds at most: the latest.
const MAX_LINES: usize = 60;

/// Characters kept of a request or of what the worker said, of a call's
/// arguments, and of a call's result.
const TEXT: usize = 400;
const ARGS: usize = 160;
const RESULT: usize = 200;

const HELP: &str = "Attach a supervisor: /supervise [calls=N] [minutes=M] [model=vendor/model] \
                    <goal>, or /supervise off";

const PROMPT: &str = "You supervise another coding agent, the worker, on behalf of the user. \
    Each check gives you the user's goal and what the worker did since your last check: what \
    it was asked, the tools it called with the start of their results, and what it said.\n\
    \n\
    - Answer each check with exactly one `verdict` call, then end your turn with one short \
    line.\n\
    - `continue` when the worker makes progress toward the goal.\n\
    - `steer` with a short, concrete instruction when it drifts from the goal, skips \
    something the goal needs, or repeats itself.\n\
    - `stop` with the reason when it is stuck, loops without progress, or does something \
    harmful or far outside the goal.\n\
    - The worker keeps working while you think: judge the trend, not one step.";

/// The plugin's stage systems, to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SupervisorSystems {
    /// In [`PrepareSystems::Edit`]: offers `verdict` to supervisors only.
    Prepare,
    /// In [`Stage::Review`], after [`ReviewSystems::Act`], so a check shows
    /// the conversation as the reviewers left it: marks a hired supervisor,
    /// and sends a check every so many tool calls.
    Review,
}

/// Adds `/supervise`, the `verdict` tool and the checks.
#[derive(Default)]
pub struct SupervisorPlugin;

impl Plugin for SupervisorPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SubagentsPlugin>() {
            app.add_plugins(SubagentsPlugin);
        }
        let options = ToolOptions {
            footprint: Some(Touches::Independent),
        };
        let describe = "Give your verdict on the worker's progress: continue, steer it with a \
                        message, or stop its turn with a reason. Answer every check with one \
                        call.";
        app.add_open_tool(VERDICT, describe, options, on_verdict)
            .add_command("supervise", HELP, supervise)
            .add_observer(arm_clock)
            .add_observer(check_by_time)
            .configure_sets(
                AgentPass,
                (
                    SupervisorSystems::Prepare.in_set(PrepareSystems::Edit),
                    SupervisorSystems::Review
                        .in_set(Stage::Review)
                        .after(ReviewSystems::Act),
                ),
            )
            .add_systems(
                AgentPass,
                (
                    verdict_for_supervisors.in_set(SupervisorSystems::Prepare),
                    (hired, count_calls).in_set(SupervisorSystems::Review),
                ),
            );
    }
}

/// On a supervisor, a helper [`SpawnedBy`] the agent it supervises: what it
/// holds the worker to, how often it looks, and where its last check left
/// off.
#[derive(Component, Reflect, Clone, Debug, Default, Serialize, Deserialize)]
#[reflect(
    opaque,
    Component,
    Saved,
    Default,
    Clone,
    Debug,
    Serialize,
    Deserialize
)]
pub struct Supervisor {
    /// The goal the supervisor holds the worker to.
    pub goal: String,
    /// Tool calls between checks; 0 checks by time only.
    pub every_calls: u32,
    /// Minutes between checks; 0 checks by calls only.
    pub every_minutes: u32,
    /// The last message of the worker's conversation it has seen.
    pub seen: Option<MessageId>,
    /// The worker's tool calls since the last check.
    pub calls: u32,
    /// The checks it was sent.
    pub checks: u32,
}

/// On the request for a supervisor: the [`Supervisor`] it becomes.
#[derive(Component)]
struct Hire(Supervisor);

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct VerdictArgs {
    /// `continue`, `steer` or `stop`.
    decision: Ruling,
    /// For `steer`, what the worker should do; for `stop`, why it stops.
    #[serde(default)]
    message: Option<String>,
}

#[derive(Deserialize, JsonSchema, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum Ruling {
    Continue,
    Steer,
    Stop,
}

/// The supervisor among the agents `worker` spawned.
fn supervisor_of(
    spawned: Option<&Spawned>,
    supervisors: &Query<&mut Supervisor>,
) -> Option<Entity> {
    spawned
        .into_iter()
        .flat_map(Spawned::iter)
        .find(|&agent| supervisors.contains(agent))
}

/// The pace and goal `/supervise` was given, and the model, if any.
fn parse(args: &str) -> Result<(u32, u32, Option<String>, String), String> {
    let (mut calls, mut minutes, mut model) = (EVERY_CALLS, 0, None);
    let mut rest = args.trim();
    while let Some((word, after)) = rest.split_once(char::is_whitespace).or(Some((rest, ""))) {
        let number = |value: &str| value.parse().map_err(|_| format!("{word}: not a number"));
        match word.split_once('=') {
            Some(("calls", value)) => calls = number(value)?,
            Some(("minutes", value)) => minutes = number(value)?,
            Some(("model", value)) => model = Some(value.to_owned()),
            _ => break,
        }
        rest = after.trim_start();
    }
    if rest.is_empty() {
        return Err(format!("/supervise needs a goal. {HELP}."));
    }
    if calls == 0 && minutes == 0 {
        return Err("calls=0 needs minutes=M.".to_owned());
    }
    Ok((calls, minutes, model, rest.to_owned()))
}

/// `/supervise [calls=N] [minutes=M] [model=vendor/model] <goal>`, or
/// `/supervise off`.
fn supervise(
    In(args): In<CommandArgs>,
    workers: Query<(&Conversation, Option<&Spawned>)>,
    supervisors: Query<&mut Supervisor>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let worker = args.agent;
    let Ok((conversation, spawned)) = workers.get(worker) else {
        return;
    };
    let current = supervisor_of(spawned, &supervisors);
    if args.args.trim() == "off" {
        match current {
            Some(supervisor) => {
                commands.trigger(Retire { entity: supervisor });
                notices.write(Notice::info(worker, "Supervision off."));
            }
            None => {
                notices.write(Notice::error(worker, "This agent has no supervisor."));
            }
        }
        return;
    }
    if supervisors.contains(worker) {
        notices.write(Notice::error(worker, "A supervisor is not supervised."));
        return;
    }
    let (every_calls, every_minutes, model, goal) = match parse(&args.args) {
        Ok(parsed) => parsed,
        Err(why) => {
            notices.write(Notice::error(worker, why));
            return;
        }
    };
    let settings = Supervisor {
        goal: goal.clone(),
        every_calls,
        every_minutes,
        seen: last_message(conversation),
        calls: 0,
        checks: 0,
    };
    match current {
        Some(supervisor) => {
            let checks = supervisors.get(supervisor).map_or(0, |old| old.checks);
            commands
                .entity(supervisor)
                .insert(Supervisor { checks, ..settings });
            if let Some(model) = model {
                commands.trigger(SetModel {
                    entity: supervisor,
                    model,
                });
            }
        }
        None => {
            let request = SpawnSubagent {
                tools: Some(vec![VERDICT.to_owned()]),
                model,
                ..SpawnSubagent::helper(NAME).system_prompt(PROMPT)
            };
            commands.spawn((request, ForAgent(worker), Hire(settings)));
        }
    }
    let mut pace = Vec::new();
    if every_calls > 0 {
        pace.push(format!("every {every_calls} tool calls"));
    }
    if every_minutes > 0 {
        pace.push(format!("every {every_minutes} minutes"));
    }
    let said = format!(
        "Supervising toward \"{goal}\", checking {}.",
        pace.join(" and ")
    );
    notices.write(Notice::info(worker, said));
}

/// Makes each helper spawned for a `/supervise` its worker's supervisor,
/// or tells the user why none was.
fn hired(
    requests: Query<
        (&Hire, &Outcome, Option<&SpawnedAgent>, &ForAgent),
        (Added<Outcome>, PluginEffects),
    >,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (Hire(settings), outcome, spawned, &ForAgent(worker)) in &requests {
        match (spawned, &outcome.0) {
            (Some(&SpawnedAgent(supervisor)), _) => {
                commands.entity(supervisor).insert(settings.clone());
            }
            (None, Err(report)) => {
                let why = format!("No supervisor was started: {}", report.message);
                notices.write(Notice::error(worker, why));
            }
            (None, Ok(_)) => {}
        }
    }
}

/// Counts the tool results committed to each supervised worker's
/// conversation, and sends its supervisor a check every `every_calls`.
fn count_calls(
    mut committed: MessageReader<ConversationChanged>,
    workers: Query<(&Conversation, Option<&Spawned>)>,
    mut supervisors: Query<&mut Supervisor>,
    mut commands: Commands,
) {
    for change in committed.read() {
        let Change::Push { entries } = &change.change.change else {
            continue;
        };
        let results = entries
            .iter()
            .filter_map(|entry| match &*entry.message {
                message::Message::User { content } => Some(content),
                _ => None,
            })
            .flatten()
            .filter(|item| matches!(item, UserContent::ToolResult(_)))
            .count();
        if results == 0 {
            continue;
        }
        let Ok((conversation, spawned)) = workers.get(change.agent) else {
            continue;
        };
        let Some(supervisor) = supervisor_of(spawned, &supervisors) else {
            continue;
        };
        let Ok(mut settings) = supervisors.get_mut(supervisor) else {
            continue;
        };
        settings.calls += u32::try_from(results).unwrap_or(u32::MAX);
        if settings.every_calls > 0 && settings.calls >= settings.every_calls {
            send_check(&mut commands, supervisor, &mut settings, conversation);
        }
    }
}

/// A supervisor that checks by time is due `every_minutes` from when it
/// was given its settings, on its own entity.
fn arm_clock(set: On<Insert<Supervisor>>, supervisors: Query<&Supervisor>, mut commands: Commands) {
    if let Ok(settings) = supervisors.get(set.entity)
        && settings.every_minutes > 0
    {
        commands.entity(set.entity).insert(minutes(settings));
    }
}

fn minutes(settings: &Supervisor) -> WakeAt {
    WakeAt::after(Duration::from_secs(60 * u64::from(settings.every_minutes)))
}

/// Sends a supervisor whose time came a check, when its worker did
/// anything since the last one, and arms its clock again.
fn check_by_time(
    due: On<Add<Due>>,
    mut supervisors: Query<(&mut Supervisor, &SpawnedBy)>,
    workers: Query<&Conversation>,
    mut commands: Commands,
) {
    let supervisor = due.entity;
    let Ok((mut settings, &SpawnedBy(worker))) = supervisors.get_mut(supervisor) else {
        return;
    };
    if let Ok(conversation) = workers.get(worker)
        && unseen(conversation, settings.seen) < conversation.messages().len()
    {
        send_check(&mut commands, supervisor, &mut settings, conversation);
    }
    commands.entity(supervisor).insert(minutes(&settings));
}

/// The id of the last message of `conversation`.
fn last_message(conversation: &Conversation) -> Option<MessageId> {
    let last = conversation.messages().len().checked_sub(1)?;
    conversation.id(last)
}

/// Where the messages after `seen` begin in `conversation`: at the start
/// when an edit took `seen` out, as a compaction's summary then stands for
/// what came before.
fn unseen(conversation: &Conversation, seen: Option<MessageId>) -> usize {
    seen.and_then(|seen| conversation.index_of(seen))
        .map_or(0, |at| at + 1)
}

/// Sends `supervisor` the goal and what the worker did since the last
/// check. A busy supervisor reads it once its turn would end.
fn send_check(
    commands: &mut Commands,
    supervisor: Entity,
    settings: &mut Supervisor,
    conversation: &Conversation,
) {
    let messages = conversation.messages();
    let text = digest(
        &settings.goal,
        messages,
        unseen(conversation, settings.seen),
    );
    let check = Deliver::new(supervisor, text, DeliveryMode::Queue);
    commands.trigger(check.with_author(Author::plugin(NAME)));
    settings.checks += 1;
    settings.seen = last_message(conversation);
    settings.calls = 0;
}

/// The text of a check: the goal, the worker's activity in `messages` from
/// `from` on, and its last text.
fn digest(goal: &str, messages: &[std::sync::Arc<message::Message>], from: usize) -> String {
    let mut lines = Vec::new();
    for message in messages.get(from.min(messages.len())..).unwrap_or_default() {
        match &**message {
            message::Message::User { content } => lines.extend(content.iter().filter_map(asked)),
            message::Message::Assistant(reply) => {
                lines.extend(reply.content.iter().filter_map(did))
            }
            message::Message::System { .. } => {}
        }
    }
    let left_out = lines.len().saturating_sub(MAX_LINES);
    let mut activity = lines.split_off(left_out).join("\n");
    if left_out > 0 {
        activity = format!("({left_out} earlier lines left out)\n{activity}");
    }
    if activity.is_empty() {
        activity = "(nothing since the last check)".to_owned();
    }
    let last = messages.iter().rev().find_map(|message| match &**message {
        message::Message::Assistant(reply) => {
            let text = reply.text();
            (!text.trim().is_empty()).then(|| clip(&text, TEXT))
        }
        message::Message::User { .. } | message::Message::System { .. } => None,
    });
    let last = last.unwrap_or_else(|| "(none yet)".to_owned());
    format!(
        "Goal: {goal}\n\nThe worker's activity since your last check:\n{activity}\n\nIts last \
         text: {last}\n\nGive your `verdict`."
    )
}

/// A line for what the worker was asked, or a tool's result.
fn asked(item: &UserContent) -> Option<String> {
    match item {
        UserContent::Text(text) => Some(format!("- request: {}", clip(&text.text, TEXT))),
        UserContent::ToolResult(result) => {
            let text: Vec<&str> = result
                .content
                .iter()
                .filter_map(|part| match part {
                    message::ToolResultContent::Text(text) => Some(text.text.as_str()),
                    _ => None,
                })
                .collect();
            let failed = if result.is_error { "failed: " } else { "" };
            let name = result.name.as_str();
            Some(format!(
                "  {name} -> {failed}{}",
                clip(&text.join(" "), RESULT)
            ))
        }
        _ => None,
    }
}

/// A line for what the worker said, or a tool it called.
fn did(item: &AssistantContent) -> Option<String> {
    match item {
        AssistantContent::Text(text) if !text.text.trim().is_empty() => {
            Some(format!("- said: {}", clip(&text.text, TEXT)))
        }
        AssistantContent::ToolCall(call) => {
            let args = clip(&call.function.raw_arguments(), ARGS);
            Some(format!("- called {} {args}", call.function.name.as_str()))
        }
        _ => None,
    }
}

/// `text` on one line, cut to `max` characters.
fn clip(text: &str, max: usize) -> String {
    let line = text.split_whitespace().collect::<Vec<_>>().join(" ");
    match line.char_indices().nth(max) {
        Some((at, _)) => format!("{}…", line.get(..at).unwrap_or_default()),
        None => line,
    }
}

/// `verdict` is offered to supervisors only: every other agent may use
/// every tool.
fn verdict_for_supervisors(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, ModelCalls, With<TurnCall>)>,
    supervisors: Query<(), With<Supervisor>>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        let offered = |call: &ModelRequest| {
            let tools = &call.request.tools;
            tools.iter().any(|tool| tool.name.as_str() == VERDICT)
        };
        if !supervisors.contains(agent) && offered(&call) {
            call.request_mut()
                .tools
                .retain(|tool| tool.name.as_str() != VERDICT);
        }
    }
}

/// Carries out a supervisor's verdict on the worker it supervises.
fn on_verdict(
    called: On<ToolCalled<VerdictArgs>>,
    supervisors: Query<&SpawnedBy, With<Supervisor>>,
    workers: Query<&ActiveTurn>,
    mut decisions: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let mut answer = |said: Result<&str, &str>| {
        let outcome = match said {
            Ok(said) => Outcome::from(ToolResult::success(said.to_owned().into())),
            Err(why) => Outcome::refused(why),
        };
        if let Ok(mut decision) = decisions.get_mut(called.call) {
            decision.decide(Decider::plugin("supervisor"), outcome);
        }
    };
    let Ok(&SpawnedBy(worker)) = supervisors.get(called.agent) else {
        answer(Err("Only a supervisor gives verdicts."));
        return;
    };
    let turn = workers.get(worker).ok().map(ActiveTurn::turn);
    let message = called.args.message.as_deref().map(str::trim);
    match (
        called.args.decision,
        message.filter(|text| !text.is_empty()),
    ) {
        (Ruling::Continue, _) => answer(Ok("Noted: the worker carries on.")),
        (Ruling::Steer | Ruling::Stop, None) => {
            answer(Err(
                "`steer` and `stop` need a `message`; nothing was done.",
            ));
        }
        (Ruling::Steer, Some(text)) => {
            // A worker whose turn ended reads it with its next turn's
            // request; it starts none.
            let mode = match turn {
                Some(_) => DeliveryMode::Steer,
                None => DeliveryMode::Inform,
            };
            let author = Author::agent(called.agent_id.clone(), None).titled(NAME);
            answer(Ok("Sent; the worker reads it with its next model call."));
            commands.trigger(Deliver::new(worker, text, mode).with_author(author));
        }
        (Ruling::Stop, Some(reason)) => {
            answer(Ok("The worker's turn is stopped and the user told why."));
            // Its calls that have not started never start, the others are
            // told why they stopped, and the turn ends.
            if let Some(turn) = turn {
                let why = format!("the supervisor stopped it: {reason}");
                commands
                    .entity(turn)
                    .try_insert(TurnOutcome::Stopped(Some(why)));
            }
            let told = format!("The supervisor stopped the turn: {reason}");
            notices.write(Notice::info(worker, told));
        }
    }
}

#[cfg(test)]
mod tests;
