//! A supervisor agent: `/supervise <goal>` gives the agent it is typed for
//! (the worker) a supervisor of its own, an agent [`SpawnedBy`] it with its
//! own conversation, model and one tool, `verdict`. Every `calls` tool calls
//! of the worker (10 by default), and every `minutes` while it works when
//! that is set, the supervisor is sent the goal and a digest of what the
//! worker did since the last check: what it was asked, the tools it called
//! with short results, and what it said. It answers with `verdict`:
//! `continue`, `steer` with a message the worker's running turn reads with
//! its next model call, or `stop` with a reason, which interrupts the
//! worker's turn and tells the user why. The worker never waits for it.
//!
//! [`Supervised`] on the worker and [`Supervisor`] on its supervisor are
//! saved with the session, and so is the supervisor's conversation, so it
//! remembers its earlier checks. `/supervise off` detaches it; the
//! supervisor stays in the session, idle, and the next `/supervise` of
//! that worker takes it on again.

use rig_harness::prelude::*;
use rig_harness::rig_core::transcript::assistant_text_from_choice;
use schemars::JsonSchema;
use serde::Deserialize;

/// The supervisor's tool.
pub const VERDICT: &str = "verdict";

/// Who the checks and steering messages come from.
const NAME: &str = "supervisor";

/// Tool calls between checks unless `/supervise` says `calls=N`.
pub const EVERY_CALLS: u32 = 10;

/// How often a supervised turn is looked at for a check that is due by
/// time.
const TICK: Duration = Duration::from_secs(15);

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

const RULE: &str = "Answer every check with one `verdict` call.";

/// Adds `/supervise`, the `verdict` tool and the checks.
#[derive(Default)]
pub struct SupervisorPlugin;

impl Plugin for SupervisorPlugin {
    fn build(&self, app: &mut App) {
        let options = ToolOptions {
            rules: &[RULE],
            footprint: Footprint::Independent,
        };
        let describe = "Give your verdict on the worker's progress: continue, steer it with a \
                        message, or stop its turn with a reason.";
        app.add_open_tool(VERDICT, describe, options, on_verdict)
            .add_command("supervise", HELP, supervise)
            .add_observer(count_call)
            .add_observer(keep_awake)
            .add_observer(verdict_for_supervisors)
            .add_observer(retry_after_check)
            .add_systems(
                Update,
                (due_by_time.run_if(on_real_timer(TICK)), send_checks)
                    .chain()
                    .after(PollCalls),
            );
    }
}

/// On a supervised agent: what its supervisor watches for, and how often
/// it looks.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
#[require(Tally)]
pub struct Supervised {
    /// The goal the supervisor holds the worker to.
    pub goal: String,
    /// Tool calls between checks; 0 checks by time only.
    pub every_calls: u32,
    /// Minutes between checks while a turn runs; 0 checks by calls only.
    pub every_minutes: u32,
    /// The supervisor's model, when it is not the worker's.
    pub model: Option<String>,
    /// The messages of the worker's conversation the supervisor has seen.
    pub seen: usize,
}

/// On a supervisor, which is [`SpawnedBy`] the agent it supervises.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default, Clone, Debug)]
pub struct Supervisor {
    /// The checks it was sent.
    pub checks: u32,
}

/// On a supervised agent: the tool calls since the last check, and when
/// that was (by the real clock). Not saved: a restart starts counting
/// again.
#[derive(Component, Debug, Default)]
struct Tally {
    calls: u32,
    since: Duration,
}

/// On a supervised agent whose check is due; it goes once its supervisor
/// is idle.
#[derive(Component, Debug)]
#[component(storage = "SparseSet")]
struct CheckDue;

#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct VerdictArgs {
    /// `continue`, `steer` or `stop`.
    decision: Decision,
    /// For `steer`, what the worker should do; for `stop`, why it stops.
    #[serde(default)]
    message: Option<String>,
}

#[derive(Deserialize, JsonSchema, Clone, Copy)]
#[serde(rename_all = "snake_case")]
enum Decision {
    Continue,
    Steer,
    Stop,
}

/// The worker's model, reasoning setting and connection.
type Staff<'a> = (Option<&'a ModelChoice>, Effort, Option<&'a Connection>);

/// The supervisor's model and reasoning setting: `model` when given, else
/// the worker's, whose connection it then shares.
fn staff(
    models: &Models,
    (choice, effort, connection): Staff,
    model: Option<&str>,
) -> Result<(ModelChoice, Effort, Option<Connection>), String> {
    let (picked, effort) = ModelChoice::inherit(&models.0, (choice, effort), model, None)?;
    let picked =
        picked.ok_or("The agent has no model to give its supervisor; add model=vendor/model")?;
    let shared = connection.filter(|_| choice == Some(&picked)).cloned();
    Ok((picked, effort, shared))
}

/// Spawns the supervisor of `worker`.
fn hire(
    commands: &mut Commands,
    worker: Entity,
    (choice, effort, shared): (ModelChoice, Effort, Option<Connection>),
) {
    let mut agent = commands.spawn_empty();
    // Before the model choice, so connecting it keeps this connection.
    if let Some(connection) = shared {
        agent.insert(connection);
    }
    agent.insert((
        Agent,
        Name::new(NAME),
        SpawnedBy(worker),
        Supervisor::default(),
        ToolAccess::Only(vec![VERDICT.to_owned()]),
        SystemPrompt(PROMPT.to_owned()),
        choice,
        effort,
    ));
}

/// The supervisor among `spawned`, the agents a worker spawned.
fn find_supervisor(
    spawned: Option<&Spawned>,
    is_supervisor: impl Fn(Entity) -> bool,
) -> Option<Entity> {
    spawned
        .into_iter()
        .flat_map(|spawned| spawned.iter())
        .find(|&agent| is_supervisor(agent))
}

/// `/supervise [calls=N] [minutes=M] [model=vendor/model] <goal>`, or
/// `/supervise off`.
fn supervise(
    In(args): In<CommandArgs>,
    workers: Query<(
        Option<&ModelChoice>,
        &Effort,
        Option<&Connection>,
        &Conversation,
        Option<&Spawned>,
    )>,
    marked: Query<(Has<Supervised>, Has<Supervisor>)>,
    supervisors: Query<(), With<Supervisor>>,
    models: Res<Models>,
    time: Res<Time<Real>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let worker = args.agent;
    let (supervised, is_supervisor) = marked.get(worker).unwrap_or_default();
    if args.args == "off" {
        if supervised {
            commands
                .entity(worker)
                .remove::<(Supervised, Tally, CheckDue)>();
            notices.write(Notice::info(worker, "Supervision off."));
        } else {
            notices.write(Notice::error(worker, "This agent has no supervisor."));
        }
        return;
    }
    if is_supervisor {
        notices.write(Notice::error(worker, "A supervisor is not supervised."));
        return;
    }
    let Ok((choice, &effort, connection, conversation, spawned)) = workers.get(worker) else {
        return;
    };
    let (mut every_calls, mut every_minutes, mut model) = (EVERY_CALLS, 0, None);
    let mut rest = args.args.as_str();
    while let Some((word, after)) = rest.split_once(char::is_whitespace).or(Some((rest, ""))) {
        let number = |value: &str| {
            value
                .parse::<u32>()
                .map_err(|_| format!("{word}: not a number"))
        };
        let set = match word.split_once('=') {
            Some(("calls", value)) => number(value).map(|n| every_calls = n),
            Some(("minutes", value)) => number(value).map(|n| every_minutes = n),
            Some(("model", value)) => {
                model = Some(value.to_owned());
                Ok(())
            }
            _ => break,
        };
        if let Err(why) = set {
            notices.write(Notice::error(worker, why));
            return;
        }
        rest = after.trim_start();
    }
    let goal = rest.trim();
    if goal.is_empty() {
        notices.write(Notice::error(
            worker,
            format!("/supervise needs a goal. {HELP}."),
        ));
        return;
    }
    if every_calls == 0 && every_minutes == 0 {
        notices.write(Notice::error(worker, "calls=0 needs minutes=M."));
        return;
    }
    match find_supervisor(spawned, |agent| supervisors.contains(agent)) {
        Some(supervisor) => {
            if let Some(model) = &model {
                commands.trigger(SetModel {
                    entity: supervisor,
                    model: model.clone(),
                });
            }
        }
        None => match staff(&models, (choice, effort, connection), model.as_deref()) {
            Ok(staffed) => hire(&mut commands, worker, staffed),
            Err(why) => {
                notices.write(Notice::error(worker, why));
                return;
            }
        },
    }
    commands.entity(worker).insert((
        Supervised {
            goal: goal.to_owned(),
            every_calls,
            every_minutes,
            model,
            seen: conversation.messages().len(),
        },
        Tally {
            calls: 0,
            since: time.elapsed(),
        },
    ));
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

/// Counts each finished tool call of a supervised agent; a check is due
/// every `every_calls`.
fn count_call(
    done: On<Add<ToolOutput>>,
    calls: Query<&CallOf, With<ToolCallRun>>,
    turns: Query<&TurnOf>,
    mut workers: Query<(&Supervised, &mut Tally)>,
    mut commands: Commands,
) {
    let Ok(&CallOf(turn)) = calls.get(done.entity) else {
        return;
    };
    let Ok(&TurnOf(worker)) = turns.get(turn) else {
        return;
    };
    let Ok((supervised, mut tally)) = workers.get_mut(worker) else {
        return;
    };
    tally.calls += 1;
    if supervised.every_calls > 0 && tally.calls >= supervised.every_calls {
        commands.entity(worker).insert(CheckDue);
    }
}

/// A supervised turn checked by time keeps the loop awake for its clock.
fn keep_awake(
    started: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    workers: Query<&Supervised>,
    mut commands: Commands,
) {
    let timed = turns
        .get(started.entity)
        .and_then(|&TurnOf(worker)| workers.get(worker))
        .is_ok_and(|supervised| supervised.every_minutes > 0);
    if timed {
        commands.entity(started.entity).insert(KeepAwake(TICK));
    }
}

/// A check is due for a working agent whose last one is `every_minutes`
/// old.
fn due_by_time(
    workers: Query<(Entity, &Supervised, &Tally), With<ActiveTurn>>,
    time: Res<Time<Real>>,
    mut commands: Commands,
) {
    for (worker, supervised, tally) in &workers {
        let every = Duration::from_secs(60 * u64::from(supervised.every_minutes));
        if supervised.every_minutes > 0 && time.elapsed().saturating_sub(tally.since) >= every {
            commands.entity(worker).insert(CheckDue);
        }
    }
}

/// Sends each due check to its idle supervisor: the goal and what the
/// worker did since the last one. A supervisor gone, such as one that
/// had not logged a message before a restart, is hired again first.
fn send_checks(
    mut due: Query<
        (
            Entity,
            &mut Supervised,
            &mut Tally,
            &Conversation,
            Option<&Spawned>,
            (Option<&ModelChoice>, &Effort, Option<&Connection>),
        ),
        With<CheckDue>,
    >,
    mut supervisors: Query<(&mut Supervisor, Has<ActiveTurn>)>,
    models: Res<Models>,
    time: Res<Time<Real>>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (worker, mut supervised, mut tally, conversation, spawned, (choice, &effort, connection)) in
        &mut due
    {
        let Some(supervisor) = find_supervisor(spawned, |agent| supervisors.contains(agent)) else {
            match staff(
                &models,
                (choice, effort, connection),
                supervised.model.as_deref(),
            ) {
                Ok(staffed) => {
                    hire(&mut commands, worker, staffed);
                    wake.wake();
                }
                Err(why) => {
                    notices.write(Notice::error(worker, why));
                    commands.entity(worker).remove::<CheckDue>();
                }
            }
            continue;
        };
        let Ok((mut checks, busy)) = supervisors.get_mut(supervisor) else {
            continue;
        };
        if busy {
            continue;
        }
        let messages = conversation.messages();
        let text = check(&supervised.goal, messages, supervised.seen);
        let check = Deliver::new(supervisor, text, DeliveryMode::Queue);
        commands.trigger(check.with_origin(Origin::plugin(NAME)));
        checks.checks += 1;
        supervised.seen = messages.len();
        *tally = Tally {
            calls: 0,
            since: time.elapsed(),
        };
        commands.entity(worker).remove::<CheckDue>();
    }
}

/// A check whose supervisor was busy goes once it is idle.
fn retry_after_check(
    end: On<TurnEnded>,
    supervisors: Query<(), With<Supervisor>>,
    wake: Res<Wake>,
) {
    if supervisors.contains(end.entity) {
        wake.wake();
    }
}

/// The text of a check: the goal, the worker's activity in `messages` from
/// `seen` on, and its last text.
fn check(goal: &str, messages: &[message::Message], seen: usize) -> String {
    let mut lines = Vec::new();
    for message in messages.get(seen.min(messages.len())..).unwrap_or_default() {
        if let message::Message::User { content } = message {
            for item in content {
                if let UserContent::Text(text) = item {
                    lines.push(format!("- request: {}", clip(&text.text, TEXT)));
                } else if let UserContent::ToolResult(result) = item {
                    let text: String = result
                        .content
                        .iter()
                        .filter_map(|part| match part {
                            message::ToolResultContent::Text(text) => Some(text.text.as_str()),
                            _ => None,
                        })
                        .collect::<Vec<_>>()
                        .join(" ");
                    let failed = if result.is_error { "failed: " } else { "" };
                    let name = result.name.as_str();
                    lines.push(format!("  {name} -> {failed}{}", clip(&text, RESULT)));
                }
            }
        } else if let message::Message::Assistant(reply) = message {
            for item in &reply.content {
                if let AssistantContent::Text(text) = item
                    && !text.text.trim().is_empty()
                {
                    lines.push(format!("- said: {}", clip(&text.text, TEXT)));
                } else if let AssistantContent::ToolCall(call) = item {
                    let args = clip(&call.function.raw_arguments(), ARGS);
                    lines.push(format!("- called {} {args}", call.function.name.as_str()));
                }
            }
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
    let last = messages.iter().rev().find_map(|message| match message {
        message::Message::Assistant(reply) => {
            let text = assistant_text_from_choice(&reply.content);
            (!text.trim().is_empty()).then(|| clip(&text, TEXT))
        }
        _ => None,
    });
    let last = last.unwrap_or_else(|| "(none yet)".to_owned());
    format!(
        "Goal: {goal}\n\nThe worker's activity since your last check:\n{activity}\n\nIts last \
         text: {last}\n\nGive your `verdict`."
    )
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
    mut prepare: On<PrepareRequest>,
    supervisors: Query<(), With<Supervisor>>,
) {
    if !supervisors.contains(prepare.agent) {
        prepare.tools.retain(|tool| tool.name.as_str() != VERDICT);
    }
}

/// Carries out a supervisor's verdict on the worker it supervises.
fn on_verdict(
    called: On<ToolCalled<VerdictArgs>>,
    supervisors: Query<(&SpawnedBy, &AgentId), With<Supervisor>>,
    workers: Query<Has<ActiveTurn>, With<Supervised>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let call = called.call;
    let mut answer = |said: Result<&str, &str>| {
        let result = match said {
            Ok(said) => ToolResult::success(said.to_owned().into()),
            Err(why) => ToolResult::failed(ToolExecutionError::other(why.to_owned())),
        };
        commands.entity(call).insert_if_new(ToolOutput(result));
    };
    let Ok((&SpawnedBy(worker), id)) = supervisors.get(called.agent) else {
        answer(Err("Only a supervisor gives verdicts."));
        return;
    };
    let Ok(busy) = workers.get(worker) else {
        answer(Ok("The worker is no longer supervised; nothing was done."));
        return;
    };
    let message = called
        .args
        .message
        .as_deref()
        .map(str::trim)
        .filter(|text| !text.is_empty());
    match (called.args.decision, message) {
        (Decision::Continue, _) => answer(Ok("Noted: the worker carries on.")),
        (Decision::Steer | Decision::Stop, None) => {
            answer(Err(
                "`steer` and `stop` need a `message`; nothing was done.",
            ));
        }
        (Decision::Steer, Some(text)) => {
            // A worker whose turn ended reads it with its next turn's
            // request; it starts none.
            let mode = if busy {
                DeliveryMode::Steer
            } else {
                DeliveryMode::Note
            };
            let origin = Origin::agent(id.clone(), None).titled(NAME);
            let steer = Deliver::new(worker, text, mode).with_origin(origin);
            answer(Ok("Sent; the worker reads it with its next model call."));
            commands.trigger(steer);
        }
        (Decision::Stop, Some(reason)) => {
            answer(Ok("The worker's turn is stopped and the user told why."));
            if busy {
                commands.trigger(Interrupt { entity: worker });
            }
            notices.write(Notice::info(
                worker,
                format!("The supervisor stopped the turn: {reason}"),
            ));
        }
    }
}

#[cfg(test)]
mod tests;
