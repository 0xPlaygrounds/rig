//! Subagents ask their parent: every subagent gets an `ask_parent` tool,
//! whose call waits, with the subagent's turn paused, until the agent that
//! started it answers with `answer_child`, or passes the question on to the
//! user, or the question times out.
//!
//! A question is a hold on the subagent's `ask_parent` call ([`Question`],
//! `HoldOf` the call), spawned by a Gate system, so the call never starts
//! until it is answered: its answer decides the call. The parent reads the
//! question as a message from the subagent, delivered into its running
//! turn's next model call, or starting a turn when it is idle. A parent's
//! open `wait` calls end once the question is in its inbox: `rig_basics::subagents`
//! ends a wait for the subagent that asked (a message from it), and this
//! plugin ends its waits for any other agent ([`Awaits`]), so the question
//! goes with the parent's next model call.
//!
//! After a crash the subagent's unanswered call is restored and gated
//! again, so the question is asked again, and its turn carries on
//! ([`CarryOn`]), which the kernel would otherwise stop. Stopping the parent stops
//! the subagents waiting on its answers, as the kernel stops every agent
//! below a stopped one, and `rig_basics::subagents` keeps their reports for the
//! parent's next turn instead of starting one.

use rig_basics::subagents::{Awaits, SpawnSubagent};
use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

/// The tool a subagent asks its parent with.
pub const ASK_PARENT: &str = "ask_parent";

/// The tool a parent answers its subagent's question with.
pub const ANSWER_CHILD: &str = "answer_child";

/// Who decides the calls this plugin answers.
const DECIDER: &str = "ask-parent";

const ASK_DESCRIPTION: &str = "Ask the agent that started you a question and wait for its \
    answer, which is this call's result. Use it only when you cannot go on without a decision \
    you cannot make yourself and only that agent can, such as which of two files is the real \
    one; ask one short, self-contained question. Your work pauses until the answer comes.";

const ANSWER_DESCRIPTION: &str = "Answer a question one of your subagents asked you, by the \
    question's id: your `answer` becomes the result of its `ask_parent` call. With `ask_user`, \
    the question goes to the user instead, and the user's reply goes to the subagent. The \
    subagent waits for it: answer before anything else, from what you know, and ask the user \
    only when you cannot.";

/// The `[ask_parent]` table of the settings.
///
/// ```toml
/// [ask_parent]
/// timeout_secs = 600
/// ```
#[derive(Resource, Clone, Debug, PartialEq, Eq, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct AskParentSettings {
    /// How long a question waits for its answer before its call returns
    /// that none came, in seconds. Ten minutes when not set.
    pub timeout_secs: u64,
}

impl Default for AskParentSettings {
    fn default() -> Self {
        Self { timeout_secs: 600 }
    }
}

/// The plugin's stage systems, for others to order against: the Gate
/// system that holds each `ask_parent` call with its [`Question`], and the
/// Prepare edits that give `ask_parent` to subagents only.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum AskParentSystems {
    /// Gives a subagent spawned with a list of tools `ask_parent` too,
    /// offers `ask_parent` and `answer_child` only to agents that have a
    /// parent or subagents, and makes the report of a subagent stopped with
    /// its parent a note.
    Prepare,
    /// Holds each `ask_parent` call with its question.
    Gate,
}

/// Adds `ask_parent` for subagents and `answer_child` for their parents,
/// the questions and their timeouts (`[ask_parent] timeout_secs`).
#[derive(Default)]
pub struct AskParentPlugin;

impl Plugin for AskParentPlugin {
    fn build(&self, app: &mut App) {
        let independent = || ToolOptions {
            footprint: Some(Touches::Independent),
        };
        app.add_setting::<AskParentSettings>("ask_parent")
            .configure_sets(
                AgentPass,
                (
                    AskParentSystems::Prepare.in_set(PrepareSystems::Edit),
                    AskParentSystems::Gate.in_set(Stage::Gate),
                ),
            )
            .add_open_tool(ASK_PARENT, ASK_DESCRIPTION, independent(), unrouted)
            .add_open_tool(ANSWER_CHILD, ANSWER_DESCRIPTION, independent(), on_answer)
            .add_systems(
                AgentPass,
                (
                    (with_ask_parent, offer_where_useful).in_set(AskParentSystems::Prepare),
                    (escalated, ask).chain().in_set(AskParentSystems::Gate),
                ),
            )
            .add_observer(time_out)
            .add_observer(question_accepted)
            .add_observer(wait_started);
    }
}

/// An open question: the hold on a subagent's `ask_parent` call, which its
/// answer decides.
#[derive(Component, Clone, Debug)]
pub struct Question {
    /// The id the parent answers it by: the `ask_parent` call's id.
    pub id: String,
    /// The subagent that asked.
    pub child: Entity,
    /// The agent that started it, which is asked.
    pub parent: Entity,
    /// The question.
    pub text: String,
    /// How long it waits, in seconds.
    pub timeout_secs: u64,
}

/// On an `ask_parent` call held with its [`Question`].
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Asked;

/// On a question's message to the parent: the subagent that asked.
#[derive(Component)]
struct QuestionFrom(Entity);

/// On the user's question a parent passed a subagent's question on as: the
/// [`Question`] and the parent's `answer_child` call, both answered with
/// the user's reply.
#[derive(Component)]
struct Escalated {
    question: Entity,
    call: Entity,
}

/// The arguments of `ask_parent`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct AskArgs {
    /// The question, complete on its own: the agent that started you sees
    /// only this.
    question: String,
}

/// The arguments of `answer_child`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct AnswerArgs {
    /// The question's id, as its message gives it.
    id: String,
    /// The answer; the subagent reads it as its call's result.
    #[serde(default)]
    answer: String,
    /// Ask the user instead, and pass the user's reply to the subagent.
    #[serde(default)]
    ask_user: bool,
}

/// A call's answer.
fn answered(text: impl Into<String>) -> Outcome {
    ToolResult::success(text.into().into()).into()
}

/// A call's refusal.
fn refused(why: impl Into<String>) -> Outcome {
    ToolResult::failed(ToolExecutionError::other(why.into())).into()
}

/// A subagent spawned with a list of tools may ask its parent too, when
/// its parent may.
fn with_ask_parent(
    mut requests: Query<(&mut SpawnSubagent, &ForAgent), Added<Admitted>>,
    parents: Query<&ToolAccess>,
) {
    for (mut request, &ForAgent(parent)) in &mut requests {
        let may = parents
            .get(parent)
            .is_ok_and(|access| access.allows(ASK_PARENT));
        if let Some(tools) = request.tools.as_mut().filter(|_| may)
            && !tools.iter().any(|tool| tool == ASK_PARENT)
        {
            tools.push(ASK_PARENT.to_owned());
        }
    }
}

/// Offers `ask_parent` only to an agent that has a parent, and
/// `answer_child` only to one that started subagents.
fn offer_where_useful(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, With<TurnCall>)>,
    agents: Query<(Has<SpawnedBy>, Has<Spawned>)>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        let (child, parent) = agents.get(agent).unwrap_or_default();
        let tools = &mut call.request_mut().tools;
        tools.retain(|tool| match tool.name.as_str() {
            ASK_PARENT => child,
            ANSWER_CHILD => parent,
            _ => true,
        });
    }
}

/// Holds each `ask_parent` call about to run with its [`Question`], and
/// sends the question to the caller's parent; refuses one from an agent
/// that has none. A restored call is asked again, its turn carried on.
fn ask(
    mut calls: Query<
        (
            Entity,
            &ToolRequest,
            &ForAgent,
            &mut Decision,
            Option<&CallOf>,
            Has<Restored>,
        ),
        (ReadyToStart, ToolCalls, Without<Asked>),
    >,
    agents: Query<(&AgentId, Option<&SpawnedBy>, Option<&AgentTitle>)>,
    settings: Res<AskParentSettings>,
    mut commands: Commands,
) {
    for (call, request, &ForAgent(child), mut decision, turn, restored) in &mut calls {
        if request.call.function.name.as_str() != ASK_PARENT || decision.is_decided() {
            continue;
        }
        let args = serde_json::Value::Object(request.call.function.arguments.clone());
        let args = serde_json::from_value::<AskArgs>(args);
        let question = args
            .map(|args| args.question.trim().to_owned())
            .unwrap_or_default();
        let Ok((id, Some(&SpawnedBy(parent)), title)) = agents.get(child) else {
            decision.decide(
                Decider::plugin(DECIDER),
                refused("No agent started you, so there is nobody to ask."),
            );
            continue;
        };
        if question.is_empty() {
            decision.decide(
                Decider::plugin(DECIDER),
                refused("`question` must not be empty."),
            );
            continue;
        }
        let task = title.map(|title| &title.0);
        let question = Question {
            id: request.call.id.to_string(),
            child,
            parent,
            text: question,
            timeout_secs: settings.timeout_secs,
        };
        let text = format!(
            "Question from subagent `{}` (task: {}): {}\n\nAnswer it with `{ANSWER_CHILD}` and id \
             `{}`; it waits for your answer. With `ask_user`, the user answers instead.",
            id.short(),
            task.map_or("none", String::as_str),
            question.text,
            question.id,
        );
        // From the subagent, so a parent's `wait` for it ends with it.
        let author = Author::agent(id.0.clone(), None);
        let author = match task {
            Some(task) => author.titled(task.clone()),
            None => author,
        };
        let message = Incoming::new(text, DeliveryMode::Steer).with_author(author);
        commands.spawn((message, ForAgent(parent), QuestionFrom(child)));
        let wait = WakeAt::after(Duration::from_secs(settings.timeout_secs));
        let name = Name::new(format!(
            "waiting for its parent's answer to `{}`",
            question.id
        ));
        commands.entity(call).insert(Asked);
        commands.spawn((question, HoldOf(call), wait, name));
        if let Some(&CallOf(turn)) = turn.filter(|_| restored) {
            commands.entity(turn).try_insert(CarryOn);
        }
    }
}

/// An `ask_parent` call that started, which only one [`ask`] did not hold
/// does: nothing was asked.
fn unrouted(called: On<ToolCalled<AskArgs>>, mut calls: Query<&mut Decision, ToolCalls>) {
    if let Ok(mut decision) = calls.get_mut(called.call) {
        decision.decide(
            Decider::plugin(DECIDER),
            refused("The question was not sent."),
        );
    }
}

/// Answers a subagent's question: the answer decides its `ask_parent`
/// call; with `ask_user`, the user is asked, and the call waits.
fn on_answer(
    called: On<ToolCalled<AnswerArgs>>,
    questions: Query<(Entity, &Question, &HoldOf)>,
    ids: Query<&AgentId>,
    answerers: Answerers,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let (parent, args) = (called.agent, &called.args);
    let mine = || {
        questions
            .iter()
            .filter(|(_, question, _)| question.parent == parent)
    };
    let found = mine().find(|(_, question, _)| question.id == args.id.trim());
    let short = |agent| {
        ids.get(agent)
            .map(|id| id.short().to_owned())
            .unwrap_or_default()
    };
    let output = match found {
        None => {
            let open: Vec<String> = mine()
                .map(|(_, question, _)| format!("`{}`", question.id))
                .collect();
            let open = if open.is_empty() {
                "none".to_owned()
            } else {
                open.join(", ")
            };
            refused(format!(
                "`{}` is no open question of your subagents. Open: {open}.",
                args.id
            ))
        }
        Some((entity, question, _)) if args.ask_user => {
            if !answerers.can_answer(Some(parent)) {
                refused("Nobody can answer the user's questions now: answer it yourself.")
            } else {
                let ask = Ask::new(format!("Subagent `{}` asks", short(question.child)))
                    .body(question.text.clone())
                    .free_text();
                let escalated = Escalated {
                    question: entity,
                    call: called.call,
                };
                commands.spawn((ask, ForAgent(parent), escalated));
                return;
            }
        }
        Some(_) if args.answer.trim().is_empty() => {
            refused("`answer` must not be empty; set `ask_user` to ask the user instead.")
        }
        Some((entity, question, &HoldOf(asked))) => {
            if let Ok(mut decision) = calls.get_mut(asked) {
                decision.decide(Decider::plugin(DECIDER), answered(args.answer.trim()));
            }
            commands.entity(entity).despawn();
            answered(format!("Sent to `{}`.", short(question.child)))
        }
    };
    if let Ok(mut decision) = calls.get_mut(called.call) {
        decision.decide(Decider::plugin(DECIDER), output);
    }
}

/// Passes the user's reply to a question a parent passed on to the
/// subagent that asked, and tells the parent.
fn escalated(
    asks: Query<(Entity, &Outcome, &Escalated), (With<Ask>, Added<Outcome>)>,
    questions: Query<&HoldOf, With<Question>>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    for (ask, outcome, escalated) in &asks {
        let reply = Answer::of(outcome).and_then(|answer| answer.text);
        let asked = questions.get(escalated.question).ok();
        let (to_child, to_parent) = match (reply, asked) {
            (_, None) => (
                None,
                refused("The subagent stopped waiting before the user answered."),
            ),
            (Some(text), Some(_)) => {
                let text = format!("The user answered: {}", text.trim());
                (
                    Some(answered(text.clone())),
                    answered(format!("{text}\nIt went to the subagent.")),
                )
            }
            (None, Some(_)) => {
                let none = "The user gave no answer; decide for yourself and say what you assumed.";
                (
                    Some(answered(none)),
                    answered("The user gave no answer; the subagent was told."),
                )
            }
        };
        if let (Some(answer), Some(&HoldOf(call))) = (to_child, asked)
            && let Ok(mut decision) = calls.get_mut(call)
        {
            decision.decide(Decider::plugin(DECIDER), answer);
            commands.entity(escalated.question).despawn();
        }
        if let Ok(mut decision) = calls.get_mut(escalated.call) {
            decision.decide(Decider::plugin(DECIDER), to_parent);
        }
        commands.entity(ask).despawn();
    }
}

/// A question unanswered when it is due: its call returns that no answer
/// came, and the subagent goes on.
fn time_out(
    due: On<Add<Due>>,
    questions: Query<(&Question, &HoldOf)>,
    mut calls: Query<&mut Decision, ToolCalls>,
    mut commands: Commands,
) {
    let Ok((question, &HoldOf(call))) = questions.get(due.entity) else {
        return;
    };
    if let Ok(mut decision) = calls.get_mut(call) {
        let text = format!(
            "no answer from parent within {} s; decide for yourself and say what you assumed.",
            question.timeout_secs
        );
        decision.decide(Decider::plugin(DECIDER), answered(text));
    }
    commands.entity(due.entity).despawn();
}

/// A question accepted into its parent's inbox ends the parent's open
/// `wait` calls for other agents, so it goes with the next model call.
fn question_accepted(
    accepted: On<Add<Started>>,
    messages: Query<(&ForAgent, &QuestionFrom)>,
    mut waits: Query<(Entity, &Awaits, &ForAgent, &mut Decision), ToolCalls>,
    mut commands: Commands,
) {
    let Ok((&ForAgent(parent), &QuestionFrom(child))) = messages.get(accepted.entity) else {
        return;
    };
    for (call, &Awaits(awaited), &ForAgent(waiter), mut decision) in &mut waits {
        if waiter == parent && awaited != child {
            end_wait(call, &mut decision, &mut commands);
        }
    }
}

/// A `wait` started while a question for its agent is in the agent's
/// inbox ends at once, unless it waits for the subagent that asked
/// (`rig_basics::subagents` ends that one).
fn wait_started(
    added: On<Add<Awaits>>,
    mut waits: Query<(&Awaits, &ForAgent, &mut Decision), ToolCalls>,
    inbox: Query<(&ForAgent, &QuestionFrom), (With<Started>, Without<Outcome>)>,
    mut commands: Commands,
) {
    let Ok((&Awaits(awaited), &ForAgent(waiter), mut decision)) = waits.get_mut(added.entity)
    else {
        return;
    };
    let asked = inbox
        .iter()
        .any(|(&ForAgent(to), &QuestionFrom(child))| to == waiter && child != awaited);
    if asked {
        end_wait(added.entity, &mut decision, &mut commands);
    }
}

/// Ends the open `wait` call `call` for a subagent's question.
fn end_wait(call: Entity, decision: &mut Decision, commands: &mut Commands) {
    let text = "A subagent asked you a question, which follows: answer it with `answer_child`, \
                then `wait` again if you still need to.";
    if decision.decide(Decider::plugin(DECIDER), answered(text)) {
        commands.entity(call).remove::<Awaits>();
    }
}

#[cfg(test)]
mod tests;
