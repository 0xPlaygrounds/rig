//! Subagents ask their parent: every subagent gets an `ask_parent` tool,
//! whose call waits, with the subagent's turn paused, until the agent that
//! started it answers with `answer_child`, or passes the question on to the
//! user, or the question times out.
//!
//! A question is a hold on the subagent's `ask_parent` call ([`Question`],
//! `HoldOf` the call), spawned by a Gate system, so the call never starts
//! until it is answered: its answer decides the call. The parent reads the
//! question as a message from the subagent, delivered into its running
//! turn's next model call, or starting a turn when it is idle; a message
//! from the subagent also ends the parent's `wait` for it (rig-subagents),
//! so a parent waiting on its subagent reads the question at once.
//!
//! After a restart the subagent's unanswered call is restored and gated
//! again, so the question is asked again, and its hold claims the restored
//! turn, which rig-subagents would otherwise stop. Interrupting the parent
//! interrupts the subagents waiting on its answers.

use rig_harness::prelude::*;
use rig_subagents::SpawnSubagent;
use schemars::JsonSchema;
use serde::Deserialize;

/// The tool a subagent asks its parent with.
pub const ASK_PARENT: &str = "ask_parent";

/// The tool a parent answers its subagent's question with.
pub const ANSWER_CHILD: &str = "answer_child";

/// Who decides the calls this plugin answers.
const DECIDER: &str = "ask-parent";

const ASK_DESCRIPTION: &str = "Ask the agent that started you a question and wait for its \
    answer, which is this call's result. Use it when you cannot go on without a decision only \
    that agent can make, such as which of two files is the real one; your work pauses until \
    the answer comes.";

const ANSWER_DESCRIPTION: &str = "Answer a question one of your subagents asked you, by the \
    question's id: your `answer` becomes the result of its `ask_parent` call. With `ask_user`, \
    the question goes to the user instead, and the user's reply goes to the subagent.";

const ASK_RULES: &[&str] = &[
    "Call `ask_parent` only for a decision you cannot make yourself and that the agent that \
     started you can; ask one short, self-contained question.",
];

const ANSWER_RULES: &[&str] = &[
    "A subagent's question waits for your `answer_child`: answer it before anything else, from \
     what you know, and ask the user (`ask_user`) only when you cannot.",
];

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
    /// Gives a subagent spawned with a list of tools `ask_parent` too, and
    /// offers `ask_parent` and `answer_child` only to agents that have a
    /// parent or subagents.
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
        let independent = |rules| ToolOptions {
            rules,
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
            .add_open_tool(
                ASK_PARENT,
                ASK_DESCRIPTION,
                independent(ASK_RULES),
                unrouted,
            )
            .add_open_tool(
                ANSWER_CHILD,
                ANSWER_DESCRIPTION,
                independent(ANSWER_RULES),
                on_answer,
            )
            .add_systems(
                AgentPass,
                (
                    (with_ask_parent, offer_where_useful).in_set(AskParentSystems::Prepare),
                    (escalated, ask).chain().in_set(AskParentSystems::Gate),
                ),
            )
            .add_observer(time_out)
            .add_observer(cancel_children);
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
        call.request.tools.retain(|tool| match tool.name.as_str() {
            ASK_PARENT => child,
            ANSWER_CHILD => parent,
            _ => true,
        });
    }
}

/// Holds each `ask_parent` call about to run with its [`Question`], and
/// sends the question to the caller's parent; refuses one from an agent
/// that has none. A restored call is asked again.
fn ask(
    mut calls: Query<
        (Entity, &ToolRequest, &ForAgent, &mut Decision),
        (Undecided, ToolCalls, Without<Asked>),
    >,
    agents: Query<(&AgentId, Option<&SpawnedBy>, Option<&AgentTitle>)>,
    settings: Res<AskParentSettings>,
    mut commands: Commands,
) {
    for (call, request, &ForAgent(child), mut decision) in &mut calls {
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
        let deliver = Deliver::new(parent, text, DeliveryMode::Steer).with_author(author);
        commands.trigger(deliver);
        let wait = WakeAt::after(Duration::from_secs(settings.timeout_secs));
        let name = Name::new(format!(
            "waiting for its parent's answer to `{}`",
            question.id
        ));
        commands.entity(call).insert(Asked);
        commands.spawn((question, HoldOf(call), wait, name));
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

/// Interrupting an agent interrupts each subagent waiting on its answer,
/// which cancels the subagent's `ask_parent` call and stops its turn.
fn cancel_children(interrupt: On<Interrupt>, questions: Query<&Question>, mut commands: Commands) {
    let parent = interrupt.entity;
    for question in questions
        .iter()
        .filter(|question| question.parent == parent)
    {
        let reason = Some("its parent was stopped while it waited for an answer".to_owned());
        commands.trigger(Interrupt {
            entity: question.child,
            reason,
        });
    }
}

#[cfg(test)]
mod tests;
