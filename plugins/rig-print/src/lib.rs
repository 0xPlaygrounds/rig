//! `--print`: one prompt to the primary agent, then exit once no agent
//! works, so the subagents it started have answered. The answer's text
//! goes to stdout and failures to stderr; the exit code
//! is 0 when the turn ended with an answer, 2 for a refused slash command
//! (unknown, or arguments it does not take), 1 otherwise. Text piped in on
//! stdin follows the prompt, so `git diff | rig -p "review this"` works.
//!
//! The agent uses `--model`, else the model its restored session chose,
//! else the first model the environment has a key for.
//!
//! It is the front ([`FRONT`]) of a run that is not interactive
//! ([`Invoked::interactive`]): one with `--print`, or with stdin piped in,
//! which answers what is piped in, unless another front's option chose
//! that one.
//!
//! Nobody sits at a print run to answer questions, so by default it is no
//! [`Answerer`]: a plugin applies its own policy for a run nobody answers,
//! such as refusing, and a question asked anyway is refused, with why on
//! stderr. `--answer=ID[,ID…]` ([`ANSWER`]) answers each question with the
//! first of those choices it offers instead, and refuses one that offers
//! none. After the answer, the run prints what other plugins add as
//! [`PrintSection`]s, such as the agent's todo list.

use std::io::{IsTerminal, Read as _};

use bevy_ecs::system::SystemParam;
use rig_harness::prelude::*;

/// `--print`'s name as a front.
pub const FRONT: &str = "print";

/// The option that answers a print run's questions.
pub const ANSWER: &str = "answer";

/// What a print run writes to stdout after the answer, such as a todo
/// list: spawn one and keep its `text` current. Sections follow the
/// answer by `tag`, each after an empty line; an empty `text`, or Bevy's
/// `Disabled` on the entity, leaves one out.
#[derive(Component, Clone, Debug, Default, PartialEq, Eq)]
pub struct PrintSection {
    /// What sorts it, such as `todo`.
    pub tag: String,
    /// What is printed.
    pub text: String,
}

impl PrintSection {
    /// `text` printed after the answer, sorted by `tag`.
    pub fn new(tag: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            text: text.into(),
        }
    }
}

/// Sends the prompt and exits when the turn ends, when it is the run's
/// front.
#[derive(Default)]
pub struct PrintPlugin;

impl Plugin for PrintPlugin {
    fn build(&self, app: &mut App) {
        app.add_front(Front::new(FRONT).headless()).add_flag(
            Flag::new(
                ANSWER,
                "Answer the questions of a --print run with the first of these choices offered",
            )
            .value("ID[,ID…]"),
        );
    }

    fn finish(&self, app: &mut App) {
        if !app.runs_front(FRONT) {
            return;
        }
        let invoked = app.world().get_resource::<Invoked>().cloned();
        let answers: Vec<String> = invoked
            .as_ref()
            .and_then(|invoked| invoked.flag_value(ANSWER))
            .map(|ids| ids.split(',').map(|id| id.trim().to_owned()).collect())
            .unwrap_or_default();
        if !answers.is_empty() {
            app.world_mut().spawn(Answerer::default());
        }
        let args = invoked.map(|invoked| invoked.args).unwrap_or_default();
        app.insert_resource(Answers(answers))
            .add_systems(AgentPass, answer.in_set(Stage::Gate))
            .insert_resource(PrintRun {
                prompt: args.print.unwrap_or_default(),
                model: args.model.is_some(),
                step: Step::Start,
                failed: false,
                refused: false,
            })
            .add_systems(Update, (print_notices, drive).chain());
    }
}

/// The choices `--answer` gives, in order; none without it.
#[derive(Resource)]
struct Answers(Vec<String>);

/// Answers each question with the first of the `--answer` choices it
/// offers, or refuses it, saying why on stderr.
fn answer(asks: Query<(Entity, &Ask), Undecided>, answers: Res<Answers>, mut commands: Commands) {
    for (entity, ask) in &asks {
        let offered = |id: &&String| ask.choices.iter().any(|choice| &choice.id == *id);
        let outcome = match answers.0.iter().find(offered) {
            Some(id) => Answer::Choice(id.clone()).outcome(),
            None => {
                let why = match answers.0.is_empty() {
                    true => format!(
                        "nobody answers questions in a --print run, such as “{}”: \
                         --answer=ID answers them",
                        ask.title
                    ),
                    false => format!("--answer offers none of the choices of “{}”", ask.title),
                };
                eprintln!("rig: {why}");
                Outcome::refused(why)
            }
        };
        commands.entity(entity).insert_if_new(outcome);
    }
}

/// Where the print run is.
#[derive(Resource)]
struct PrintRun {
    prompt: String,
    /// Whether `--model` named the model, which rig-models chooses.
    model: bool,
    step: Step,
    /// Whether an error notice about the agent came while it ran.
    failed: bool,
    /// Whether the prompt was a refused slash command.
    refused: bool,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    /// Nothing done yet.
    Start,
    /// A model was chosen for `agent`, which connects it at once.
    Connecting { agent: Entity },
    /// The prompt went to `agent`, whose conversation had `before`
    /// messages.
    Sent { agent: Entity, before: usize },
    /// Exited.
    Done,
}

/// The prompt with what was piped in on stdin after it.
fn full_prompt(prompt: &str) -> String {
    let stdin = std::io::stdin();
    if stdin.is_terminal() {
        return prompt.to_owned();
    }
    let mut piped = String::new();
    if stdin.lock().read_to_string(&mut piped).is_err() || piped.trim().is_empty() {
        return prompt.to_owned();
    }
    if prompt.trim().is_empty() {
        piped
    } else {
        format!("{prompt}\n\n{piped}")
    }
}

/// The agents as the print run reads them.
#[derive(SystemParam)]
struct Agents<'w, 's> {
    primary: PrimaryQuery<'w, 's>,
    choices: Query<'w, 's, (Option<&'static ModelChoice>, Has<Connection>)>,
    conversations: Query<'w, 's, &'static Conversation>,
    /// Running turns.
    turns: Query<'w, 's, (), With<TurnOf>>,
    /// Work in progress, such as a sign-in.
    running: Query<'w, 's, (), With<Running>>,
    /// Messages on their way to an agent, such as the prompt itself.
    messages: Query<'w, 's, (), (With<Incoming>, Without<Settled>)>,
    /// What other plugins print after the answer.
    sections: Query<'w, 's, &'static PrintSection>,
}

/// Chooses the model, sends the prompt, and exits after the turn.
fn drive(
    mut run: ResMut<PrintRun>,
    agents: Agents,
    models: Res<Models>,
    mut commands: Commands,
    mut exits: MessageWriter<AppExit>,
) {
    // A command such as `/login` runs without a model.
    let command = run.prompt.trim_start().starts_with('/');
    match run.step {
        Step::Start => {
            let Some(agent) = primary(&agents.primary) else {
                return;
            };
            let Ok((chosen, _)) = agents.choices.get(agent) else {
                return;
            };
            let model = match (run.model, chosen) {
                // `--model` chose it, or the restored session did, and it
                // was connected, or refused with a notice, already.
                (true, _) | (false, Some(_)) => None,
                (false, None) => match models.0.reachable().first() {
                    Some(spec) => Some(spec.reference()),
                    None if command => None,
                    None => {
                        eprintln!(
                            "rig: no model can be reached: set a provider's API key, or name one \
                             with --model"
                        );
                        run.step = Step::Done;
                        exits.write(AppExit::from_code(1));
                        return;
                    }
                },
            };
            if let Some(model) = model {
                commands.trigger(SetModel {
                    entity: agent,
                    model,
                });
            }
            run.step = Step::Connecting { agent };
        }
        Step::Connecting { agent } => {
            let connected = command
                || agents
                    .choices
                    .get(agent)
                    .is_ok_and(|(_, connected)| connected);
            if !connected {
                // The notice of the failed choice says why.
                run.step = Step::Done;
                exits.write(AppExit::from_code(1));
                return;
            }
            let text = full_prompt(&run.prompt);
            if text.trim().is_empty() {
                eprintln!("rig: no prompt: give one after --print, or pipe it in");
                run.step = Step::Done;
                exits.write(AppExit::from_code(2));
                return;
            }
            let before = agents
                .conversations
                .get(agent)
                .map_or(0, |conversation| conversation.messages().len());
            commands.trigger(Deliver::new(agent, text, DeliveryMode::Steer));
            run.step = Step::Sent { agent, before };
        }
        Step::Sent { agent, before } => {
            // The turn starts with the request once the agent accepted
            // it; a command may start none, or other work, such as a
            // sign-in. A subagent's answer starts another turn of the agent
            // that started it.
            if !agents.turns.is_empty() || !agents.running.is_empty() || !agents.messages.is_empty()
            {
                return;
            }
            let answer = agents
                .conversations
                .get(agent)
                .ok()
                .filter(|conversation| conversation.messages().len() > before)
                .and_then(|conversation| conversation.messages().last())
                .and_then(final_answer);
            let ok = !run.failed || answer.is_some();
            if let Some(output) = output(answer.as_deref(), agents.sections) {
                println!("{output}");
            }
            run.step = Step::Done;
            exits.write(match (run.refused, ok) {
                (true, _) => AppExit::from_code(2),
                (false, true) => AppExit::Success,
                (false, false) => AppExit::from_code(1),
            });
        }
        Step::Done => {}
    }
}

/// What the run prints: the answer, then each [`PrintSection`] by tag,
/// each after an empty line; `None` when there is nothing.
fn output<'a>(
    answer: Option<&str>,
    sections: impl IntoIterator<Item = &'a PrintSection>,
) -> Option<String> {
    let mut sections: Vec<&PrintSection> = sections
        .into_iter()
        .filter(|section| !section.text.is_empty())
        .collect();
    sections.sort_by(|a, b| a.tag.cmp(&b.tag));
    let parts: Vec<&str> = answer
        .into_iter()
        .chain(sections.iter().map(|section| section.text.as_str()))
        .map(str::trim_end)
        .collect();
    (!parts.is_empty()).then(|| parts.join("\n\n"))
}

/// Notes failures and writes them to stderr, with every notice of a
/// `/command` prompt and why it was refused.
fn print_notices(
    mut run: ResMut<PrintRun>,
    mut notices: MessageReader<Notice>,
    mut recalled: MessageReader<Recalled>,
) {
    for why in recalled.read().filter_map(|recalled| recalled.why.as_ref()) {
        eprintln!("rig: {why}");
        run.refused = true;
    }
    // A command's answers are its notices.
    let command = run.prompt.trim_start().starts_with('/');
    for notice in notices.read() {
        if notice.level != NoticeLevel::Error {
            if command {
                eprintln!("{}", notice.text);
            }
            continue;
        }
        if let Step::Sent { agent, .. } | Step::Connecting { agent, .. } = run.step
            && notice.agent.is_none_or(|about| about == agent)
        {
            run.failed = true;
        }
        eprintln!("rig: {}", notice.text);
    }
}

#[cfg(test)]
mod tests;
