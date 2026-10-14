//! `--print`: one prompt to the primary agent, driven by rig-harness's
//! [`headless`](rig_harness::headless) run: it exits once no agent works,
//! so the subagents it started have answered, unless a plugin keeps it
//! going ([`KeepRunning`]); such a run may start without a prompt. Each
//! answer's text goes to stdout as it comes, and failures to stderr; the exit code is 0 when the turn ended
//! with an answer, 2 for no prompt or a refused slash command (unknown, or
//! arguments it does not take), 1 otherwise. Text piped in on stdin
//! follows the prompt, so `git diff | rig -p "review this"` works.
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
//! first of those choices it offers instead (a question that takes several
//! with every one it offers), and refuses one that offers none. After the
//! answer, the run prints what other plugins add as [`PrintSection`]s,
//! such as the agent's todo list. While it runs, each change of a running
//! call's [`ToolProgress`] goes to stderr.

use std::io::{IsTerminal, Read as _};

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
        let prompt = invoked
            .and_then(|invoked| invoked.args.print)
            .unwrap_or_default();
        // A command's answers are its notices.
        let command = prompt.trim_start().starts_with('/');
        app.insert_resource(Answers(answers))
            .configure_sets(
                AgentPass,
                PrintSystems.in_set(Stage::Gate).after(AskSystems),
            )
            .add_systems(AgentPass, answer.in_set(PrintSystems))
            .add_systems(Update, (print_notices(command), print_progress))
            .init_resource::<Printed>()
            .add_observer(print_answer)
            .add_observer(print_output)
            .run_headless(Some(full_prompt(&prompt)));
    }
}

/// The Gate system that answers each question of a `--print` run, after
/// [`AskSystems`], so a line the user typed wins.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrintSystems;

/// The choices `--answer` gives, in order; none without it.
#[derive(Resource)]
struct Answers(Vec<String>);

/// Answers each question with the first of the `--answer` choices it
/// offers, or refuses it, saying why on stderr.
fn answer(
    mut asks: Query<(&Ask, &mut Decision), (ReadyToStart, PluginEffects)>,
    answers: Res<Answers>,
) {
    for (ask, mut decision) in &mut asks {
        if decision.is_decided() {
            continue;
        }
        let offered = |id: &&String| ask.choices.iter().any(|choice| &choice.id == *id);
        let taken = ask.choices.iter().map(|choice| &choice.id);
        let taken: Vec<String> = taken.filter(|id| answers.0.contains(id)).cloned().collect();
        let answer = match ask.multiple {
            true => Some(taken)
                .filter(|taken| !taken.is_empty())
                .map(Answer::choices),
            false => answers.0.iter().find(offered).cloned().map(Answer::choice),
        };
        let outcome = match answer {
            Some(answer) => answer.outcome(),
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
        decision.decide(Decider::plugin("rig-print"), outcome);
    }
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

/// Whether the run wrote anything to stdout yet.
#[derive(Resource, Default)]
struct Printed(bool);

impl Printed {
    /// Writes `block` to stdout, after an empty line when something came
    /// before it.
    fn print(&mut self, block: Option<String>) {
        if let Some(block) = block {
            match self.0 {
                true => println!("\n{block}"),
                false => println!("{block}"),
            }
            self.0 = true;
        }
    }
}

/// Writes each answer to stdout as it comes.
fn print_answer(ended: On<HeadlessTurnEnded>, mut printed: ResMut<Printed>) {
    printed.print(output(ended.answer.as_deref(), []));
}

/// Writes why the run could not send the prompt to stderr, and the
/// [`PrintSection`]s to stdout.
fn print_output(
    ended: On<HeadlessEnded>,
    sections: Query<&PrintSection>,
    mut printed: ResMut<Printed>,
) {
    if let Some(why) = &ended.why {
        eprintln!("rig: {why}");
    }
    printed.print(output(None, sections));
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

/// Writes each new line of a running call's [`ToolProgress`] to stderr,
/// after the tool's name.
fn print_progress(progress: Query<(&ToolProgress, Option<&ToolRequest>), Changed<ToolProgress>>) {
    for (ToolProgress(line), tool) in &progress {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match tool {
            Some(tool) => eprintln!("rig: {}: {line}", tool.call.function.name),
            None => eprintln!("rig: {line}"),
        }
    }
}

/// Writes error notices and why a command was refused to stderr, with
/// every notice of a `/command` prompt.
fn print_notices(
    command: bool,
) -> impl FnMut(MessageReader<Notice>, MessageReader<Recalled>) + Send + Sync + 'static {
    move |mut notices, mut recalled| {
        for why in recalled.read().filter_map(|recalled| recalled.why.as_ref()) {
            eprintln!("rig: {why}");
        }
        for notice in notices.read() {
            match notice.level {
                NoticeLevel::Warn | NoticeLevel::Error => eprintln!("rig: {}", notice.text),
                _ if command => eprintln!("{}", notice.text),
                _ => {}
            }
        }
    }
}

#[cfg(test)]
mod tests;
