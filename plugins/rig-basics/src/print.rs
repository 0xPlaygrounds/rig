//! The agent over its pipes, for a script or another agent; `rig
//! help-agents` is the guide a driver reads. Logs go to stderr.
//!
//! `-p` ([`FRONT`]), the front of every run that is not interactive
//! ([`Invoked::interactive`]) unless another front's option chose one,
//! answers its prompt, with what was piped in on stdin after it (`git diff
//! | rig -p "review this"`), and exits. The answer goes to stdout, then the
//! [`PrintSection`]s; failures, notices and each [`ToolProgress`] line go
//! to stderr. `--json` ([`JSON`]) writes the run as JSON lines instead,
//! each with the protocol's [`VERSION`] as `v`. Without `-p` it is
//! long-lived: each line on stdin is a request, answered by a `reply`
//! line, and the end of stdin ends the agent at once. Nobody answers a
//! one-shot run's questions: each is refused, with why, unless
//! `--answer=ID[,ID…]` ([`ANSWER`]) gives choices to answer with. A
//! long-lived run is an [`Answerer`]: a question is an `ask` event, and
//! the driver's `answer` request answers it.
//!
//! [`AppHeadlessExt::run_headless`] is the run loop of both, and of any
//! front nobody sits at. It sends the prompt to the main agent once it has
//! a model (`--model`, else its restored session's, else the first one
//! reachable; a `/command` needs none) and exits once no agent works, so
//! the subagents it started have answered. A plugin that sends the agent
//! work over time, such as a file watch, keeps the run going with a
//! [`KeepRunning`] entity. Such a run may start without a prompt, and so
//! may one whose restored session left a turn unfinished (`rig -r <id>
//! -p`): that turn's answer is the run's; a run the launcher restarted on a
//! reload with nothing left ends with 0. The exit code is 0 answered, 2
//! for no prompt or a refused slash command, else 1, such as for an error
//! notice about the agent and no answer; a run of several turns exits
//! with the worst.

use std::io::{IsTerminal, Read, Write};

use async_channel::{Receiver, Sender};
use bevy_ecs::message::{MessageCursor, Messages};
use bevy_ecs::system::SystemParam;
use rig_harness::prelude::*;
use rig_harness::rig_core::completion::Message;
use serde::Deserialize;
use serde_json::{Map, Value, json};

use crate::telemetry::usage::{SessionUsage, TurnUsage};

/// `--print`'s name as a front.
pub const FRONT: &str = "print";

/// `--json`'s name as a front, and its option.
pub const JSON: &str = "json";

/// The name the print front decides effects as ([`Decider::plugin`]), in
/// effect logs: the module's.
pub const ORIGIN: &str = "print";

/// The option that answers a one-shot run's questions.
pub const ANSWER: &str = "answer";

/// The option that intersects every agent's tool access with the names given.
/// Omission leaves access unchanged; an empty value permits no tools. Unknown
/// names grant nothing. Subagents retain inherited restrictions and this limit.
pub const TOOLS: &str = "tools";

/// The version of the JSON lines, `v` on every line both ways.
pub const VERSION: u64 = 1;

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

/// While any entity has it, a run nobody sits at does not end when its
/// agents are idle: a plugin that sends the agent work over time spawns
/// one, and despawns it to let the run end.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct KeepRunning;

/// Triggered each time the run's work settles (no agent works) after a
/// turn of the agent its prompts go to ended.
#[derive(Event, Clone, Debug, PartialEq, Eq)]
pub struct HeadlessTurnEnded {
    /// The agent the prompts go to.
    pub agent: Entity,
    /// The answer of its last turn, if it had one.
    pub answer: Option<String>,
}

/// How the run ended, triggered as it asks to exit ([`Exit`]).
#[derive(Event, Clone, Debug, PartialEq, Eq)]
pub struct HeadlessEnded {
    /// The agent the prompt went to, if it got that far.
    pub agent: Option<Entity>,
    /// The answer of the last [`HeadlessTurnEnded`], if it had one.
    pub answer: Option<String>,
    /// Why the run could not send the prompt, such as no model.
    pub why: Option<String>,
    /// The exit code, the worst of its turns'.
    pub code: u8,
}

/// Answers `--print` and `--json` runs.
#[derive(Default)]
pub struct PrintPlugin;

impl Plugin for PrintPlugin {
    fn build(&self, app: &mut App) {
        let json = "Write the run as JSON lines; without -p, read requests as JSON lines on \
                    stdin (rig help-agents)";
        let answer = "Answer the questions of a -p run with these choices";
        app.add_front(Front::new(FRONT).headless())
            .add_front(Front::new(JSON).on_flag(JSON, json))
            .add_flag(Flag::new(ANSWER, answer).value("ID[,ID…]"))
            .add_flag(Flag::new(TOOLS, "Give the agent only these tools").value("NAME[,NAME…]"));
    }

    fn finish(&self, app: &mut App) {
        let invoked = app.world().get_resource::<Invoked>().cloned();
        let list = |name| -> Vec<String> {
            let list = invoked
                .as_ref()
                .and_then(|invoked| invoked.flag_value(name));
            let items = list.unwrap_or_default().split(',').map(str::trim);
            items
                .filter(|item| !item.is_empty())
                .map(str::to_owned)
                .collect()
        };
        let (answers, tools) = (list(ANSWER), list(TOOLS));
        if invoked.as_ref().is_some_and(|invoked| invoked.flag(TOOLS)) {
            app.insert_resource(Tools(tools)).add_systems(
                AgentPass,
                (
                    limit_tools.in_set(PrepareSystems::Input),
                    limit_offers.in_set(PrepareSystems::Edit),
                ),
            );
        }
        let json = app.runs_front(JSON);
        if !json && !app.runs_front(FRONT) {
            return;
        }
        let prompt = invoked.and_then(|invoked| invoked.args.print);
        if json {
            if !app.world().contains_resource::<JsonOut>() {
                app.insert_resource(JsonOut::new(std::io::stdout()));
            }
            app.add_observer(turn_started)
                .add_observer(started)
                .add_observer(settled)
                .add_observer(json_ended)
                .add_systems(
                    Update,
                    (ready.run_if(resource_added::<SessionRestored>), notices),
                )
                .init_resource::<TextRead>()
                .add_systems(PostUpdate, stream_text);
        } else {
            let command = prompt.as_deref().unwrap_or_default().trim_start();
            app.init_resource::<Printed>()
                .add_systems(
                    Update,
                    (print_notices(command.starts_with('/')), print_progress),
                )
                .add_observer(print_answer)
                .add_observer(print_ended);
        }
        if json && prompt.is_none() {
            // A long-lived `--json` run: the driver answers questions, and
            // stdin keeps it going until it ends.
            app.world_mut()
                .spawn((Answerer::default(), Requests, KeepRunning));
            if !app.world().contains_resource::<JsonIn>() {
                let (sender, input) = JsonIn::channel();
                read_stdin(sender);
                app.insert_resource(input);
            }
            app.add_systems(Startup, listen)
                .add_observer(request)
                .add_observer(input_ended)
                .run_headless(None);
            return;
        }
        let prompt = prompt.unwrap_or_default();
        if !answers.is_empty() {
            app.world_mut().spawn(Answerer::default());
        }
        app.insert_resource(Answers(answers))
            .configure_sets(
                AgentPass,
                PrintSystems.in_set(Stage::Gate).after(AskSystems),
            )
            .add_systems(AgentPass, answer.in_set(PrintSystems))
            .run_headless(Some(full_prompt(&prompt)));
    }
}

/// The Gate system that answers each question of a one-shot run, after
/// [`AskSystems`], so a line the user typed wins.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct PrintSystems;

/// The choices `--answer` gives, in order; none without it.
#[derive(Resource)]
struct Answers(Vec<String>);

/// The tools `--tools` leaves the agent.
#[derive(Resource)]
struct Tools(Vec<String>);

/// Intersects access before dispatch, including agents admitted this pass.
fn limit_tools(mut agents: Query<&mut ToolAccess, With<Agent>>, tools: Res<Tools>) {
    for mut access in &mut agents {
        let names = match &*access {
            ToolAccess::All => tools.0.clone(),
            ToolAccess::Only(names) => {
                if names.iter().all(|name| tools.0.contains(name)) {
                    continue;
                }
                names
                    .iter()
                    .filter(|name| tools.0.contains(name))
                    .cloned()
                    .collect()
            }
        };
        *access = ToolAccess::Only(names);
    }
}

/// Removes offers built at Admit before the agent's access was intersected.
fn limit_offers(
    agents: Query<&ToolAccess, With<Agent>>,
    mut calls: Query<(&ForAgent, &mut ModelRequest), Added<Admitted>>,
) {
    for (agent, mut call) in &mut calls {
        if let Ok(access) = agents.get(agent.0) {
            call.request_mut()
                .tools
                .retain(|tool| access.allows(&tool.name));
        }
    }
}

/// Answers each question with the first of the `--answer` choices it
/// offers, or refuses it, saying why.
fn answer(
    mut asks: Query<(&Ask, &mut Decision), (ReadyToStart, PluginEffects)>,
    answers: Res<Answers>,
    mut notices: MessageWriter<Notice>,
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
                        "nobody answers questions in a -p run, such as “{}”: --answer=ID \
                         answers them",
                        ask.title
                    ),
                    false => format!("--answer offers none of the choices of “{}”", ask.title),
                };
                notices.write(Notice::warn(None, why.clone()));
                Outcome::refused(why)
            }
        };
        decision.decide(Decider::plugin(ORIGIN), outcome);
    }
}

/// How long a prompt waits for the first of what is piped in on stdin: a
/// pipe its driver never writes to nor closes, as an agent's shell may
/// leave stdin, must not hold the run.
const PIPED_WAIT: Duration = Duration::from_secs(1);

/// The prompt with what was piped in on stdin after it.
fn full_prompt(prompt: &str) -> String {
    match std::io::stdin().is_terminal() {
        true => prompt.to_owned(),
        false => with_piped(prompt, std::io::stdin(), PIPED_WAIT),
    }
}

/// `prompt` with what `piped` holds after it. Without a prompt, `piped` is
/// the prompt, read to its end; with one, it is read only when something
/// comes within `wait`.
fn with_piped(prompt: &str, mut piped: impl Read + Send + 'static, wait: Duration) -> String {
    let (sender, read) = std::sync::mpsc::channel();
    let (first, rest) = (sender.clone(), sender);
    std::thread::spawn(move || {
        let mut text = Vec::new();
        let mut byte = [0];
        let got = piped.read(&mut byte).unwrap_or_default();
        first.send(None).ok();
        text.extend(byte.get(..got).unwrap_or_default());
        if got > 0 {
            piped.read_to_end(&mut text).ok();
        }
        rest.send(Some(String::from_utf8_lossy(&text).into_owned()))
            .ok();
    });
    let wait = match prompt.trim().is_empty() {
        true => Duration::MAX,
        false => wait,
    };
    if read.recv_timeout(wait).is_err() {
        eprintln!(
            "rig: nothing came on stdin within {}s, so the prompt goes alone; give -p stdin \
             from /dev/null, or pipe in what goes with it",
            wait.as_secs_f32()
        );
        return prompt.to_owned();
    }
    match read.recv().ok().flatten().unwrap_or_default() {
        piped if piped.trim().is_empty() => prompt.to_owned(),
        piped if prompt.trim().is_empty() => piped,
        piped => format!("{prompt}\n\n{piped}"),
    }
}

/// Starts a run nobody sits at on an [`App`].
pub trait AppHeadlessExt {
    /// Sends `prompt` to the main agent once it has a model, and exits when
    /// no agent works any more and no [`KeepRunning`] exists. Without a
    /// prompt, the run finishes the turns its restored session left
    /// unfinished, and waits for the work a `KeepRunning` plugin sends.
    /// Call it once, from the front's `finish`.
    fn run_headless(&mut self, prompt: Option<String>) -> &mut Self;
}

impl AppHeadlessExt for App {
    fn run_headless(&mut self, prompt: Option<String>) -> &mut Self {
        let invoked = self.world().get_resource::<Invoked>();
        let model = invoked.is_some_and(|invoked| invoked.args.model.is_some());
        let world = self.world_mut();
        let kept = world.query::<&KeepRunning>().iter(world).next().is_some();
        self.insert_resource(Run {
            prompt: prompt.filter(|prompt| !prompt.trim().is_empty()),
            model,
            kept,
            step: Step::Start,
            answer: None,
            turned: false,
            last: None,
            failed: false,
            refused: false,
            worst: 0,
        })
        .add_systems(Update, drive.after(AgentLoop))
        .add_observer(keep_answer)
        .add_observer(|_: On<Add<KeepRunning>>, mut run: ResMut<Run>| run.kept = true)
    }
}

/// Where the run is.
#[derive(Resource)]
struct Run {
    prompt: Option<String>,
    /// Whether `--model` named the model, which a models plugin chooses.
    model: bool,
    /// Whether a [`KeepRunning`] ever existed: a run something kept going
    /// has work to wait for, even when it ended before the run first
    /// looked.
    kept: bool,
    step: Step,
    /// What the agent's last turn since the work last settled answered.
    answer: Option<Message>,
    /// Whether a turn of the agent ended since the work last settled.
    turned: bool,
    /// The answer of the last [`HeadlessTurnEnded`].
    last: Option<String>,
    /// Whether an error notice about the agent, or a refused slash
    /// command, came since the work last settled.
    failed: bool,
    refused: bool,
    /// The worst exit code so far.
    worst: u8,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Step {
    Start,
    /// A model was chosen for `agent`, which connects it at once.
    Connecting {
        agent: Entity,
    },
    /// The prompt went to `agent`, or other work came, and the agents work.
    Sent {
        agent: Entity,
    },
    /// The agents are idle, and a [`KeepRunning`] keeps the run going.
    Waiting {
        agent: Entity,
    },
    Done,
}

/// The agents as the run reads them.
#[derive(SystemParam)]
struct Agents<'w, 's> {
    primary: MainAgentQuery<'w, 's>,
    choices: Query<'w, 's, (Option<&'static ModelChoice>, Has<Connection>)>,
    turns: Query<'w, 's, (), With<TurnOf>>,
    /// Work in progress, such as a sign-in; not a long-lived stream, such
    /// as a file watch.
    running: Query<'w, 's, &'static Running>,
    /// Messages on their way to an agent, such as the prompt itself.
    messages: Query<'w, 's, (), (With<Incoming>, Without<Settled>)>,
    /// Effects not started yet, such as one held at Gate, or answered and
    /// not settled, such as one held for review; not an exit.
    held: Query<
        'w,
        's,
        (),
        (
            With<Effect>,
            Without<Settled>,
            Without<ExitRequest>,
            Or<(Without<Started>, With<Outcome>)>,
        ),
    >,
    keep: Query<'w, 's, (), With<KeepRunning>>,
}

impl Agents<'_, '_> {
    /// Whether no agent works: no turn, task, held effect or message on its
    /// way. An open effect that started and waits on the world outside,
    /// such as a question nobody answers, is no work.
    fn idle(&self) -> bool {
        self.turns.is_empty()
            && self.messages.is_empty()
            && self.held.is_empty()
            && self.running.iter().all(Running::is_stream)
    }
}

/// Chooses the model, sends the prompt, and exits after the turn.
fn drive(
    mut run: ResMut<Run>,
    agents: Agents,
    models: Res<Models>,
    (start, wake): (Option<Res<AppStart>>, Res<Wake>),
    (mut notices, mut recalled): (MessageReader<Notice>, MessageReader<Recalled>),
    mut commands: Commands,
) {
    if recalled.read().any(|recalled| recalled.why.is_some()) {
        run.refused = true;
    }
    for notice in notices.read() {
        if let Step::Sent { agent } | Step::Waiting { agent } | Step::Connecting { agent } =
            run.step
            && notice.level == NoticeLevel::Error
            && notice.agent.is_none_or(|about| about == agent)
        {
            run.failed = true;
        }
    }
    // The launcher restarts a run whose agent reloaded without its prompt.
    let restarted = start
        .is_some_and(|start| matches!(start.0, StartReason::Reloaded | StartReason::Restarted));
    // A command such as `/login` runs without a model.
    let command = run
        .prompt
        .as_ref()
        .is_some_and(|prompt| prompt.trim_start().starts_with('/'));
    let ended = match run.step {
        Step::Start => {
            let Some(agent) = main_agent(&agents.primary) else {
                return;
            };
            let Ok((chosen, connected)) = agents.choices.get(agent) else {
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
                        let why = "no model can be reached: set a provider's API key, or name \
                                   one with --model";
                        return end(&mut run, ended(Some(agent), Some(why), 1), commands);
                    }
                },
            };
            // Connected already, as `--model` or the restored session
            // leaves it: the prompt goes out in the first frame, so a kill
            // soon after leaves it in the session to resume.
            if model.is_none() && (connected || command) {
                send(&mut run, agent, &agents, command, restarted, &mut commands)
            } else {
                if let Some(model) = model {
                    commands.trigger(SetModel {
                        entity: agent,
                        model,
                    });
                }
                run.step = Step::Connecting { agent };
                // The model connects as this frame ends, or a refusal says
                // why not; nothing else may ask for the frame that sees it.
                wake.wake();
                return;
            }
        }
        Step::Connecting { agent } => {
            send(&mut run, agent, &agents, command, restarted, &mut commands)
        }
        Step::Sent { agent } | Step::Waiting { agent } => {
            // The turn starts with the request once the agent accepted
            // it; a command may start none, or other work, such as a
            // sign-in. A subagent's answer starts another turn of the agent
            // that started it.
            if !agents.idle() {
                run.step = Step::Sent { agent };
                return;
            }
            let keep = !agents.keep.is_empty();
            // A turn may start and end between two runs of this system, as
            // a fast model's can within one run of the agent loop: the
            // turn's end, not having seen it work, says it happened.
            if run.step == (Step::Sent { agent }) || run.turned || !keep {
                settle(&mut run, agent, &mut commands);
            }
            if keep {
                run.step = Step::Waiting { agent };
                return;
            }
            let answered = run.last.clone();
            Some(HeadlessEnded {
                answer: answered,
                ..ended(Some(agent), None, run.worst)
            })
        }
        Step::Done => None,
    };
    if let Some(ended) = ended {
        end(&mut run, ended, commands);
    }
}

/// Sends the prompt to `agent` once it is connected, or else moves on to
/// the work there is; how the run ends, when it ends here.
fn send(
    run: &mut Run,
    agent: Entity,
    agents: &Agents,
    command: bool,
    restarted: bool,
    commands: &mut Commands,
) -> Option<HeadlessEnded> {
    let connected = command || agents.choices.get(agent).is_ok_and(|(_, on)| on);
    match run.prompt.clone() {
        // The notice of the failed choice says why.
        _ if !connected => return Some(ended(Some(agent), None, 1)),
        Some(text) => {
            commands.trigger(Deliver::new(agent, text, DeliveryMode::Steer));
            run.step = Step::Sent { agent };
        }
        // Something keeps the run going, or already did: a turn it started
        // may have ended while the model was connecting.
        None if !agents.keep.is_empty() || run.turned || run.kept => {
            run.step = Step::Waiting { agent }
        }
        // Work the restored session left runs, such as an unfinished turn:
        // it finishes, and its answer is the run's.
        None if !agents.idle() => run.step = Step::Sent { agent },
        // The run restarted on a reload with nothing left to do.
        None if restarted => return Some(ended(Some(agent), None, 0)),
        None => {
            let why = "no prompt: give one after --print, or pipe it in";
            return Some(ended(Some(agent), Some(why), 2));
        }
    }
    None
}

/// A run that ended with `code`, for `why` when no notice says it.
fn ended(agent: Option<Entity>, why: Option<&str>, code: u8) -> HeadlessEnded {
    HeadlessEnded {
        agent,
        answer: None,
        why: why.map(str::to_owned),
        code,
    }
}

/// Notes each turn of the agent the prompts go to that ended, with its
/// answer when it answered, whatever its conversation looks like now, such
/// as after a compaction.
fn keep_answer(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    primary: MainAgentQuery,
    wake: Res<Wake>,
    mut run: ResMut<Run>,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    // While connecting too: a turn a plugin started may end before the
    // model the run chose is connected. And before the run first looked:
    // a turn the restored session left may end within the first frame,
    // which then asks for the frame that settles it.
    let to = match run.step {
        Step::Start => main_agent(&primary),
        Step::Connecting { agent } | Step::Sent { agent } | Step::Waiting { agent } => Some(agent),
        Step::Done => None,
    };
    if to != Some(agent) {
        return;
    }
    if run.step == Step::Start {
        wake.wake();
    }
    run.turned = true;
    if let TurnOutcome::Answered(message) = outcome {
        run.answer = Some(message.clone());
    }
}

/// Takes what came since the work last settled: the exit code it earns,
/// and a [`HeadlessTurnEnded`] when a turn of `agent` ended.
fn settle(run: &mut Run, agent: Entity, commands: &mut Commands) {
    let answer = run.answer.take().and_then(|message| message.final_answer());
    let code = match (run.refused, run.failed && answer.is_none()) {
        (true, _) => 2,
        (false, true) => 1,
        (false, false) => 0,
    };
    run.worst = run.worst.max(code);
    if run.turned {
        let ended = HeadlessTurnEnded {
            agent,
            answer: answer.clone(),
        };
        commands.trigger(ended);
        run.last = answer;
    }
    (run.turned, run.failed, run.refused) = (false, false, false);
}

/// Ends the run as `ended` says.
fn end(run: &mut Run, ended: HeadlessEnded, mut commands: Commands) {
    run.step = Step::Done;
    let exit = match ended.code {
        0 => ExitReason::Quit,
        code => ExitReason::Failed(code),
    };
    commands.trigger(Exit(exit));
    commands.trigger(ended);
}

/// Whether the run wrote anything to stdout yet.
#[derive(Resource, Default)]
struct Printed(bool);

impl Printed {
    /// Writes `block` to stdout, after an empty line when something came
    /// before it.
    fn print(&mut self, block: Option<String>) {
        if let Some(block) = block {
            let gap = if self.0 { "\n" } else { "" };
            println!("{gap}{block}");
            self.0 = true;
        }
    }
}

/// Writes each answer to stdout as it comes.
fn print_answer(ended: On<HeadlessTurnEnded>, mut printed: ResMut<Printed>) {
    printed.print(output(ended.answer.as_deref(), []));
}

/// Once the run ends, writes the [`PrintSection`]s to stdout, and why it
/// could not send the prompt to stderr.
fn print_ended(
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

/// Writes warnings, errors and why a command was refused to stderr, with
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

/// Where `--json` lines go: stdout unless the app inserted another.
#[derive(Resource)]
struct JsonOut(Box<dyn Write + Send + Sync>);

impl JsonOut {
    fn new(out: impl Write + Send + Sync + 'static) -> Self {
        Self(Box::new(out))
    }
}

/// The request lines of a long-lived `--json` run: stdin's unless the app
/// inserted another. They end when every sender dropped.
#[derive(Resource)]
struct JsonIn(Receiver<String>);

impl JsonIn {
    fn channel() -> (Sender<String>, Self) {
        let (sender, receiver) = async_channel::unbounded();
        (sender, Self(receiver))
    }
}

/// How far the text events have read the model calls' [`StreamDelta`]s.
#[derive(Resource, Default)]
struct TextRead(MessageCursor<StreamDelta>);

/// Writes events, each naming its agent, after the text streamed before
/// them.
#[derive(SystemParam)]
struct Out<'w, 's> {
    out: ResMut<'w, JsonOut>,
    ids: Query<'w, 's, &'static AgentId>,
    deltas: Res<'w, Messages<StreamDelta>>,
    read: ResMut<'w, TextRead>,
    turn_calls: Query<'w, 's, (), With<TurnCall>>,
}

impl Out<'_, '_> {
    /// Writes the event `kind` with `fields` about `agent`.
    fn emit(&mut self, agent: Option<Entity>, kind: &str, fields: Value) {
        self.text();
        self.write(agent, kind, fields);
    }

    /// Writes the text each turn's model call streamed since the last
    /// event, one line per agent's run of it.
    fn text(&mut self) {
        let mut runs: Vec<(Option<Entity>, String)> = Vec::new();
        let deltas = self.read.0.read(&self.deltas);
        for delta in deltas.filter(|delta| self.turn_calls.contains(delta.effect)) {
            let StreamEvent::Text { text, .. } = &delta.event else {
                continue;
            };
            match runs.last_mut() {
                Some((agent, run)) if *agent == delta.agent => run.push_str(text),
                _ => runs.push((delta.agent, text.clone())),
            }
        }
        for (agent, text) in runs {
            self.write(agent, "text", json!({ "text": text }));
        }
    }

    fn write(&mut self, agent: Option<Entity>, kind: &str, fields: Value) {
        let mut line = Map::from_iter([
            ("v".to_owned(), VERSION.into()),
            ("type".to_owned(), kind.into()),
        ]);
        if let Value::Object(fields) = fields {
            line.extend(fields);
        }
        if let Some(AgentId(id)) = agent.and_then(|agent| self.ids.get(agent).ok()) {
            line.insert("agent".to_owned(), id.clone().into());
        }
        writeln!(self.out.0, "{}", Value::Object(line)).ok();
        self.out.0.flush().ok();
    }
}

/// The session's id, once it has one.
fn session_id(paths: Option<Res<SessionPaths>>) -> Option<String> {
    Some(paths?.0.id()?.as_str().to_owned())
}

/// The first line, once the session is restored.
fn ready(primary: MainAgentQuery, paths: Option<Res<SessionPaths>>, mut out: Out) {
    let session = session_id(paths);
    out.emit(main_agent(&primary), "ready", json!({ "session": session }));
}

/// The last line, as the run asks to exit.
fn json_ended(
    ended: On<HeadlessEnded>,
    (paths, usage): (Option<Res<SessionPaths>>, Option<Res<SessionUsage>>),
    mut out: Out,
) {
    let fields = json!({
        "code": ended.code,
        "answer": ended.answer,
        "error": ended.why,
        "session": session_id(paths),
        "usage": usage.map(|usage| usage.0.total),
    });
    out.emit(ended.agent, "exit", fields);
}

fn turn_started(turn: On<Add<TurnOf>>, turns: Query<&TurnOf>, mut out: Out) {
    if let Ok(&TurnOf(agent)) = turns.get(turn.entity) {
        out.emit(Some(agent), "turn_start", json!({}));
    }
}

/// A tool call starts, or a question waits for its answer.
fn started(
    start: On<Add<Started>>,
    calls: Query<(&ToolRequest, &ForAgent)>,
    asks: Query<(&Ask, Option<&ForAgent>)>,
    mut out: Out,
) {
    if let Ok((request, &ForAgent(agent))) = calls.get(start.entity) {
        let call = &request.call;
        let fields = json!({
            "id": call.id.wire(),
            "name": call.function.name.as_str(),
            "args": call.function.arguments_value(),
        });
        out.emit(Some(agent), "tool_call", fields);
    } else if let Ok((ask, agent)) = asks.get(start.entity) {
        let choices = ask.choices.iter();
        let fields = json!({
            "ask": start.entity.to_bits(),
            "title": ask.title,
            "body": ask.body,
            "choices": choices.map(|choice| json!({ "id": choice.id, "label": choice.label })).collect::<Vec<_>>(),
            "text": ask.free_text,
            "multiple": ask.multiple,
        });
        out.emit(agent.map(|agent| agent.0), "ask", fields);
    }
}

/// A tool call or a turn ended.
fn settled(
    settled: On<Add<Settled>>,
    effects: Query<(
        &ForAgent,
        Option<&Outcome>,
        Option<&TurnOutcome>,
        Option<&TurnUsage>,
    )>,
    calls: Query<&ToolRequest>,
    mut out: Out,
) {
    let Ok((&ForAgent(agent), outcome, ended, used)) = effects.get(settled.entity) else {
        return;
    };
    if let (Ok(request), Some(outcome)) = (calls.get(settled.entity), outcome) {
        let result = outcome.tool_result();
        let fields = json!({
            "id": request.call.id.wire(),
            "name": request.call.function.name.as_str(),
            "ok": result.is_success(),
            "output": result.output().render(),
        });
        out.emit(Some(agent), "tool_result", fields);
    } else if let Some(ended) = ended {
        let (outcome, answer, error) = match ended {
            TurnOutcome::Answered(message) => ("answered", message.final_answer(), None),
            TurnOutcome::Failed(why) => ("failed", None, Some(why.clone())),
            TurnOutcome::Stopped(why) => ("stopped", None, why.clone()),
        };
        let usage = used.map(|used| used.0);
        let fields =
            json!({ "outcome": outcome, "answer": answer, "error": error, "usage": usage });
        out.emit(Some(agent), "turn_end", fields);
    }
}

/// Writes the text that streamed this frame.
fn stream_text(mut out: Out) {
    out.text();
}

fn notices(mut notices: MessageReader<Notice>, mut out: Out) {
    for notice in notices.read() {
        let level = match notice.level {
            NoticeLevel::Info => "info",
            NoticeLevel::Warn => "warn",
            NoticeLevel::Error => "error",
        };
        let fields = json!({ "level": level, "text": notice.text });
        out.emit(notice.agent, "notice", fields);
    }
}

/// The request lines of a long-lived run, a stream on this entity.
#[derive(Component)]
struct Requests;

/// Starts the thread that sends stdin's lines to `sender`.
fn read_stdin(sender: Sender<String>) {
    std::thread::spawn(move || {
        for line in std::io::stdin().lines() {
            let Ok(line) = line else {
                break;
            };
            if sender.send_blocking(line).is_err() {
                break;
            }
        }
    });
}

/// Reads the request lines as a stream on the [`Requests`] entity.
fn listen(
    input: Option<Res<JsonIn>>,
    requests: Query<Entity, With<Requests>>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let (Some(input), Ok(entity)) = (input, requests.single()) else {
        return;
    };
    let pool = bevy_tasks::IoTaskPool::get_or_init(bevy_tasks::TaskPool::default);
    let stream = Running::stream(pool, &wake, input.0.clone());
    commands.entity(entity).insert(stream);
    commands.remove_resource::<JsonIn>();
}

/// The end of stdin ends the run.
fn input_ended(
    ended: On<Remove<Running>>,
    requests: Query<(), With<Requests>>,
    mut commands: Commands,
) {
    if requests.contains(ended.entity) {
        commands.trigger(Exit(ExitReason::Quit));
    }
}

/// A request a driver writes on stdin, as `rig help-agents` lists them.
#[derive(Deserialize, Debug, Clone, PartialEq, Eq)]
#[serde(tag = "type", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Send {
        text: String,
        #[serde(default)]
        agent: Option<String>,
    },
    Stop {
        #[serde(default)]
        agent: Option<String>,
    },
    Answer {
        ask: u64,
        #[serde(default)]
        choices: Vec<String>,
        #[serde(default)]
        text: Option<String>,
    },
    State,
}

/// What a request needs of the app.
#[derive(SystemParam)]
struct Requested<'w, 's> {
    agents: Query<
        'w,
        's,
        (
            Entity,
            &'static AgentId,
            Has<ActiveTurn>,
            Option<&'static ModelChoice>,
        ),
    >,
    primary: MainAgentQuery<'w, 's>,
    asks: Query<
        'w,
        's,
        (
            Entity,
            &'static Ask,
            Option<&'static ForAgent>,
            &'static mut Decision,
        ),
        (Without<Outcome>, Without<Settled>),
    >,
    commands: Commands<'w, 's>,
}

impl Requested<'_, '_> {
    /// The agent named `id`, else the main agent.
    fn agent(&self, id: Option<String>) -> Result<Entity, String> {
        let agent = match id {
            Some(id) => {
                let ids = self.agents.iter().map(|(entity, id, ..)| (entity, id));
                AgentId(id).find_in(ids)
            }
            None => main_agent(&self.primary),
        };
        agent.ok_or_else(|| "no such agent".to_owned())
    }

    /// The agents and the open questions.
    fn state(&self) -> Value {
        let id = |agent: Entity| self.agents.get(agent).ok().map(|(_, id, ..)| &id.0);
        let agents: Vec<Value> = self
            .agents
            .iter()
            .map(|(_, id, working, model)| {
                json!({ "agent": id.0, "working": working, "model": model.map(|model| &model.0) })
            })
            .collect();
        let asks: Vec<Value> = self
            .asks
            .iter()
            .map(|(ask, question, agent, _)| {
                let agent = agent.and_then(|agent| id(agent.0));
                json!({ "ask": ask.to_bits(), "agent": agent, "title": question.title })
            })
            .collect();
        json!({ "agents": agents, "asks": asks })
    }
}

/// Carries out a request line and answers it with a `reply`.
fn request(
    line: On<Insert<Done<String>>>,
    lines: Query<&Done<String>, With<Requests>>,
    mut requested: Requested,
    mut out: Out,
) {
    let Ok(Done(line)) = lines.get(line.entity) else {
        return;
    };
    if line.trim().is_empty() {
        return;
    }
    let (id, done) = match serde_json::from_str::<Value>(line) {
        Ok(value) => (
            value.get("id").cloned().unwrap_or(Value::Null),
            versioned(value).and_then(|request| carry_out(&mut requested, request)),
        ),
        Err(error) => (Value::Null, Err(format!("not a JSON line: {error}"))),
    };
    let fields = match done {
        Ok(None) => json!({ "id": id, "ok": true }),
        Ok(Some(state)) => json!({ "id": id, "ok": true, "state": state }),
        Err(error) => json!({ "id": id, "ok": false, "error": error }),
    };
    out.emit(None, "reply", fields);
}

/// The request a line of this version holds, or why it is refused.
fn versioned(mut value: Value) -> Result<Request, String> {
    let fields = value.as_object_mut().ok_or("a request is a JSON object")?;
    let version = fields.shift_remove("v");
    fields.shift_remove("id");
    match version.as_ref().and_then(Value::as_u64) {
        Some(VERSION) => {}
        Some(newer) if newer > VERSION => {
            return Err(format!(
                "this line is version {newer} and this rig speaks version {VERSION}: update rig"
            ));
        }
        _ => {
            return Err(format!(
                "this line is version {} and this rig speaks version {VERSION}: update the \
                 driver, whose every line carries \"v\":{VERSION}",
                version.map_or("none".to_owned(), |version| version.to_string())
            ));
        }
    }
    serde_json::from_value(value).map_err(|error| format!("not a request: {error}"))
}

/// Carries out `request`; what `state` asked for.
fn carry_out(requested: &mut Requested, request: Request) -> Result<Option<Value>, String> {
    match request {
        Request::Send { text, agent } => {
            let agent = requested.agent(agent)?;
            let message = Deliver::new(agent, text, DeliveryMode::Steer);
            requested.commands.trigger(message);
        }
        Request::Stop { agent } => {
            let agent = requested.agent(agent)?;
            requested.commands.trigger(StopTurn::new(agent));
        }
        Request::Answer { ask, choices, text } => {
            let entity = Entity::try_from_bits(ask).ok_or("no such question")?;
            let (_, question, _, mut decision) = requested
                .asks
                .get_mut(entity)
                .map_err(|_| "no open question with that id")?;
            let offered = |choice: &&String| question.choices.iter().any(|c| &c.id == *choice);
            if let Some(choice) = choices.iter().find(|choice| !offered(choice)) {
                return Err(format!("the question offers no choice `{choice}`"));
            }
            if choices.is_empty() && !(text.is_some() && question.free_text) {
                return Err("the answer needs a choice".to_owned());
            }
            let answer = Answer { choices, text };
            decision.decide(Decider::Person, answer.outcome());
        }
        Request::State => return Ok(Some(requested.state())),
    }
    Ok(None)
}

#[cfg(test)]
mod tests;
