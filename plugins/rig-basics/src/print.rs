//! The agent over its pipes, for a script or another agent; `rig
//! help-agents` is the guide a driver reads. Logs go to stderr.
//!
//! `-p` ([`FRONT`]), the front of every run that is not interactive
//! ([`Invoked::interactive`]) unless another front's option chose one,
//! answers its prompt, with what was piped in on stdin after it (`git diff
//! | rig -p "review this"`), and exits. The answer goes to stdout, then the
//! [`PrintSection`]s; failures, notices and each [`ToolProgress`] line go
//! to stderr. `--json` ([`JSON`]) writes the run as JSON lines instead,
//! each with the protocol's [`VERSION`] as `v` and the time it was written
//! as `ms`, after the notices before it; `exit` is the last, written as
//! the agent exits, after its session log, with why the run failed as its
//! `error`, and nothing is written after it. Without `-p` it is
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
//! [`KeepRunning`] entity, and one whose tools come in the background holds
//! the start until they are in with a [`Starting`] entity. A kept run may
//! start without a prompt, and so may one whose restored session has a turn
//! a plugin carries on (`CarryOn`, `rig -r <id> -p`): that turn's answer is
//! the run's. A turn an exit or a crash cut short is stopped, not finished,
//! so without a prompt such a run has none; a run the launcher restarted on
//! a reload with nothing left ends with 0. The exit code is 0 answered, 2
//! for no prompt, a refused slash command, a `--tools` name no tool has or
//! a `--model` the catalog lacks, 130 when cut off, as by a signal or a
//! reader of stdout gone, else 1, such as for a failed turn, a turn that
//! ended with no answer, or an error notice about the agent and no answer;
//! a run of several turns exits with the worst. A `--json` run that exits
//! before its work ends still closes each tool call and turn it started,
//! and a one-shot one writes its `exit` line.

use std::collections::HashSet;
use std::io::{IsTerminal, Read, Write};
use std::sync::mpsc::RecvTimeoutError;
use std::time::Instant;

use async_channel::{Receiver, Sender};
use bevy_ecs::message::{MessageCursor, Messages};
use bevy_ecs::system::SystemParam;
use rig_core::json_utils::remove_preserving_order;
use rig_ecs::{StopTurnsOnExit, WriteLogs};
use rig_harness::prelude::*;
use rig_harness::rig_core::catalog::edit_distance;
use rig_harness::rig_core::redact;
use rig_harness::rig_core::tool::feedback::not_finished;
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
/// Omission leaves access unchanged; an empty value permits no tools. A run
/// refuses a name no registered tool has, with exit code 2, before any
/// model or tool call, headless or at a terminal. A headless run checks
/// once every plugin finished and no [`Starting`] is left, so a tool a
/// plugin registers in the background, such as an MCP server's, can be
/// named while its plugin holds the start; a run at a terminal checks once
/// every plugin finished, so such a tool cannot be named there. Subagents
/// retain inherited restrictions and this limit.
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

/// While any entity has it, a run nobody sits at waits before it checks
/// `--tools` and sends its prompt: a plugin whose tools arrive in the
/// background, such as an MCP client, spawns one for each source and
/// removes it once their tools are in or it gave up, and bounds that wait
/// itself.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Starting;

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
    /// Why the run failed, such as no model or the failed turn's error;
    /// `None` only when it exits with 0.
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
        app.add_systems(
            AgentPass,
            hold_calls
                .in_set(Stage::Gate)
                .run_if(resource_exists::<HoldCalls>),
        );
        if invoked.as_ref().is_some_and(|invoked| invoked.flag(TOOLS)) {
            app.insert_resource(Tools(tools)).add_systems(
                AgentPass,
                (
                    limit_tools.in_set(PrepareSystems::Input),
                    limit_offers.in_set(PrepareSystems::Edit),
                )
                    .distributive_run_if(resource_exists::<Tools>),
            );
        }
        let json = app.runs_front(JSON);
        if !json && !app.runs_front(FRONT) {
            return;
        }
        // Output an app sends elsewhere has no reader of stdout to lose.
        if !json || !app.world().contains_resource::<JsonOut>() {
            app.init_resource::<ExitWhenUnread>();
        }
        let unread = app
            .world()
            .get_resource::<ExitWhenUnread>()
            .map(|unread| unread.0);
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
                    ready
                        .run_if(resource_added::<SessionRestored>)
                        .before(AgentLoop),
                )
                .init_resource::<LinesRead>()
                .add_systems(PostUpdate, flush)
                .add_systems(
                    Last,
                    (
                        note_cut_off.before(StopTurnsOnExit),
                        json_exit.after(WriteLogs),
                    )
                        .in_set(OnAppExitSystems)
                        .distributive_run_if(on_message::<AppExit>),
                );
            if prompt.is_some() {
                app.world_mut().resource_mut::<JsonOut>().one_shot = true;
            }
        } else {
            let command = prompt.as_deref().unwrap_or_default().trim_start();
            app.init_resource::<Printed>()
                .add_systems(
                    Update,
                    (
                        // Before the run can end on what they say, so its
                        // why knows them.
                        print_notices(command.starts_with('/'))
                            .after(AgentLoop)
                            .before(drive),
                        print_progress,
                    ),
                )
                .add_observer(print_answer)
                .add_observer(print_ended)
                .add_systems(
                    Last,
                    print_interrupted
                        .in_set(OnAppExitSystems)
                        .run_if(on_message::<AppExit>),
                );
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
        // Read before the first frame, so no work starts before the
        // prompt is known; a signal, the launcher's death or a reader of
        // stdout gone ends the wait, and the first frame exits.
        let signalled = signals::interruption(app.world());
        let interrupted = move || signalled() || unread.is_some_and(|unread| unread());
        let piped = app.world_mut().remove_resource::<PipedIn>();
        let prompt = full_prompt(&prompt, piped, interrupted);
        let interrupted = prompt.is_none();
        if interrupted {
            app.insert_resource(HoldCalls);
        }
        app.insert_resource(Answers(answers))
            .configure_sets(
                AgentPass,
                PrintSystems.in_set(Stage::Gate).after(AskSystems),
            )
            .add_systems(AgentPass, answer.in_set(PrintSystems))
            .run_headless(Some(prompt.unwrap_or_default()));
        if let Some(mut run) = app.world_mut().get_resource_mut::<Run>() {
            run.interrupted = interrupted;
            run.unread = unread;
        }
    }

    /// Refuses unknown tools after every plugin finished in a run someone
    /// sits at, before its view starts, holding all model and tool calls,
    /// including restored work. The session keeps every agent's tool access
    /// unchanged. A headless run checks once its start is over
    /// (`check_start`).
    fn cleanup(&self, app: &mut App) {
        let world = app.world_mut();
        if world.contains_resource::<Run>() {
            return;
        }
        let mut tools = world.query::<&ToolDef>();
        let names: Vec<String> = tools
            .iter(world)
            .map(|tool| tool.0.name.to_string())
            .collect();
        let refused = world
            .get_resource::<Tools>()
            .and_then(|tools| tools.unknown(names.iter().map(String::as_str)));
        let Some(why) = refused else {
            return;
        };
        world.remove_resource::<Tools>();
        world.insert_resource(HoldCalls);
        app.refuse_run(why);
    }
}

/// Once no [`Starting`] is left, checks the names `--tools` gives against
/// the tools registered then: the agent loop runs from the frame they all
/// are. A run they are not [`drive`] ends, keeping every agent's tool
/// access as the session has it.
fn check_start(
    starting: Query<(), With<Starting>>,
    tools: Option<Res<Tools>>,
    defs: Query<&ToolDef>,
    (mut run, exit): (ResMut<Run>, Option<Res<ExitReason>>),
    mut commands: Commands,
) {
    // A run asked to exit never starts, though its last start ended in the
    // same frame. A `/command` waits too: it may start a turn, as a skill's
    // does.
    let waiting = !starting.is_empty() || exit.is_some();
    if run.checked || run.step != Step::Start || run.refused.is_some() || waiting {
        return;
    }
    let names = defs.iter().map(|tool| tool.0.name.as_str());
    match tools.and_then(|tools| tools.unknown(names)) {
        Some(why) => {
            run.refused = Some(why);
            commands.remove_resource::<Tools>();
        }
        None => run.checked = true,
    }
}

/// While it exists, model and tool calls are held: for a run whose tool
/// list was refused, or that was interrupted before its first frame.
#[derive(Resource)]
struct HoldCalls;

/// Holds model and tool calls while [`HoldCalls`] exists.
fn hold_calls(
    calls: Query<Entity, (ReadyToStart, Or<(ModelCalls, ToolCalls)>)>,
    mut commands: Commands,
) {
    for call in &calls {
        commands.spawn(HoldOf(call));
    }
}

/// Exits a run asked to exit before its start was checked, such as on a
/// signal or a refused `--tools`, as its exit effect would, without the
/// agent loop: restored work stays as the session has it, never answered
/// for tools still coming.
fn exit_unstarted(
    run: Res<Run>,
    reason: Option<Res<ExitReason>>,
    mut exits: MessageWriter<AppExit>,
    mut exited: Local<bool>,
) {
    if let Some(reason) = reason.filter(|_| !run.checked && !*exited) {
        *exited = true;
        exits.write(reason.exit());
    }
}

/// On a signal before the start was checked, lets each restored turn
/// carry on ([`CarryOn`]): no plugin had its say about it yet, so the exit
/// leaves it as the session has it, for the next run to decide. A run that
/// ends itself first, as on a refused `--tools`, stops it as any exit does.
fn keep_restored(
    run: Res<Run>,
    reason: Option<Res<ExitReason>>,
    turns: Query<Entity, (With<Restored>, With<TurnOf>, Without<CarryOn>)>,
    mut commands: Commands,
) {
    if run.checked || !matches!(reason.as_deref(), Some(ExitReason::Signal)) {
        return;
    }
    for turn in &turns {
        commands.entity(turn).insert(CarryOn);
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

impl Tools {
    /// Why some of the names no tool of `registered` has, if any does:
    /// each unknown name, then the names they likely mean, or else every
    /// tool there is.
    fn unknown<'a>(&self, registered: impl IntoIterator<Item = &'a str>) -> Option<String> {
        let mut registered: Vec<&str> = registered.into_iter().collect();
        registered.sort_unstable();
        let unknown: Vec<&str> = self
            .0
            .iter()
            .map(String::as_str)
            .filter(|name| !registered.contains(name))
            .collect();
        if unknown.is_empty() {
            return None;
        }
        let names = unknown.iter().map(|name| format!("`{name}`"));
        let names = names.collect::<Vec<_>>().join(", ");
        let tools = if unknown.len() == 1 { "tool" } else { "tools" };
        let meant: Option<Vec<String>> = unknown
            .iter()
            .map(|name| closest(name, &registered).map(|close| format!("`{close}`")))
            .collect();
        let hint = match (meant, registered.is_empty()) {
            (Some(meant), _) => format!("did you mean {}?", meant.join(", ")),
            (None, true) => "this build has no tools".to_owned(),
            (None, false) => format!("the tools are {}", registered.join(", ")),
        };
        Some(format!("--tools: unknown {tools} {names}; {hint}"))
    }
}

/// The name of `names` closest to `name`, within a quarter of its length
/// (at least one edit), so a typo finds its tool and another word none, as
/// the model catalog suggests.
fn closest<'a>(name: &str, names: &[&'a str]) -> Option<&'a str> {
    let limit = (name.chars().count() / 4).max(1);
    let distances = names
        .iter()
        .map(|close| (edit_distance(name, close), *close));
    let close = distances.filter(|(distance, _)| *distance <= limit).min();
    close.map(|(_, close)| close)
}

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

/// What a one-shot run reads after its prompt: stdin unless the app
/// inserted another.
#[derive(Resource)]
struct PipedIn(Box<dyn Read + Send + Sync>);

/// The prompt with what was piped in after it, on stdin or `piped`;
/// `None` once `interrupted` says the run is.
fn full_prompt(
    prompt: &str,
    piped: Option<PipedIn>,
    interrupted: impl Fn() -> bool,
) -> Option<String> {
    match piped {
        Some(PipedIn(piped)) => with_piped(prompt, piped, PIPED_WAIT, interrupted),
        None if std::io::stdin().is_terminal() => Some(prompt.to_owned()),
        None => with_piped(prompt, std::io::stdin(), PIPED_WAIT, interrupted),
    }
}

/// How often a wait for what is piped in looks whether the run was
/// interrupted.
const INTERRUPT_CHECK: Duration = Duration::from_millis(50);

/// `prompt` with what `piped` holds after it. Without a prompt, `piped` is
/// the prompt, read to its end; with one, it is read only when something
/// comes within `wait`. `None` once `interrupted` says the run is, however
/// long the wait was.
fn with_piped(
    prompt: &str,
    mut piped: impl Read + Send + 'static,
    wait: Duration,
    interrupted: impl Fn() -> bool,
) -> Option<String> {
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
    let deadline = (!prompt.trim().is_empty()).then(|| Instant::now() + wait);
    let mut came = false;
    let piped = loop {
        if interrupted() {
            return None;
        }
        match read.recv_timeout(INTERRUPT_CHECK) {
            Ok(Some(piped)) => break piped,
            Ok(None) => came = true,
            Err(RecvTimeoutError::Disconnected) => break String::new(),
            Err(RecvTimeoutError::Timeout) => {
                if !came && deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                    // A closed stderr leaves nothing to tell it on.
                    let _ = writeln!(
                        std::io::stderr(),
                        "rig: nothing came on stdin within {}s, so the prompt goes alone; give \
                         -p stdin from /dev/null, or pipe in what goes with it",
                        wait.as_secs_f32()
                    );
                    return Some(prompt.to_owned());
                }
            }
        }
    };
    Some(match piped {
        piped if piped.trim().is_empty() => prompt.to_owned(),
        piped if prompt.trim().is_empty() => piped,
        piped => format!("{prompt}\n\n{piped}"),
    })
}

/// Starts a run nobody sits at on an [`App`].
pub trait AppHeadlessExt {
    /// Sends `prompt` to the main agent once it has a model, and exits when
    /// no agent works any more and no [`KeepRunning`] exists. Without a
    /// prompt, the run finishes the turns a plugin carries on from its
    /// restored session (`CarryOn`), and waits for the work a `KeepRunning`
    /// plugin sends.
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
            ..Run::default()
        })
        .add_systems(
            Update,
            (
                check_start.after(PollTasks).before(AgentLoop),
                drive.after(AgentLoop),
            ),
        )
        // Nothing runs before the start: no restored turn, request or
        // prompt sees the tools still coming, or a refused `--tools`.
        .configure_sets(Update, AgentLoop.run_if(|run: Res<Run>| run.checked))
        .add_systems(Update, exit_unstarted.after(drive))
        .add_systems(
            Last,
            keep_restored
                .in_set(OnAppExitSystems)
                .before(StopTurnsOnExit)
                .run_if(on_message::<AppExit>),
        )
        .add_observer(keep_answer)
        .add_observer(|_: On<Add<KeepRunning>>, mut run: ResMut<Run>| run.kept = true)
        // The frame after the last start ended looks again.
        .add_observer(|_: On<Remove<Starting>>, wake: Res<Wake>| wake.wake())
    }
}

/// Where the run is.
#[derive(Resource, Default)]
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
    answer: Option<String>,
    /// Whether a turn of the agent ended since the work last settled.
    turned: bool,
    /// The answer of the last [`HeadlessTurnEnded`].
    last: Option<String>,
    /// The last error notice about the agent since the work last settled,
    /// such as a refused model: why the run failed when nothing else says.
    notice: Option<String>,
    /// The worst failure so far.
    failed: Option<Failure>,
    /// Whether a line already told why the run that ended failed.
    told: bool,
    /// Why `--tools` is refused, which ends the run at once.
    refused: Option<String>,
    /// Whether the start is over and `--tools` names only registered
    /// tools: the agent loop runs from then on.
    checked: bool,
    /// Whether the agent's last turn ended with no answer text.
    unanswered: bool,
    /// Whether the run was interrupted before its first frame, which then
    /// exits: nothing is sent.
    interrupted: bool,
    /// What says that nobody reads the run's output any more, which cuts
    /// it off ([`ExitWhenUnread`]).
    unread: Option<fn() -> bool>,
}

/// Why a run fails: its exit code, why, and whether a line already said so.
#[derive(Clone, Debug, PartialEq, Eq)]
struct Failure {
    code: u8,
    why: String,
    told: bool,
}

impl Run {
    /// Notes `code` for `why`, `told` already, if it is worse than any so
    /// far.
    fn fail(&mut self, code: u8, why: String, told: bool) {
        if self.failed.as_ref().is_none_or(|worst| code > worst.code) {
            self.failed = Some(Failure { code, why, told });
        }
    }

    /// Notes a turn of the agent that ended: a failed one fails the run,
    /// told already by the kernel's error notice, and takes back an earlier
    /// answer, as one with no answer text does; a stopped one changes
    /// neither.
    fn turn_ended(&mut self, outcome: &TurnOutcome) {
        self.turned = true;
        self.unanswered = false;
        match TurnEnd::of(outcome) {
            TurnEnd::Answered(answer) => self.answer = Some(answer),
            TurnEnd::Unanswered => {
                self.unanswered = true;
                self.answer = None;
            }
            TurnEnd::Failed(why) => {
                self.fail(1, why, true);
                self.answer = None;
            }
            TurnEnd::Stopped(_) => {}
        }
    }

    /// Takes what came since the work last settled: the answer, and the
    /// exit code an error notice, or a last turn with no answer text, and
    /// no answer earns. A turn with no text that a later one answers after,
    /// such as once a subagent reported, fails nothing.
    fn settled(&mut self) -> Option<String> {
        let answer = self.answer.take();
        if std::mem::take(&mut self.unanswered) && answer.is_none() {
            self.fail(1, NO_ANSWER.to_owned(), false);
        }
        self.notice_fails(answer.is_some());
        answer
    }

    /// Takes the latest error notice, which its own line told, and fails
    /// the run with code 1 for it unless the work `answered` since: what a
    /// run earns for one with no answer.
    fn notice_fails(&mut self, answered: bool) {
        if let Some(why) = self.notice.take().filter(|_| !answered) {
            self.fail(1, why, true);
        }
    }

    /// The agent the run follows: the main agent before the run first
    /// looked, whose turn may end or model be refused within the first
    /// frame; while connecting too, as a turn a plugin started may end then.
    fn agent(&self, primary: &MainAgentQuery) -> Option<Entity> {
        match self.step {
            Step::Start => main_agent(primary),
            Step::Connecting { agent } | Step::Sent { agent } | Step::Waiting { agent } => {
                Some(agent)
            }
            Step::Done => None,
        }
    }
}

/// Why a turn that ended with no final text fails a headless run.
const NO_ANSWER: &str = "the model ended the turn without an answer";

/// How a turn ended, as a headless run reports it, both in its exit code
/// and in its `turn_end` line.
enum TurnEnd {
    /// With this final text.
    Answered(String),
    /// Answered with no final text, such as a reply of reasoning alone: a
    /// failed turn, [`NO_ANSWER`].
    Unanswered,
    /// For this reason.
    Failed(String),
    /// Stopped, and why when something said.
    Stopped(Option<String>),
}

impl TurnEnd {
    fn of(outcome: &TurnOutcome) -> Self {
        match outcome {
            TurnOutcome::Answered(message) => message
                .final_answer()
                .map_or(Self::Unanswered, Self::Answered),
            TurnOutcome::Failed(why) => Self::Failed(why.clone()),
            TurnOutcome::Stopped(why) => Self::Stopped(why.clone()),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Default)]
enum Step {
    #[default]
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
    /// Why an agent's model was refused, as its notice said.
    refused: Query<'w, 's, &'static ModelRefused>,
    /// Running turns; a restored one the kernel stops is none (one that goes
    /// on is no longer `Restored`).
    turns: Query<'w, 's, (), (With<TurnOf>, Without<Restored>)>,
    /// Work in progress, such as a sign-in; not a long-lived stream, such
    /// as a file watch.
    running: Query<'w, 's, &'static Running>,
    /// Messages on their way to an agent, such as the prompt itself.
    messages: Query<'w, 's, (), (With<Incoming>, Without<Settled>)>,
    /// Effects not started yet, such as one held at Gate, or answered and
    /// not settled, such as one held for review; not an exit, nor a
    /// restored call, which is its turn's.
    held: Query<
        'w,
        's,
        (),
        (
            With<Effect>,
            Without<Settled>,
            Without<ExitRequest>,
            Without<Restored>,
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

/// Once [`check_start`] passed, ends a run whose `--tools` was refused,
/// chooses the model, sends the prompt, and exits after the turn.
fn drive(
    mut run: ResMut<Run>,
    agents: Agents,
    models: Res<Models>,
    (start, wake): (Option<Res<AppStart>>, Res<Wake>),
    (mut notices, mut recalled): (MessageReader<Notice>, MessageReader<Recalled>),
    mut commands: Commands,
) {
    if run.interrupted {
        return;
    }
    for why in recalled.read().filter_map(|recalled| recalled.why.clone()) {
        run.fail(2, why, true);
    }
    // A command such as `/login` runs without a model.
    let command = run
        .prompt
        .as_ref()
        .is_some_and(|prompt| prompt.trim_start().starts_with('/'));
    let to = run.agent(&agents.primary);
    for notice in notices.read() {
        if let Some(agent) = to
            && notice.level == NoticeLevel::Error
            && notice.agent.is_none_or(|about| about == agent)
        {
            run.notice = Some(notice.text.clone());
        }
    }
    // The launcher restarts a run whose agent reloaded without its prompt.
    let restarted = start
        .is_some_and(|start| matches!(start.0, StartReason::Reloaded | StartReason::Restarted));
    let ended = match run.step {
        Step::Start => {
            if !run.checked && run.refused.is_none() {
                return;
            }
            let Some(agent) = main_agent(&agents.primary) else {
                return;
            };
            if let Some(why) = run.refused.take() {
                let ended = ended(Some(agent), 2);
                return end(&mut run, (ended, Reason::Said(&why)), commands);
            }
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
                        let ended = ended(Some(agent), 1);
                        return end(&mut run, (ended, Reason::Said(why)), commands);
                    }
                },
            };
            // Connected already, as `--model` or the restored session
            // leaves it: the prompt goes out in the first frame after the
            // start, so a kill soon after leaves it in the session to resume.
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
            let code = run.failed.as_ref().map_or(0, |failed| failed.code);
            let ended = HeadlessEnded {
                answer: run.last.clone(),
                ..ended(Some(agent), code)
            };
            Some((ended, Reason::Worst))
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
) -> Option<(HeadlessEnded, Reason<'static>)> {
    let connected = command || agents.choices.get(agent).is_ok_and(|(_, on)| on);
    if !connected {
        // The model's refusal, which its notice told, is why, over any
        // failure before it.
        let refused = agents.refused.get(agent).ok();
        let reason = refused.map_or(Reason::Worst, |ModelRefused(why)| Reason::Told(why.clone()));
        // `--model` named a model the catalog lacks, which leaves the agent
        // with none: a command line refused, not a failed run.
        let unknown = run.model
            && agents
                .choices
                .get(agent)
                .is_ok_and(|(chosen, _)| chosen.is_none());
        return Some((ended(Some(agent), if unknown { 2 } else { 1 }), reason));
    }
    match run.prompt.clone() {
        Some(text) => {
            // Nor does a restored turn that failed before it: this run's
            // own prompt has its own end. Without one, that turn's is the
            // run's.
            run.notice = None;
            run.failed = None;
            commands.trigger(Deliver::new(agent, text, DeliveryMode::Steer));
            run.step = Step::Sent { agent };
        }
        // Something keeps the run going, or already did: a turn it started
        // may have ended while the model was connecting.
        None if !agents.keep.is_empty() || run.turned || run.kept => {
            run.step = Step::Waiting { agent }
        }
        // Work the restored session left runs, such as a turn a plugin
        // carries on after a crash: it finishes, and its answer is the
        // run's.
        None if !agents.idle() => run.step = Step::Sent { agent },
        // The run restarted on a reload with nothing left to do.
        None if restarted => return Some((ended(Some(agent), 0), Reason::Worst)),
        None => {
            let why = "no prompt: give one after --print, or pipe it in";
            return Some((ended(Some(agent), 2), Reason::Said(why)));
        }
    }
    None
}

/// A run that ended with `code`, its reason named by [`end`].
fn ended(agent: Option<Entity>, code: u8) -> HeadlessEnded {
    HeadlessEnded {
        agent,
        answer: None,
        why: None,
        code,
    }
}

/// Why a run that failed did, as [`end`] names it.
enum Reason<'a> {
    /// This, which no line told yet.
    Said(&'a str),
    /// This, which a notice told already, such as the agent's model's
    /// refusal.
    Told(String),
    /// The worst failure so far, else the latest error notice.
    Worst,
}

/// Notes each turn of the agent the prompts go to that ended, with its
/// answer when it answered, whatever its conversation looks like now, such
/// as after a compaction; not a restored turn the kernel stopped, which was
/// no work of this run.
fn keep_answer(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent), Without<Restored>>,
    primary: MainAgentQuery,
    wake: Res<Wake>,
    mut run: ResMut<Run>,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    if run.agent(&primary) != Some(agent) {
        return;
    }
    // A turn the restored session left may end within the first frame,
    // which then asks for the frame that settles it.
    if run.step == Step::Start {
        wake.wake();
    }
    run.turn_ended(outcome);
}

/// Takes what came since the work last settled: the exit code it earns,
/// and a [`HeadlessTurnEnded`] when a turn of `agent` ended.
fn settle(run: &mut Run, agent: Entity, commands: &mut Commands) {
    let answer = run.settled();
    if std::mem::take(&mut run.turned) {
        let ended = HeadlessTurnEnded { agent, answer };
        run.last.clone_from(&ended.answer);
        commands.trigger(ended);
    }
}

/// Ends the run as `ended` says, naming why, by `reason`, when it failed.
fn end(run: &mut Run, (mut ended, reason): (HeadlessEnded, Reason), mut commands: Commands) {
    run.step = Step::Done;
    if ended.code != 0 {
        let picked = match reason {
            Reason::Said(why) => Some((why.to_owned(), false)),
            Reason::Told(why) => Some((why, true)),
            Reason::Worst => {
                run.notice_fails(false);
                let failed = run.failed.take();
                failed.map(|failed| (failed.why, failed.told))
            }
        };
        let (why, told) =
            picked.unwrap_or_else(|| ("the run failed without saying why".to_owned(), false));
        ended.why = Some(why);
        run.told = told;
    }
    let exit = match ended.code {
        // Nobody reads the answer: the run was cut off, as by SIGPIPE.
        _ if run.unread.is_some_and(|unread| unread()) => ExitReason::Signal,
        0 => ExitReason::Quit,
        code => ExitReason::Failed(code),
    };
    commands.trigger(Exit(exit));
    commands.trigger(ended);
}

/// Whether the run wrote anything to stdout yet, and where its notices
/// go: stderr unless the app inserted another.
#[derive(Resource)]
struct Printed {
    out: bool,
    err: Box<dyn Write + Send + Sync>,
}

impl Default for Printed {
    fn default() -> Self {
        Self::to(std::io::stderr())
    }
}

impl Printed {
    /// Notices written to `err`.
    fn to(err: impl Write + Send + Sync + 'static) -> Self {
        Self {
            out: false,
            err: Box::new(err),
        }
    }

    /// Writes `block` to stdout, after an empty line when something came
    /// before it.
    fn print(&mut self, block: Option<String>) {
        if let Some(block) = block {
            let gap = if self.out { "\n" } else { "" };
            // A reader gone ends the run on its own; nobody is left to tell.
            let _ = writeln!(std::io::stdout(), "{gap}{block}");
            self.out = true;
        }
    }

    /// Writes the notice `text` to stderr.
    fn note(&mut self, text: &str) {
        // A closed stderr leaves nothing to tell it on.
        let _ = writeln!(self.err, "{text}");
    }

    /// Writes the error or warning `text` to stderr.
    fn error(&mut self, text: &str) {
        self.note(&format!("rig: {text}"));
    }
}

/// Writes each answer to stdout as it comes.
fn print_answer(ended: On<HeadlessTurnEnded>, mut printed: ResMut<Printed>) {
    printed.print(output(ended.answer.as_deref(), []));
}

/// Once the run ends, writes the [`PrintSection`]s to stdout, and why it
/// failed to stderr unless a line already told it.
fn print_ended(
    ended: On<HeadlessEnded>,
    sections: Query<&PrintSection>,
    run: Res<Run>,
    mut printed: ResMut<Printed>,
) {
    if let Some(why) = ended.why.as_ref().filter(|_| !run.told) {
        printed.error(why);
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
fn print_progress(
    progress: Query<(&ToolProgress, Option<&ToolRequest>), Changed<ToolProgress>>,
    mut printed: ResMut<Printed>,
) {
    for (ToolProgress(line), tool) in &progress {
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        match tool {
            Some(tool) => printed.error(&format!("{}: {line}", tool.call.function.name)),
            None => printed.error(line),
        }
    }
}

/// Writes warnings, errors, why a command was refused, and the task a
/// failed turn handed back to stderr, with every notice of a `/command`
/// prompt.
fn print_notices(
    command: bool,
) -> impl FnMut(MessageReader<Notice>, MessageReader<Recalled>, ResMut<Printed>) + Send + Sync + 'static
{
    move |mut notices, mut recalled, mut printed| {
        for recalled in recalled.read() {
            match &recalled.why {
                Some(why) => printed.error(why),
                None => {
                    let text = rig_harness::rig_core::redact::text(&recalled.text);
                    printed.error(&format!("not answered: {}", excerpt(&text)));
                }
            }
        }
        for notice in notices.read() {
            match notice.level {
                NoticeLevel::Error | NoticeLevel::Warn => printed.error(&notice.text),
                _ if command => printed.note(&notice.text),
                _ => {}
            }
        }
    }
}

/// The most lines of a recalled text stderr shows.
const EXCERPT_LINES: usize = 10;

/// The most bytes of a recalled text stderr shows.
const EXCERPT_BYTES: usize = 1000;

/// `text` as stderr shows it: its first [`EXCERPT_LINES`] lines, at most
/// [`EXCERPT_BYTES`] bytes, never cut inside a character, then how much was
/// left out, so a long input piped in after the prompt does not flood the
/// terminal.
fn excerpt(text: &str) -> String {
    let lines_end = text
        .match_indices('\n')
        .nth(EXCERPT_LINES - 1)
        .map_or(text.len(), |(at, _)| at);
    let mut end = lines_end.min(EXCERPT_BYTES);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    let (shown, left) = text.split_at(end);
    match left.trim_end().len() {
        0 => text.trim_end().to_owned(),
        _ => format!("{}\n… ({} more bytes)", shown.trim_end(), left.len()),
    }
}

/// Plain `-p`: says on stderr that a signal, the launcher's death or a
/// reader gone cut the run off, which no line said otherwise.
fn print_interrupted(reason: Option<Res<ExitReason>>, mut printed: ResMut<Printed>) {
    if reason.as_deref() == Some(&ExitReason::Signal) {
        printed.error("the run was interrupted");
    }
}

/// Where `--json` lines go: stdout unless the app inserted another.
#[derive(Resource)]
struct JsonOut {
    out: Box<dyn Write + Send + Sync>,
    /// The time of the last line, so `ms` never goes back.
    ms: u64,
    /// Whether the `exit` line was written, after which nothing is.
    closed: bool,
    /// Whether the run answers one prompt, and so writes `exit` however it
    /// exits.
    one_shot: bool,
    /// How the run ended, for its `exit` line.
    ended: Option<HeadlessEnded>,
    /// The running calls the exit cut off before they answered: their
    /// `tool_result` says `cancelled`.
    cancelled: HashSet<Entity>,
    /// The lines that close the turns the exit leaves for the restart, each
    /// with its agent.
    closing: Vec<(Entity, &'static str, Value)>,
}

impl JsonOut {
    fn new(out: impl Write + Send + Sync + 'static) -> Self {
        Self {
            out: Box::new(out),
            ms: 0,
            closed: false,
            one_shot: false,
            ended: None,
            cancelled: HashSet::new(),
            closing: Vec::new(),
        }
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

/// How far the lines have read the model calls' [`StreamDelta`]s, the
/// [`Notice`]s and the [`Recalled`] texts.
#[derive(Resource, Default)]
struct LinesRead {
    text: MessageCursor<StreamDelta>,
    notices: MessageCursor<Notice>,
    recalled: MessageCursor<Recalled>,
}

/// Writes events, each naming its agent, after the text streamed and the
/// notices written before them.
#[derive(SystemParam)]
struct Out<'w, 's> {
    out: ResMut<'w, JsonOut>,
    ids: Query<'w, 's, &'static AgentId>,
    deltas: Res<'w, Messages<StreamDelta>>,
    notices: Res<'w, Messages<Notice>>,
    recalled: Res<'w, Messages<Recalled>>,
    read: ResMut<'w, LinesRead>,
    turn_calls: Query<'w, 's, (), With<TurnCall>>,
}

impl Out<'_, '_> {
    /// Writes the event `kind` with `fields` about `agent`.
    fn emit(&mut self, agent: Option<Entity>, kind: &str, fields: Value) {
        self.flush();
        self.write(agent, kind, fields);
    }

    /// Writes the notices written, then the text streamed, since the last
    /// line, so every line comes after the notices pending before it.
    fn flush(&mut self) {
        let notices: Vec<Notice> = self.read.notices.read(&self.notices).cloned().collect();
        for notice in notices {
            let level = match notice.level {
                NoticeLevel::Info => "info",
                NoticeLevel::Warn => "warn",
                NoticeLevel::Error => "error",
            };
            let fields = json!({ "level": level, "text": notice.text });
            self.write(notice.agent, "notice", fields);
        }
        let recalled: Vec<Recalled> = self.read.recalled.read(&self.recalled).cloned().collect();
        for recalled in recalled {
            let text = rig_harness::rig_core::redact::text(&recalled.text);
            let fields = json!({ "text": text, "why": recalled.why });
            self.write(Some(recalled.agent), "recalled", fields);
        }
        self.text();
    }

    /// Writes the text each turn's model call streamed since the last
    /// event, one line per agent's run of it.
    fn text(&mut self) {
        let mut runs: Vec<(Option<Entity>, String)> = Vec::new();
        let deltas = self.read.text.read(&self.deltas);
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
        if self.out.closed {
            return;
        }
        self.out.ms = self.out.ms.max(now_ms());
        let mut line = Map::from_iter([
            ("v".to_owned(), VERSION.into()),
            ("type".to_owned(), kind.into()),
            ("ms".to_owned(), self.out.ms.into()),
        ]);
        if let Value::Object(fields) = fields {
            line.extend(fields);
        }
        if let Some(AgentId(id)) = agent.and_then(|agent| self.ids.get(agent).ok()) {
            line.insert("agent".to_owned(), id.clone().into());
        }
        writeln!(self.out.out, "{}", Value::Object(line)).ok();
        self.out.out.flush().ok();
    }
}

/// The session's id, once it has one.
fn session_id(paths: Option<Res<SessionPaths>>) -> Option<String> {
    Some(paths?.0.id()?.as_str().to_owned())
}

/// The first line, once the session is restored, before the agent loop
/// starts or resumes a turn.
fn ready(primary: MainAgentQuery, paths: Option<Res<SessionPaths>>, mut out: Out) {
    let session = session_id(paths);
    // First: what came before it follows.
    out.write(main_agent(&primary), "ready", json!({ "session": session }));
}

/// Keeps how the run ended for its `exit` line, which the real exit
/// writes.
fn json_ended(ended: On<HeadlessEnded>, mut out: ResMut<JsonOut>) {
    out.ended = Some(ended.event().clone());
}

/// Why an exit cuts off the work the run started, as the lines that close
/// it say; `None` for a reload, after which the launcher starts the agent
/// again and the run goes on.
fn cut_off(reason: Option<&ExitReason>) -> Option<&'static str> {
    match reason {
        Some(ExitReason::Reload) => None,
        reason => Some(reason.map_or(ExitReason::Quit.stops_because(), ExitReason::stops_because)),
    }
}

/// On the exit frame, before the exit stops the running turns as a stop
/// does: writes what streamed, and notes each tool call that started and
/// has no answer, whose `tool_result` says `cancelled`. The stop settles
/// every call and turn it cuts off, which writes their lines with the
/// results the session keeps. A run that ended by itself, such as on a
/// refused `--tools`, gives its own reason as the stop's ([`Exiting`]),
/// unless something else, such as a signal, ended it first. A turn a
/// plugin carries on across the exit (`CarryOn`), and a call of no turn,
/// are left unended: this keeps their closing lines, a started call's with
/// what a stop says of it, which [`json_exit`] writes.
fn note_cut_off(
    reason: Option<Res<ExitReason>>,
    exiting: Option<ResMut<Exiting>>,
    calls: Query<
        (
            Entity,
            (&ToolRequest, &ForAgent, Option<&CallOf>),
            Option<&Outcome>,
            Option<&IfStopped>,
            Option<&Footprint>,
        ),
        (With<Started>, Without<Settled>),
    >,
    left: Query<(&TurnOf, Option<&TurnUsage>, Option<&FirstStreamed>), With<CarryOn>>,
    mut out: Out,
) {
    let Some(why) = cut_off(reason.as_deref()) else {
        return;
    };
    let code = match reason.as_deref().map(ExitReason::exit) {
        Some(AppExit::Error(code)) => code.get(),
        _ => 0,
    };
    let ended = out.out.ended.as_ref().filter(|ended| ended.code == code);
    let why = ended
        .and_then(|ended| ended.why.clone())
        .unwrap_or_else(|| why.to_owned());
    if let Some(mut exiting) = exiting {
        exiting.0.clone_from(&why);
    }
    out.flush();
    let mut closing = Vec::new();
    for (call, (request, &ForAgent(agent), turn), outcome, if_stopped, footprint) in &calls {
        if outcome.is_none() {
            out.out.cancelled.insert(call);
        }
        if turn.is_some_and(|&CallOf(turn)| !left.contains(turn)) {
            continue;
        }
        let kept = outcome.or(if_stopped.map(|stopped| &stopped.0));
        let kept = kept.map(Outcome::tool_result);
        let kept = kept.map(|result| redact::tool_result(&result).unwrap_or(result));
        let (ok, output) = match kept {
            Some(result) => (result.is_success(), result.output().render()),
            None => {
                let footprint = footprint.cloned().unwrap_or_default();
                (false, not_finished(&why, true, &footprint.0))
            }
        };
        let fields = json!({
            "id": request.call.id.wire(),
            "name": request.call.function.name.as_str(),
            "ok": ok,
            "refused": false,
            "cancelled": outcome.is_none(),
            "output": output,
        });
        closing.push((agent, "tool_result", fields));
    }
    for (&TurnOf(agent), used, first) in &left {
        let ended = TurnEnd::Stopped(Some(why.clone()));
        closing.push((agent, "turn_end", turn_end_fields(ended, used, first)));
    }
    out.out.closing = closing;
}

/// On the exit frame, once the turns are stopped and the session log is
/// written: the lines [`note_cut_off`] kept, then the `exit` line of a run
/// that ended ([`HeadlessEnded`]) or of a one-shot run cut off, with the
/// agent's exit code.
fn json_exit(
    reason: Option<Res<ExitReason>>,
    primary: MainAgentQuery,
    (paths, usage): (Option<Res<SessionPaths>>, Option<Res<SessionUsage>>),
    mut out: Out,
) {
    let reason = reason.as_deref();
    let Some(why) = cut_off(reason) else {
        return;
    };
    for (agent, kind, fields) in std::mem::take(&mut out.out.closing) {
        out.emit(Some(agent), kind, fields);
    }
    let ended = out.out.ended.take();
    if ended.is_none() && !out.out.one_shot {
        return;
    }
    let code = match reason.map(ExitReason::exit) {
        Some(AppExit::Error(code)) => code.get(),
        _ => 0,
    };
    let (agent, answer, error) = match ended {
        Some(ended) if ended.code == code => (ended.agent, ended.answer, ended.why),
        // Something else ended it first, such as a signal.
        ended => (
            ended
                .as_ref()
                .and_then(|ended| ended.agent)
                .or(main_agent(&primary)),
            ended.and_then(|ended| ended.answer),
            (code != 0).then(|| why.to_owned()),
        ),
    };
    let fields = json!({
        "code": code,
        "answer": answer,
        "error": error,
        "session": session_id(paths),
        "usage": usage.map(|usage| usage.0.total),
    });
    out.emit(agent, "exit", fields);
    // Nothing after it.
    out.out.closed = true;
}

fn turn_started(turn: On<Add<TurnOf>>, turns: Query<&TurnOf>, mut out: Out) {
    if let Ok(&TurnOf(agent)) = turns.get(turn.entity) {
        out.emit(Some(agent), "turn_start", json!({}));
    }
}

/// The fields of a `tool_call` line.
fn call_fields(call: &ToolCall) -> Value {
    json!({
        "id": call.id.wire(),
        "name": call.function.name.as_str(),
        "args": call.function.arguments_value(),
    })
}

/// A tool call starts, or a question waits for its answer.
fn started(
    start: On<Add<Started>>,
    calls: Query<(&ToolRequest, &ForAgent)>,
    asks: Query<(&Ask, Option<&ForAgent>)>,
    mut out: Out,
) {
    if let Ok((request, &ForAgent(agent))) = calls.get(start.entity) {
        out.emit(Some(agent), "tool_call", call_fields(&request.call));
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

/// A tool call or a turn ended. A call refused before it started, such as
/// an unknown or denied tool, has its `tool_call` line first.
fn settled(
    settled: On<Add<Settled>>,
    effects: Query<(
        &ForAgent,
        Option<&Outcome>,
        Option<&TurnOutcome>,
        Option<&TurnUsage>,
        Option<&FirstStreamed>,
    )>,
    calls: Query<(&ToolRequest, Has<Started>)>,
    mut out: Out,
) {
    let Ok((&ForAgent(agent), outcome, ended, used, first)) = effects.get(settled.entity) else {
        return;
    };
    if let (Ok((request, started)), Some(outcome)) = (calls.get(settled.entity), outcome) {
        if !started {
            out.emit(Some(agent), "tool_call", call_fields(&request.call));
        }
        let result = outcome.tool_result();
        let cancelled = out.out.cancelled.contains(&settled.entity);
        let fields = json!({
            "id": request.call.id.wire(),
            "name": request.call.function.name.as_str(),
            "ok": result.is_success(),
            // Refused before it ran: the agent may not use it, or a gate
            // refused it, as its report says.
            "refused": !started && outcome.0.as_ref().err().is_some_and(|report| report.refusal),
            "cancelled": cancelled,
            "output": result.output().render(),
        });
        out.emit(Some(agent), "tool_result", fields);
    } else if let Some(ended) = ended {
        let fields = turn_end_fields(TurnEnd::of(ended), used, first);
        out.emit(Some(agent), "turn_end", fields);
    }
}

/// The fields of a `turn_end` line.
fn turn_end_fields(
    ended: TurnEnd,
    used: Option<&TurnUsage>,
    first: Option<&FirstStreamed>,
) -> Value {
    let (outcome, answer, error) = match ended {
        TurnEnd::Answered(answer) => ("answered", Some(answer), None),
        TurnEnd::Unanswered => ("failed", None, Some(NO_ANSWER.to_owned())),
        TurnEnd::Failed(why) => ("failed", None, Some(why)),
        TurnEnd::Stopped(why) => ("stopped", None, why),
    };
    json!({
        "outcome": outcome,
        "answer": answer,
        "error": error,
        "usage": used.map(|used| used.0),
        "first_token_ms": first.map(|first| rig_harness::rig_core::millis(first.0)),
    })
}

/// Writes the text that streamed and the notices written this frame.
fn flush(mut out: Out) {
    out.flush();
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
    let version = remove_preserving_order(fields, "v");
    remove_preserving_order(fields, "id");
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
