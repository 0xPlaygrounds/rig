//! The run contract every front keeps: when a run's start is over, which
//! tools `--tools` ([`TOOLS`]) leaves its agents, and when a run nobody
//! sits at has work left. [`RunPlugin`] owns it.
//!
//! A plugin whose tools come in the background, such as an MCP client,
//! holds the start with a [`Starting`] entity for each source. It removes
//! it once their tools are in or it gave up, and bounds that wait itself.
//! Once none is left, [`RunStart`] checks the names `--tools` gives
//! against the tools registered then, once, in every front. While
//! `--tools` is given, the agent loop waits for that check, so no turn,
//! request or tool call sees the tools still coming; a front nobody sits
//! at makes it wait in any case ([`AppRunExt::run_after_start`]). A run
//! asked to exit meanwhile exits without the agent loop, and a signal then
//! leaves each restored turn to carry on in the next run.
//!
//! `--tools` limits only its run: once the start passes, it is the run's
//! [`RunTools`], which the kernel keeps every agent within and the session
//! never saves, so a later run has every tool the session kept. A name no
//! tool has refuses the run with exit code 2. A front that says why a run
//! failed itself, such as `--json`'s `exit` line, takes the refusal
//! ([`RunStart::take_refusal`]) in `Update`; else the harness refuses the
//! run on stderr, and the launcher says it again once the screen is back.
//! A run that keeps no session runs no agent and checks nothing: `rig
//! acp`'s router hands `--tools` to its sessions.
//!
//! ```
//! use rig_harness::prelude::*;
//! use rig_harness::run::Starting;
//!
//! /// Holds the start until the tools of its server are in.
//! fn connect(mut commands: Commands) {
//!     commands.spawn((Name::new("my server"), Starting));
//! }
//! ```

use bevy_app::OnAppExitSystems;
use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use rig_core::catalog::edit_distance;
use rig_ecs::StopTurnsOnExit;
use rig_ecs::agent::{RunTools, TurnOf};
use rig_ecs::inbox::Incoming;
use rig_ecs::restore::{CarryOn, Restored};
use rig_ecs::stage::{AgentLoop, Effect, Outcome, Settled, Started};
use rig_ecs::tasks::{PollTasks, Running, Wake};
use rig_ecs::tools::ToolDef;

use crate::host::cli::{self, AppCliExt, Cli, Flag};
use crate::host::exit::{ExitReason, ExitRequest};
use crate::host::launcher::{self, Invoked};

/// The option that leaves the run only the tools named ([`RunTools`]):
/// every agent keeps the access the session saved, within these, for this
/// run only. Omission leaves every tool; an empty value leaves none. A run
/// refuses a name no registered tool has once its start is over
/// ([`RunStart`]), before any model or tool call, in every front.
pub const TOOLS: &str = "tools";

/// While any entity has it, the run's start is not over: [`RunStart`]
/// waits before it checks `--tools`, and a run nobody sits at before it
/// sends its prompt.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct Starting;

/// While any entity has it, a run nobody sits at does not end when its
/// agents are idle: a plugin that sends the agent work over time spawns
/// one, and despawns it to let the run end.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct KeepRunning;

/// Where the run's start is: over once no [`Starting`] is left and
/// `--tools` names only registered tools, or refused for a name that is
/// not one.
#[derive(Resource, Debug, Default)]
pub struct RunStart {
    over: bool,
    refused: Option<String>,
}

impl RunStart {
    /// Whether the start is over and `--tools` passed.
    pub fn is_over(&self) -> bool {
        self.over
    }

    /// Why `--tools` was refused, for a front that ends the run itself
    /// with it; the harness refuses a run whose refusal nobody took by
    /// `PostUpdate`.
    pub fn take_refusal(&mut self) -> Option<String> {
        self.refused.take()
    }
}

/// The `Update` system that ends the start ([`RunStart`]), after the tasks
/// polled and before the agent loop. A front that reads [`RunStart`] in
/// the frame it ends orders itself after it.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RunStartSystems;

/// The agents' work in progress, for a front nobody sits at that waits for
/// the work it sent to settle, as a headless run does.
#[derive(SystemParam)]
pub struct Work<'w, 's> {
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
}

impl Work<'_, '_> {
    /// Whether no agent works: no turn, task, held effect or message on its
    /// way. An open effect that started and waits on the world outside,
    /// such as a question nobody answers, is no work.
    pub fn idle(&self) -> bool {
        self.turns.is_empty()
            && self.messages.is_empty()
            && self.held.is_empty()
            && self.running.iter().all(Running::is_stream)
    }
}

/// Declares `--tools` and keeps the run contract: see [`run`](self). Part
/// of [`RigHarnessPlugins`](crate::RigHarnessPlugins); a plugin that reads
/// [`RunStart`] adds it when it is missing, as a test app may leave it out.
pub struct RunPlugin;

impl Plugin for RunPlugin {
    fn build(&self, app: &mut App) {
        app.add_flag(
            Flag::new(
                TOOLS,
                "Give this run only these tools; an empty list gives none",
            )
            .value("NAME[,NAME…]"),
        );
        start(app);
    }

    fn finish(&self, app: &mut App) {
        cli::choose(app.world_mut());
        if app
            .world()
            .get_resource::<Cli>()
            .is_some_and(Cli::sessionless)
        {
            return;
        }
        let Some(tools) = app.world().get_resource::<Invoked>().and_then(|invoked| {
            let list = invoked.flag(TOOLS).then(|| invoked.flag_value(TOOLS));
            list.map(|list| {
                let items = list.unwrap_or_default().split(',').map(str::trim);
                items
                    .filter(|item| !item.is_empty())
                    .map(str::to_owned)
                    .collect()
            })
        }) else {
            return;
        };
        app.insert_resource(Tools(tools)).run_after_start();
    }
}

/// Running the agent loop only once the run's start is over.
pub trait AppRunExt {
    /// The agent loop runs from the frame the start is over ([`RunStart`]),
    /// as a front nobody sits at needs: no restored turn, request or prompt
    /// sees the tools still coming. A run asked to exit before then exits
    /// without the agent loop, as its exit effect would, its restored work
    /// as the session has it. Calling it again changes nothing.
    fn run_after_start(&mut self) -> &mut Self;
}

impl AppRunExt for App {
    fn run_after_start(&mut self) -> &mut Self {
        if self.world().contains_resource::<LoopAfterStart>() {
            return self;
        }
        start(self);
        self.insert_resource(LoopAfterStart)
            .configure_sets(Update, AgentLoop.run_if(|start: Res<RunStart>| start.over))
            .add_systems(PostUpdate, exit_unstarted.after(refuse_untaken))
            .add_systems(
                Last,
                keep_restored
                    .in_set(RunExitSystems)
                    .in_set(OnAppExitSystems)
                    .before(StopTurnsOnExit)
                    .run_if(on_message::<AppExit>),
            )
    }
}

/// Makes the app end its start ([`RunStart`]), once.
fn start(app: &mut App) {
    if app.world().contains_resource::<RunStart>() {
        return;
    }
    app.init_resource::<RunStart>()
        .configure_sets(Update, RunStartSystems.after(PollTasks).before(AgentLoop))
        .add_systems(Update, check_start.in_set(RunStartSystems))
        .add_systems(PostUpdate, refuse_untaken)
        // The frame after the last start ended looks again.
        .add_observer(|_: On<Remove<Starting>>, wake: Option<Res<Wake>>| {
            if let Some(wake) = wake {
                wake.wake();
            }
        });
}

/// That the agent loop waits for the start.
#[derive(Resource)]
struct LoopAfterStart;

/// Exits a run asked to exit before its start was over, such as on a
/// signal or a refused `--tools`, as its exit effect would, without the
/// agent loop.
fn exit_unstarted(
    start: Res<RunStart>,
    reason: Option<Res<ExitReason>>,
    mut exits: MessageWriter<AppExit>,
    mut exited: Local<bool>,
) {
    if let Some(reason) = reason.filter(|_| !start.over && !*exited) {
        *exited = true;
        exits.write(reason.exit());
    }
}

/// The `Last` system that, on a signal before the start was over, leaves
/// each restored turn to carry on, before the exit stops the running turns
/// ([`StopTurnsOnExit`]).
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct RunExitSystems;

/// On a signal before the start was over, lets each restored turn carry on
/// ([`CarryOn`]): no plugin had its say about it yet, so the exit leaves it
/// as the session has it, for the next run to decide. A run that ends
/// itself first, as on a refused `--tools`, stops it as any exit does.
fn keep_restored(
    start: Res<RunStart>,
    reason: Option<Res<ExitReason>>,
    turns: Query<Entity, (With<Restored>, With<TurnOf>, Without<CarryOn>)>,
    mut commands: Commands,
) {
    if start.over || !matches!(reason.as_deref(), Some(ExitReason::Signal)) {
        return;
    }
    for turn in &turns {
        commands.entity(turn).insert(CarryOn);
    }
}

/// The names `--tools` gives, until the start checks them.
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
/// (at least one edit), so a typo finds its name and another word none, as
/// the model catalog suggests.
pub(crate) fn closest<'a>(name: &str, names: &[&'a str]) -> Option<&'a str> {
    let limit = (name.chars().count() / 4).max(1);
    let distances = names
        .iter()
        .map(|close| (edit_distance(name, close), *close));
    let close = distances.filter(|(distance, _)| *distance <= limit).min();
    close.map(|(_, close)| close)
}

/// Once no [`Starting`] is left, checks the names `--tools` gives against
/// the tools registered then, and makes them the run's tools
/// ([`RunTools`]) before the first request is built. A refusal leaves the
/// run every tool.
fn check_start(
    starting: Query<(), With<Starting>>,
    tools: Option<Res<Tools>>,
    defs: Query<&ToolDef>,
    (mut start, exit): (ResMut<RunStart>, Option<Res<ExitReason>>),
    mut commands: Commands,
) {
    // A run asked to exit never starts, though its last start ended in the
    // same frame.
    if start.over || start.refused.is_some() || !starting.is_empty() || exit.is_some() {
        return;
    }
    let names = defs.iter().map(|tool| tool.0.name.as_str());
    match tools.as_ref().and_then(|tools| tools.unknown(names)) {
        Some(why) => start.refused = Some(why),
        None => {
            start.over = true;
            if let Some(tools) = tools {
                commands.insert_resource(RunTools::new(tools.0.iter().map(String::as_str)));
            }
        }
    }
    commands.remove_resource::<Tools>();
}

/// Refuses the run whose `--tools` was refused when no front took the
/// refusal.
fn refuse_untaken(mut start: ResMut<RunStart>, mut commands: Commands) {
    if let Some(why) = start.refused.take() {
        commands.queue(move |world: &mut World| launcher::refuse_running(world, &why));
    }
}

#[cfg(test)]
mod tests;
