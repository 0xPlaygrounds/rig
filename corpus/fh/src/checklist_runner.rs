//! Runs a markdown checklist as tasks: `/run-plan PLAN.md [--jobs N]`
//! does each unchecked top-level `- [ ]` item of the plan in a one-job
//! helper agent of its own, at most N at once (2 when not given), in the
//! order written and as `(after: 2, 3)`, `(after: "title prefix")` and
//! `(parallel)` say, and ticks the item in the file as it finishes.
//!
//! A run is a [`PlanRun`] entity: the plan's items and where each stands
//! ([`ItemState`]). Each item is asked for as a [`SpawnSubagent`] helper
//! [`ForAgent`] the agent that typed the command, briefed with the text
//! above the checklist, the item and its nested lines; once spawned it is
//! a [`PlanItemAgent`] and gets the `fail_item` tool, which it calls when
//! it cannot do the item. Its turn's end settles the item: `- [x]` with
//! the first line of its answer under it as `> ..`, or `- [!]` with the
//! reason, and the items that wait for it with `(after: ..)` are blocked.
//! The file is read again for each mark, so the user's edits to its other
//! lines stay, and it is the run's only record: a second `/run-plan` runs
//! the items still `[ ]`, and offers to run the `[!]` ones again.
//!
//! `/run-plan stop` lets the running items finish and starts no more; an
//! interrupt of the agent that started the run retires its running items
//! and leaves their boxes as they were. Once a run ends, its agent gets a
//! message saying what finished, what failed and what was blocked. Each
//! change of a run is published as an `x/plan/runs` event ([`PlanRuns`]),
//! from which the view half ([`ChecklistRunnerView`], with the `tui`
//! feature) shows each run as a panel listing its items as they stand.

use std::path::{Path, PathBuf};

use rig_basics::subagents::{SpawnSubagent, SpawnedAgent};
use rig_harness::channel::{ChannelAppExt, ChannelEvent};
use rig_harness::prelude::*;
use rig_harness::worker::{Publish, WorkerOutbox};
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

mod plan;

use plan::Checklist;
pub use plan::Mark;

/// The tool a plan item's agent calls when it cannot do its item.
pub const FAIL_ITEM: &str = "fail_item";

/// Items run at once when the command does not say.
const DEFAULT_JOBS: usize = 2;

const HELP: &str = "Run a markdown checklist's unchecked items, each in an agent of its own: \
    /run-plan PLAN.md [--jobs N] [--retry], or /run-plan stop";

const FAIL_DESCRIPTION: &str = "Report that you cannot do your plan item, and why, then end your \
    turn. The item is marked failed in the plan, and the items that wait for it do not start.";

/// What a plan item's agent is told, besides its brief.
const ITEM_ROLE: &str = "You do one item of a markdown plan; other agents do its other items. \
    Nobody answers questions while you work. When the item is done, end with one line saying \
    what you did: it is written under the item in the plan. If you cannot do it, call \
    `fail_item` with the reason and end your turn.";

/// Adds `/run-plan`, the `fail_item` tool its agents get, and with the
/// `tui` feature a panel for each run.
#[derive(Default)]
pub struct ChecklistRunnerPlugin;

/// The plugin's stage systems, for other plugins to order against.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ChecklistSystems;

impl Plugin for ChecklistRunnerPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<rig_basics::subagents::SubagentsPlugin>() {
            app.add_plugins(rig_basics::subagents::SubagentsPlugin);
        }
        let options = ToolOptions {
            footprint: Some(Touches::Independent),
        };
        app.add_command("run-plan", HELP, run_plan)
            .add_open_tool(FAIL_ITEM, FAIL_DESCRIPTION, options, on_fail_item)
            .add_observer(start_plan)
            .add_observer(on_retry_answer)
            .add_observer(on_item_spawned)
            .add_observer(on_item_end)
            .add_observer(on_interrupt)
            .configure_sets(AgentPass, ChecklistSystems.in_set(PrepareSystems::Edit))
            .add_systems(AgentPass, offer_fail_item_to_items.in_set(ChecklistSystems))
            .add_systems(Update, (retire_orphans, schedule).before(AgentLoop))
            .add_channel_event::<PlanRuns>()
            .add_systems(Last, publish.run_if(resource_exists::<WorkerOutbox>));
    }
}

/// The runs as the views show them: each run's title and its items with
/// where each stands, in the order the runs started.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct PlanRuns {
    /// The runs.
    pub runs: Vec<RunShown>,
}

/// One run of [`PlanRuns`].
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq, Eq)]
pub struct RunShown {
    /// The plan file's name and whether it runs, stops or finished.
    pub title: String,
    /// Each item, numbered, with where it stands on the line below.
    pub items: Vec<String>,
}

impl ChannelEvent for PlanRuns {
    const KIND: &'static str = "x/plan/runs";
}

/// Publishes every run after one changed.
fn publish(
    changed: Query<(), Changed<PlanRun>>,
    runs: Query<(Entity, &PlanRun)>,
    names: Query<&Name>,
    mut out: Publish,
) {
    if changed.is_empty() {
        return;
    }
    let mut runs: Vec<(Entity, &PlanRun)> = runs.iter().collect();
    runs.sort_by_key(|(entity, _)| *entity);
    let runs = runs
        .into_iter()
        .map(|(_, run)| shown(run, &names))
        .collect();
    out.event(&PlanRuns { runs });
}

/// A run's title and items as the panel shows them.
fn shown(run: &PlanRun, names: &Query<&Name>) -> RunShown {
    let items = run
        .items
        .iter()
        .enumerate()
        .map(|(index, item)| {
            let state = match &item.state {
                ItemState::Checked => "done before".to_owned(),
                ItemState::Waiting => "waiting".to_owned(),
                ItemState::Starting => "starting".to_owned(),
                ItemState::Running(agent) => {
                    let name = names.get(*agent).map(Name::as_str).unwrap_or("agent");
                    format!("running ({name})")
                }
                ItemState::Done(_) => "done".to_owned(),
                ItemState::Failed(why) => format!("failed: {why}"),
                ItemState::Blocked(why) => format!("blocked: {why}"),
                ItemState::Cancelled => "cancelled".to_owned(),
            };
            format!("{}. {}\n   {state}", index + 1, item.title)
        })
        .collect();
    let file = run
        .path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned());
    let status = match (run.finished, run.stopping) {
        (true, _) => "finished",
        (false, true) => "stopping",
        (false, false) => "running",
    };
    RunShown {
        title: format!("{} ({status})", file.unwrap_or_default()),
        items,
    }
}

/// A run of a plan: its items and where each stands.
#[derive(Component, Debug)]
pub struct PlanRun {
    /// The plan file.
    pub path: PathBuf,
    /// The agent that started it, whose helpers the items' agents are.
    pub owner: Entity,
    /// The most items running at once.
    pub jobs: usize,
    /// The text above the checklist, every item's context.
    pub context: String,
    /// The top-level items, in order.
    pub items: Vec<PlanItem>,
    /// No more items start: `/run-plan stop`, or an interrupt.
    pub stopping: bool,
    /// The agent that started it was interrupted.
    pub interrupted: bool,
    /// Nothing runs or will start; its agent was told.
    pub finished: bool,
}

/// A top-level item of a [`PlanRun`].
#[derive(Clone, Debug)]
pub struct PlanItem {
    /// Its text, without its annotations.
    pub title: String,
    /// Its nested lines.
    pub body: String,
    /// The items it waits for with `(after: ..)`, by index; each must be
    /// done first.
    pub after: Vec<usize>,
    /// Whether it starts without waiting for the item before it.
    pub parallel: bool,
    /// Where it stands.
    pub state: ItemState,
}

/// Where a [`PlanItem`] stands in its run.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum ItemState {
    /// Checked in the file before the run: skipped.
    Checked,
    /// Not started yet.
    Waiting,
    /// Its agent was asked for.
    Starting,
    /// Its agent works on it.
    Running(Entity),
    /// Its agent finished it, with this line.
    Done(String),
    /// It failed, for this reason, in this run or an earlier one.
    Failed(String),
    /// An item it waits for did not finish, as this says.
    Blocked(String),
    /// It was stopped before it finished; its box stays unchecked.
    Cancelled,
}

impl ItemState {
    /// Whether the items that wait for it may start.
    fn done(&self) -> bool {
        matches!(self, Self::Checked | Self::Done(_))
    }

    /// Whether it has not settled yet.
    fn pending(&self) -> bool {
        matches!(self, Self::Waiting | Self::Starting | Self::Running(_))
    }

    /// Whether it is started.
    fn active(&self) -> bool {
        matches!(self, Self::Starting | Self::Running(_))
    }
}

/// On a plan item's agent: the run and the item's index.
#[derive(Component, Clone, Debug)]
pub struct PlanItemAgent {
    /// The [`PlanRun`].
    pub run: Entity,
    /// The item, from 0.
    pub index: usize,
    /// Why it failed, once it called `fail_item`.
    pub failure: Option<String>,
}

/// Saved on a plan item's agent. A restart brings back an agent the exit
/// cut off mid-turn, but not its run, so this plugin retires it; its box
/// stays unchecked for the next `/run-plan`.
#[derive(Component, Reflect, Default)]
#[reflect(Component, Saved, Default)]
pub struct WasPlanItem;

/// On a request for an item's agent: the run and the item's index.
#[derive(Component, Clone, Copy, Debug)]
struct ItemRequest {
    run: Entity,
    index: usize,
}

/// On the question whether to retry the failed items: the run it starts.
#[derive(Component, Clone, Debug)]
struct RetryAsk {
    path: PathBuf,
    jobs: usize,
}

/// Starts a run of the plan at `path` for `agent`.
#[derive(Event, Clone, Debug)]
struct StartPlan {
    agent: Entity,
    path: PathBuf,
    jobs: usize,
    retry: bool,
}

/// The arguments of `fail_item`.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct FailArgs {
    /// Why the item cannot be done, in a line.
    reason: String,
}

/// What `/run-plan` was asked.
#[derive(Debug, PartialEq, Eq)]
enum Request {
    Stop,
    Run {
        path: String,
        jobs: usize,
        retry: Option<bool>,
    },
}

/// Reads `/run-plan`'s arguments.
fn parse_args(args: &str) -> Result<Request, String> {
    let mut words = args.split_whitespace();
    let (mut path, mut jobs, mut retry) = (None, DEFAULT_JOBS, None);
    while let Some(word) = words.next() {
        match word {
            "stop" if path.is_none() => return Ok(Request::Stop),
            "--retry" => retry = Some(true),
            "--no-retry" => retry = Some(false),
            "--jobs" => {
                let count = words.next().and_then(|count| count.parse().ok());
                jobs = count
                    .filter(|&count| count > 0)
                    .ok_or("--jobs needs a number above 0")?;
            }
            flag if flag.starts_with("--") => return Err(format!("unknown option {flag}")),
            word if path.is_none() => path = Some(word.to_owned()),
            _ => return Err("one plan at a time".to_owned()),
        }
    }
    let path = path.ok_or("name the plan file: /run-plan PLAN.md")?;
    Ok(Request::Run { path, jobs, retry })
}

/// `/run-plan PLAN.md [--jobs N] [--retry | --no-retry]` and
/// `/run-plan stop`.
fn run_plan(
    In(args): In<CommandArgs>,
    mut runs: Query<(Entity, &mut PlanRun)>,
    dirs: Query<&WorkDir>,
    root: Option<Res<ProjectRoot>>,
    answerers: Answerers,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let agent = args.agent;
    let (path, jobs, retry) = match parse_args(&args.args) {
        Ok(Request::Run { path, jobs, retry }) => (path, jobs, retry),
        Ok(Request::Stop) => {
            let mut said = "No plan runs.".to_owned();
            for (entity, mut run) in runs.iter_mut().filter(|(_, run)| run.owner == agent) {
                if run.finished {
                    commands.entity(entity).despawn();
                } else if !run.stopping {
                    run.stopping = true;
                    said = format!(
                        "{}: the running items finish; no more start.",
                        run.path.display()
                    );
                }
            }
            notices.write(Notice::info(agent, said));
            return;
        }
        Err(why) => {
            notices.write(Notice::error(agent, format!("/run-plan: {why}")));
            return;
        }
    };
    let base = dirs
        .get(agent)
        .map(|dir| dir.0.clone())
        .ok()
        .or_else(|| root.map(|root| root.0.clone()))
        .unwrap_or_default();
    let path = base.join(path);
    let text = match std::fs::read_to_string(&path) {
        Ok(text) => text,
        Err(error) => {
            let why = format!("/run-plan: cannot read {}: {error}", path.display());
            notices.write(Notice::error(agent, why));
            return;
        }
    };
    let failed = plan::parse(&text)
        .items
        .iter()
        .filter(|item| item.mark == Mark::Failed)
        .count();
    match retry {
        None if failed > 0 && answerers.can_answer(Some(agent)) => {
            let title = format!("Run the {failed} failed items of {} again?", path.display());
            let ask = Ask::yes_no(title).body("No runs only the unchecked items.");
            commands.spawn((ask, ForAgent(agent), RetryAsk { path, jobs }));
        }
        retry => {
            if failed > 0 && retry.is_none() {
                let why = format!(
                    "{failed} failed items stay failed; /run-plan {} --retry runs them again.",
                    path.display()
                );
                notices.write(Notice::info(agent, why));
            }
            let retry = retry.unwrap_or(false);
            commands.trigger(StartPlan {
                agent,
                path,
                jobs,
                retry,
            });
        }
    }
}

/// Starts the run the user's answer on retrying failed items says.
fn on_retry_answer(
    settled: On<Add<Settled>>,
    asks: Query<(&RetryAsk, &Outcome, &ForAgent)>,
    mut commands: Commands,
) {
    let Ok((RetryAsk { path, jobs }, outcome, &ForAgent(agent))) = asks.get(settled.entity) else {
        return;
    };
    if let Some(answer) = Answer::of(outcome) {
        commands.trigger(StartPlan {
            agent,
            path: path.clone(),
            jobs: *jobs,
            retry: answer.is("yes"),
        });
    }
}

/// Spawns the run of a plan, unless one of it runs already or nothing in
/// it is left to do.
fn start_plan(
    start: On<StartPlan>,
    runs: Query<(Entity, &PlanRun)>,
    mut notices: MessageWriter<Notice>,
    (wake, mut commands): (Res<Wake>, Commands),
) {
    let StartPlan {
        agent,
        path,
        jobs,
        retry,
    } = start.event().clone();
    let same = || runs.iter().filter(|(_, run)| run.path == path);
    if same().any(|(_, run)| !run.finished) {
        let why = format!("/run-plan: {} is running already", path.display());
        notices.write(Notice::error(agent, why));
        return;
    }
    for (old, _) in same() {
        commands.entity(old).despawn();
    }
    let checklist = match std::fs::read_to_string(&path) {
        Ok(text) => plan::parse(&text),
        Err(error) => {
            let why = format!("/run-plan: cannot read {}: {error}", path.display());
            notices.write(Notice::error(agent, why));
            return;
        }
    };
    let run = match new_run(checklist, (agent, path.clone(), jobs), retry) {
        Ok(run) => run,
        Err(why) => {
            notices.write(Notice::error(agent, format!("/run-plan: {why}")));
            return;
        }
    };
    let waiting = run
        .items
        .iter()
        .filter(|item| item.state == ItemState::Waiting);
    let waiting = waiting.count();
    if waiting == 0 {
        let why = format!("{}: nothing is left to run.", path.display());
        notices.write(Notice::info(agent, why));
        return;
    }
    let said = format!(
        "{}: running {waiting} items, {} at once.",
        path.display(),
        run.jobs
    );
    notices.write(Notice::info(agent, said));
    let name = Name::new(format!("plan run: {}", path.display()));
    commands.spawn((name, run));
    wake.wake();
}

/// The run of `checklist` for `(owner, path, jobs)`: its checked items
/// skipped, its failed ones run again with `retry`.
fn new_run(
    checklist: Checklist,
    (owner, path, jobs): (Entity, PathBuf, usize),
    retry: bool,
) -> Result<PlanRun, String> {
    let dependencies = checklist.dependencies()?;
    let items = checklist
        .items
        .into_iter()
        .zip(dependencies)
        .map(|(entry, after)| {
            let state = match entry.mark {
                Mark::Done => ItemState::Checked,
                Mark::Failed if !retry => ItemState::Failed("failed in an earlier run".to_owned()),
                Mark::Open | Mark::Failed => ItemState::Waiting,
            };
            PlanItem {
                title: entry.title,
                body: entry.body,
                after,
                parallel: entry.parallel,
                state,
            }
        })
        .collect();
    Ok(PlanRun {
        path,
        owner,
        jobs,
        context: checklist.context,
        items,
        stopping: false,
        interrupted: false,
        finished: false,
    })
}

impl PlanRun {
    /// Blocks each waiting item that waits, by `(after: ..)`, for one that
    /// will not be done, until none is left to block.
    fn block(&mut self) -> bool {
        let mut changed = false;
        loop {
            let blocked = self.items.iter().enumerate().find_map(|(index, item)| {
                let stuck = item.after.iter().find(|&&after| {
                    let state = self.items.get(after).map(|item| &item.state);
                    state.is_some_and(|state| !state.done() && !state.pending())
                });
                (item.state == ItemState::Waiting)
                    .then_some(())
                    .and(stuck)
                    .map(|&after| (index, after))
            });
            let Some(item) = blocked.and_then(|(index, after)| {
                let why = format!("item {} did not finish", after + 1);
                Some((self.items.get_mut(index)?, why))
            }) else {
                return changed;
            };
            item.0.state = ItemState::Blocked(item.1);
            changed = true;
        }
    }

    /// The waiting items that may start now, in order.
    fn ready(&self) -> Vec<usize> {
        let state = |index: usize| self.items.get(index).map(|item| &item.state);
        let ready = self.items.iter().enumerate().filter(|(index, item)| {
            let after_done = item
                .after
                .iter()
                .all(|&after| state(after).is_some_and(ItemState::done));
            let in_order = !item.after.is_empty()
                || item.parallel
                || index
                    .checked_sub(1)
                    .and_then(state)
                    .is_none_or(|before| !before.pending());
            item.state == ItemState::Waiting && after_done && in_order
        });
        ready.map(|(index, _)| index).collect()
    }

    /// How many items are started.
    fn active(&self) -> usize {
        self.items.iter().filter(|item| item.state.active()).count()
    }

    /// What the run's agent is told when it ends.
    fn summary(&self) -> String {
        let list = |pick: &dyn Fn(&ItemState) -> Option<String>| {
            let picked = self.items.iter().enumerate().filter_map(|(index, item)| {
                let detail = pick(&item.state)?;
                Some(format!("- {} {}{detail}", index + 1, item.title))
            });
            picked.collect::<Vec<_>>()
        };
        let sections = [
            (
                "Finished",
                list(&|state| match state {
                    ItemState::Done(line) => Some(format!(": {line}")),
                    _ => None,
                }),
            ),
            (
                "Failed",
                list(&|state| match state {
                    ItemState::Failed(why) => Some(format!(": {why}")),
                    _ => None,
                }),
            ),
            (
                "Blocked",
                list(&|state| match state {
                    ItemState::Blocked(why) => Some(format!(": {why}")),
                    _ => None,
                }),
            ),
            (
                "Not run",
                list(&|state| {
                    matches!(state, ItemState::Waiting | ItemState::Cancelled).then(String::new)
                }),
            ),
        ];
        let mut text = format!("The plan {} ran.", self.path.display());
        for (heading, lines) in sections.iter().filter(|(_, lines)| !lines.is_empty()) {
            text.push_str(&format!("\n{heading}:\n{}", lines.join("\n")));
        }
        text
    }

    /// The brief of item `index`'s agent.
    fn brief(&self, index: usize) -> String {
        let Some(item) = self.items.get(index) else {
            return String::new();
        };
        let mut brief = String::new();
        if !self.context.is_empty() {
            brief.push_str(&format!("The plan's context:\n\n{}\n\n", self.context));
        }
        brief.push_str(&format!(
            "Your item, number {} of the plan {}:\n\n{}",
            index + 1,
            self.path.display(),
            item.title
        ));
        if !item.body.is_empty() {
            brief.push_str(&format!("\n{}", item.body));
        }
        brief
    }

    /// Settles item `index` as `state`, and marks it in the file when it
    /// is done or failed.
    fn settle(&mut self, index: usize, state: ItemState, notices: &mut MessageWriter<Notice>) {
        let Some(item) = self.items.get_mut(index) else {
            return;
        };
        item.state = state.clone();
        let (mark, note) = match state {
            ItemState::Done(line) => (Mark::Done, line),
            ItemState::Failed(why) => (Mark::Failed, format!("failed: {why}")),
            _ => return,
        };
        if let Err(why) = mark_in_file(&self.path, index + 1, &item.title, mark, &note) {
            notices.write(Notice::warn(self.owner, why));
        }
    }
}

/// Marks item `number`, titled `title`, in the plan at `path` as the file
/// is now.
fn mark_in_file(
    path: &Path,
    number: usize,
    title: &str,
    mark: Mark,
    note: &str,
) -> Result<(), String> {
    let cannot =
        |error: std::io::Error| format!("cannot mark item {number} in {}: {error}", path.display());
    let text = std::fs::read_to_string(path).map_err(cannot)?;
    let marked = plan::mark(&text, number, title, mark, note)
        .ok_or_else(|| format!("{}: item {number} \"{title}\" is gone", path.display()))?;
    std::fs::write(path, marked).map_err(cannot)
}

/// Starts the items of each run that may start, up to its jobs, blocks
/// those whose items failed, and ends a run once nothing runs or will
/// start, telling its agent how it went.
fn schedule(
    mut runs: Query<(Entity, &mut PlanRun), Changed<PlanRun>>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for (entity, mut run) in &mut runs {
        // Only a real change marks the run changed, so this runs again
        // only when something happened.
        let run_now = run.bypass_change_detection();
        if run_now.finished {
            continue;
        }
        let mut changed = run_now.block();
        if !run_now.stopping {
            for index in run_now.ready() {
                if run_now.active() >= run_now.jobs {
                    break;
                }
                let title = run_now.items.get(index).map(|item| item.title.clone());
                let title = format!("item {}: {}", index + 1, title.unwrap_or_default());
                let short: String = title.chars().take(40).collect();
                let request = SpawnSubagent::helper(short)
                    .brief(run_now.brief(index))
                    .once();
                let item = ItemRequest { run: entity, index };
                commands.spawn((request, ForAgent(run_now.owner), item));
                if let Some(item) = run_now.items.get_mut(index) {
                    item.state = ItemState::Starting;
                }
                changed = true;
            }
        }
        if run_now.active() == 0 && (run_now.stopping || run_now.ready().is_empty()) {
            if !run_now.stopping {
                // Nothing runs and nothing may start: what still waits,
                // waits on itself.
                for item in &mut run_now.items {
                    if item.state == ItemState::Waiting {
                        item.state =
                            ItemState::Blocked("waits on an item that cannot start".into());
                    }
                }
            }
            run_now.finished = true;
            changed = true;
            let summary = run_now.summary();
            if run_now.interrupted {
                notices.write(Notice::info(run_now.owner, summary));
            } else {
                let message = Deliver::new(run_now.owner, summary, DeliveryMode::Queue);
                commands.trigger(message.with_author(Author::plugin("run-plan")));
            }
        }
        if changed {
            run.set_changed();
        }
    }
}

/// An item's agent was spawned, or could not be: it starts on its item
/// with its role and `fail_item`, or the item fails. One spawned for a run
/// that cancelled the item meanwhile is retired.
fn on_item_spawned(
    settled: On<Add<Settled>>,
    requests: Query<(&ItemRequest, &Outcome, Option<&SpawnedAgent>)>,
    mut runs: Query<&mut PlanRun>,
    mut notices: MessageWriter<Notice>,
    (wake, mut commands): (Res<Wake>, Commands),
) {
    let Ok((&ItemRequest { run, index }, outcome, spawned)) = requests.get(settled.entity) else {
        return;
    };
    // The next frame's `schedule` sees the run changed.
    wake.wake();
    let starting = |plan: &PlanRun| {
        let state = plan.items.get(index).map(|item| &item.state);
        state == Some(&ItemState::Starting)
    };
    let Some(mut plan) = runs.get_mut(run).ok().filter(|plan| starting(plan)) else {
        if let Some(&SpawnedAgent(agent)) = spawned {
            commands.trigger(Retire { entity: agent });
        }
        return;
    };
    match (spawned, &outcome.0) {
        (Some(&SpawnedAgent(agent)), Ok(_)) => {
            let role = PromptSection::new(PromptSection::ORDER_ROLE, "plan_item_role", ITEM_ROLE);
            commands.spawn((Name::new("plan_item_role"), role, SectionOf(agent)));
            let item = PlanItemAgent {
                run,
                index,
                failure: None,
            };
            commands.entity(agent).insert((item, WasPlanItem));
            if let Some(item) = plan.items.get_mut(index) {
                item.state = ItemState::Running(agent);
            }
        }
        (_, outcome) => {
            let why = match outcome {
                Err(report) => format!("not started: {}", first_line(&report.message)),
                Ok(_) => "not started".to_owned(),
            };
            plan.settle(index, ItemState::Failed(why), &mut notices);
        }
    }
}

/// An item's agent ended its turn: the item is done with its answer's
/// first line, failed with the reason it gave `fail_item` or its turn's
/// failure, or, stopped, left unchecked. `rig_basics::subagents` retires the agent,
/// a one-job helper, as the same turn ends.
fn on_item_end(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    agents: Query<&PlanItemAgent>,
    mut runs: Query<&mut PlanRun>,
    mut notices: MessageWriter<Notice>,
    wake: Res<Wake>,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok(PlanItemAgent {
        run,
        index,
        failure,
    }) = agents.get(agent)
    else {
        return;
    };
    let Ok(mut plan) = runs.get_mut(*run) else {
        return;
    };
    let state = plan.items.get(*index).map(|item| &item.state);
    if state != Some(&ItemState::Running(agent)) {
        return;
    }
    let state = match (failure, outcome) {
        (Some(why), _) => ItemState::Failed(why.clone()),
        (None, TurnOutcome::Answered(message)) => {
            let answer = message.final_answer().unwrap_or_default();
            let line = first_line(&answer);
            ItemState::Done(if line.is_empty() {
                "done".to_owned()
            } else {
                line
            })
        }
        (None, TurnOutcome::Failed(why)) => ItemState::Failed(first_line(why)),
        (None, TurnOutcome::Stopped(_)) => ItemState::Cancelled,
    };
    plan.settle(*index, state, &mut notices);
    wake.wake();
}

/// The first non-empty line of `text`, trimmed.
fn first_line(text: &str) -> String {
    let line = text.lines().map(str::trim).find(|line| !line.is_empty());
    line.unwrap_or_default().to_owned()
}

/// An interrupt of a run's agent stops the run: its running items' agents
/// are retired and their boxes stay as they were.
fn on_interrupt(
    interrupt: On<StopTurn>,
    mut runs: Query<(Entity, &mut PlanRun)>,
    open: Query<(Entity, &ItemRequest), Without<Settled>>,
    (wake, mut commands): (Res<Wake>, Commands),
) {
    wake.wake();
    let owned = runs
        .iter_mut()
        .filter(|(_, run)| run.owner == interrupt.entity);
    for (entity, mut run) in owned.filter(|(_, run)| !run.finished) {
        run.stopping = true;
        run.interrupted = true;
        for item in &mut run.items {
            if let ItemState::Running(agent) = item.state {
                commands.trigger(Retire { entity: agent });
            }
            if item.state.active() {
                item.state = ItemState::Cancelled;
            }
        }
        for (request, _) in open.iter().filter(|(_, request)| request.run == entity) {
            commands.entity(request).despawn();
        }
    }
}

/// `fail_item`: an item's agent says it cannot do its item. Any other
/// caller is refused.
fn on_fail_item(
    called: On<ToolCalled<FailArgs>>,
    mut agents: Query<&mut PlanItemAgent>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let Ok(mut call) = calls.get_mut(called.call) else {
        return;
    };
    let Ok(mut item) = agents.get_mut(called.agent) else {
        call.decide(
            Decider::plugin("fh-checklist-runner"),
            Outcome::refused("fail_item is for the agents of a plan's items"),
        );
        return;
    };
    item.failure = Some(first_line(&called.args.reason)).filter(|why| !why.is_empty());
    item.failure
        .get_or_insert_with(|| "no reason given".to_owned());
    let said = format!(
        "Item {} is marked failed. End your turn now.",
        item.index + 1
    );
    call.decide(
        Decider::plugin("fh-checklist-runner"),
        ToolResult::success(said.into()),
    );
}

/// `fail_item` is offered only to the agents of plan items: every other
/// agent may use every tool, so it is taken out of their requests.
fn offer_fail_item_to_items(
    mut calls: Query<(&mut ModelRequest, &ForAgent), (Added<Admitted>, ModelCalls, With<TurnCall>)>,
    items: Query<(), With<PlanItemAgent>>,
) {
    for (mut call, &ForAgent(agent)) in &mut calls {
        let offered = |call: &ModelRequest| {
            let tools = &call.request.tools;
            tools.iter().any(|tool| tool.name.as_str() == FAIL_ITEM)
        };
        if !items.contains(agent) && offered(&call) {
            call.request_mut()
                .tools
                .retain(|tool| tool.name.as_str() != FAIL_ITEM);
        }
    }
}

/// Retires each item agent a restart brought back: its run is gone, and
/// its box stays unchecked for the next `/run-plan`. Before the agent
/// loop, so its restored turn never goes on.
fn retire_orphans(
    restored: Query<Entity, (Added<WasPlanItem>, Without<PlanItemAgent>)>,
    mut commands: Commands,
) {
    for agent in &restored {
        commands.trigger(Retire { entity: agent });
    }
}

#[cfg(feature = "tui")]
pub use view::ChecklistRunnerView;

/// The view half: a panel per run, from the agent's `x/plan/runs` events.
#[cfg(feature = "tui")]
mod view {
    use rig_harness::prelude::*;
    use rig_harness::view::Applied;
    use rig_tui::ratatui::layout::Constraint;
    use rig_tui::ratatui::widgets::{Block, Paragraph, Wrap};
    use rig_tui::{PanelCanvas, Placement, RequestRedraw, TuiPanel, TuiSystems};

    use super::{ChannelEvent, PlanRuns};

    /// A panel per run, named in `plugins.toml` as the entry's `view`.
    #[derive(Default)]
    pub struct ChecklistRunnerView;

    impl Plugin for ChecklistRunnerView {
        fn build(&self, app: &mut App) {
            app.init_resource::<Shown>()
                .add_message::<RequestRedraw>()
                .add_message::<Applied>()
                .add_systems(Update, (mirror_runs, panels).chain())
                .add_systems(PostUpdate, draw.in_set(TuiSystems::Draw));
        }
    }

    /// The runs the agent last published.
    #[derive(Resource, Default)]
    struct Shown(PlanRuns);

    /// On the panel of the run at this place in [`Shown`].
    #[derive(Component)]
    struct PlanPanel(usize);

    fn mirror_runs(
        mut applied: MessageReader<Applied>,
        mut shown: ResMut<Shown>,
        mut redraw: MessageWriter<RequestRedraw>,
    ) {
        for Applied(record) in applied.read() {
            if record.kind != PlanRuns::KIND {
                continue;
            }
            if let Ok(runs) = serde_json::from_value::<PlanRuns>(record.data.clone()) {
                shown.0 = runs;
                redraw.write(RequestRedraw);
            }
        }
    }

    /// A panel for each run, and none for a run that is gone.
    fn panels(shown: Res<Shown>, panels: Query<(Entity, &PlanPanel)>, mut commands: Commands) {
        if !shown.is_changed() {
            return;
        }
        let count = shown.0.runs.len();
        for (entity, &PlanPanel(index)) in &panels {
            if index >= count {
                commands.entity(entity).despawn();
            }
        }
        for index in panels.iter().count()..count {
            let panel = TuiPanel::new(Placement::Right(Constraint::Length(44)));
            commands.spawn((PlanPanel(index), panel));
        }
    }

    /// Each run's items, with where each stands.
    fn draw(mut panels: Query<(&mut PanelCanvas, &PlanPanel)>, shown: Res<Shown>) {
        for (mut canvas, &PlanPanel(index)) in &mut panels {
            let Some(run) = shown.0.runs.get(index) else {
                continue;
            };
            let text = Paragraph::new(run.items.join("\n")).wrap(Wrap { trim: false });
            canvas.render(text.block(Block::bordered().title(run.title.clone())));
        }
    }
}

#[cfg(test)]
mod tests;
