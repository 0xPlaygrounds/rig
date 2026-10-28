//! A background task queue: `/queue <prompt>` lines up a task that runs as
//! a fresh helper agent of its own, one after another, while the user keeps
//! talking to the main agent.
//!
//! Each task is a [`SpawnSubagent`] helper, named `task N`, on the default
//! model or the one `/queue --model m <prompt>` names, with the tools of the
//! agent that queued it. Its prompt is its first message, so its
//! conversation, logged in the session's store as every agent's is, holds
//! only its prompt and its replies; it reports to no one, so the main
//! agent's conversation never hears of it. The queue is a saved resource
//! ([`TaskQueue`]), and each task's agent carries a saved [`QueuedTask`], so
//! after a restart a pending task still runs and one that was running is
//! [`TaskState::Interrupted`], its restored turn stopped, until `/queue
//! retry N` runs it again in the same agent.
//!
//! `[queue] concurrency = N` in the settings runs N at once (1 by default).
//! `/queue cancel N`, `/queue up N`, `/queue retry N` and `/queue open N`
//! (shows that task's agent, to keep talking to it) manage the queue;
//! `/queue` lists it. A question a task's agent asks is titled with the
//! task. With the `tui` feature, a panel lists the tasks and the status
//! line says how the last one ended.

use std::time::{SystemTime, UNIX_EPOCH};

use rig_basics::subagents::{SpawnSubagent, SpawnedAgent, SubagentsPlugin};
use rig_basics::telemetry::usage::TurnUsage;
use rig_harness::prelude::*;
use serde::Deserialize;

#[cfg(feature = "tui")]
mod looks;

const HELP: &str = "Background tasks: /queue [--model m] <prompt>, /queue, /queue cancel N, \
    /queue up N, /queue retry N, /queue open N";

/// Adds `/queue`, the saved queue, the tasks' agents and, with the `tui`
/// feature, the queue's panel and status-line item.
#[derive(Default)]
pub struct TaskQueuePlugin;

impl Plugin for TaskQueuePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SubagentsPlugin>() {
            app.add_plugins(SubagentsPlugin);
        }
        app.register_type::<TaskQueue>()
            .register_type::<QueuedTask>()
            .add_setting::<QueueSettings>("queue")
            .init_resource::<TaskQueue>()
            .init_resource::<QueueNote>()
            .init_resource::<ShownAgent>()
            .add_command("queue", HELP, queue)
            .add_systems(Startup, (interrupt_running, warn_shared_tree))
            .add_systems(
                Update,
                start_next
                    .before(AgentLoop)
                    .run_if(resource_changed::<TaskQueue>),
            )
            .add_observer(adopt)
            .add_observer(finish)
            .configure_sets(AgentPass, QueueSystems::Label.in_set(PrepareSystems::Edit))
            .add_systems(AgentPass, label_asks.in_set(QueueSystems::Label));
        #[cfg(feature = "tui")]
        looks::add(app);
    }
}

/// The plugin's stage systems, for others to order against.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum QueueSystems {
    /// In `PrepareSystems::Edit`: titles each question a task's agent asks
    /// with the task.
    Label,
}

/// The `[queue]` settings table.
#[derive(Resource, Clone, Debug, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct QueueSettings {
    /// How many tasks run at once; above 1 they share the working tree.
    pub concurrency: usize,
}

impl Default for QueueSettings {
    fn default() -> Self {
        Self { concurrency: 1 }
    }
}

/// The tasks, in queue order, saved with the session.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved, Default)]
pub struct TaskQueue {
    /// The number the last task got.
    pub last: u32,
    /// Every task, the next to run first among the pending ones.
    pub tasks: Vec<Task>,
}

/// A queued prompt and how it went.
#[derive(Reflect, Clone, Debug, Default, PartialEq)]
pub struct Task {
    /// Its number, which the commands name.
    pub number: u32,
    /// What it was asked.
    pub prompt: String,
    /// The catalog model it runs on, else the queuing agent's.
    pub model: Option<String>,
    /// The [`AgentId`] of the agent that queued it, its agent's parent.
    pub owner: String,
    /// Where it stands.
    pub state: TaskState,
    /// When its last run started, in milliseconds since the Unix epoch.
    pub started: Option<u64>,
    /// When its last run ended, likewise.
    pub ended: Option<u64>,
    /// The first line of its final answer, or why it has none.
    pub outcome: String,
    /// What its last run's model calls cost, in USD, once it ended.
    pub cost: f64,
}

/// Where a [`Task`] stands.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum TaskState {
    /// Waiting for its turn.
    #[default]
    Pending,
    /// Its agent works on it.
    Running,
    /// It answered.
    Done,
    /// Its turn failed, or its agent could not start.
    Failed,
    /// The user cancelled it.
    Cancelled,
    /// It was running when rig quit, or its turn was stopped by someone
    /// other than the queue; `/queue retry` runs it again.
    Interrupted,
}

impl TaskState {
    /// How the panel and the list name it.
    pub fn label(self) -> &'static str {
        match self {
            Self::Pending => "pending",
            Self::Running => "running",
            Self::Done => "done",
            Self::Failed => "failed",
            Self::Cancelled => "cancelled",
            Self::Interrupted => "interrupted",
        }
    }

    /// Whether it ran and is over.
    fn ended(self) -> bool {
        matches!(
            self,
            Self::Done | Self::Failed | Self::Cancelled | Self::Interrupted
        )
    }
}

impl Task {
    /// How long its last run took, or runs so far, in seconds.
    pub fn elapsed(&self) -> Option<u64> {
        let started = self.started?;
        let ended = self.ended.unwrap_or_else(now);
        Some(ended.saturating_sub(started) / 1000)
    }
}

/// On a task's agent, saved with it: the task it runs.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default)]
pub struct QueuedTask(pub u32);

/// The last thing the queue said, for the status line: `task 2 done`.
#[derive(Resource, Clone, Debug, Default)]
pub struct QueueNote(pub String);

/// On a task's [`SpawnSubagent`] request until it settles.
#[derive(Component)]
struct TaskRequest(u32);

/// Wall time, in milliseconds since the Unix epoch.
fn now() -> u64 {
    let since = SystemTime::now().duration_since(UNIX_EPOCH);
    since.map_or(0, |since| {
        u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
    })
}

/// The name of task `number`'s agent.
fn title(number: u32) -> String {
    format!("task {number}")
}

/// What `/queue` was asked.
#[derive(Debug, PartialEq, Eq)]
enum Verb {
    List,
    Add(Option<String>, String),
    Cancel(u32),
    Up(u32),
    Retry(u32),
    Open(u32),
}

/// Reads `/queue`'s arguments: a verb and a task number, else a prompt.
fn parse(args: &str) -> Result<Verb, String> {
    let args = args.trim();
    if args.is_empty() || args == "list" {
        return Ok(Verb::List);
    }
    let (verb, rest) = args.split_once(' ').unwrap_or((args, ""));
    if let Ok(number) = rest.trim().parse::<u32>() {
        match verb {
            "cancel" => return Ok(Verb::Cancel(number)),
            "up" => return Ok(Verb::Up(number)),
            "retry" => return Ok(Verb::Retry(number)),
            "open" => return Ok(Verb::Open(number)),
            _ => {}
        }
    }
    let (model, prompt) = match args.strip_prefix("--model") {
        Some(rest) => {
            let (model, prompt) = rest.trim_start().split_once(' ').unwrap_or((rest, ""));
            (Some(model.trim().to_owned()), prompt.trim())
        }
        None => (None, args),
    };
    if prompt.is_empty() || model.as_deref() == Some("") {
        return Err(HELP.to_owned());
    }
    Ok(Verb::Add(model, prompt.to_owned()))
}

/// `/queue …`.
fn queue(
    In(args): In<CommandArgs>,
    mut queue: ResMut<TaskQueue>,
    ids: Query<&AgentId>,
    agents: Query<(Entity, &QueuedTask)>,
    turns: RunningTurns,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let agent_of = |number: u32| {
        agents
            .iter()
            .find_map(|(agent, task)| (task.0 == number).then_some(agent))
    };
    let said = parse(&args.args).and_then(|verb| match verb {
        Verb::List => Ok(list(&queue, |number| live_cost(agent_of(number), &turns))),
        Verb::Add(model, prompt) => {
            let owner = ids
                .get(args.agent)
                .map_err(|_| "Not an agent.".to_owned())?;
            let ahead = queue
                .tasks
                .iter()
                .filter(|task| !task.state.ended())
                .count();
            queue.last = queue.last.saturating_add(1);
            let number = queue.last;
            queue.tasks.push(Task {
                number,
                prompt,
                model,
                owner: owner.0.clone(),
                ..Task::default()
            });
            Ok(format!("Queued task {number} ({ahead} ahead of it)."))
        }
        Verb::Cancel(number) => {
            let task = find(&mut queue, number)?;
            match task.state {
                TaskState::Pending => {}
                TaskState::Running => {
                    if let Some(agent) = agent_of(number) {
                        commands.trigger(StopTurn::new(agent));
                    }
                    task.ended = Some(now());
                }
                state => return Err(format!("Task {number} is {} already.", state.label())),
            }
            task.state = TaskState::Cancelled;
            Ok(format!("Cancelled task {number}."))
        }
        Verb::Up(number) => {
            let pending = |task: &Task| task.state == TaskState::Pending;
            let at = queue.tasks.iter().position(|task| task.number == number);
            let at = at.filter(|&at| queue.tasks.get(at).is_some_and(pending));
            let at = at.ok_or_else(|| format!("Task {number} is not pending."))?;
            let before = queue
                .tasks
                .get(..at)
                .and_then(|before| before.iter().rposition(pending));
            let before = before.ok_or_else(|| format!("Task {number} is next already."))?;
            queue.tasks.swap(before, at);
            Ok(format!("Moved task {number} up."))
        }
        Verb::Retry(number) => {
            let at = queue.tasks.iter().position(|task| task.number == number);
            let mut task = at
                .map(|at| queue.tasks.remove(at))
                .ok_or_else(|| format!("No task {number}; /queue lists them."))?;
            if !task.state.ended() {
                let state = task.state.label();
                queue.tasks.insert(at.unwrap_or_default(), task);
                return Err(format!("Task {number} is {state}."));
            }
            task.state = TaskState::Pending;
            (task.started, task.ended, task.cost) = (None, None, 0.0);
            task.outcome.clear();
            queue.tasks.push(task);
            Ok(format!("Task {number} will run again."))
        }
        Verb::Open(number) => {
            let agent = agent_of(number)
                .ok_or_else(|| format!("Task {number} has no agent yet; it has not started."))?;
            show(&mut commands, agent);
            Ok(format!("Showing task {number}."))
        }
    });
    notices.write(match said {
        Ok(text) => Notice::info(args.agent, text),
        Err(why) => Notice::error(args.agent, why),
    });
}

/// The task numbered `number`.
fn find(queue: &mut TaskQueue, number: u32) -> Result<&mut Task, String> {
    let task = queue.tasks.iter_mut().find(|task| task.number == number);
    task.ok_or_else(|| format!("No task {number}; /queue lists them."))
}

/// Shows `agent` in the front the user uses: the terminal view's `ShowAgent`
/// when there is one, and the harness's [`ShownAgent`] for any other.
fn show(commands: &mut Commands, agent: Entity) {
    #[cfg(feature = "tui")]
    commands.trigger(rig_tui::ShowAgent { entity: agent });
    commands.queue(move |world: &mut World| {
        if let Some(mut shown) = world.get_resource_mut::<ShownAgent>() {
            shown.set_if_neq(ShownAgent(Some(agent)));
        }
    });
}

/// What the running turns spent so far, by their agents.
pub type RunningTurns<'w, 's> = Query<'w, 's, (&'static TurnOf, &'static TurnUsage)>;

/// What the running turn of `agent`, if any, spent so far, in USD.
pub fn live_cost(agent: Option<Entity>, turns: &RunningTurns) -> Option<f64> {
    let agent = agent?;
    let turn = turns.iter().find(|(turn, _)| turn.0 == agent);
    turn.map(|(_, spent)| spent.0.cost)
}

/// The queue, a line per task, a running one with what its turn spent so
/// far (`live`).
fn list(queue: &TaskQueue, live: impl Fn(u32) -> Option<f64>) -> String {
    if queue.tasks.is_empty() {
        return "No tasks; /queue <prompt> adds one.".to_owned();
    }
    let mut lines = vec!["Tasks:".to_owned()];
    lines.extend(queue.tasks.iter().map(|task| line(task, live(task.number))));
    lines.join("\n")
}

/// A task's line: number, state, elapsed time, what its run cost (for a
/// running one, what its turn spent so far, `live`, when usage is counted),
/// and its answer's first line, else its prompt.
pub fn line(task: &Task, live: Option<f64>) -> String {
    let mut line = format!("{:>3} {:<11}", task.number, task.state.label());
    if let Some(elapsed) = task.elapsed() {
        line.push_str(&format!(" {}m{:02}s", elapsed / 60, elapsed % 60));
    }
    let cost = match task.state {
        TaskState::Running => live,
        _ => task.started.map(|_| task.cost),
    };
    if let Some(cost) = cost {
        line.push_str(&format!(" ${cost:.3}"));
    }
    let text = if task.outcome.is_empty() {
        &task.prompt
    } else {
        &task.outcome
    };
    let first = text.lines().next().unwrap_or_default();
    line.push_str(&format!("  {first}"));
    line
}

/// Starts pending tasks, in order, while fewer than `concurrency` run: a
/// task that ran before runs again in its agent, a new one gets a helper
/// agent spawned from the agent that queued it.
fn start_next(
    mut queue: ResMut<TaskQueue>,
    settings: Res<QueueSettings>,
    agents: Query<(Entity, &QueuedTask)>,
    ids: Query<(Entity, &AgentId), With<Agent>>,
    primary_agents: MainAgentQuery,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let running = |queue: &TaskQueue| {
        let running = queue.tasks.iter();
        running
            .filter(|task| task.state == TaskState::Running)
            .count()
    };
    while running(&queue) < settings.concurrency.max(1) {
        // Found before the queue is borrowed mutably, so an idle queue is
        // not marked changed, which would rerun this and redraw each update.
        let next = queue
            .tasks
            .iter()
            .position(|task| task.state == TaskState::Pending);
        let Some(task) = next.and_then(|next| queue.tasks.get_mut(next)) else {
            break;
        };
        task.state = TaskState::Running;
        task.started = Some(now());
        let agent = agents
            .iter()
            .find_map(|(agent, queued)| (queued.0 == task.number).then_some(agent));
        if let Some(agent) = agent {
            commands.trigger(brief(agent, task));
            continue;
        }
        let owner = AgentId(task.owner.clone()).find_in(ids.iter());
        let Some(owner) = owner.or_else(|| main_agent(&primary_agents)) else {
            task.state = TaskState::Failed;
            task.outcome = "No agent to start it from.".to_owned();
            continue;
        };
        let request = SpawnSubagent {
            model: task.model.clone(),
            ..SpawnSubagent::helper(title(task.number))
        };
        commands.spawn((request, ForAgent(owner), TaskRequest(task.number)));
        let text = format!("Started task {}.", task.number);
        notices.write(Notice::info(owner, text));
    }
}

/// Task `task`'s prompt, for its agent, as the user's words.
fn brief(agent: Entity, task: &Task) -> Deliver {
    let author = Author::user().titled(title(task.number));
    Deliver::new(agent, task.prompt.clone(), DeliveryMode::Queue).with_author(author)
}

/// A task's agent was spawned, or refused: it gets its prompt, unless the
/// task was cancelled meanwhile, or the task failed.
fn adopt(
    settled: On<Add<Settled>>,
    requests: Query<(&TaskRequest, &Outcome, Option<&SpawnedAgent>)>,
    mut queue: ResMut<TaskQueue>,
    mut note: ResMut<QueueNote>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((&TaskRequest(number), outcome, spawned)) = requests.get(settled.entity) else {
        return;
    };
    let Ok(task) = find(&mut queue, number) else {
        return;
    };
    match (spawned, &outcome.0) {
        (Some(&SpawnedAgent(agent)), _) => {
            commands.entity(agent).insert(QueuedTask(number));
            if task.state == TaskState::Running {
                commands.trigger(brief(agent, task));
            }
        }
        (None, result) => {
            let why = result.as_ref().err().map(|error| error.message.clone());
            task.state = TaskState::Failed;
            task.ended = Some(now());
            task.outcome = why.unwrap_or_else(|| "Its agent did not start.".to_owned());
            note.0 = format!("task {number} failed");
            notices.write(Notice::error(
                None,
                format!("Task {number} failed: {}", task.outcome),
            ));
        }
    }
}

/// A running task's turn ended: the task is done, failed or interrupted,
/// and the user hears so.
fn finish(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent, Option<&TurnUsage>), With<Turn>>,
    agents: Query<&QueuedTask>,
    mut queue: ResMut<TaskQueue>,
    mut note: ResMut<QueueNote>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((outcome, &ForAgent(agent), spent)) = turns.get(settled.entity) else {
        return;
    };
    let Ok(&QueuedTask(number)) = agents.get(agent) else {
        return;
    };
    let Ok(task) = find(&mut queue, number) else {
        return;
    };
    if task.state != TaskState::Running {
        return;
    }
    let (state, outcome) = match outcome {
        TurnOutcome::Answered(message) => {
            let answer = message.final_answer().unwrap_or_default();
            (TaskState::Done, answer)
        }
        TurnOutcome::Failed(why) => (TaskState::Failed, why.clone()),
        TurnOutcome::Stopped(why) => {
            let why = why.as_deref().unwrap_or("stopped");
            (TaskState::Interrupted, why.to_owned())
        }
    };
    task.state = state;
    task.ended = Some(now());
    task.cost = spent.map_or(0.0, |spent| spent.0.cost);
    task.outcome = outcome.lines().next().unwrap_or_default().to_owned();
    note.0 = format!("task {number} {}", state.label());
    notices.write(match state {
        TaskState::Done => Notice::info(None, format!("Task {number} done: {}", task.outcome)),
        _ => Notice::warn(
            None,
            format!("Task {number} {}: {}", state.label(), task.outcome),
        ),
    });
}

/// The tasks running when rig quit are interrupted: they wait for `/queue
/// retry`.
fn interrupt_running(mut queue: ResMut<TaskQueue>, mut notices: MessageWriter<Notice>) {
    for task in &mut queue.tasks {
        if task.state == TaskState::Running {
            task.state = TaskState::Interrupted;
            task.outcome = "rig quit while it ran".to_owned();
            let text = format!(
                "Task {} was interrupted when rig quit; /queue retry {} runs it again.",
                task.number, task.number
            );
            notices.write(Notice::warn(None, text));
        }
    }
}

/// Says that tasks running side by side share the working tree.
fn warn_shared_tree(settings: Res<QueueSettings>, mut notices: MessageWriter<Notice>) {
    if settings.concurrency > 1 {
        let text = format!(
            "The task queue runs {} tasks at once; they share the working tree.",
            settings.concurrency
        );
        notices.write(Notice::warn(None, text));
    }
}

/// Titles each question a task's agent asks with the task, so the user
/// knows which task asks.
fn label_asks(
    mut asks: Query<(&mut Ask, &ForAgent), (Added<Admitted>, PluginEffects)>,
    agents: Query<&QueuedTask>,
) {
    for (mut ask, &ForAgent(agent)) in &mut asks {
        if let Ok(&QueuedTask(number)) = agents.get(agent) {
            ask.title = format!("[{}] {}", title(number), ask.title);
        }
    }
}

#[cfg(test)]
mod tests;
