//! Scheduled prompts: recurring or one-off jobs that each run as a fresh
//! subagent named `cron:<id>`, whose final answer reaches the agent that
//! scheduled the job as a note and is shown as a notice.
//!
//! `/cron add "<schedule>" <prompt>` (with `--catch-up` before the
//! schedule to run a tick missed while the agent was not running once at
//! startup), `/cron list`, `/cron remove <id>` and `/cron run <id>`; the
//! agent schedules with the `schedule` tool. A schedule is `every 30m`,
//! `in 1h` (once) or five cron fields in local time. The jobs are a saved
//! resource ([`CronJobs`]). The loop sleeps until the soonest tick: one
//! delayed command on Bevy's clock, armed again whenever the jobs change.
//! The clock is read as wall time from the wall time at startup, so a test's
//! manual clock moves it too.

use bevy_ecs::system::SystemParam;
use rig_harness::prelude::*;
use schemars::JsonSchema;
use serde::Deserialize;

mod schedule;

use schedule::Schedule;

/// The tool the agent schedules a job with.
pub const SCHEDULE: &str = "schedule";

const HELP: &str = "Scheduled prompts: /cron add [--catch-up] \"<schedule>\" <prompt>, \
    /cron list, /cron remove <id>, /cron run <id>";

const DESCRIPTION: &str = "Schedule a prompt to run later, once or again and again, in a \
    fresh subagent of its own; its final answer comes back to you as a note. Use it for \
    \"check again in an hour\" (`in 1h`) or recurring work (`every 30m`, or five cron fields \
    `minute hour day month weekday` in local time, such as `0 9 * * 1-5`).";

const RULES: &[&str] = &[
    "Use `schedule` to come back to something later instead of waiting; write its prompt as a \
     full brief, since the job's agent sees nothing else.",
];

/// Adds `/cron`, the `schedule` tool, the jobs' timer and their runs.
#[derive(Default)]
pub struct CronPlugin;

impl Plugin for CronPlugin {
    fn build(&self, app: &mut App) {
        let options = ToolOptions {
            rules: RULES,
            footprint: Footprint::Independent,
        };
        app.init_resource::<CronJobs>()
            .init_resource::<Clock>()
            .init_resource::<Armed>()
            .add_command("cron", HELP, cron)
            .add_open_tool(SCHEDULE, DESCRIPTION, options, on_schedule)
            .add_observer(report)
            .add_observer(on_restored)
            .add_systems(PostStartup, catch_up)
            .add_systems(Last, arm.run_if(resource_changed::<CronJobs>));
    }
}

/// The scheduled jobs, saved with the session.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved, Default)]
pub struct CronJobs {
    /// The id the next job gets.
    pub next_id: u32,
    /// The jobs, oldest first.
    pub jobs: Vec<CronJob>,
}

/// A scheduled prompt.
#[derive(Reflect, Clone, Debug, Default)]
pub struct CronJob {
    pub id: u32,
    /// As written: `every 30m`, `in 1h` or five cron fields.
    pub schedule: String,
    pub prompt: String,
    /// The [`AgentId`] of the agent that gets the answers.
    pub owner: String,
    /// Whether a tick missed while the agent was not running runs once at
    /// startup.
    pub catch_up: bool,
    /// When it runs next, in milliseconds since the Unix epoch.
    pub next: u64,
}

/// On a job's subagent, saved with it: the job it runs and the agent its
/// answer goes to.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default)]
pub struct CronRun {
    pub job: u32,
    pub owner: String,
}

/// Wall time as the app's clock moves it: the wall time at elapsed zero,
/// moved forward when the wall clock ran ahead, such as after a suspend.
#[derive(Resource)]
struct Clock(u64);

impl Default for Clock {
    fn default() -> Self {
        Self(rig_ecs::journal::now_ms())
    }
}

impl Clock {
    fn now(&mut self, time: &Time) -> u64 {
        let elapsed = u64::try_from(time.elapsed().as_millis()).unwrap_or(u64::MAX);
        let wall = rig_ecs::journal::now_ms().saturating_sub(elapsed);
        self.0 = self.0.max(wall);
        self.0.saturating_add(elapsed)
    }
}

/// The generation of the timer armed last; an older one does nothing.
#[derive(Resource, Default)]
struct Armed(u64);

/// Why the jobs are looked at.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Fire {
    /// The timer went off.
    Due,
    /// The app started: a tick missed meanwhile runs only with `catch_up`.
    Startup,
}

fn name(job: u32) -> String {
    format!("cron:{job}")
}

/// What starts a job's runs and tells its owner.
#[derive(SystemParam)]
struct Runs<'w, 's> {
    agents: Query<
        'w,
        's,
        (
            Entity,
            &'static AgentId,
            Option<&'static Connection>,
            Option<&'static ModelChoice>,
            &'static Effort,
            &'static SystemPrompt,
            &'static ToolAccess,
        ),
        With<Agent>,
    >,
    primary: PrimaryQuery<'w, 's>,
    running: Query<'w, 's, &'static CronRun>,
    notices: MessageWriter<'w, Notice>,
    commands: Commands<'w, 's>,
}

impl Runs<'_, '_> {
    /// The agent `owner` names, else the one the user talks to.
    fn owner(&self, owner: &str) -> Option<Entity> {
        let ids = self.agents.iter().map(|(entity, id, ..)| (entity, id));
        AgentId(owner.to_owned())
            .find_in(ids)
            .or_else(|| primary(&self.primary))
    }

    fn running(&self, job: u32) -> bool {
        self.running.iter().any(|run| run.job == job)
    }

    /// Starts `job` in a fresh subagent of its owner, on the owner's model,
    /// unless its last run still works.
    fn start(&mut self, job: &CronJob) {
        let Some(owner) = self.owner(&job.owner) else {
            return;
        };
        if self.running(job.id) {
            let text = format!("{} is still running; skipped this tick.", name(job.id));
            self.notices.write(Notice::info(owner, text));
            return;
        }
        let Ok((_, id, connection, Some(choice), effort, prompt, access)) = self.agents.get(owner)
        else {
            self.notices.write(Notice::no_model(owner));
            return;
        };
        let mut child = self.commands.spawn_empty();
        // Before the model choice, so connecting it keeps this connection.
        if let Some(connection) = connection {
            child.insert(connection.clone());
        }
        let run = CronRun {
            job: job.id,
            owner: id.0.clone(),
        };
        let parts = (choice.clone(), *effort, prompt.clone(), access.clone());
        child.insert((Agent, Name::new(name(job.id)), SpawnedBy(owner), run, parts));
        let child = child.id();
        let origin = Origin::plugin("cron").titled(name(job.id));
        let brief = Deliver::new(child, job.prompt.clone(), DeliveryMode::Queue);
        self.commands.trigger(brief.with_origin(origin));
        let text = format!("{} started: {}", name(job.id), job.prompt);
        self.notices.write(Notice::info(owner, text));
    }
}

/// Adds a job for `owner`, its first tick after `now`.
fn add(
    jobs: &mut CronJobs,
    (schedule, prompt, catch_up): (&str, &str, bool),
    owner: &AgentId,
    now: u64,
) -> Result<CronJob, String> {
    let (schedule, prompt) = (schedule.trim(), prompt.trim());
    if prompt.is_empty() {
        return Err("The job needs a prompt.".to_owned());
    }
    let next = Schedule::parse(schedule)?
        .next(None, now)
        .ok_or_else(|| format!("`{schedule}` never runs."))?;
    jobs.next_id = jobs.next_id.saturating_add(1);
    let job = CronJob {
        id: jobs.next_id,
        schedule: schedule.to_owned(),
        prompt: prompt.to_owned(),
        owner: owner.0.clone(),
        catch_up,
        next,
    };
    jobs.jobs.push(job.clone());
    Ok(job)
}

/// A time for the user, in local time.
fn when(at: u64) -> String {
    use chrono::TimeZone;
    let at = i64::try_from(at).ok();
    let at = at.and_then(|at| chrono::Local.timestamp_millis_opt(at).single());
    at.map_or_else(
        || "never".to_owned(),
        |at| at.format("%Y-%m-%d %H:%M:%S").to_string(),
    )
}

/// `--catch-up "<schedule>" <prompt>`, the flag optional; `every 30m` and
/// `in 1h` may go without quotes.
fn split_add(args: &str) -> Result<(String, String, bool), String> {
    let args = args.trim();
    let (catch_up, args) = match args.strip_prefix("--catch-up") {
        Some(rest) => (true, rest.trim_start()),
        None => (false, args),
    };
    let (schedule, prompt) = if let Some(quoted) = args.strip_prefix('"') {
        quoted
            .split_once('"')
            .ok_or("The schedule's quote is not closed.")?
    } else {
        let mut words = args.splitn(3, ' ');
        match (words.next(), words.next(), words.next()) {
            (Some(kind @ ("every" | "in")), Some(span), rest) => {
                let at = kind.len() + 1 + span.len();
                (args.get(..at).unwrap_or_default(), rest.unwrap_or_default())
            }
            _ => return Err("Put the schedule in quotes: /cron add \"0 9 * * *\" <prompt>".into()),
        }
    };
    Ok((schedule.to_owned(), prompt.to_owned(), catch_up))
}

/// `/cron …`.
fn cron(
    In(args): In<CommandArgs>,
    mut jobs: ResMut<CronJobs>,
    time: Res<Time>,
    mut clock: ResMut<Clock>,
    mut runs: Runs,
) {
    let agent = args.agent;
    let (verb, rest) = args.args.split_once(' ').unwrap_or((&args.args, ""));
    let said = match verb {
        "" | "list" => Ok(list(&jobs, &runs)),
        "add" => split_add(rest).and_then(|(schedule, prompt, catch_up)| {
            let owner = runs.agents.get(agent).map(|(_, id, ..)| id.clone());
            let owner = owner.map_err(|_| "Not an agent.".to_owned())?;
            let now = clock.now(&time);
            let job = add(&mut jobs, (&schedule, &prompt, catch_up), &owner, now)?;
            Ok(format!(
                "Added job {} ({}), next at {}.",
                name(job.id),
                job.schedule,
                when(job.next)
            ))
        }),
        "remove" | "run" => {
            let id = rest.trim().parse::<u32>().ok();
            let at = id.and_then(|id| jobs.jobs.iter().position(|job| job.id == id));
            match (verb, at) {
                (_, None) => Err(format!("No job `{}`; /cron list shows them.", rest.trim())),
                ("remove", Some(at)) => {
                    let job = jobs.jobs.remove(at);
                    Ok(format!("Removed job {}.", name(job.id)))
                }
                (_, Some(at)) => {
                    if let Some(job) = jobs.jobs.get(at).cloned() {
                        runs.start(&job);
                    }
                    return;
                }
            }
        }
        _ => Err(HELP.to_owned()),
    };
    let notice = match said {
        Ok(text) => Notice::info(agent, text),
        Err(why) => Notice::error(agent, why),
    };
    runs.notices.write(notice);
}

/// The jobs, each with its next tick.
fn list(jobs: &CronJobs, runs: &Runs) -> String {
    if jobs.jobs.is_empty() {
        return "No scheduled prompts; /cron add \"every 30m\" <prompt> adds one.".to_owned();
    }
    let mut lines = vec!["Scheduled prompts:".to_owned()];
    for job in &jobs.jobs {
        let running = if runs.running(job.id) {
            " (running)"
        } else {
            ""
        };
        let catch_up = if job.catch_up { ", catch-up" } else { "" };
        lines.push(format!(
            "  {} \"{}\"{catch_up}, next {}{running}: {}",
            name(job.id),
            job.schedule,
            when(job.next),
            job.prompt
        ));
    }
    lines.join("\n")
}

/// The arguments of a `schedule` call.
#[derive(Deserialize, JsonSchema)]
#[serde(deny_unknown_fields)]
struct ScheduleArgs {
    /// `every 30m` (also s, h, d), `in 1h` for once, or five cron fields
    /// `minute hour day month weekday` in local time, such as `0 9 * * *`.
    schedule: String,
    /// The job's prompt in full: its agent sees nothing else.
    prompt: String,
    /// Run a tick missed while the agent was not running once at startup.
    #[serde(default)]
    catch_up: bool,
}

/// Adds the job a `schedule` call asks for. A job's own subagent schedules
/// for that job's owner, since it is gone once it answered.
fn on_schedule(
    called: On<ToolCalled<ScheduleArgs>>,
    mut jobs: ResMut<CronJobs>,
    time: Res<Time>,
    mut clock: ResMut<Clock>,
    runs: Query<&CronRun>,
    mut commands: Commands,
) {
    let args = &called.args;
    let owner = runs.get(called.agent).map(|run| AgentId(run.owner.clone()));
    let owner = owner.unwrap_or_else(|_| called.caller.clone());
    let now = clock.now(&time);
    let fields = (args.schedule.as_str(), args.prompt.as_str(), args.catch_up);
    let output = match add(&mut jobs, fields, &owner, now) {
        Ok(job) => ToolResult::success(
            format!(
                "Scheduled job {} ({}), next at {}. Its answer will arrive as a note.",
                name(job.id),
                job.schedule,
                when(job.next)
            )
            .into(),
        ),
        Err(why) => ToolResult::failed(ToolExecutionError::other(why)),
    };
    commands
        .entity(called.call)
        .insert_if_new(ToolOutput(output));
}

/// Runs the jobs that are due, or at startup those missed meanwhile with
/// `catch_up`, and moves each to its next tick; a job that never runs
/// again goes.
fn tick(
    In(fire): In<Fire>,
    mut jobs: ResMut<CronJobs>,
    time: Res<Time>,
    mut clock: ResMut<Clock>,
    mut runs: Runs,
) {
    let now = clock.now(&time);
    if !jobs.jobs.iter().any(|job| job.next <= now) {
        return;
    }
    let mut kept = Vec::new();
    for mut job in std::mem::take(&mut jobs.jobs) {
        if job.next > now {
            kept.push(job);
            continue;
        }
        let Ok(schedule) = Schedule::parse(&job.schedule) else {
            continue;
        };
        if fire == Fire::Startup && !job.catch_up {
            if let Some(owner) = runs.owner(&job.owner) {
                let text = format!("{} missed a tick while not running; skipped.", name(job.id));
                runs.notices.write(Notice::info(owner, text));
            }
        } else {
            runs.start(&job);
        }
        if let Some(next) = schedule.next(Some(job.next), now) {
            job.next = next;
            kept.push(job);
        }
    }
    jobs.jobs = kept;
}

/// At startup, the ticks missed while the agent was not running.
fn catch_up(mut commands: Commands) {
    commands.run_system_cached_with(tick, Fire::Startup);
}

/// Sleeps until the soonest tick: one delayed command, which an older
/// arming's does nothing beside.
fn arm(
    jobs: Res<CronJobs>,
    time: Res<Time>,
    mut clock: ResMut<Clock>,
    mut armed: ResMut<Armed>,
    mut commands: Commands,
) {
    armed.0 = armed.0.wrapping_add(1);
    let Some(next) = jobs.jobs.iter().map(|job| job.next).min() else {
        return;
    };
    let wait = Duration::from_millis(next.saturating_sub(clock.now(&time)));
    let generation = armed.0;
    let mut delayed = commands.delayed();
    delayed.duration(wait).queue(move |world: &mut World| {
        if world
            .get_resource::<Armed>()
            .is_some_and(|armed| armed.0 == generation)
        {
            world.run_system_cached_with(tick, Fire::Due).ok();
        }
    });
}

/// A job's subagent ended its turn: its answer goes to the job's owner as a
/// note and a notice, and the subagent goes.
fn report(
    ended: On<TurnEnded>,
    runs: Query<(&CronRun, &AgentId)>,
    ids: Query<(Entity, &AgentId)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let agent = ended.entity;
    let Ok((run, id)) = runs.get(agent) else {
        return;
    };
    commands.entity(agent).despawn();
    let Some(owner) = AgentId(run.owner.clone()).find_in(ids) else {
        return;
    };
    let job = name(run.job);
    match &ended.outcome {
        TurnOutcome::Answered(message) => {
            let answer = final_answer(message).unwrap_or_else(|| "(no answer)".to_owned());
            notices.write(Notice::info(owner, format!("{job}: {answer}")));
            let origin = Origin::agent(id.clone(), None).titled(job);
            let note = Deliver::new(owner, answer, DeliveryMode::Note);
            commands.trigger(note.with_origin(origin));
        }
        TurnOutcome::Failed(why) => {
            notices.write(Notice::error(owner, format!("{job} failed: {why}")));
        }
        TurnOutcome::Stopped => {
            notices.write(Notice::info(owner, format!("{job} was stopped.")));
        }
    }
}

/// A job's subagent comes back with the session: one that had answered
/// goes again, since the session log keeps every agent it ever had; one cut
/// short carries on under its name.
fn on_restored(
    mut restored: On<Restored>,
    runs: Query<(&CronRun, &Conversation)>,
    mut commands: Commands,
) {
    let agent = restored.entity;
    let Ok((run, conversation)) = runs.get(agent) else {
        return;
    };
    if conversation
        .messages()
        .last()
        .and_then(final_answer)
        .is_some()
    {
        restored.event_mut().resume = false;
        commands.entity(agent).despawn();
    } else {
        commands.entity(agent).insert(Name::new(name(run.job)));
    }
}

#[cfg(test)]
mod tests;
