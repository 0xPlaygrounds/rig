//! Scheduled prompts: recurring or one-off jobs that each run as a fresh
//! helper agent named `cron:<id>`, whose final answer reaches the agent that
//! scheduled the job as a note and is shown as a notice.
//!
//! `/cron add "<schedule>" <prompt>` (with `--catch-up` before the
//! schedule to run a tick missed while the agent was not running once at
//! startup), `/cron list`, `/cron remove <id>` and `/cron run <id>`; the
//! agent schedules with the `schedule` tool. A schedule is `every 30m`,
//! `in 1h` (once) or five cron fields in local time. The jobs are a saved
//! resource ([`CronJobs`]) whose ticks are wall-clock times; one timer
//! entity's [`WakeAt`] is the soonest, so the loop sleeps until it. A run is
//! one-job [`SpawnSubagent`] helper, which retires once it answered.

use std::time::{SystemTime, UNIX_EPOCH};

use bevy_ecs::system::SystemParam;
use rig_basics::subagents::{SpawnSubagent, SpawnedAgent, SubagentsPlugin};
use rig_harness::clock::ClockPlugin;
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
    `minute hour day month weekday` in local time, such as `0 9 * * 1-5`), to come back to \
    something later instead of waiting. Write its prompt as a full brief, since the job's agent \
    sees nothing else.";

/// Adds `/cron`, the `schedule` tool, the jobs' timer and their runs.
#[derive(Default)]
pub struct CronPlugin;

impl Plugin for CronPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<ClockPlugin>() {
            app.add_plugins(ClockPlugin);
        }
        if !app.is_plugin_added::<SubagentsPlugin>() {
            app.add_plugins(SubagentsPlugin);
        }
        let options = ToolOptions {
            footprint: Some(Touches::Independent),
        };
        app.world_mut().spawn(CronTimer);
        app.init_resource::<CronJobs>()
            .add_command("cron", HELP, cron)
            .add_open_tool(SCHEDULE, DESCRIPTION, options, on_schedule)
            .add_observer(due)
            .add_systems(Startup, catch_up)
            .add_systems(PostUpdate, arm.run_if(resource_changed::<CronJobs>))
            .add_observer(report)
            .configure_sets(AgentPass, CronSystems.in_set(Stage::Review))
            .add_systems(AgentPass, adopt.in_set(CronSystems));
    }
}

/// The Review system that makes a spawned agent its job's run, to order
/// against.
#[derive(SystemSet, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CronSystems;

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

/// On a job's agent, saved with it: the job it runs. Its answer goes to
/// the agent that spawned it ([`SpawnedBy`]), the job's owner.
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Saved, Default)]
pub struct CronRun {
    pub job: u32,
}

/// On a job's [`SpawnSubagent`] request until its agent is spawned.
#[derive(Component)]
struct CronRequest(u32);

/// The entity whose [`WakeAt`] is the soonest tick.
#[derive(Component)]
struct CronTimer;

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

/// Wall time, in milliseconds since the Unix epoch.
fn now() -> u64 {
    let since = SystemTime::now().duration_since(UNIX_EPOCH);
    since.map_or(0, |since| {
        u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
    })
}

/// What starts a job's runs and tells its owner.
#[derive(SystemParam)]
struct Runs<'w, 's> {
    agents: Query<'w, 's, (Entity, &'static AgentId), With<Agent>>,
    primary: MainAgentQuery<'w, 's>,
    running: Query<'w, 's, &'static CronRun>,
    requested: Query<'w, 's, &'static CronRequest>,
    notices: MessageWriter<'w, Notice>,
    commands: Commands<'w, 's>,
}

impl Runs<'_, '_> {
    /// The agent `owner` names, else the one the user talks to.
    fn owner(&self, owner: &str) -> Option<Entity> {
        AgentId(owner.to_owned())
            .find_in(self.agents.iter())
            .or_else(|| main_agent(&self.primary))
    }

    fn running(&self, job: u32) -> bool {
        self.running.iter().any(|run| run.job == job)
            || self.requested.iter().any(|request| request.0 == job)
    }

    /// Asks for `job`'s helper agent, briefed with its prompt, unless its
    /// last run still works.
    fn start(&mut self, job: &CronJob) {
        let Some(owner) = self.owner(&job.owner) else {
            return;
        };
        if self.running(job.id) {
            let text = format!("{} is still running; skipped this tick.", name(job.id));
            self.notices.write(Notice::info(owner, text));
            return;
        }
        let request = SpawnSubagent::helper(name(job.id))
            .brief(job.prompt.clone())
            .once();
        self.commands
            .spawn((request, ForAgent(owner), CronRequest(job.id)));
    }

    /// Runs the jobs due by `at`, or at startup those missed meanwhile with
    /// `catch_up`, and moves each to its next tick after now; a job that
    /// never runs again goes.
    fn tick(&mut self, jobs: &mut CronJobs, at: u64, fire: Fire) {
        let now = now();
        let mut kept = Vec::new();
        for mut job in std::mem::take(&mut jobs.jobs) {
            if job.next > at {
                kept.push(job);
                continue;
            }
            let Ok(schedule) = Schedule::parse(&job.schedule) else {
                continue;
            };
            if fire == Fire::Startup && !job.catch_up {
                if let Some(owner) = self.owner(&job.owner) {
                    let text =
                        format!("{} missed a tick while not running; skipped.", name(job.id));
                    self.notices.write(Notice::info(owner, text));
                }
            } else {
                self.start(&job);
            }
            if let Some(next) = schedule.next(Some(job.next), now) {
                job.next = next;
                kept.push(job);
            }
        }
        jobs.jobs = kept;
    }
}

/// Adds a job for `owner`, its first tick after now.
fn add(
    jobs: &mut CronJobs,
    (schedule, prompt, catch_up): (&str, &str, bool),
    owner: &AgentId,
) -> Result<CronJob, String> {
    let (schedule, prompt) = (schedule.trim(), prompt.trim());
    if prompt.is_empty() {
        return Err("The job needs a prompt.".to_owned());
    }
    let next = Schedule::parse(schedule)?
        .next(None, now())
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
fn cron(In(args): In<CommandArgs>, mut jobs: ResMut<CronJobs>, mut runs: Runs) {
    let agent = args.agent;
    let (verb, rest) = args.args.split_once(' ').unwrap_or((&args.args, ""));
    let said = match verb {
        "" | "list" => Ok(list(&jobs, &runs)),
        "add" => split_add(rest).and_then(|(schedule, prompt, catch_up)| {
            let owner = runs.agents.get(agent).map(|(_, id)| id.clone());
            let owner = owner.map_err(|_| "Not an agent.".to_owned())?;
            let job = add(&mut jobs, (&schedule, &prompt, catch_up), &owner)?;
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
            match (verb, at.and_then(|at| jobs.jobs.get(at).cloned())) {
                (_, None) => Err(format!("No job `{}`; /cron list shows them.", rest.trim())),
                ("remove", Some(job)) => {
                    jobs.jobs.retain(|kept| kept.id != job.id);
                    Ok(format!("Removed job {}.", name(job.id)))
                }
                (_, Some(job)) => {
                    runs.start(&job);
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

/// Adds the job a `schedule` call asks for. A job's own agent schedules for
/// that job's owner, since it retires once it answered.
fn on_schedule(
    called: On<ToolCalled<ScheduleArgs>>,
    mut jobs: ResMut<CronJobs>,
    runs: Query<&SpawnedBy, With<CronRun>>,
    ids: Query<&AgentId>,
    mut calls: Query<&mut Decision, ToolCalls>,
) {
    let args = &called.args;
    let owner = runs
        .get(called.agent)
        .and_then(|&SpawnedBy(owner)| ids.get(owner));
    let owner = owner.cloned().unwrap_or_else(|_| called.agent_id.clone());
    let fields = (args.schedule.as_str(), args.prompt.as_str(), args.catch_up);
    let result = match add(&mut jobs, fields, &owner) {
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
    if let Ok(mut call) = calls.get_mut(called.call) {
        call.decide(Decider::plugin("cron-agent"), result);
    }
}

/// The timer says the soonest tick passed: runs the jobs due by then or by
/// now, whichever is later.
fn due(
    due: On<Add<Due>>,
    timers: Query<(), With<CronTimer>>,
    mut jobs: ResMut<CronJobs>,
    mut runs: Runs,
) {
    if timers.contains(due.entity) {
        let soonest = jobs.jobs.iter().map(|job| job.next).min();
        let at = soonest.unwrap_or_default().max(now());
        runs.tick(&mut jobs, at, Fire::Due);
    }
}

/// At startup, the ticks missed while the agent was not running.
fn catch_up(mut jobs: ResMut<CronJobs>, mut runs: Runs) {
    runs.tick(&mut jobs, now(), Fire::Startup);
}

/// Sets the timer to the soonest tick, or takes it off with no jobs left.
fn arm(jobs: Res<CronJobs>, timer: Single<Entity, With<CronTimer>>, mut commands: Commands) {
    let mut timer = commands.entity(*timer);
    match jobs.jobs.iter().map(|job| job.next).min() {
        Some(next) => timer.insert(WakeAt(UNIX_EPOCH + Duration::from_millis(next))),
        None => timer.remove::<(WakeAt, Due)>(),
    };
}

/// A job's agent was spawned, or refused: it is the job's run from now on,
/// or the owner hears why there is none.
fn adopt(
    requests: Query<
        (
            Entity,
            &CronRequest,
            &Outcome,
            &ForAgent,
            Option<&SpawnedAgent>,
        ),
        (ReadyToSettle, PluginEffects),
    >,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    for (request, &CronRequest(job), outcome, &ForAgent(owner), spawned) in &requests {
        commands.entity(request).remove::<CronRequest>();
        match (spawned, &outcome.0) {
            (Some(&SpawnedAgent(agent)), _) => {
                commands.entity(agent).insert(CronRun { job });
                notices.write(Notice::info(owner, format!("{} started.", name(job))));
            }
            (None, Err(why)) => {
                let text = format!("{} could not start: {}", name(job), why.message);
                notices.write(Notice::error(owner, text));
            }
            (None, Ok(_)) => {}
        }
    }
}

/// A job's agent ended its one turn, and retires: its answer goes to the
/// job's owner as a note and a notice.
fn report(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent)>,
    runs: Query<(&CronRun, &SpawnedBy, &AgentId)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((outcome, &ForAgent(agent))) = turns.get(settled.entity) else {
        return;
    };
    let Ok((run, &SpawnedBy(owner), from)) = runs.get(agent) else {
        return;
    };
    let job = name(run.job);
    match outcome {
        TurnOutcome::Answered(message) => {
            let answer = message
                .final_answer()
                .unwrap_or_else(|| "(no answer)".to_owned());
            notices.write(Notice::info(owner, format!("{job}: {answer}")));
            let author = Author::agent(from.clone(), None).titled(job);
            let note = Deliver::new(owner, answer, DeliveryMode::Inform);
            commands.trigger(note.with_author(author));
        }
        TurnOutcome::Failed(why) => {
            notices.write(Notice::error(owner, format!("{job} failed: {why}")));
        }
        TurnOutcome::Stopped(why) => {
            let why = why.as_deref().unwrap_or("by you");
            notices.write(Notice::info(owner, format!("{job} was stopped {why}.")));
        }
    }
}

#[cfg(test)]
mod tests;
