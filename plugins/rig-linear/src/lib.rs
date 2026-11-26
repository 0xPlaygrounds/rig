//! Mirrors a run into a Linear agent session, so the people watching an
//! issue see the agent's plan, notes, tool calls and result, and can write
//! to the agent or stop it from there. The plugin does nothing, and sends
//! nothing, unless it has both a token (the secret `linear`, or
//! `linear-client-id` and `linear-client-secret`) and a target
//! (`--linear-issue`, `--linear-session` or their variables). The crate's
//! README lists the settings, the plan file and what is sent and read when.
//!
//! ```no_run
//! use rig_harness::prelude::*;
//! use rig_linear::LinearPlugin;
//!
//! let mut app = App::new();
//! app.insert_resource(Secrets::from_iter([("linear", "lin_api_…")]))
//!     .add_plugins(LinearPlugin);
//! ```

use std::collections::HashSet;
use std::time::Duration;

use async_channel::{Receiver, Sender};
use bevy_log::warn;
use rig_core::effect::{Answered, EffectKind, EffectRecord};
use rig_core::tool::{ToolExecutionError, ToolOutput};
use rig_harness::prelude::*;
use rig_harness::rig_ecs::StopTurnsOnExit;
use rig_harness::rig_ecs::agent::ToolAccess;
use schemars::JsonSchema;
use serde::Deserialize;

pub use crate::plan::PlanStep;

use crate::activity::{Activity, Call, Content, Redactor, Reply, head, pull_request};
use crate::api::{Credentials, Link};
use crate::plan::{End, Plan};
use crate::send::{Config, Job, Started, Target, Told};

mod activity;
mod api;
mod plan;
mod send;

#[cfg(test)]
mod tests;

/// The option that shows the run on a new agent session of this issue.
pub const ISSUE_FLAG: &str = "linear-issue";

/// The option that shows the run on this existing agent session.
pub const SESSION_FLAG: &str = "linear-session";

/// The variable read when [`ISSUE_FLAG`] is not given.
pub const ISSUE_ENV: &str = "RIG_LINEAR_ISSUE";

/// The variable read when [`SESSION_FLAG`] is not given.
pub const SESSION_ENV: &str = "RIG_LINEAR_SESSION";

/// The option that claims the issue: delegates it to the app and moves it
/// to started. Leave it off for a run an automation started, so the issue
/// stays in triage and a person assigns it.
pub const CLAIM_FLAG: &str = "linear-claim";

/// The variable that claims the issue as [`CLAIM_FLAG`] does, when `1` or
/// `true`.
pub const CLAIM_ENV: &str = "RIG_LINEAR_CLAIM";

/// The variable naming a TOML plan file, which replaces `[linear] plan`.
pub const PLAN_ENV: &str = "RIG_LINEAR_PLAN";

/// The secret holding an access token: an OAuth or app token, or a
/// personal API key.
pub const TOKEN_SECRET: &str = "linear";

/// The secret holding an OAuth application's client id.
pub const CLIENT_ID_SECRET: &str = "linear-client-id";

/// The secret holding an OAuth application's client secret.
pub const CLIENT_SECRET_SECRET: &str = "linear-client-secret";

/// The tool with which the agent ends the run.
pub const OUTCOME_TOOL: &str = "report_outcome";

/// Linear's GraphQL endpoint.
pub const DEFAULT_API_URL: &str = "https://api.linear.app/graphql";

/// The scopes the client credentials ask for by default.
pub const DEFAULT_SCOPE: &str = "read,write,app:assignable,app:mentionable";

/// The label of the session's own link, which the plugin publishes as a
/// [`SessionLink`].
pub const SESSION_LINK_LABEL: &str = "Linear agent session";

/// The tag of the prompt section.
const SECTION_TAG: &str = "linear";

/// The plugin's name as a decider of its tool's calls.
const ORIGIN: &str = "linear";

/// What the run's error says when it stopped before it finished.
const STOPPED: &str = "The run was stopped before it finished.";

/// The most characters of a message the acknowledgement quotes.
const ACK_MOST: usize = 100;

/// The most answers a question offers, and the most characters of each.
const OPTIONS_MOST: (usize, usize) = (100, 255);

/// The longest the app waits on exit for what is queued to be sent.
const FLUSH_WAIT: Duration = Duration::from_secs(10);

/// The least time between two requests.
const SPACING: Duration = Duration::from_secs(1);

/// What the outcome tool says to the model.
const OUTCOME_DESCRIPTION: &str = "End the run by reporting its outcome on the Linear issue \
it is shown on. `status` is `done` when the task is complete, `gave_up` when it cannot be \
completed, `needs_input` when a person must answer first. `summary` is posted as written, in \
Markdown, for the people watching the issue: what was done or why not, and what is needed from \
them. With `needs_input`, `options` may list short answers for them to pick from. Call it \
once, as the last thing you do.";

/// Shows the run on a Linear agent session, when it has a token and a
/// target. See the [crate] docs.
#[derive(Default)]
pub struct LinearPlugin;

/// `[linear]` in `settings.toml`.
#[derive(Resource, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct LinearSettings {
    /// The GraphQL endpoint; the OAuth token endpoint is `/oauth/token` on
    /// its origin. [`DEFAULT_API_URL`] by default.
    pub api_url: String,
    /// Minutes without a request after which a heartbeat thought goes, so
    /// the session does not go stale; 10 by default.
    pub heartbeat_minutes: u64,
    /// Seconds between two reads of the session's new prompts, for a
    /// person's message or stop; 15 by default, 0 for none.
    pub poll_seconds: u64,
    /// The scopes the client credentials ask for, comma-separated;
    /// [`DEFAULT_SCOPE`] by default.
    pub scope: String,
    /// The plan's steps, unless [`PLAN_ENV`] names a file.
    pub plan: Vec<PlanStep>,
}

impl Default for LinearSettings {
    fn default() -> Self {
        Self {
            api_url: DEFAULT_API_URL.to_owned(),
            heartbeat_minutes: 10,
            poll_seconds: 15,
            scope: DEFAULT_SCOPE.to_owned(),
            plan: Vec::new(),
        }
    }
}

/// The plan file [`PLAN_ENV`] names: `[[plan]]` tables, as in settings.
#[derive(Deserialize, Debug, Default)]
#[serde(deny_unknown_fields)]
struct PlanFile {
    plan: Vec<PlanStep>,
}

/// The outcome tool's arguments.
#[derive(Deserialize, JsonSchema, Debug)]
struct OutcomeArgs {
    /// How the run ended.
    status: OutcomeStatus,
    /// What to tell the people watching the issue, in Markdown.
    summary: String,
    /// With `needs_input` only: short answers the people watching may pick
    /// from. They may still answer otherwise.
    #[serde(default)]
    options: Vec<String>,
}

/// How the run ended, as the agent reports it.
#[derive(Deserialize, JsonSchema, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
enum OutcomeStatus {
    /// The task is complete.
    Done,
    /// The task cannot be completed.
    GaveUp,
    /// A person must answer first.
    NeedsInput,
}

/// What the plugin needs to start, once it has a token and a target.
#[derive(Debug)]
struct Setup {
    config: Config,
    plan: Vec<PlanStep>,
    redactor: Redactor,
}

/// The Linear agent session id saved with this rig session, so a reload
/// reattaches to the same Linear session instead of opening a second one.
#[derive(Resource, Reflect, Default, Clone, Debug, PartialEq, Eq)]
#[reflect(Resource, Saved, Default)]
struct SavedLinearSession {
    session_id: Option<String>,
    issue: Option<String>,
}

/// The plugin's state, present only when it is active.
#[derive(Resource)]
struct Linear {
    /// The queue to the task.
    jobs: Sender<Job>,
    /// What the task tells: the session is there, a person wrote or asked
    /// to stop.
    told: Receiver<Told>,
    /// What [`start`] hands the task.
    pending: Option<(Config, Receiver<Job>, Sender<Told>, Redactor)>,
    /// Scrubs the plugin's own credentials before any truncation.
    redactor: Redactor,
    plan: Plan,
    /// The issue, as the prompt names it.
    issue: Option<String>,
    /// The issue target as requested (`--linear-issue` / `RIG_LINEAR_ISSUE`),
    /// kept so reloads match even when the target was a UUID.
    target_issue: Option<String>,
    /// The session's own link.
    session_url: Option<String>,
    /// The links sent.
    linked: HashSet<String>,
    /// Whether the agent reported the outcome since its turn started.
    reported: bool,
    /// The text of the main agent's last reply.
    last_reply: Option<String>,
    /// Why its last model call failed, if it did.
    last_error: Option<String>,
    /// Whether the main agent was in a turn as the app exited.
    working: bool,
    /// Whether nothing more is sent.
    closed: bool,
    /// Whether a person stopped the run, so nothing more is sent but the
    /// close.
    halted: bool,
}

impl Linear {
    fn new(setup: Setup) -> Self {
        let (jobs, queued) = async_channel::unbounded();
        let (tell, told) = async_channel::unbounded();
        let issue = match &setup.config.target {
            Target::Issue(issue) => Some(issue.clone()),
            Target::Session(_) | Target::Saved(_) => None,
        };
        let redactor = setup.redactor.clone();
        Self {
            jobs,
            told,
            pending: Some((setup.config, queued, tell, setup.redactor)),
            redactor,
            plan: Plan::new(setup.plan),
            target_issue: issue.clone(),
            issue,
            session_url: None,
            linked: HashSet::new(),
            reported: false,
            last_reply: None,
            last_error: None,
            working: false,
            closed: false,
            halted: false,
        }
    }

    /// Queues `job` for the task, unless the run is closed, or stopped and
    /// `job` is not the close.
    fn queue(&self, job: Job) {
        if !self.closed && (!self.halted || matches!(job, Job::Close(_))) {
            self.jobs.try_send(job).ok();
        }
    }

    /// A person on Linear, `by`, stopped the run: cancels the plan, posts the
    /// stop as the run's end, mutes everything after it, and exits the app
    /// as a signal does, which stops the turn and its tool calls.
    fn stop(&mut self, by: Option<&str>, commands: &mut Commands) {
        if self.halted {
            return;
        }
        if self.plan.end(End::Failed) {
            self.send_plan();
        }
        let by = by.unwrap_or("a person on Linear");
        let stopped = Content::Error(format!("Stopped at the request of {by}."));
        self.queue(Job::Activity(Activity::end(stopped)));
        self.reported = true;
        self.halted = true;
        commands.trigger(Exit(ExitReason::Signal));
    }

    /// Queues `activity`, unless the agent reported the outcome since its
    /// turn started.
    fn note(&self, activity: Activity) {
        if !self.reported {
            self.queue(Job::Activity(activity));
        }
    }

    /// Queues the plan whole.
    fn send_plan(&self) {
        if !self.plan.is_empty() {
            self.queue(Job::Plan(self.plan.json()));
        }
    }

    /// Posts the run's end and closes the plan to match.
    fn end(&mut self, content: Content, end: End) {
        self.queue(Job::Activity(Activity::end(content)));
        if self.plan.end(end) {
            self.send_plan();
        }
        self.reported = true;
    }

    /// Takes in one effect record: the main agent's replies and tool calls
    /// become activities, and any agent's `gh pr create` publishes its
    /// pull request.
    fn record(&mut self, record: &EffectRecord, main: bool, commands: &mut Commands) {
        let outcome = record.reviewed.as_ref().unwrap_or(&record.outcome);
        match &record.kind {
            EffectKind::Completion { .. }
                if main
                    && record.purpose.as_deref() == Some(rig_harness::rig_ecs::turn::PURPOSE) =>
            {
                match outcome {
                    Ok(Answered::Completion(response)) => {
                        let reply = Reply::of(response);
                        self.last_error = response.error.clone();
                        if !reply.text.is_empty() {
                            self.last_reply = Some(reply.text.clone());
                        }
                        for activity in reply.activities(&self.redactor) {
                            self.note(activity);
                        }
                    }
                    Err(report) => self.last_error = Some(report.message.clone()),
                    Ok(_) => {}
                }
            }
            EffectKind::ToolCall { name, args } => {
                let call = Call {
                    name,
                    args,
                    outcome,
                    published: record.published.as_ref(),
                };
                let command = (name == "shell").then(|| call.arg("command")).flatten();
                if let Some(link) = command
                    .as_deref()
                    .and_then(|command| pull_request(command, &call.output()))
                {
                    commands.spawn(link);
                }
                if !main || name == OUTCOME_TOOL || self.reported {
                    return;
                }
                let moved = match (name.as_str(), command, call.arg("path")) {
                    ("shell", Some(command), _) => self.plan.ran(&command),
                    ("read", _, Some(path)) => self.plan.read(&path),
                    _ => false,
                };
                if moved {
                    self.send_plan();
                }
                self.note(call.activity(&self.redactor));
            }
            _ => {}
        }
    }
}

impl Plugin for LinearPlugin {
    fn build(&self, app: &mut App) {
        app.add_flag(
            Flag::new(
                ISSUE_FLAG,
                "Show this run on a new Linear agent session of this issue",
            )
            .value("ID"),
        )
        .add_flag(
            Flag::new(
                SESSION_FLAG,
                "Show this run on this existing Linear agent session",
            )
            .value("ID"),
        )
        .add_flag(Flag::new(
            CLAIM_FLAG,
            "Delegate the Linear issue to the app and move it to started",
        ))
        .add_setting::<LinearSettings>("linear")
        .init_resource::<SavedLinearSession>()
        .register_type::<SavedLinearSession>();
        let env = |name: &str| std::env::var(name).ok();
        let Some(target) = target(app.world().get_resource::<Invoked>(), &env) else {
            return;
        };
        let world = app.world();
        let claims = claims(world.get_resource::<Invoked>(), &env);
        let Some(setup) = configure(
            target,
            claims,
            world.get_resource::<Secrets>(),
            world.get_resource::<LinearSettings>(),
            &env,
        ) else {
            return;
        };
        app.insert_resource(Linear::new(setup))
            .add_open_tool(
                OUTCOME_TOOL,
                OUTCOME_DESCRIPTION,
                ToolOptions::default(),
                report_outcome,
            )
            .add_observer(publish_link)
            .add_observer(turn_started)
            .add_systems(Startup, start)
            .add_systems(Update, (take_told, sync_section).chain())
            .add_systems(AgentPass, sync_section.in_set(PrepareSystems::Input))
            .add_systems(
                Last,
                (
                    note_working
                        .before(StopTurnsOnExit)
                        .run_if(on_message::<AppExit>),
                    (read_records, close.run_if(on_message::<AppExit>))
                        .chain()
                        .after(StopTurnsOnExit)
                        .after(RecordRedactSystems),
                ),
            );
    }
}

/// Text that is there and not blank.
fn given(value: Option<String>) -> Option<String> {
    value.filter(|value| !value.trim().is_empty())
}

/// The session the run shows on, if any: the options before `env`'s
/// variables, a session before an issue.
fn target(invoked: Option<&Invoked>, env: &dyn Fn(&str) -> Option<String>) -> Option<Target> {
    let flag =
        |name| given(invoked.and_then(|invoked| invoked.flag_value(name).map(str::to_owned)));
    flag(SESSION_FLAG)
        .map(Target::Session)
        .or_else(|| flag(ISSUE_FLAG).map(Target::Issue))
        .or_else(|| given(env(SESSION_ENV)).map(Target::Session))
        .or_else(|| given(env(ISSUE_ENV)).map(Target::Issue))
}

/// Whether the run claims its issue: [`CLAIM_FLAG`] given, or
/// [`CLAIM_ENV`] `1` or `true`.
fn claims(invoked: Option<&Invoked>, env: &dyn Fn(&str) -> Option<String>) -> bool {
    let set = env(CLAIM_ENV).is_some_and(|value| {
        let value = value.trim();
        value == "1" || value.eq_ignore_ascii_case("true")
    });
    set || invoked.is_some_and(|invoked| invoked.flag(CLAIM_FLAG))
}

/// What the plugin starts with on `target`, claiming its issue when
/// `claim`, or `None` without a token: an access token before client
/// credentials.
fn configure(
    target: Target,
    claim: bool,
    secrets: Option<&Secrets>,
    settings: Option<&LinearSettings>,
    env: &dyn Fn(&str) -> Option<String>,
) -> Option<Setup> {
    let secret = |name| given(secrets.and_then(|secrets| secrets.get(name).map(str::to_owned)));
    let defaults = LinearSettings::default();
    let settings = settings.unwrap_or(&defaults);
    let token = secret(TOKEN_SECRET);
    let client_id = secret(CLIENT_ID_SECRET);
    let client_secret = secret(CLIENT_SECRET_SECRET);
    let redactor = Redactor::new(token.iter().chain(client_secret.iter()).cloned());
    let credentials = match token {
        Some(token) => Credentials::Token(token),
        None => Credentials::Client {
            id: client_id?,
            secret: client_secret?,
            scope: settings.scope.clone(),
        },
    };
    let plan = match given(env(PLAN_ENV)) {
        Some(path) => match read_plan(&path) {
            Ok(plan) => plan,
            Err(why) => {
                warn!("Linear: the plan file {path} was not read: {why}");
                settings.plan.clone()
            }
        },
        None => settings.plan.clone(),
    };
    let plan_json = (!plan.is_empty()).then(|| Plan::new(plan.clone()).json());
    Some(Setup {
        config: Config {
            api_url: settings.api_url.clone(),
            credentials,
            target,
            plan: plan_json,
            heartbeat: Duration::from_secs(settings.heartbeat_minutes.max(1) * 60),
            spacing: SPACING,
            poll: (settings.poll_seconds > 0).then(|| Duration::from_secs(settings.poll_seconds)),
            claim,
        },
        plan,
        redactor,
    })
}

/// Why a plan file was not read.
#[derive(Debug, thiserror::Error)]
enum PlanFileError {
    #[error("{0}")]
    Read(#[from] std::io::Error),
    #[error("{0}")]
    Parse(#[from] toml::de::Error),
}

/// The steps of the TOML plan file at `path`.
fn read_plan(path: &str) -> Result<Vec<PlanStep>, PlanFileError> {
    let text = std::fs::read_to_string(path)?;
    Ok(toml::from_str::<PlanFile>(&text)?.plan)
}

/// The prompt section for the main agent, naming `issue` when known, and
/// mentioning [`OUTCOME_TOOL`] only when the agent has access to it.
fn section(issue: Option<&str>, can_report_outcome: bool) -> String {
    let shown = match issue {
        Some(issue) => format!("This run is shown on the Linear issue {issue} as an agent session"),
        None => "This run is shown on a Linear issue as an agent session".to_owned(),
    };
    let mut text = format!(
        "{shown}: the people watching the issue see your plan, notes and tool calls. The \
         harness posts your result there, so do not post a comment of your own on the issue. \
         They may write to you there while you work: their messages reach you from the plugin \
         {ORIGIN}, titled with their name; answer them as you would the user."
    );
    if can_report_outcome {
        text.push_str(&format!(
            " End the run by calling `{OUTCOME_TOOL}` with `done`, `gave_up` or `needs_input` \
             and a summary for those people."
        ));
    }
    text
}

/// On the plugin's prompt section.
#[derive(Component)]
struct LinearSection;

/// Starts the task on rig's tokio runtime, reusing a Linear session saved
/// with this rig session across reloads when targeting the same issue.
fn start(
    mut linear: ResMut<Linear>,
    saved: Res<SavedLinearSession>,
    runtime: Option<Res<TokioRuntime>>,
) {
    let Some((mut config, jobs, told, redactor)) = linear.pending.take() else {
        return;
    };
    let Some(runtime) = runtime else {
        warn!("Linear: no tokio runtime, so the run is not shown on Linear");
        linear.closed = true;
        return;
    };
    if let Target::Issue(issue) = &config.target
        && let Some(session_id) = saved.session_id.as_ref().filter(|id| !id.trim().is_empty())
        && saved
            .issue
            .as_deref()
            .is_none_or(|saved_issue| saved_issue == issue)
    {
        config.target = Target::Saved(session_id.clone());
    }
    runtime.0.spawn(send::run(
        config,
        rig_reqwest::shared(),
        redactor,
        jobs,
        told,
    ));
}

/// Keeps the plugin's prompt section scoped to the main agent and aligned
/// with its current issue name and access to [`OUTCOME_TOOL`].
fn sync_section(
    linear: Res<Linear>,
    agents: MainAgentQuery,
    access: Query<&ToolAccess>,
    mut sections: Query<(Entity, &SectionOf, &mut PromptSection), With<LinearSection>>,
    mut commands: Commands,
) {
    let Some(main) = main_agent(&agents) else {
        return;
    };
    let can_report = access
        .get(main)
        .map_or(true, |access| access.allows(OUTCOME_TOOL));
    let text = section(linear.issue.as_deref(), can_report);
    let mut matched = false;
    for (entity, owner, mut section_text) in &mut sections {
        if owner.0 == main && !matched {
            matched = true;
            if section_text.text != text {
                section_text.text = text.clone();
            }
        } else {
            commands.entity(entity).despawn();
        }
    }
    if !matched {
        commands.spawn((
            Name::new("linear"),
            LinearSection,
            SectionOf(main),
            PromptSection::new(PromptSection::ORDER_PROJECT, SECTION_TAG, text),
        ));
    }
}

/// Takes what the task tells: once the session is there, saves its id, names
/// the issue and publishes the session's link; a person's message goes to
/// the main agent, acknowledged on Linear; a stop ends the run.
fn take_told(
    mut linear: ResMut<Linear>,
    mut saved: ResMut<SavedLinearSession>,
    agents: MainAgentQuery,
    mut commands: Commands,
) {
    while let Ok(told) = linear.told.try_recv() {
        match told {
            Told::Started(started) => {
                linear.started(started, &mut saved, &mut commands);
            }
            Told::Prompt { body, by } => {
                let quoted = quote(&body, &linear.redactor);
                let ack = match main_agent(&agents) {
                    Some(agent) => {
                        let title = format!("{} on Linear", by.as_deref().unwrap_or("a person"));
                        let author = Author::plugin(ORIGIN).titled(title);
                        let message = Deliver::new(agent, body, DeliveryMode::Steer);
                        commands.trigger(message.with_author(author));
                        format!("Got your message: {quoted}")
                    }
                    None => format!("Got your message: {quoted} No agent is running to read it."),
                };
                linear.queue(Job::Activity(Activity::note(ack)));
            }
            Told::Stop { by } => linear.stop(by.as_deref(), &mut commands),
        }
    }
}

/// The start of a person's message, as its acknowledgement quotes it: its
/// first line, scrubbed before it is cut short.
fn quote(body: &str, redactor: &Redactor) -> String {
    let scrubbed = redactor.text(body);
    let mut lines = scrubbed.trim().lines();
    let mut quoted = head(lines.next().unwrap_or_default(), ACK_MOST);
    if lines.next().is_some() && !quoted.ends_with('…') {
        quoted.push('…');
    }
    quoted
}

impl Linear {
    /// Saves the started session id and requested issue target for reloads.
    fn save_started(&mut self, started: &Started, saved: &mut SavedLinearSession) {
        if !started.id.trim().is_empty() {
            saved.session_id = Some(started.id.clone());
        }
        if let Some(issue) = &started.issue {
            self.issue = Some(issue.clone());
        }
        if let Some(issue) = self.target_issue.as_ref().or(self.issue.as_ref()) {
            saved.issue = Some(issue.clone());
        }
    }

    /// The session is there: saves its id for reloads, names the issue and
    /// publishes the session's link.
    fn started(
        &mut self,
        started: Started,
        saved: &mut SavedLinearSession,
        commands: &mut Commands,
    ) {
        self.save_started(&started, saved);
        if let Some(url) = started.url {
            self.session_url = Some(url.clone());
            commands.spawn(SessionLink::new(SESSION_LINK_LABEL, url));
        }
    }
}

/// Adds each link a plugin publishes to the session's external links,
/// once, but the session's own.
fn publish_link(
    added: On<Add<SessionLink>>,
    links: Query<&SessionLink>,
    mut linear: ResMut<Linear>,
) {
    let Ok(link) = links.get(added.entity) else {
        return;
    };
    if linear.session_url.as_deref() == Some(link.url.as_str())
        || !linear.linked.insert(link.url.clone())
    {
        return;
    }
    linear.queue(Job::Links(vec![Link {
        label: link.label.clone(),
        url: link.url.clone(),
    }]));
}

/// A turn of the main agent starts: what it does shows again, and it may
/// report again.
fn turn_started(
    added: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    agents: MainAgentQuery,
    mut linear: ResMut<Linear>,
) {
    if let Ok(TurnOf(agent)) = turns.get(added.entity)
        && main_agent(&agents) == Some(*agent)
    {
        linear.reported = false;
    }
}

/// Takes in the records written since the last frame.
fn read_records(
    mut records: MessageReader<EffectRecorded>,
    agents: MainAgentQuery,
    ids: Query<&AgentId>,
    mut linear: ResMut<Linear>,
    mut commands: Commands,
) {
    let main = main_agent(&agents)
        .and_then(|agent| ids.get(agent).ok())
        .map(|id| id.0.as_str());
    for EffectRecorded(record) in records.read() {
        let is_main = main.is_some() && record.scope.as_deref() == main;
        linear.record(record, is_main, &mut commands);
    }
}

/// `report_outcome`: posts the outcome at once. Only the main agent may.
fn report_outcome(
    called: On<ToolCalled<OutcomeArgs>>,
    agents: MainAgentQuery,
    mut linear: ResMut<Linear>,
    mut calls: Query<&mut Decision>,
) {
    let summary = called.args.summary.trim();
    let result = if main_agent(&agents) != Some(called.agent) {
        ToolResult::failed(ToolExecutionError::refused(
            "Only the main agent reports the run's outcome.",
        ))
    } else if summary.is_empty() {
        ToolResult::failed(ToolExecutionError::refused("`summary` must not be empty."))
    } else {
        let (content, end) = outcome(
            called.args.status,
            summary.to_owned(),
            &called.args.options,
            &linear.redactor,
        );
        linear.end(content, end);
        ToolResult::success(ToolOutput::text(
            "Posted to the Linear issue. End the run now, without a comment of your own.",
        ))
    };
    if let Ok(mut decision) = calls.get_mut(called.call) {
        decision.decide(Decider::plugin(ORIGIN), result);
    }
}

/// The activity and plan end of a reported outcome. A question offers
/// `options`, scrubbed, trimmed, without blanks or repeats, at most
/// [`OPTIONS_MOST`]; any other outcome leaves them out.
fn outcome(
    status: OutcomeStatus,
    summary: String,
    options: &[String],
    redactor: &Redactor,
) -> (Content, End) {
    match status {
        OutcomeStatus::Done => (Content::Response(summary), End::Done),
        OutcomeStatus::GaveUp => (Content::Error(summary), End::Failed),
        OutcomeStatus::NeedsInput => {
            let (most, longest) = OPTIONS_MOST;
            let mut offered: Vec<String> = Vec::new();
            for option in options
                .iter()
                .map(|option| head(&redactor.text(option), longest))
            {
                if !option.is_empty() && !offered.contains(&option) && offered.len() < most {
                    offered.push(option);
                }
            }
            let question = Content::Elicitation {
                body: summary,
                options: offered,
            };
            (question, End::NeedsInput)
        }
    }
}

/// The run's end when the agent did not report one: stopped while it
/// worked, else its last failure, else its last reply.
fn fallback(stopped: bool, error: Option<&str>, reply: Option<&str>) -> (Content, End) {
    match (stopped, error, reply) {
        (false, Some(error), _) => (Content::Error(error.to_owned()), End::Failed),
        (false, None, Some(reply)) => (Content::Response(reply.to_owned()), End::Done),
        _ => (Content::Error(STOPPED.to_owned()), End::Failed),
    }
}

/// Notes whether the main agent is in a turn as the app exits, before the
/// exit stops it.
fn note_working(
    agents: MainAgentQuery,
    active: Query<(), With<ActiveTurn>>,
    mut linear: ResMut<Linear>,
) {
    linear.working = main_agent(&agents).is_some_and(|agent| active.contains(agent));
}

/// Posts the run's end unless the agent reported it, then waits up to
/// [`FLUSH_WAIT`] for the queue to be sent. A reload posts no end: the run
/// reattaches to the saved Linear session in the next process.
fn close(
    mut linear: ResMut<Linear>,
    mut saved: ResMut<SavedLinearSession>,
    reason: Option<Res<ExitReason>>,
) {
    if linear.closed || linear.pending.is_some() {
        return;
    }
    let goes_on = matches!(reason.as_deref(), Some(ExitReason::Reload));
    if !linear.reported && !goes_on {
        let stopped = linear.working || matches!(reason.as_deref(), Some(ExitReason::Signal));
        let (content, end) = fallback(
            stopped,
            linear.last_error.as_deref(),
            linear.last_reply.as_deref(),
        );
        linear.end(content, end);
    }
    let (done, wait) = std::sync::mpsc::channel();
    linear.queue(Job::Close(done));
    linear.closed = true;
    let timed_out = wait.recv_timeout(FLUSH_WAIT).is_err();
    while let Ok(told) = linear.told.try_recv() {
        if let Told::Started(started) = told {
            linear.save_started(&started, &mut saved);
        }
    }
    if timed_out {
        if goes_on && saved.session_id.is_none() {
            warn!(
                "Linear: session did not finish starting within {} s before reload; the next \
                 process may open a new session",
                FLUSH_WAIT.as_secs()
            );
        } else {
            warn!(
                "Linear: not everything was sent within {} s",
                FLUSH_WAIT.as_secs()
            );
        }
    }
}
