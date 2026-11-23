//! Mirrors a run into a Linear agent session, so the people watching an
//! issue see the agent's plan, notes, tool calls and result. The plugin
//! does nothing, and sends nothing, unless it has both a token (the secret
//! `linear`, or `linear-client-id` and `linear-client-secret`) and a target
//! (`--linear-issue`, `--linear-session` or their variables). The crate's
//! README lists the settings, the plan file and what is sent when.
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
use schemars::JsonSchema;
use serde::Deserialize;

pub use crate::plan::PlanStep;

use crate::activity::{Activity, Call, Content, Redactor, Reply, pull_request};
use crate::api::{Credentials, Link};
use crate::plan::{End, Plan};
use crate::send::{Config, Job, Started, Target};

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

/// The longest the app waits on exit for what is queued to be sent.
const FLUSH_WAIT: Duration = Duration::from_secs(10);

/// The least time between two requests.
const SPACING: Duration = Duration::from_secs(1);

/// What the outcome tool says to the model.
const OUTCOME_DESCRIPTION: &str = "End the run by reporting its outcome on the Linear issue \
it is shown on. `status` is `done` when the task is complete, `gave_up` when it cannot be \
completed, `needs_input` when a person must answer first. `summary` is posted as written, in \
Markdown, for the people watching the issue: what was done or why not, and what is needed from \
them. Call it once, as the last thing you do.";

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

/// The plugin's state, present only when it is active.
#[derive(Resource)]
struct Linear {
    /// The queue to the task.
    jobs: Sender<Job>,
    /// What the task tells once the session is there.
    started: Receiver<Started>,
    /// What [`start`] hands the task.
    pending: Option<(Config, Receiver<Job>, Sender<Started>, Redactor)>,
    plan: Plan,
    /// The issue, as the prompt names it.
    issue: Option<String>,
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
}

impl Linear {
    fn new(setup: Setup) -> Self {
        let (jobs, queued) = async_channel::unbounded();
        let (tell, started) = async_channel::unbounded();
        let issue = match &setup.config.target {
            Target::Issue(issue) => Some(issue.clone()),
            Target::Session(_) => None,
        };
        Self {
            jobs,
            started,
            pending: Some((setup.config, queued, tell, setup.redactor)),
            plan: Plan::new(setup.plan),
            issue,
            session_url: None,
            linked: HashSet::new(),
            reported: false,
            last_reply: None,
            last_error: None,
            working: false,
            closed: false,
        }
    }

    /// Queues `job` for the task, unless the run is closed.
    fn queue(&self, job: Job) {
        if !self.closed {
            self.jobs.try_send(job).ok();
        }
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
                        for activity in reply.activities() {
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
                self.note(call.activity());
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
        .add_setting::<LinearSettings>("linear");
        let env = |name: &str| std::env::var(name).ok();
        // Only a run shown on Linear asks for its secrets, so a build with
        // the plugin on prompts nobody else.
        let Some(target) = target(app.world().get_resource::<Invoked>(), &env) else {
            return;
        };
        app.add_secret(Secret::new(
            TOKEN_SECRET,
            "Linear access token or API key, empty to leave Linear off",
        ))
        .add_secret(Secret::new(
            CLIENT_ID_SECRET,
            "Linear OAuth client id, empty unless signing in as an app",
        ))
        .add_secret(Secret::new(
            CLIENT_SECRET_SECRET,
            "Linear OAuth client secret, empty unless signing in as an app",
        ));
        let world = app.world();
        let Some(setup) = configure(
            target,
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
            .add_systems(Update, take_started)
            .add_systems(
                Last,
                (
                    note_working
                        .before(StopTurnsOnExit)
                        .run_if(on_message::<AppExit>),
                    (read_records, close.run_if(on_message::<AppExit>))
                        .chain()
                        .after(StopTurnsOnExit),
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

/// What the plugin starts with on `target`, or `None` without a token: an
/// access token before client credentials.
fn configure(
    target: Target,
    secrets: Option<&Secrets>,
    settings: Option<&LinearSettings>,
    env: &dyn Fn(&str) -> Option<String>,
) -> Option<Setup> {
    let secret = |name| given(secrets.and_then(|secrets| secrets.get(name).map(str::to_owned)));
    let defaults = LinearSettings::default();
    let settings = settings.unwrap_or(&defaults);
    let credentials = match secret(TOKEN_SECRET) {
        Some(token) => Credentials::Token(token),
        None => Credentials::Client {
            id: secret(CLIENT_ID_SECRET)?,
            secret: secret(CLIENT_SECRET_SECRET)?,
            scope: settings.scope.clone(),
        },
    };
    let redactor = Redactor::new(match &credentials {
        Credentials::Token(token) => vec![token.clone()],
        Credentials::Client { secret, .. } => vec![secret.clone()],
    });
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

/// The prompt section, naming `issue` when known.
fn section(issue: Option<&str>) -> String {
    let shown = match issue {
        Some(issue) => format!("This run is shown on the Linear issue {issue} as an agent session"),
        None => "This run is shown on a Linear issue as an agent session".to_owned(),
    };
    format!(
        "{shown}: the people watching the issue see your plan, notes and tool calls. The \
         harness posts your result there, so do not post a comment of your own on the issue. \
         End the run by calling `{OUTCOME_TOOL}` with `done`, `gave_up` or `needs_input` and a \
         summary for those people."
    )
}

/// On the plugin's prompt section.
#[derive(Component)]
struct LinearSection;

/// Starts the task on rig's tokio runtime and adds the prompt section.
fn start(mut linear: ResMut<Linear>, runtime: Option<Res<TokioRuntime>>, mut commands: Commands) {
    let Some((config, jobs, started, redactor)) = linear.pending.take() else {
        return;
    };
    let Some(runtime) = runtime else {
        warn!("Linear: no tokio runtime, so the run is not shown on Linear");
        linear.closed = true;
        return;
    };
    commands.spawn((
        Name::new("linear"),
        LinearSection,
        PromptSection::new(
            PromptSection::ORDER_PROJECT,
            SECTION_TAG,
            section(linear.issue.as_deref()),
        ),
    ));
    runtime.0.spawn(send::run(
        config,
        rig_reqwest::shared(),
        redactor,
        jobs,
        started,
    ));
}

/// Takes what the task tells once the session is there: names the issue
/// in the prompt section and publishes the session's link.
fn take_started(
    mut linear: ResMut<Linear>,
    mut sections: Query<&mut PromptSection, With<LinearSection>>,
    mut commands: Commands,
) {
    while let Ok(started) = linear.started.try_recv() {
        if let Some(issue) = started.issue {
            for mut section_text in &mut sections {
                let text = section(Some(&issue));
                if section_text.text != text {
                    section_text.text = text;
                }
            }
            linear.issue = Some(issue);
        }
        if let Some(url) = started.url {
            linear.session_url = Some(url.clone());
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
        let (content, end) = outcome(called.args.status, summary.to_owned());
        linear.end(content, end);
        ToolResult::success(ToolOutput::text(
            "Posted to the Linear issue. End the run now, without a comment of your own.",
        ))
    };
    if let Ok(mut decision) = calls.get_mut(called.call) {
        decision.decide(Decider::plugin(ORIGIN), result);
    }
}

/// The activity and plan end of a reported outcome.
fn outcome(status: OutcomeStatus, summary: String) -> (Content, End) {
    match status {
        OutcomeStatus::Done => (Content::Response(summary), End::Done),
        OutcomeStatus::GaveUp => (Content::Error(summary), End::Failed),
        OutcomeStatus::NeedsInput => (Content::Elicitation(summary), End::NeedsInput),
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
/// [`FLUSH_WAIT`] for the queue to be sent. A reload or a session switch
/// posts no end: the run goes on in the next process.
fn close(mut linear: ResMut<Linear>, reason: Option<Res<ExitReason>>) {
    if linear.closed || linear.pending.is_some() {
        return;
    }
    let goes_on = matches!(
        reason.as_deref(),
        Some(ExitReason::Reload | ExitReason::Switch(_))
    );
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
    if wait.recv_timeout(FLUSH_WAIT).is_err() {
        warn!(
            "Linear: not everything was sent within {} s",
            FLUSH_WAIT.as_secs()
        );
    }
}
