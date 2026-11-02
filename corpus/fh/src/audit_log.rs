//! FH-22: a tamper-evident audit log. Every tool call that may change
//! something or reach outside, every question that decided whether one
//! runs, and the session's start, resume and end are appended to
//! `RIG_HOME/audit/<date>.jsonl` ([`chain`]), hash-chained, before the
//! call's result reaches the model. When the log cannot be written, the
//! calls it would record are refused. `rig audit verify [from-date]`
//! checks the chain and `rig audit show --session <id>` lists a session's
//! entries.

pub mod chain;

use std::path::{Path, PathBuf};

use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::redact;
use rig_core::tool::{ChangeKind, ExitStatus, FilesChanged};
use rig_harness::prelude::*;
use serde::Deserialize;

use chain::{Entry, Event, FileHashes, Ran};

/// This plugin, as a decider.
const NAME: &str = "fh-audit-log";

/// The `[audit]` table of the user's settings.
#[derive(Resource, Deserialize, Default, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct AuditSettings {
    /// Log calls that only read too.
    pub log_reads: bool,
    /// Tools that reach the network, besides those whose annotations say
    /// they reach an open world.
    pub network_tools: Vec<String>,
}

/// Where the log is: `RIG_HOME/audit`, and the project, which it must not
/// be in.
#[derive(Resource, Clone, Debug)]
pub struct AuditLog {
    /// The log's directory.
    pub dir: PathBuf,
    /// The project directory, if known.
    pub project: Option<PathBuf>,
    /// Why the last append failed, until one succeeds; audited calls are
    /// refused meanwhile.
    pub failed: Option<String>,
}

impl AuditLog {
    /// Whether an entry could be written now; why not otherwise.
    pub fn writable(&self) -> Result<(), String> {
        if let Some(why) = &self.failed {
            return Err(why.clone());
        }
        self.outside_project()?;
        chain::probe(&self.dir).map_err(|error| error.to_string())
    }

    /// Appends `entry`, or says why it could not.
    pub fn append(&mut self, entry: Entry) -> Result<String, String> {
        let appended = self
            .outside_project()
            .and_then(|()| chain::append(&self.dir, entry).map_err(|error| error.to_string()));
        self.failed = appended.as_ref().err().cloned();
        appended
    }

    fn outside_project(&self) -> Result<(), String> {
        match &self.project {
            Some(project) if self.dir.starts_with(project) => {
                Err("the audit log may not be in the project".to_owned())
            }
            _ => Ok(()),
        }
    }
}

/// The plugin's stage systems.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum AuditSystems {
    /// In Gate: refuses audited calls while the log cannot be written, and
    /// notes the content of the files a call may write before it runs.
    Gate,
    /// In Review, after every reviewer: writes each result's entry.
    Record,
}

/// The audit log, and the command `rig audit`.
#[derive(Default)]
pub struct AuditLogPlugin;

impl Plugin for AuditLogPlugin {
    fn build(&self, app: &mut App) {
        let dir = RigHome::of(app.world()).root().join("audit");
        let project = app
            .world()
            .get_resource::<ProjectRoot>()
            .map(|root| root.0.clone());
        let front = Front::new("audit").sessionless().on_command(
            "audit",
            "verify [from-date] | show --session <id>",
            "Check the audit log's chain, or list a session's entries",
        );
        app.add_setting::<AuditSettings>("audit")
            .insert_resource(AuditLog {
                dir,
                project,
                failed: None,
            })
            .add_front(front);
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front("audit") {
            app.add_systems(Startup, audit_command);
            return;
        }
        app.configure_sets(
            AgentPass,
            (
                AuditSystems::Gate.in_set(Stage::Gate),
                AuditSystems::Record
                    .after(ReviewSystems::Act)
                    .after(rig_basics::redact::RedactSystems)
                    .in_set(Stage::Review),
            ),
        )
        .add_systems(
            AgentPass,
            (
                guard.in_set(AuditSystems::Gate),
                record.in_set(AuditSystems::Record),
            ),
        )
        .add_observer(record_stopped)
        .add_observer(record_answer)
        .add_systems(PostStartup, record_start)
        .add_systems(
            Last,
            record_end
                .in_set(OnAppExitSystems)
                .run_if(on_message::<AppExit>),
        );
    }
}

/// On a call: each file it may write, with the hash of its content when
/// the call was about to start.
#[derive(Component)]
struct Before(Vec<(PathBuf, Option<String>)>);

/// On a call whose entry is written, or that is not audited.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Audited;

/// On a call a question held: who answered it, and the question and
/// answer.
#[derive(Component)]
struct Approval {
    by: Option<Decider>,
    question: String,
}

/// What deciding whether a call is audited and writing its entry read.
#[derive(bevy_ecs::system::SystemParam)]
struct Context<'w, 's> {
    tools: Query<'w, 's, &'static ToolDef>,
    agents: Query<'w, 's, (&'static AgentId, Option<&'static SpawnedBy>)>,
    settings: Res<'w, AuditSettings>,
    log: ResMut<'w, AuditLog>,
    session: Option<Res<'w, SessionPaths>>,
}

impl Context<'_, '_> {
    /// Whether calls of `tool` with `footprint` are logged: what may
    /// change something or reach the network, and reads when asked.
    fn audited(&self, tool: &str, footprint: Option<&Footprint>) -> bool {
        let network = self.settings.network_tools.iter().any(|name| name == tool)
            || self.tools.iter().any(|def| {
                def.0.name.as_str() == tool && def.0.annotations.open_world_hint == Some(true)
            });
        self.settings.log_reads || network || !footprint.is_some_and(|f| f.0.reads_only())
    }

    /// An entry of `event` for `agent`, with its parents.
    fn entry(&self, event: Event, agent: Option<Entity>) -> Entry {
        let id = |agent| self.agents.get(agent).ok().map(|(id, _)| id.0.clone());
        let by = |agent| self.agents.get(agent).ok()?.1.map(|by| by.0);
        let parents = std::iter::successors(agent.and_then(by), |&parent| by(parent))
            .filter_map(id)
            .collect();
        let session = self.session.as_ref().and_then(|paths| paths.id());
        Entry {
            event,
            session: session.map(|id| id.to_string()),
            agent: agent.and_then(id),
            parents,
            ..Entry::default()
        }
    }

    /// The entry of a call as it settles.
    fn call_entry(&self, call: CallItem<'_>) -> Entry {
        let function = &call.request.call.function;
        let mut arguments = serde_json::Value::Object(function.arguments.clone());
        redact::json(&mut arguments);
        let (outcome, error) = how(call.outcome);
        let published = call.published.map(|published| &published.0);
        let exit = published.and_then(|published| published.get::<ExitStatus>().ok().flatten());
        let changed = published
            .and_then(|published| published.get::<FilesChanged>().ok().flatten())
            .unwrap_or_default();
        let before = call.before.map_or(&[][..], |Before(paths)| paths);
        // What the call left: the hash its tool published for each file it
        // changed, the content from before for each it did not.
        let mut files: Vec<FileHashes> = before
            .iter()
            .map(|(path, before)| FileHashes {
                path: path.display().to_string(),
                before: before.clone(),
                after: before.clone(),
            })
            .collect();
        for change in &changed.0 {
            let after = match change.kind {
                ChangeKind::Deleted => None,
                _ => change
                    .after
                    .as_ref()
                    .and_then(|it| Some(it.hash.as_ref()?.0.clone())),
            };
            match files.iter_mut().find(|file| file.path == change.path) {
                Some(file) => file.after = after,
                None => files.push(FileHashes {
                    path: change.path.clone(),
                    before: None,
                    after,
                }),
            }
        }
        let (decided_by, question) = match call.approval {
            Some(approval) => (approval.by.clone(), Some(approval.question.clone())),
            None => (None, None),
        };
        Entry {
            tool: Some(function.name.as_str().to_owned()),
            arguments: Some(arguments),
            outcome: Some(outcome),
            exit_code: exit.and_then(|status| status.code()),
            error: error.map(|error| redact::text(&error).into_owned()),
            files,
            decided_by: call.decision.and_then(Decision::by).cloned().or(decided_by),
            detail: question,
            ..self.entry(Event::ToolCall, Some(call.agent))
        }
    }

    /// Appends `entry`; a notice to `agent` says why it could not.
    fn append(&mut self, entry: Entry, agent: Option<Entity>, notices: &mut MessageWriter<Notice>) {
        if let Err(why) = self.log.append(entry) {
            let text = format!("The audit log could not record an entry: {why}.");
            notices.write(Notice::error(agent, text));
        }
    }
}

/// What a call's entry is made of.
struct CallItem<'a> {
    request: &'a ToolRequest,
    agent: Entity,
    outcome: &'a Outcome,
    published: Option<&'a Published>,
    decision: Option<&'a Decision>,
    before: Option<&'a Before>,
    approval: Option<&'a Approval>,
}

/// How a call went, and why when it did not run.
fn how(outcome: &Outcome) -> (Ran, Option<String>) {
    match &outcome.0 {
        Err(report) if report.kind == ErrorKind::Cancelled => {
            (Ran::Cancelled, Some(report.message.clone()))
        }
        Err(report) if report.refusal => (Ran::Refused, Some(report.message.clone())),
        Err(report) => (Ran::Failed, Some(report.message.clone())),
        Ok(_) => {
            let result = outcome.tool_result();
            let text = || Some(result.output().render());
            if result.is_refused() {
                (Ran::Refused, text())
            } else if result.is_error() {
                (Ran::Failed, text())
            } else {
                (Ran::Ran, None)
            }
        }
    }
}

/// Refuses each audited call about to start while the log cannot be
/// written, and notes the content of the files it may write.
fn guard(
    mut calls: Query<
        (Entity, &ToolRequest, Option<&Footprint>, &mut Decision),
        (ReadyToStart, ToolCalls),
    >,
    context: Context,
    mut commands: Commands,
) {
    for (call, request, footprint, mut decision) in &mut calls {
        let name = request.call.function.name.as_str();
        if decision.is_decided() || !context.audited(name, footprint) {
            continue;
        }
        if let Err(why) = context.log.writable() {
            let why = format!(
                "The audit log cannot be written ({why}), so nothing that changes files or \
                 reaches outside may run."
            );
            decision.decide(Decider::plugin(NAME), Outcome::refused(why));
            continue;
        }
        let before = match footprint.map(|footprint| &footprint.0) {
            Some(Touches::Writes(Scope::Paths(paths))) => paths
                .iter()
                .map(|path| (path.clone(), chain::file_hash(path)))
                .collect(),
            _ => Vec::new(),
        };
        commands.entity(call).insert(Before(before));
    }
}

/// Writes each audited call's entry once its outcome is reviewed, before
/// it is committed; a call whose entry cannot be written answers with
/// that failure instead.
fn record(
    mut calls: Query<
        (
            Entity,
            (&ToolRequest, &ForAgent, &mut Outcome, Option<&Footprint>),
            (
                Option<&Published>,
                Option<&Decision>,
                Option<&Before>,
                Option<&Approval>,
            ),
        ),
        (ReadyToSettle, ToolCalls, Without<Audited>),
    >,
    mut context: Context,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, (request, &ForAgent(agent), mut outcome, footprint), facts) in &mut calls {
        commands.entity(call).insert(Audited);
        if !context.audited(request.call.function.name.as_str(), footprint) {
            continue;
        }
        let (published, decision, before, approval) = facts;
        let item = CallItem {
            request,
            agent,
            outcome: &outcome,
            published,
            decision,
            before,
            approval,
        };
        let entry = context.call_entry(item);
        let Err(why) = context.log.append(entry) else {
            continue;
        };
        // A refused call did nothing, and its refusal says so.
        if how(&outcome).0 != Ran::Refused {
            let text = format!("The call ran, but the audit log could not record it: {why}.");
            notices.write(Notice::error(agent, text.clone()));
            *outcome = Outcome(Err(ErrorReport::new(ErrorKind::Internal, text)));
        }
    }
}

/// Writes the entry of an audited call that settles without a review,
/// such as one stopped by an interrupt.
fn record_stopped(
    settled: On<Add<Settled>>,
    calls: Query<
        (
            (&ToolRequest, &ForAgent, &Outcome, Option<&Footprint>),
            (
                Option<&Published>,
                Option<&Decision>,
                Option<&Before>,
                Option<&Approval>,
            ),
        ),
        (Without<Audited>, Without<SupersededBy>),
    >,
    mut context: Context,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(((request, &ForAgent(agent), outcome, footprint), facts)) = calls.get(settled.entity)
    else {
        return;
    };
    if !context.audited(request.call.function.name.as_str(), footprint) {
        return;
    }
    let (published, decision, before, approval) = facts;
    let item = CallItem {
        request,
        agent,
        outcome,
        published,
        decision,
        before,
        approval,
    };
    let entry = context.call_entry(item);
    context.append(entry, Some(agent), &mut notices);
}

/// Writes each question's answer and who gave it, and puts them on the
/// call it held.
fn record_answer(
    settled: On<Add<Settled>>,
    asks: Query<(
        &Ask,
        &Outcome,
        &Decision,
        Option<&HoldOf>,
        Option<&ForAgent>,
    )>,
    calls: Query<(), With<ToolRequest>>,
    mut context: Context,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((ask, outcome, decision, held, agent)) = asks.get(settled.entity) else {
        return;
    };
    let answer = match Answer::of(outcome) {
        Some(Answer { choices, text }) => {
            let choices = choices.join(", ");
            match text.as_deref().map(redact::text) {
                Some(text) if !choices.is_empty() => format!("{choices}: {text}"),
                Some(text) => text.into_owned(),
                None => choices,
            }
        }
        None => "no answer".to_owned(),
    };
    let question = format!("question \"{}\" answered: {answer}", ask.title);
    let by = decision.by().cloned();
    if let Some(&HoldOf(call)) = held
        && calls.contains(call)
    {
        let approval = Approval {
            by: by.clone(),
            question: question.clone(),
        };
        commands.entity(call).insert(approval);
    }
    let agent = agent.map(|agent| agent.0);
    let entry = Entry {
        decided_by: by,
        detail: Some(format!("{question}; {}", redact::text(&ask.body))),
        ..context.entry(Event::Approval, agent)
    };
    context.append(entry, agent, &mut notices);
}

/// Writes that the session started, or started again.
fn record_start(
    started: Option<Res<AppStart>>,
    mut context: Context,
    mut notices: MessageWriter<Notice>,
) {
    let event = if started.is_some_and(|started| started.0 == StartReason::New) {
        Event::SessionStart
    } else {
        Event::SessionResume
    };
    let project = context.log.project.as_ref().map(|root| root.display());
    let entry = Entry {
        detail: project.map(|root| format!("project {root}")),
        ..context.entry(event, None)
    };
    context.append(entry, None, &mut notices);
}

/// Writes that the app exits, and why.
fn record_end(reason: Option<Res<ExitReason>>, mut context: Context) {
    let entry = Entry {
        detail: reason.map(|reason| format!("{:?}", *reason)),
        ..context.entry(Event::SessionEnd, None)
    };
    // Nothing is left to tell on the way out.
    context.log.append(entry).ok();
}

/// `rig audit …`: prints what [`run_command`] answers, and exits with 1
/// when it failed.
fn audit_command(invoked: Option<Res<Invoked>>, log: Res<AuditLog>, mut commands: Commands) {
    let words = invoked
        .as_ref()
        .and_then(|invoked| invoked.command())
        .map_or(&[][..], |(_, words)| words);
    let reason = match run_command(&log.dir, words) {
        Ok(text) => {
            println!("{text}");
            ExitReason::Quit
        }
        Err(text) => {
            eprintln!("{text}");
            ExitReason::Failed(1)
        }
    };
    commands.trigger(Exit(reason));
}

/// What `rig audit` with `words` prints: `verify [from-date]` checks the
/// chain in `dir`, `show --session <id>` lists that session's entries.
///
/// # Errors
///
/// The first broken entry, unreadable logs, or the usage.
pub fn run_command(dir: &Path, words: &[String]) -> Result<String, String> {
    let words: Vec<&str> = words.iter().map(String::as_str).collect();
    match words.as_slice() {
        ["verify", rest @ ..] if rest.len() <= 1 => match chain::verify(dir, rest.first().copied())
        {
            Ok(count) => Ok(format!("OK, {count} entries")),
            Err(broken) => Err(format!("Broken entry at {broken}")),
        },
        ["show", "--session", id] => match chain::show(dir, id) {
            Ok(lines) if lines.is_empty() => Ok(format!("No entries of {id}.")),
            Ok(lines) => Ok(lines.join("\n")),
            Err(error) => Err(format!("Could not read the audit log: {error}")),
        },
        _ => Err("usage: rig audit verify [from-date] | rig audit show --session <id>".to_owned()),
    }
}

#[cfg(test)]
mod tests;
