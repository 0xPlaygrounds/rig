//! FH-22: a tamper-evident audit log. Every tool call that may change
//! something or reach outside, every question that decided whether one
//! runs, and the session's start, resume and end are appended to
//! `RIG_HOME/audit/<date>.jsonl` ([`chain`]), hash-chained, before the
//! call's result reaches the model. When the log cannot be written, the
//! calls it would record are refused. `/audit verify [from-date]` checks
//! the chain and `/audit show [session]` lists a session's entries.

pub mod chain;

use std::path::{Path, PathBuf};

use rig_core::error::{ErrorKind, ErrorReport};
use rig_core::redact;
use rig_core::tool::{ExitStatus, FilesChanged};
use rig_harness::prelude::*;
use serde::Deserialize;

use chain::{Entry, Event, FileHashes, Ran};

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
}

impl AuditLog {
    /// Whether an entry could be written now; why not otherwise.
    pub fn writable(&self) -> Result<(), String> {
        if self
            .project
            .as_ref()
            .is_some_and(|project| self.dir.starts_with(project))
        {
            return Err("the audit log may not be in the project".to_owned());
        }
        chain::probe(&self.dir).map_err(|error| error.to_string())
    }

    /// Appends `entry`, or says why it could not.
    pub fn append(&self, entry: Entry) -> Result<String, String> {
        self.writable()?;
        chain::append(&self.dir, entry).map_err(|error| error.to_string())
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

/// The audit log.
#[derive(Default)]
pub struct AuditLogPlugin;

impl Plugin for AuditLogPlugin {
    fn build(&self, app: &mut App) {
        let dir = RigHome::of(app.world()).root().join("audit");
        let project = app
            .world()
            .get_resource::<ProjectRoot>()
            .map(|root| root.0.clone());
        app.add_setting::<AuditSettings>("audit")
            .insert_resource(AuditLog { dir, project })
            .configure_sets(
                AgentPass,
                (
                    AuditSystems::Gate.in_set(Stage::Gate),
                    AuditSystems::Record
                        .after(ReviewSystems::Act)
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
            )
            .add_command(
                "audit",
                "The audit log: /audit verify [date], /audit show [session]",
                audit,
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

/// On a call a question let run or refused: the question and its answer.
#[derive(Component)]
struct Approval(String);

/// What deciding whether a call is audited and writing its entry read.
#[derive(bevy_ecs::system::SystemParam)]
struct Context<'w, 's> {
    tools: Query<'w, 's, &'static ToolDef>,
    agents: Query<'w, 's, (&'static AgentId, Option<&'static SpawnedBy>)>,
    settings: Res<'w, AuditSettings>,
    log: Res<'w, AuditLog>,
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
        let mut parents = Vec::new();
        let mut at = agent.and_then(|agent| self.agents.get(agent).ok()?.1.map(|by| by.0));
        while let Some(parent) = at {
            parents.extend(id(parent));
            at = self
                .agents
                .get(parent)
                .ok()
                .and_then(|(_, by)| by.map(|by| by.0));
        }
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
        let mut files: Vec<FileHashes> = call
            .before
            .map(|Before(paths)| paths.iter())
            .into_iter()
            .flatten()
            .map(|(path, before)| hashes(path, before.clone()))
            .collect();
        for path in changed.paths() {
            if !files.iter().any(|file| file.path == path) {
                files.push(hashes(Path::new(path), None));
            }
        }
        Entry {
            tool: Some(function.name.as_str().to_owned()),
            arguments: Some(arguments),
            outcome: Some(outcome),
            exit_code: exit.and_then(|status| status.code()),
            error: error.map(|error| redact::text(&error).into_owned()),
            files,
            decided_by: call.approval.map(|Approval(by)| by.clone()),
            ..self.entry(Event::ToolCall, Some(call.agent))
        }
    }
}

/// What a call's entry is made of.
struct CallItem<'a> {
    request: &'a ToolRequest,
    agent: Entity,
    outcome: &'a Outcome,
    published: Option<&'a Published>,
    before: Option<&'a Before>,
    approval: Option<&'a Approval>,
}

/// A file's hashes: `before`, and its content now.
fn hashes(path: &Path, before: Option<String>) -> FileHashes {
    FileHashes {
        path: path.display().to_string(),
        before,
        after: chain::file_hash(path),
    }
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
        (Undecided, ToolCalls),
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
            decision.decide(
                Decider::plugin("fh-audit-log"),
                Outcome::refused(format!(
                    "The audit log cannot be written ({why}), so nothing that changes files or \
                 reaches outside may run."
                )),
            );
            continue;
        }
        let paths = match footprint.map(|footprint| &footprint.0) {
            Some(Touches::Writes(Scope::Paths(paths))) => paths.clone(),
            _ => Vec::new(),
        };
        let before = paths
            .into_iter()
            .map(|path| {
                let hash = chain::file_hash(&path);
                (path, hash)
            })
            .collect();
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
            &ToolRequest,
            &ForAgent,
            &mut Outcome,
            Option<&Footprint>,
            Option<&Published>,
            Option<&Before>,
            Option<&Approval>,
        ),
        (Unreviewed, ToolCalls, Without<Audited>),
    >,
    context: Context,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    for (call, request, &ForAgent(agent), mut outcome, footprint, published, before, approval) in
        &mut calls
    {
        commands.entity(call).insert(Audited);
        if !context.audited(request.call.function.name.as_str(), footprint) {
            continue;
        }
        let item = CallItem {
            request,
            agent,
            outcome: &outcome,
            published,
            before,
            approval,
        };
        let Err(why) = context.log.append(context.call_entry(item)) else {
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
            &ToolRequest,
            &ForAgent,
            &Outcome,
            Option<&Footprint>,
            Option<&Published>,
            Option<&Before>,
            Option<&Approval>,
        ),
        (Without<Audited>, Without<SupersededBy>),
    >,
    context: Context,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((request, &ForAgent(agent), outcome, footprint, published, before, approval)) =
        calls.get(settled.entity)
    else {
        return;
    };
    if !context.audited(request.call.function.name.as_str(), footprint) {
        return;
    }
    let item = CallItem {
        request,
        agent,
        outcome,
        published,
        before,
        approval,
    };
    if let Err(why) = context.log.append(context.call_entry(item)) {
        notices.write(Notice::error(
            agent,
            format!("The audit log could not record a call: {why}."),
        ));
    }
}

/// Writes each question's answer, and puts it on the call it held.
fn record_answer(
    settled: On<Add<Settled>>,
    asks: Query<(&Ask, &Outcome, Option<&HoldOf>, Option<&ForAgent>)>,
    calls: Query<(), With<ToolRequest>>,
    context: Context,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((ask, outcome, held, agent)) = asks.get(settled.entity) else {
        return;
    };
    let answer = match Answer::of(outcome) {
        Some(Answer::Choice(id)) => id,
        Some(Answer::Text(text)) => redact::text(&text).into_owned(),
        None => "no answer".to_owned(),
    };
    let by = format!("question \"{}\" answered: {answer}", ask.title);
    if let Some(&HoldOf(call)) = held
        && calls.contains(call)
    {
        commands.entity(call).insert(Approval(by.clone()));
    }
    let entry = Entry {
        decided_by: Some(by),
        detail: Some(redact::text(&ask.body).into_owned()),
        ..context.entry(Event::Approval, agent.map(|agent| agent.0))
    };
    if let Err(why) = context.log.append(entry) {
        notices.write(Notice::error(
            None,
            format!("The audit log could not record an answer: {why}."),
        ));
    }
}

/// Writes that the session started, or started again.
fn record_start(
    first: Option<Res<FirstStart>>,
    context: Context,
    mut notices: MessageWriter<Notice>,
) {
    let event = match first {
        Some(_) => Event::SessionStart,
        None => Event::SessionResume,
    };
    let project = context
        .log
        .project
        .as_ref()
        .map(|root| root.display().to_string());
    let entry = Entry {
        detail: project.map(|root| format!("project {root}")),
        ..context.entry(event, None)
    };
    if let Err(why) = context.log.append(entry) {
        notices.write(Notice::error(
            None,
            format!("The audit log cannot be written: {why}."),
        ));
    }
}

/// Writes that the app exits, and why.
fn record_end(reason: Option<Res<ExitReason>>, context: Context) {
    let entry = Entry {
        detail: reason.map(|reason| format!("{:?}", *reason)),
        ..context.entry(Event::SessionEnd, None)
    };
    // Nothing is left to tell on the way out.
    context.log.append(entry).ok();
}

/// `/audit verify [from-date]` and `/audit show [session]`.
fn audit(
    In(args): In<CommandArgs>,
    log: Res<AuditLog>,
    session: Option<Res<SessionPaths>>,
    mut notices: MessageWriter<Notice>,
) {
    let mut words = args.args.split_whitespace();
    let notice = match (words.next(), words.next()) {
        (Some("verify"), from) => match chain::verify(&log.dir, from) {
            Ok(count) => Notice::info(args.agent, format!("OK, {count} entries")),
            Err(broken) => Notice::error(args.agent, format!("Broken entry at {broken}")),
        },
        (Some("show"), session_id) => {
            let current = session
                .and_then(|paths| paths.id())
                .map(|id| id.to_string());
            let Some(id) = session_id.map(str::to_owned).or(current) else {
                notices.write(Notice::error(args.agent, "/audit show needs a session id"));
                return;
            };
            match chain::show(&log.dir, &id) {
                Ok(lines) if lines.is_empty() => {
                    Notice::info(args.agent, format!("No entries of {id}."))
                }
                Ok(lines) => Notice::info(args.agent, lines.join("\n")),
                Err(error) => {
                    Notice::error(args.agent, format!("Could not read the audit log: {error}"))
                }
            }
        }
        _ => Notice::error(
            args.agent,
            "/audit verify [from-date] or /audit show [session]",
        ),
    };
    notices.write(notice);
}

#[cfg(test)]
mod tests;
