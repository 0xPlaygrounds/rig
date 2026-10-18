//! Session provenance: each session carries a record of the setup it
//! started with (rig's version, the plugins and their versions, each agent's
//! model and provider, the working directory, git and the OS) and of every
//! change to it since, appended and never overwritten: a model changed, the
//! plugins of a `/reload` or a resume, git's HEAD moved as a turn starts.
//! `/session-info` shows it. The record is a saved value of the agent the
//! user talks to, which no model reads.

use std::collections::BTreeMap;
use std::fmt;
use std::path::Path;

use rig_harness::harness_protocol::VERSION;
use rig_harness::prelude::*;
use rig_harness::rig_ecs::WriteLogs;
use serde::{Deserialize, Serialize};

/// The key the record is kept under.
pub const KEY: &str = "session-provenance.record";

/// Records the setup of each session and its changes; adds `/session-info`.
#[derive(Default)]
pub struct SessionProvenancePlugin;

impl Plugin for SessionProvenancePlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<Provenance>();
        // Not listed: the record is too long for a listing.
        let _key = app.saved_as::<Provenance>(KEY);
        app.add_command(
            "session-info",
            "Show the setup this session started with and every change since",
            session_info,
        )
        .add_observer(head_at_turn_start)
        .add_systems(
            Last,
            (record_setup, track_models)
                .chain()
                .before(WriteLogs)
                .run_if(resource_exists::<SessionRestored>),
        );
    }
}

/// The provenance record, on the agent the user talks to, saved with the
/// session.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[reflect(opaque, Component, Saved, Default, Clone, Serialize, Deserialize)]
pub struct Provenance {
    /// The setup the session started with; `None` for a session saved
    /// before this plugin ran.
    pub start: Option<Setup>,
    /// The changes since, oldest first.
    pub entries: Vec<ProvenanceEntry>,
    /// The last known setup: the start with every change applied.
    pub current: Setup,
}

/// A setup an agent ran with.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Setup {
    /// rig's version.
    pub agent_version: String,
    /// The plugins `plugins.toml` loaded, by crate.
    pub plugins: BTreeMap<String, PluginVersion>,
    /// Each agent the user started, by id, with its model.
    pub models: BTreeMap<String, AgentModel>,
    /// The project's directory.
    pub work_dir: String,
    /// Git's state there.
    pub git: Git,
    /// The operating system and architecture.
    pub os: String,
}

/// A plugin crate as loaded.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct PluginVersion {
    /// Its version, or where it is built from when no version is named.
    pub version: String,
    /// The plugins of the crate loaded.
    pub plugins: Vec<String>,
}

/// An agent's model.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct AgentModel {
    /// The catalog reference, `vendor/model`.
    pub model: String,
    /// The provider serving it.
    pub provider: String,
}

/// Git's state in the project's directory.
#[derive(Clone, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum Git {
    /// Not in a git work tree.
    #[default]
    NotARepo,
    /// The project is in another workspace (named by its host), whose git
    /// is not read.
    Remote(String),
    /// A work tree.
    Repo {
        /// The branch checked out; `None` when detached.
        branch: Option<String>,
        /// HEAD's commit; `None` before the first commit.
        head: Option<String>,
        /// Whether the tree has changes not committed.
        dirty: bool,
    },
}

/// One change of the setup.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProvenanceEntry {
    /// When, in milliseconds since the Unix epoch.
    pub at_ms: u64,
    /// What kind of change.
    pub kind: EntryKind,
    /// What changed, a line each.
    pub lines: Vec<String>,
}

/// What kind of change an entry is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum EntryKind {
    /// An agent's model or provider changed.
    Model,
    /// Git's HEAD moved, noticed as a turn started.
    Git,
    /// The session was resumed.
    Resumed,
    /// The agent was rebuilt by `/reload`.
    Reloaded,
}

impl fmt::Display for EntryKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Model => "model",
            Self::Git => "git",
            Self::Resumed => "resumed",
            Self::Reloaded => "reloaded",
        })
    }
}

impl fmt::Display for Git {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::NotARepo => f.write_str("not a git repo"),
            Self::Remote(host) => write!(f, "on {host}, not read"),
            Self::Repo {
                branch,
                head,
                dirty,
            } => {
                let branch = branch.as_deref().unwrap_or("detached");
                let head = head.as_deref().map_or("no commit", short);
                let tree = if *dirty { "dirty" } else { "clean" };
                write!(f, "{branch} at {head}, {tree}")
            }
        }
    }
}

impl fmt::Display for AgentModel {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{} via {}", self.model, self.provider)
    }
}

/// A commit's short hash.
fn short(head: &str) -> &str {
    head.get(..7).unwrap_or(head)
}

impl ProvenanceEntry {
    fn now(kind: EntryKind, lines: Vec<String>) -> Self {
        Self {
            at_ms: now_ms(),
            kind,
            lines,
        }
    }
}

impl Setup {
    /// The whole setup, a line each.
    fn lines(&self) -> Vec<String> {
        let mut lines = vec![
            format!("rig {} on {}", self.agent_version, self.os),
            format!("dir {}", self.work_dir),
            format!("git {}", self.git),
        ];
        lines.extend(
            self.plugins
                .iter()
                .map(|(krate, plugin)| format!("plugin {krate} {}", plugin.version)),
        );
        lines.extend(
            self.models
                .iter()
                .map(|(agent, model)| format!("model {agent}: {model}")),
        );
        lines
    }

    /// What differs from `old` to `self`, a line each.
    fn diff(&self, old: &Self) -> Vec<String> {
        let mut lines = Vec::new();
        if self.agent_version != old.agent_version {
            lines.push(format!(
                "rig {} → {}",
                old.agent_version, self.agent_version
            ));
        }
        if self.os != old.os {
            lines.push(format!("os {} → {}", old.os, self.os));
        }
        if self.work_dir != old.work_dir {
            lines.push(format!("dir {} → {}", old.work_dir, self.work_dir));
        }
        if self.git != old.git {
            lines.push(format!("git {} → {}", old.git, self.git));
        }
        changed(&old.plugins, &self.plugins, "plugin", &mut lines, |p| {
            p.version.clone()
        });
        changed(&old.models, &self.models, "model", &mut lines, |m| {
            m.to_string()
        });
        lines
    }
}

/// Lines for what `new` added, removed or changed from `old`.
fn changed<T: PartialEq>(
    old: &BTreeMap<String, T>,
    new: &BTreeMap<String, T>,
    what: &str,
    lines: &mut Vec<String>,
    show: impl Fn(&T) -> String,
) {
    for (key, value) in new {
        match old.get(key) {
            None => lines.push(format!("{what} {key} {} added", show(value))),
            Some(was) if was != value => {
                lines.push(format!("{what} {key} {} → {}", show(was), show(value)));
            }
            Some(_) => {}
        }
    }
    for (key, was) in old {
        if !new.contains_key(key) {
            lines.push(format!("{what} {key} {} removed", show(was)));
        }
    }
}

/// The record has been set up this run: later changes are recorded.
#[derive(Resource)]
struct Ready;

/// What the setup is read from.
type SetupInputs<'w, 's> = (
    Query<'w, 's, (&'static Name, &'static PluginSource)>,
    Query<'w, 's, (&'static AgentId, &'static Connection), (With<Agent>, Without<SpawnedBy>)>,
    Option<Res<'w, ProjectRoot>>,
    Option<Res<'w, ProjectWorkspace>>,
    Option<Res<'w, Processes>>,
);

/// The setup now.
fn read_setup((plugins, models, root, workspace, processes): &SetupInputs) -> Setup {
    let mut by_crate: BTreeMap<String, PluginVersion> = BTreeMap::new();
    for (name, source) in plugins {
        let entry = by_crate.entry(source.package.clone()).or_default();
        entry.version = version_of(&source.source);
        let short = name.as_str().rsplit("::").next().unwrap_or(name.as_str());
        entry.plugins.push(short.to_owned());
    }
    let models = models
        .iter()
        .map(|(id, connection)| {
            let model = AgentModel {
                model: connection.spec.reference(),
                provider: connection.spec.provider.to_string(),
            };
            (id.0.clone(), model)
        })
        .collect();
    let dir = root
        .as_ref()
        .map(|root| root.0.clone())
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_default();
    Setup {
        agent_version: VERSION.to_owned(),
        plugins: by_crate,
        models,
        work_dir: dir.display().to_string(),
        git: git_state(&dir, workspace.as_deref(), processes.as_deref()),
        os: format!("{} {}", std::env::consts::OS, std::env::consts::ARCH),
    }
}

/// A plugin's version from where `plugins.toml` builds it: `version 0.3`
/// is `0.3`; a path or git source is kept as it is.
fn version_of(source: &str) -> String {
    match source.strip_prefix("version ") {
        Some(version) => version.trim().to_owned(),
        None if source.trim().is_empty() => "(rig's)".to_owned(),
        None => source.trim().to_owned(),
    }
}

/// Git's state in `dir`, read on this machine; another workspace's is not
/// read (it may wait on the network).
fn git_state(
    dir: &Path,
    workspace: Option<&ProjectWorkspace>,
    processes: Option<&Processes>,
) -> Git {
    if let Some(host) = workspace.and_then(|workspace| workspace.0.host()) {
        return Git::Remote(host.to_owned());
    }
    let launcher = processes.map(Processes::launcher).unwrap_or_default();
    let git = |args: &[&str]| -> Option<String> {
        let output = launcher
            .clone()
            .in_dir(dir)
            .command("git")
            .args(args)
            .output()
            .ok()?;
        output
            .status
            .success()
            .then(|| String::from_utf8_lossy(&output.stdout).trim().to_owned())
    };
    if git(&["rev-parse", "--is-inside-work-tree"]).as_deref() != Some("true") {
        return Git::NotARepo;
    }
    Git::Repo {
        branch: git(&["branch", "--show-current"]).filter(|branch| !branch.is_empty()),
        head: git(&["rev-parse", "HEAD"]),
        dirty: git(&["status", "--porcelain"]).is_some_and(|status| !status.is_empty()),
    }
}

/// Once the session is restored and has an agent: starts the record of a
/// new session, or appends how this start differs from the last known
/// setup (a resume, a `/reload`).
fn record_setup(
    ready: Option<Res<Ready>>,
    inputs: SetupInputs,
    agents: MainAgentQuery,
    records: Query<&Provenance>,
    start: Option<Res<AppStart>>,
    mut commands: Commands,
) {
    if ready.is_some() {
        return;
    }
    let Some(main) = main_agent(&agents) else {
        return;
    };
    let now = read_setup(&inputs);
    let reason = start.map_or(StartReason::New, |start| start.0);
    let mut record = records.get(main).cloned().unwrap_or_default();
    match (&record.start, reason) {
        (None, StartReason::New) => record.start = Some(now.clone()),
        (None, ..) => {
            let mut lines = vec!["no provenance recorded before; the setup now:".to_owned()];
            lines.extend(now.lines());
            record
                .entries
                .push(ProvenanceEntry::now(EntryKind::Resumed, lines));
        }
        (Some(_), StartReason::New) => {}
        (Some(_), StartReason::Reloaded) => {
            let mut lines = now.diff(&record.current);
            if lines.is_empty() {
                lines.push("nothing changed".to_owned());
            }
            record
                .entries
                .push(ProvenanceEntry::now(EntryKind::Reloaded, lines));
        }
        // Resumed, restarted, or a reason added later: the session went on.
        (Some(_), _) => {
            let mut lines = now.diff(&record.current);
            if lines.is_empty() {
                lines.push("nothing changed".to_owned());
            }
            record
                .entries
                .push(ProvenanceEntry::now(EntryKind::Resumed, lines));
        }
    }
    record.current = now;
    commands.entity(main).insert(record);
    commands.insert_resource(Ready);
}

/// Appends each change of a user-started agent's model or provider.
fn track_models(
    ready: Option<Res<Ready>>,
    changed: Query<(&AgentId, &Connection), (With<Agent>, Without<SpawnedBy>, Changed<Connection>)>,
    agents: MainAgentQuery,
    mut records: Query<&mut Provenance>,
) {
    if ready.is_none() || changed.is_empty() {
        return;
    }
    let Some(mut record) = main_agent(&agents).and_then(|main| records.get_mut(main).ok()) else {
        return;
    };
    for (id, connection) in &changed {
        let model = AgentModel {
            model: connection.spec.reference(),
            provider: connection.spec.provider.to_string(),
        };
        let was = record.current.models.get(&id.0);
        if was == Some(&model) {
            continue;
        }
        let from = was.map_or_else(|| "none".to_owned(), AgentModel::to_string);
        let line = format!("{}: {from} → {model}", id.0);
        record
            .entries
            .push(ProvenanceEntry::now(EntryKind::Model, vec![line]));
        record.current.models.insert(id.0.clone(), model);
    }
}

/// As a turn of an agent the user started begins: appends git's HEAD if it
/// moved since the last known setup.
fn head_at_turn_start(
    turn: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    users: Query<(), (With<Agent>, Without<SpawnedBy>)>,
    ready: Option<Res<Ready>>,
    (root, workspace, processes): (
        Option<Res<ProjectRoot>>,
        Option<Res<ProjectWorkspace>>,
        Option<Res<Processes>>,
    ),
    agents: MainAgentQuery,
    mut records: Query<&mut Provenance>,
) {
    let of_user = turns
        .get(turn.entity)
        .is_ok_and(|&TurnOf(agent)| users.contains(agent));
    if ready.is_none() || !of_user {
        return;
    }
    let Some(mut record) = main_agent(&agents).and_then(|main| records.get_mut(main).ok()) else {
        return;
    };
    let dir = root
        .map(|root| root.0.clone())
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_default();
    let git = git_state(&dir, workspace.as_deref(), processes.as_deref());
    let head = |git: &Git| match git {
        Git::Repo { branch, head, .. } => Some((branch.clone(), head.clone())),
        _ => None,
    };
    if head(&git) == head(&record.current.git) {
        return;
    }
    let line = format!("{} → {git}", record.current.git);
    record
        .entries
        .push(ProvenanceEntry::now(EntryKind::Git, vec![line]));
    record.current.git = git;
}

/// `/session-info`: the start record and the changes since, oldest first.
fn session_info(
    In(args): In<CommandArgs>,
    agents: MainAgentQuery,
    records: Query<&Provenance>,
    mut notices: MessageWriter<Notice>,
) {
    let record = main_agent(&agents).and_then(|main| records.get(main).ok());
    notices.write(Notice::info(args.agent, describe(record)));
}

/// The record as `/session-info` shows it.
pub fn describe(record: Option<&Provenance>) -> String {
    let mut text = String::from("Session provenance\n");
    match record.and_then(|record| record.start.as_ref()) {
        Some(start) => {
            text.push_str("Started with:\n");
            for line in start.lines() {
                text.push_str(&format!("  {line}\n"));
            }
        }
        None => text.push_str("no provenance recorded\n"),
    }
    for entry in record.iter().flat_map(|record| &record.entries) {
        let mut lines = entry.lines.iter();
        let first = lines.next().map(String::as_str).unwrap_or_default();
        text.push_str(&format!("{} {}: {first}\n", time(entry.at_ms), entry.kind));
        for line in lines {
            text.push_str(&format!("  {line}\n"));
        }
    }
    text.trim_end().to_owned()
}

/// A time as UTC, to the second.
fn time(at_ms: u64) -> String {
    let at = i64::try_from(at_ms)
        .ok()
        .and_then(chrono::DateTime::from_timestamp_millis);
    at.map_or_else(
        || format!("{at_ms} ms"),
        |at| at.format("%Y-%m-%d %H:%M:%S").to_string(),
    )
}

#[cfg(test)]
mod tests;
