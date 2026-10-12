//! FH-17: import sessions from Claude Code and Codex.
//!
//! `/import claude-code [path]` and `/import codex [path]` read the other
//! tool's session files (its default directory, or a file or directory
//! given) and make each a normal saved session of this harness, which
//! `/resume` lists: the user's messages, the model's text, reasoning and
//! tool calls, and the results, in order. `/import` alone lists the
//! sessions found in a picker. `rig --import=claude-code[:PATH]` does the
//! same as the agent starts. Common tools are mapped onto rig's (`Bash` and
//! Codex's `shell` onto `shell`, `Edit` onto `edit`, …); others keep their
//! name and arguments. An imported model turn says where it came from in
//! its rig-core `Origin` (`api` `import`, `provider` the source tool), so
//! no provider replays it as its own. A subagent's (sidechain)
//! conversation becomes a subagent of the imported session. An entry the
//! importer does not understand becomes a note in the transcript, and each
//! import reports a count per kind ([`SessionImported`]). Importing the
//! same source session again appends what is new to the session made the
//! first time.
//!
//! A session is written the one public way an agent log is written: by
//! the kernel's [`Commit`], here in a small app of the kernel's own
//! plugins on the new session's store, which restores what an earlier
//! import wrote first.

mod parse;

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use bevy_ecs::schedule::SingleThreadedExecutor;
use rig_cassette::journal::JsonlDirStore;
use rig_core::message::{Message, Origin as MadeBy};
use rig_core::transcript::{close_pending_with, pending_calls};
use rig_harness::harness_protocol::{Home, SessionDir, SessionId};
use rig_harness::prelude::*;
use rig_harness::rig_ecs::agent::{Halt, MessageId};
use rig_harness::rig_ecs::journal::SessionStore;
use rig_sessions::{Meta, SessionTitle};
use rig_tools::fs::{read_json, write_json};
use serde::{Deserialize, Serialize};

use parse::{Parsed, Step, StepKind};

/// The result a call gets that the source session never answered.
const NO_RESULT: &str = "[no result in the imported session]";
/// Most sessions `/import` lists.
const LISTED: usize = 100;

/// A coding agent sessions are imported from.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Source {
    /// Claude Code: `~/.claude/projects/<project>/<session>.jsonl`.
    ClaudeCode,
    /// Codex: `~/.codex/sessions/<y>/<m>/<d>/rollout-….jsonl`.
    Codex,
}

impl Source {
    /// Every source.
    pub const ALL: [Self; 2] = [Self::ClaudeCode, Self::Codex];

    /// The source named `name` on the command line.
    pub fn named(name: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|source| source.name() == name)
    }

    /// Its name on the command line.
    pub fn name(self) -> &'static str {
        match self {
            Self::ClaudeCode => "claude-code",
            Self::Codex => "codex",
        }
    }

    /// Its name for people.
    pub fn label(self) -> &'static str {
        match self {
            Self::ClaudeCode => "Claude Code",
            Self::Codex => "Codex",
        }
    }

    /// Where it keeps its sessions.
    pub fn default_dir(self) -> Option<PathBuf> {
        let home = std::env::home_dir()?;
        Some(match self {
            Self::ClaudeCode => home.join(".claude/projects"),
            Self::Codex => home.join(".codex/sessions"),
        })
    }
}

/// On an imported agent: the source conversation it was made from, how
/// much of it is imported, and when each message was first made there.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq)]
#[reflect(Component, Saved, Default)]
pub struct ImportedFrom {
    /// The source tool's name, such as `claude-code`.
    pub source: String,
    /// The source's id of the conversation: its session id, then
    /// `#<subagent>` for a subagent's.
    pub key: String,
    /// How many source steps are imported, so a re-import appends only
    /// the rest.
    pub consumed: usize,
    /// The source's time of each message, in milliseconds since the Unix
    /// epoch.
    pub stamps: Vec<(MessageId, u64)>,
}

/// Import the sessions of `source` at `path` (a file or directory), or in
/// its default directory.
#[derive(Event, Clone, Debug)]
pub struct ImportSessions {
    /// The tool they come from.
    pub source: Source,
    /// A session file or a directory of them; `None` for the default.
    pub path: Option<PathBuf>,
    /// The agent the import reports to.
    pub agent: Option<Entity>,
}

/// A session imported, or updated by importing it again.
#[derive(Message, Clone, Debug)]
pub struct SessionImported {
    /// The tool it came from.
    pub source: Source,
    /// The source's id of it.
    pub source_id: String,
    /// The session it is now.
    pub session: SessionId,
    /// Whether the session was made now, not updated.
    pub created: bool,
    /// The messages added to its conversations.
    pub added: usize,
    /// How many source entries of each kind were read; an entry the
    /// importer did not understand is counted as `unknown: <its type>`.
    pub counts: BTreeMap<String, usize>,
}

impl SessionImported {
    /// How many entries the importer did not understand, kept as notes.
    pub fn unknown(&self) -> usize {
        let unknown = self
            .counts
            .iter()
            .filter(|(kind, _)| kind.starts_with("unknown: "));
        unknown.map(|(_, n)| n).sum()
    }

    /// What a notice says of it.
    pub fn summary(&self) -> String {
        let counts: Vec<String> = self
            .counts
            .iter()
            .map(|(kind, n)| format!("{n} × {kind}"))
            .collect();
        let what = if self.created { "Imported" } else { "Updated" };
        format!(
            "{what} {} session {} as {} ({} messages added): {}.",
            self.source.label(),
            self.source_id,
            self.session,
            self.added,
            counts.join(", ")
        )
    }
}

/// `/import`, `--import` and the [`ImportSessions`] event.
#[derive(Default)]
pub struct ImportSessionsPlugin;

impl Plugin for ImportSessionsPlugin {
    fn build(&self, app: &mut App) {
        app.register_type::<ImportedFrom>()
            .add_message::<SessionImported>()
            .init_resource::<RigHome>()
            .add_flag(
                Flag::new(
                    "import",
                    "Import sessions of claude-code or codex, from PATH or their default place",
                )
                .value("TOOL[:PATH]"),
            )
            .add_command(
                "import",
                "Import Claude Code or Codex sessions: /import [claude-code|codex] [path]",
                import_command,
            )
            .add_observer(import)
            .add_systems(Startup, import_on_start);
    }
}

/// `--import=TOOL[:PATH]`.
fn import_on_start(
    invoked: Option<Res<Invoked>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Some(value) = invoked
        .as_deref()
        .and_then(|invoked| invoked.flag_value("import"))
    else {
        return;
    };
    let (name, path) = value
        .split_once(':')
        .map_or((value, None), |(name, path)| (name, Some(path)));
    match Source::named(name) {
        Some(source) => commands.trigger(ImportSessions {
            source,
            path: path.map(expand),
            agent: None,
        }),
        None => {
            notices.write(Notice::error(
                None,
                format!("--import takes claude-code or codex, not `{name}`."),
            ));
        }
    }
}

/// `/import <tool> [path]`, or a picker of the sessions found.
fn import_command(
    In(args): In<CommandArgs>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let line = args.args.trim();
    if line.is_empty() {
        commands.spawn(picker(args.agent));
        return;
    }
    let (name, path) = line
        .split_once(' ')
        .map_or((line, ""), |(name, path)| (name, path.trim()));
    let Some(source) = Source::named(name) else {
        notices.write(Notice::error(
            args.agent,
            "/import takes claude-code or codex, then a path.",
        ));
        return;
    };
    commands.trigger(ImportSessions {
        source,
        path: (!path.is_empty()).then(|| expand(path)),
        agent: Some(args.agent),
    });
}

/// `path` with a leading `~/` made the home directory.
fn expand(path: &str) -> PathBuf {
    match (path.strip_prefix("~/"), std::env::home_dir()) {
        (Some(rest), Some(home)) => home.join(rest),
        _ => PathBuf::from(path),
    }
}

/// The sessions found in each tool's default directory, newest first, with
/// date, project and first message; each imports itself.
fn picker(agent: Entity) -> Picker {
    let mut found: Vec<(u64, Source, PathBuf, Parsed)> = Source::ALL
        .into_iter()
        .flat_map(|source| {
            let files = source
                .default_dir()
                .map(|dir| session_files(&dir))
                .unwrap_or_default();
            files.into_iter().map(move |file| (source, file))
        })
        .filter_map(|(source, file)| {
            let parsed = parse::parse(source, &fs::read_to_string(&file).ok()?);
            Some((parsed.span().1, source, file, parsed))
        })
        .collect();
    found.sort_by_key(|(last, ..)| std::cmp::Reverse(*last));
    found.truncate(LISTED);
    let picker = Picker::new("Import a session").for_agent(agent);
    let picker = if found.is_empty() {
        picker.body("No Claude Code or Codex sessions found.")
    } else {
        picker
    };
    found
        .into_iter()
        .fold(picker, |picker, (last, source, file, parsed)| {
            let date =
                chrono::DateTime::from_timestamp_millis(i64::try_from(last).unwrap_or_default())
                    .map(|at| at.format("%Y-%m-%d %H:%M").to_string())
                    .unwrap_or_default();
            let project = parsed
                .cwd
                .as_deref()
                .and_then(Path::file_name)
                .map(|name| name.to_string_lossy().into_owned());
            let first = parsed
                .first_message()
                .unwrap_or_default()
                .split_whitespace()
                .collect::<Vec<_>>()
                .join(" ");
            let label = format!("{date} · {} · {first}", project.unwrap_or_default());
            let mut picker = picker.item(
                label,
                format!("import {} {}", source.name(), file.display()),
            );
            if let Some(item) = picker.items.last_mut() {
                item.detail = Some(source.label().to_owned());
            }
            picker
        })
}

/// The session files at `path`: it, or every `*.jsonl` under it but a
/// subagent's file of its own.
fn session_files(path: &Path) -> Vec<PathBuf> {
    if path.is_file() {
        return vec![path.to_owned()];
    }
    let Ok(entries) = fs::read_dir(path) else {
        return Vec::new();
    };
    let mut files: Vec<PathBuf> = entries
        .filter_map(Result::ok)
        .map(|entry| entry.path())
        .flat_map(|path| match () {
            () if path.is_dir() && path.file_name().is_some_and(|name| name != "subagents") => {
                session_files(&path)
            }
            () if path.extension().is_some_and(|ext| ext == "jsonl") => vec![path],
            () => Vec::new(),
        })
        .collect();
    files.sort();
    files
}

/// Imports every session [`ImportSessions`] names, and says what came of
/// each.
fn import(
    asked: On<ImportSessions>,
    home: Res<RigHome>,
    mut imported: MessageWriter<SessionImported>,
    mut notices: MessageWriter<Notice>,
) {
    let source = asked.source;
    let Some(path) = asked.path.clone().or_else(|| source.default_dir()) else {
        notices.write(Notice::error(
            asked.agent,
            "No home directory to find sessions in.",
        ));
        return;
    };
    let files = session_files(&path);
    if files.is_empty() {
        let text = format!("No {} sessions at {}.", source.label(), path.display());
        notices.write(Notice::error(asked.agent, text));
    }
    for file in files {
        match import_file(&home.0, source, &file) {
            Ok(done) => {
                notices.write(Notice::info(asked.agent, done.summary()));
                imported.write(done);
            }
            Err(why) => {
                let text = format!("Could not import {}: {why}", file.display());
                notices.write(Notice::error(asked.agent, text));
            }
        }
    }
}

/// What an imported session's directory says of its source, in the
/// plugin's own directory there.
#[derive(Serialize, Deserialize)]
struct Marker {
    source: Source,
    id: String,
    file: PathBuf,
}

fn marker_path(dir: &SessionDir) -> PathBuf {
    dir.path().join("import").join("source.json")
}

/// Imports the session in `file` into its session under `home`: a new one,
/// or the one an earlier import of it made.
fn import_file(home: &Home, source: Source, file: &Path) -> Result<SessionImported, String> {
    let text = fs::read_to_string(file).map_err(|failure| failure.to_string())?;
    let mut parsed = parse::parse(source, &text);
    if parsed.id.is_empty() {
        let stem = file
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned());
        parsed.id = stem.unwrap_or_default();
    }
    let existing = imported_session(home, source, &parsed.id);
    let created = existing.is_none();
    let (session, dir) = match existing {
        Some(found) => found,
        None => new_session(home)?,
    };
    let written = write_session(&dir, source, &parsed).and_then(|added| {
        let marker = Marker {
            source,
            id: parsed.id.clone(),
            file: file.to_owned(),
        };
        fs::create_dir_all(dir.path().join("import")).map_err(|failure| failure.to_string())?;
        write_json(&marker_path(&dir), &marker).map_err(|failure| failure.to_string())?;
        if let Some(cwd) = &parsed.cwd {
            fs::write(dir.directory(), cwd.to_string_lossy().as_bytes())
                .map_err(|failure| failure.to_string())?;
        }
        write_meta(&dir, source, &parsed).map_err(|failure| failure.to_string())?;
        Ok(added)
    });
    let added = match written {
        Ok(added) => added,
        Err(why) => {
            if created {
                fs::remove_dir_all(dir.path()).ok();
            }
            return Err(why);
        }
    };
    Ok(SessionImported {
        source,
        source_id: parsed.id,
        session,
        created,
        added,
        counts: parsed.counts,
    })
}

/// The session an earlier import of `source`'s session `id` made.
fn imported_session(home: &Home, source: Source, id: &str) -> Option<(SessionId, SessionDir)> {
    fs::read_dir(home.sessions())
        .ok()?
        .filter_map(Result::ok)
        .find_map(|entry| {
            let session: SessionId = entry.file_name().to_str()?.parse().ok()?;
            let dir = home.session(&session);
            let marker: Marker = read_json(&marker_path(&dir))?;
            (marker.source == source && marker.id == id).then_some((session, dir))
        })
}

/// A new session's directory under `home`.
fn new_session(home: &Home) -> Result<(SessionId, SessionDir), String> {
    let base = SessionId::generate();
    let mut id = base.clone();
    let mut n = 0;
    while home.session(&id).path().exists() {
        n += 1;
        id = format!("{base}-{n}").parse()?;
    }
    let dir = home.session(&id);
    fs::create_dir_all(dir.path()).map_err(|failure| failure.to_string())?;
    Ok((id, dir))
}

/// The listing `/resume` shows: the first message, `from Claude Code`, and
/// the source's last change as when it was updated; a name given since is
/// kept.
fn write_meta(dir: &SessionDir, source: Source, parsed: &Parsed) -> std::io::Result<()> {
    let mut meta: Meta = read_json(&dir.meta()).unwrap_or_default();
    if meta.title.typed.is_empty() {
        let first = parsed
            .first_message()
            .unwrap_or_default()
            .split_whitespace()
            .collect::<Vec<_>>()
            .join(" ");
        meta.title = SessionTitle {
            typed: rig_tools::shorten(&first, 60),
            ..meta.title
        };
    }
    meta.labels
        .insert("imported".to_owned(), format!("from {}", source.label()));
    let (_, last) = parsed.span();
    meta.updated = meta.updated.max(last);
    write_json(&dir.meta(), &meta)
}

/// One conversation to commit: the agent, its source key and its steps.
struct Job {
    agent: Entity,
    source: Source,
    key: String,
    steps: Vec<Step>,
}

/// Writes `parsed` into the session at `dir` through the kernel: an app of
/// the agent core and the journal on its store restores what is there,
/// then commits each conversation's new steps, a subagent's to an agent
/// spawned by the main one, and writes the logs. The messages added.
fn write_session(dir: &SessionDir, source: Source, parsed: &Parsed) -> Result<usize, String> {
    let mut app = App::new();
    app.insert_resource(SessionStore::new(JsonlDirStore::new(dir.path())))
        .add_plugins((AgentPlugin, JournalPlugin))
        .register_type::<ImportedFrom>();
    let single = |schedule: &mut Schedule| {
        schedule.set_executor(SingleThreadedExecutor::new());
    };
    app.edit_schedule(PreStartup, single)
        .edit_schedule(Startup, single)
        .edit_schedule(Last, single);
    app.finish();
    app.cleanup();
    let world = app.world_mut();
    world.run_schedule(PreStartup);
    world.run_schedule(Startup);
    let main = agent_of(world, &parsed.id, None).ok_or("the session has no agent")?;
    let job = Job {
        agent: main,
        source,
        key: parsed.id.clone(),
        steps: parsed.main.clone(),
    };
    let mut added = world
        .run_system_cached_with(commit_steps, job)
        .map_err(|failure| failure.to_string())?;
    for (chain, steps) in &parsed.sidechains {
        let key = format!("{}#{chain}", parsed.id);
        let agent = agent_of(world, &key, Some(main)).ok_or("no subagent")?;
        let job = Job {
            agent,
            source,
            key,
            steps: steps.clone(),
        };
        added += world
            .run_system_cached_with(commit_steps, job)
            .map_err(|failure| failure.to_string())?;
    }
    world.run_schedule(Last);
    Ok(added)
}

/// The agent imported from `key`; else, for the main conversation, the
/// agent the core spawned, and for a subagent's, a new agent spawned by
/// `parent`.
fn agent_of(world: &mut World, key: &str, parent: Option<Entity>) -> Option<Entity> {
    let mut agents =
        world.query_filtered::<(Entity, Option<&ImportedFrom>, Has<SpawnedBy>), With<Agent>>();
    let mut fresh = None;
    for (agent, imported, spawned) in agents.iter(world) {
        match imported {
            Some(imported) if imported.key == key => return Some(agent),
            None if !spawned => fresh = fresh.or(Some(agent)),
            _ => {}
        }
    }
    match parent {
        None => fresh,
        Some(parent) => {
            let agent = world
                .spawn((Name::new("imported subagent"), Agent, SpawnedBy(parent)))
                .id();
            world.flush();
            Some(agent)
        }
    }
}

/// Commits the steps of `job` its agent has not imported yet: each result
/// paired with its call, a call the source never answered closed with
/// [`NO_RESULT`], an orphan result and an unknown entry as a note once no
/// call waits for its result, and a last user message left for the next
/// turn, so a resume asks nothing by itself. The messages added.
fn commit_steps(
    In(job): In<Job>,
    mut agents: Query<(&mut Conversation, Option<&ImportedFrom>)>,
    mut commit: Commit,
    mut commands: Commands,
) -> usize {
    let agent = job.agent;
    let Ok((mut conversation, imported)) = agents.get_mut(agent) else {
        return 0;
    };
    let mut imported = imported.cloned().unwrap_or_default();
    let mut added = 0;
    let mut notes: Vec<(String, String)> = Vec::new();
    let mut add =
        |conversation: &mut Conversation, message: Message, origin: Option<Origin>, at_ms: u64| {
            commit.message_from(agent, conversation, message, origin);
            added += 1;
            let last = conversation.messages().len().checked_sub(1);
            if let Some(id) = last.and_then(|at| conversation.id(at))
                && !imported.stamps.iter().any(|(known, _)| *known == id)
            {
                imported.stamps.push((id, at_ms));
            }
        };
    let close = |conversation: &mut Conversation,
                 add: &mut dyn FnMut(&mut Conversation, Message, Option<Origin>, u64),
                 at_ms| {
        let pending = pending_calls(conversation.messages());
        if !pending.is_empty() {
            add(
                conversation,
                close_pending_with(&pending, NO_RESULT),
                None,
                at_ms,
            );
        }
    };
    let flush = |conversation: &mut Conversation,
                 add: &mut dyn FnMut(&mut Conversation, Message, Option<Origin>, u64),
                 notes: &mut Vec<(String, String)>,
                 at_ms| {
        for (kind, raw) in notes.drain(..) {
            let text = format!("[imported {} entry `{kind}`] {raw}", job.source.label());
            let origin = Origin::plugin("import").titled(kind);
            add(conversation, Message::user(text), Some(origin), at_ms);
        }
    };
    let skip = imported.consumed;
    for step in job.steps.iter().skip(skip) {
        let at_ms = step.at_ms;
        match &step.kind {
            StepKind::User(text) => {
                close(&mut conversation, &mut add, at_ms);
                flush(&mut conversation, &mut add, &mut notes, at_ms);
                add(&mut conversation, Message::user(text.clone()), None, at_ms);
            }
            StepKind::Assistant { content, model } => {
                close(&mut conversation, &mut add, at_ms);
                flush(&mut conversation, &mut add, &mut notes, at_ms);
                let mut turn = AssistantMessage::new(content.clone());
                turn.origin = Some(MadeBy::new("import", job.source.name(), model.clone()));
                add(&mut conversation, Message::Assistant(turn), None, at_ms);
            }
            StepKind::Results(results) => {
                let pending = pending_calls(conversation.messages());
                let (paired, orphans): (Vec<_>, Vec<_>) = results
                    .iter()
                    .cloned()
                    .partition(|result| pending.iter().any(|call| call.id == result.call));
                if !paired.is_empty() {
                    add(
                        &mut conversation,
                        Message::tool_results(paired),
                        None,
                        at_ms,
                    );
                }
                for orphan in orphans {
                    let text = orphan
                        .content
                        .iter()
                        .filter_map(|content| content.as_text())
                        .collect::<Vec<_>>()
                        .join("\n");
                    notes.push((
                        "tool result of no call".to_owned(),
                        format!("{}: {text}", orphan.call),
                    ));
                }
                if pending_calls(conversation.messages()).is_empty() {
                    flush(&mut conversation, &mut add, &mut notes, at_ms);
                }
            }
            StepKind::Note { kind, raw } => {
                notes.push((kind.clone(), raw.clone()));
                if pending_calls(conversation.messages()).is_empty() {
                    flush(&mut conversation, &mut add, &mut notes, at_ms);
                }
            }
        }
    }
    let at_ms = job.steps.last().map(|step| step.at_ms).unwrap_or_default();
    close(&mut conversation, &mut add, at_ms);
    flush(&mut conversation, &mut add, &mut notes, at_ms);
    commit.halt(agent, &mut conversation, Halt::Kept);
    imported.source = job.source.name().to_owned();
    imported.key = job.key;
    imported.consumed = job.steps.len();
    commands.entity(agent).insert(imported);
    added
}

#[cfg(test)]
mod tests;
