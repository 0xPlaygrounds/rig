//! FH-17: import sessions from Claude Code and Codex.
//!
//! `rig import claude-code [path]` and `rig import codex [path]` read the
//! other tool's session files (its default directory, or a file or
//! directory given) and make each a normal saved session of this harness,
//! which `/resume` lists: the user's messages, the model's text, reasoning
//! and tool calls, and the results, in order, each logged at the source's
//! time. `/import <tool> [path]` does the same inside a session, and
//! `/import` alone lists the sessions found in a picker. Common tools are
//! mapped onto rig's (`Bash` and Codex's `shell` onto `shell`, `Edit` onto
//! `edit`, …); others keep their name and arguments. An imported model
//! turn says where it came from in its rig-core `Origin` (`api` `import`,
//! `provider` the source tool), so no provider replays it as its own. A
//! subagent's (sidechain) conversation becomes a subagent of the imported
//! session. An entry the importer does not understand becomes a note in
//! the transcript, and each import reports a count per kind
//! ([`SessionImported`]). Importing the same source session again appends
//! what is new to the session made the first time.
//!
//! A session is written with [`Session::save`] into a new session's store
//! ([`Home::new_session`], [`StoreOpener`]), with the source it came from
//! in the plugin's own file there.

mod parse;

use std::collections::BTreeMap;
use std::fs;
use std::io;
use std::path::{Path, PathBuf};

use rig_core::conversation::Conversation;
use rig_core::conversation::pairing::{close_pending_with, pending_calls};
use rig_core::message::{Message, Origin as MadeBy};
use rig_harness::harness_protocol::{SessionDir, SessionId};
use rig_harness::prelude::*;
use rig_harness::rig_memory::{SessionMeta, title_of};
use serde::{Deserialize, Serialize};

use parse::{Parsed, Step, StepKind};

/// The result a call gets that the source session never answered.
const NO_RESULT: &str = "[no result in the imported session]";
/// Most sessions `/import` lists.
const LISTED: usize = 100;
/// The command, and the front that runs it.
const COMMAND: &str = "import";
/// The plugin's file in an imported session's store: its [`Marker`].
const MARKER: &str = "import/source.json";

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

/// `rig import`, `/import` and the [`ImportSessions`] event.
#[derive(Default)]
pub struct ImportSessionsPlugin;

impl Plugin for ImportSessionsPlugin {
    fn build(&self, app: &mut App) {
        let front = Front::new(COMMAND).sessionless();
        app.add_message::<SessionImported>()
            .init_resource::<RigHome>()
            .add_front(front.on_command(
                COMMAND,
                "claude-code|codex [path]",
                "Import Claude Code or Codex sessions, from path or their default place",
            ))
            .add_command(
                COMMAND,
                "Import Claude Code or Codex sessions: /import [claude-code|codex] [path]",
                import_command,
            )
            .add_observer(import);
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front(COMMAND) {
            app.add_systems(Startup, import_command_line);
        }
    }
}

/// `rig import <tool> [path]`: prints what it imported, and exits.
fn import_command_line(
    invoked: Res<Invoked>,
    (home, stores): (Res<RigHome>, Res<StoreOpener>),
    mut imported: MessageWriter<SessionImported>,
    mut commands: Commands,
) {
    let arguments = invoked
        .command()
        .map_or(&[][..], |(_, arguments)| arguments);
    let (name, path) = match arguments {
        [name] => (name, None),
        [name, path] => (name, Some(PathBuf::from(path))),
        _ => (&String::new(), None),
    };
    let reason = match Source::named(name) {
        None => {
            eprintln!("usage: rig import claude-code|codex [path]");
            ExitReason::Failed(2)
        }
        Some(source) => match import_all((&home.0, &stores), source, path) {
            Err(why) => {
                eprintln!("rig import: {why}");
                ExitReason::Failed(1)
            }
            Ok(done) => {
                let mut failed = false;
                for result in done {
                    match result {
                        Ok(done) => {
                            println!("{}", done.summary());
                            imported.write(done);
                        }
                        Err(why) => {
                            failed = true;
                            eprintln!("rig import: {why}");
                        }
                    }
                }
                if failed {
                    ExitReason::Failed(1)
                } else {
                    ExitReason::Quit
                }
            }
        },
    };
    commands.trigger(Exit(reason));
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
    let mut picker = Picker::new("Import a session").for_agent(agent);
    if found.is_empty() {
        picker = picker.body("No Claude Code or Codex sessions found.");
    }
    for (last, source, file, parsed) in found {
        let date = chrono::DateTime::from_timestamp_millis(i64::try_from(last).unwrap_or_default())
            .map(|at| at.format("%Y-%m-%d %H:%M").to_string())
            .unwrap_or_default();
        let project = parsed
            .cwd
            .as_deref()
            .and_then(Path::file_name)
            .map(|name| name.to_string_lossy().into_owned());
        picker.items.push(PickItem {
            label: format!(
                "{date} · {} · {}",
                project.unwrap_or_default(),
                first_line(&parsed)
            ),
            command: format!("{COMMAND} {} {}", source.name(), file.display()),
            detail: Some(source.label().to_owned()),
            checked: false,
        });
    }
    picker
}

/// The first thing the user said, as `/resume` titles it.
fn first_line(parsed: &Parsed) -> String {
    title_of(parsed.first_message().unwrap_or_default())
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
    (home, stores): (Res<RigHome>, Res<StoreOpener>),
    mut imported: MessageWriter<SessionImported>,
    mut notices: MessageWriter<Notice>,
) {
    let done = match import_all((&home.0, &stores), asked.source, asked.path.clone()) {
        Ok(done) => done,
        Err(why) => {
            notices.write(Notice::error(asked.agent, why));
            return;
        }
    };
    for result in done {
        match result {
            Ok(done) => {
                notices.write(Notice::info(asked.agent, done.summary()));
                imported.write(done);
            }
            Err(why) => {
                notices.write(Notice::error(asked.agent, why));
            }
        }
    }
}

/// Imports each session file of `source` at `path`, or in its default
/// directory; what came of each. An error when there is none.
fn import_all(
    (home, stores): (&Home, &StoreOpener),
    source: Source,
    path: Option<PathBuf>,
) -> Result<Vec<Result<SessionImported, String>>, String> {
    let path = path
        .or_else(|| source.default_dir())
        .ok_or("no home directory to find sessions in")?;
    let files = session_files(&path);
    if files.is_empty() {
        return Err(format!(
            "no {} sessions at {}",
            source.label(),
            path.display()
        ));
    }
    let import = |file: PathBuf| {
        import_file((home, stores), source, &file)
            .map_err(|why| format!("could not import {}: {why}", file.display()))
    };
    Ok(files.into_iter().map(import).collect())
}

/// What an imported session keeps of its source, in its store.
#[derive(Serialize, Deserialize)]
struct Marker {
    source: Source,
    id: String,
    file: PathBuf,
    /// Each source conversation imported.
    conversations: Vec<Imported>,
}

/// A source conversation imported into an agent log of the session.
#[derive(Serialize, Deserialize)]
struct Imported {
    /// The source's id of it: its session id, then `#<subagent>` for a
    /// subagent's.
    key: String,
    /// The agent whose log holds it.
    agent: String,
    /// How many source steps are imported, so a re-import appends only
    /// the rest.
    consumed: usize,
}

/// The marker of the session in `store`, if an import made it.
fn marker_of(store: &dyn Store) -> Option<Marker> {
    serde_json::from_slice(&store.read_file(MARKER).ok()?).ok()
}

/// Imports the session in `file` into its session under `home`, kept in a
/// store `stores` open: a new one, or the one an earlier import of it made.
fn import_file(
    (home, stores): (&Home, &StoreOpener),
    source: Source,
    file: &Path,
) -> io::Result<SessionImported> {
    let mut parsed = parse::parse(source, &fs::read_to_string(file)?);
    if parsed.id.is_empty() {
        let stem = file
            .file_stem()
            .map(|stem| stem.to_string_lossy().into_owned());
        parsed.id = stem.unwrap_or_default();
    }
    let (session, dir, marker) = match imported_session((home, stores), source, &parsed.id) {
        Some((session, dir, _)) if dir.runs_elsewhere() => {
            return Err(io::Error::other(format!(
                "its session {session} is open; close it to import again"
            )));
        }
        Some(found) => found,
        None => {
            let (session, dir) = home.new_session()?;
            let marker = Marker {
                source,
                id: parsed.id.clone(),
                file: file.to_owned(),
                conversations: Vec::new(),
            };
            (session, dir, marker)
        }
    };
    let created = marker.conversations.is_empty();
    let written = stores
        .open(&dir)
        .map_err(io::Error::from)
        .and_then(|store| write(&*store, &dir, marker, &parsed))
        .and_then(|added| dir.mark_saved().map(|()| added));
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

/// The session an earlier import of `source`'s session `id` made, with its
/// marker.
fn imported_session(
    (home, stores): (&Home, &StoreOpener),
    source: Source,
    id: &str,
) -> Option<(SessionId, SessionDir, Marker)> {
    fs::read_dir(home.sessions())
        .ok()?
        .filter_map(Result::ok)
        .find_map(|entry| {
            let session: SessionId = entry.file_name().to_str()?.parse().ok()?;
            let dir = home.session(&session);
            let marker = marker_of(&*stores.open(&dir).ok()?)?;
            (marker.source == source && marker.id == id).then_some((session, dir, marker))
        })
}

/// Brings the session in `store` up to `parsed`: each conversation's new
/// steps appended to its agent's log, a subagent's spawned by the main
/// agent; then its marker, its directory and its `/resume` listing. The
/// messages added.
fn write(
    store: &dyn Store,
    dir: &SessionDir,
    mut marker: Marker,
    parsed: &Parsed,
) -> io::Result<usize> {
    let mut session = Session::read(store)?;
    let source = marker.source;
    let mut conversations = vec![(parsed.id.clone(), &parsed.main)];
    for (chain, steps) in &parsed.sidechains {
        conversations.push((format!("{}#{chain}", parsed.id), steps));
    }
    let mut main = None;
    let mut added = 0;
    for (key, steps) in conversations {
        let known = marker
            .conversations
            .iter()
            .position(|known| known.key == key);
        let (agent, consumed) = match known.and_then(|at| marker.conversations.get(at)) {
            Some(known) => (known.agent.clone(), known.consumed),
            None => (AgentId::default().0, 0),
        };
        let at = match session.agents.iter().position(|saved| saved.id == agent) {
            Some(at) => at,
            None => {
                session.agents.push(SavedAgent {
                    id: agent.clone(),
                    parent: main.clone(),
                    ..SavedAgent::default()
                });
                session.agents.len().saturating_sub(1)
            }
        };
        if let Some(saved) = session.agents.get_mut(at) {
            let mut importing = Importing {
                conversation: &mut saved.conversation,
                source,
                notes: Vec::new(),
                added: 0,
            };
            importing.steps(steps.get(consumed..).unwrap_or_default());
            added += importing.added;
        }
        let imported = Imported {
            key,
            agent: agent.clone(),
            consumed: steps.len(),
        };
        match known.and_then(|at| marker.conversations.get_mut(at)) {
            Some(known) => *known = imported,
            None => marker.conversations.push(imported),
        }
        main = main.or(Some(agent));
    }
    session.save(store)?;
    store.write_file(MARKER, &serde_json::to_vec_pretty(&marker)?)?;
    if let Some(cwd) = &parsed.cwd {
        fs::write(dir.directory(), cwd.to_string_lossy().as_bytes())?;
    }
    let mut meta = SessionMeta::read(store).unwrap_or_default();
    if meta.title.is_empty() {
        meta.title = first_line(parsed);
    }
    meta.labels
        .insert("imported".to_owned(), format!("from {}", source.label()));
    meta.updated = meta.updated.max(parsed.span().1);
    meta.write(store)?;
    Ok(added)
}

/// Appends a source conversation's steps to an agent's conversation, each
/// message stamped with the source's time.
struct Importing<'a> {
    conversation: &'a mut Conversation,
    source: Source,
    /// Notes held back while a call waits for its result.
    notes: Vec<(String, String)>,
    /// The messages added.
    added: usize,
}

impl Importing<'_> {
    /// Adds each step: each result paired with its call, a call the source
    /// never answered closed with [`NO_RESULT`], an orphan result and an
    /// unknown entry as a note once no call waits for its result, and a
    /// last user message left for the next turn, so a resume asks nothing
    /// by itself.
    fn steps(&mut self, steps: &[Step]) {
        for step in steps {
            self.step(step);
        }
        let at_ms = steps.last().map(|step| step.at_ms).unwrap_or_default();
        self.close(at_ms);
        self.flush(at_ms);
        self.dated(at_ms, |conversation| conversation.halt(Halt::Kept));
    }

    fn step(&mut self, step: &Step) {
        let at_ms = step.at_ms;
        match &step.kind {
            StepKind::User(text) => {
                self.close(at_ms);
                self.flush(at_ms);
                self.add(Message::user(text.clone()), None, at_ms);
            }
            StepKind::Assistant { content, model } => {
                self.close(at_ms);
                self.flush(at_ms);
                let mut turn = AssistantMessage::new(content.clone());
                let source = self.source.name();
                turn.origin = Some(MadeBy::new("import", source, model.clone()));
                self.add(Message::Assistant(turn), None, at_ms);
            }
            StepKind::Results(results) => {
                let pending = pending_calls(self.conversation.messages());
                let (paired, orphans): (Vec<_>, Vec<_>) = results
                    .iter()
                    .cloned()
                    .partition(|result| pending.iter().any(|call| call.id == result.call));
                if !paired.is_empty() {
                    self.add(Message::tool_results(paired), None, at_ms);
                }
                for orphan in orphans {
                    let text = orphan
                        .content
                        .iter()
                        .filter_map(|content| content.as_text())
                        .collect::<Vec<_>>()
                        .join("\n");
                    let kind = "tool result of no call".to_owned();
                    self.notes.push((kind, format!("{}: {text}", orphan.call)));
                }
                self.flush_when_answered(at_ms);
            }
            StepKind::Note { kind, raw } => {
                self.notes.push((kind.clone(), raw.clone()));
                self.flush_when_answered(at_ms);
            }
        }
    }

    /// Adds `message`, by `author` (the user's own without one), logged
    /// at `at_ms`.
    fn add(&mut self, message: Message, author: Option<Author>, at_ms: u64) {
        let entry = Entry::new(message);
        let entry = match author {
            Some(author) => entry.by(author),
            None => entry,
        };
        self.dated(at_ms, |conversation| conversation.push(entry));
        self.added += 1;
    }

    /// Makes `edit` as made at `at_ms`, the source's time; now when the
    /// source gave none.
    fn dated<R>(&mut self, at_ms: u64, edit: impl FnOnce(&mut Conversation) -> R) -> R {
        if at_ms == 0 {
            return edit(self.conversation);
        }
        self.conversation.dated(at_ms, edit)
    }

    /// Answers the calls still waiting with [`NO_RESULT`].
    fn close(&mut self, at_ms: u64) {
        let pending = pending_calls(self.conversation.messages());
        if !pending.is_empty() {
            self.add(close_pending_with(&pending, NO_RESULT), None, at_ms);
        }
    }

    /// Adds the notes held back.
    fn flush(&mut self, at_ms: u64) {
        let label = self.source.label();
        for (kind, raw) in std::mem::take(&mut self.notes) {
            let text = format!("[imported {label} entry `{kind}`] {raw}");
            let author = Author::plugin("import").titled(kind);
            self.add(Message::user(text), Some(author), at_ms);
        }
    }

    /// Adds the notes held back once no call waits for its result.
    fn flush_when_answered(&mut self, at_ms: u64) {
        if pending_calls(self.conversation.messages()).is_empty() {
            self.flush(at_ms);
        }
    }
}

#[cfg(test)]
mod tests;
