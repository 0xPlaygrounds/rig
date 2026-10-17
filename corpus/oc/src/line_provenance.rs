//! OC-21: which prompt wrote this line? `git blame` for agent work.
//!
//! Every file change an agent's tool call makes is recorded per project,
//! in the project's [`PluginData`] (outside the repository, kept across
//! sessions): the session, agent, turn, the prompt the user typed to start
//! the turn, the tool call and the time, and which lines it wrote. The
//! file tools publish what they changed (`FilesChanged`), and so does any
//! call that may write anywhere, such as `shell`, through the harness's
//! checkpoint around it; each change carries the file's text before and
//! its hunks.
//!
//! Each line's origin is carried forward through every later change, by
//! the change's hunks for an agent's and by a line diff against the text
//! last seen for the user's, so it follows code that moves. A change the
//! user made is found where the text an agent's change found, or the file
//! when asked, is not what was last seen: a line of an agent's they
//! changed is marked "changed by you", a line they added is no agent's.
//!
//! - `/provenance <file>[:<line>[-<line>]]` lists the lines in runs of one
//!   origin, and offers a picker of the agents' entries: choosing one opens
//!   that turn's session, read from its store, in a read-only
//!   [`TranscriptPane`] over the screen scrolled to the turn; Esc closes it.
//! - `/provenance --forget <session>` drops one session's records.
//! - `rig provenance <file>[:<line>[-<line>]] [--json]` prints the same from
//!   the command line, without a session; `--forget` works there too.

use std::io::Write as _;
use std::path::Path;

use bevy_log::warn;
use rig_basics::checkpoint::CheckpointPlugin;
use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::Conversation as Log;
use rig_harness::rig_core::now_ms;
use rig_harness::rig_core::tool::{ChangeKind, FilesChanged};
use rig_harness::rig_memory::Session;
use rig_tui::crossterm::event::KeyCode;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::{
    ItemRow, KeyBinding, KeyFocus, KeyPress, KeyPressed, KeyScope, PaneKeys, Placement, ScrollTo,
    TranscriptItem, TranscriptPane, TranscriptSource, TuiPanel,
};
use std::collections::HashMap;

mod store;
pub use store::{LineOrigin, Logged, ProvenanceStore, Record, Span, TrackedFile};

#[cfg(test)]
mod tests;

/// The slash command, the command-line command, and the author of notes.
pub const NAME: &str = "provenance";

/// The log's file in the project's plugin data, a [`Logged`] per line.
pub const STORE: &str = "line-provenance/log.jsonl";

/// Its usage.
const USAGE: &str = "<file>[:<line>[-<line>]] [--json] | --forget <session>";

/// Records which tool call wrote each line, and answers `/provenance` and
/// `rig provenance`.
#[derive(Default)]
pub struct LineProvenancePlugin;

impl Plugin for LineProvenancePlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<CheckpointPlugin>() {
            app.add_plugins(CheckpointPlugin);
        }
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .add_observer(call_settled)
            .add_command(
                NAME,
                "Which agent, session, turn and prompt wrote these lines: /provenance \
                 <file>[:<line>[-<line>]] | --forget <session>",
                provenance,
            )
            .add_front(Front::new(NAME).sessionless().on_command(
                NAME,
                USAGE,
                "Which agent, session, turn and prompt wrote a file's lines",
            ));
    }

    fn finish(&self, app: &mut App) {
        if app.runs_front(NAME) {
            app.add_systems(Startup, command_line);
        }
    }
}

/// What a run of `rig provenance` printed, and its exit status; inserted
/// once it ran.
#[derive(Resource, Clone, Debug, Default, PartialEq, Eq)]
pub struct ProvenanceRun {
    /// What went to stdout.
    pub stdout: String,
    /// 0, or 2 when the arguments were refused.
    pub status: u8,
}

/// The project's store, folded from its log; empty when there is none yet.
pub fn load(data: &PluginData) -> ProvenanceStore {
    let log = data.project.read_file(STORE).unwrap_or_default();
    ProvenanceStore::fold(&log)
}

/// Adds a line to the project's log.
fn append(data: &PluginData, logged: &Logged) {
    let appended = serde_json::to_vec(logged)
        .map_err(std::io::Error::from)
        .and_then(|mut line| {
            line.push(b'\n');
            data.project.append_file(STORE, &line)
        });
    if let Err(why) = appended {
        warn!("line provenance not saved: {why}");
    }
}

/// `path` as the store names it: from the project root, with `/`.
fn key(root: &Path, path: &Path) -> String {
    let canonical = root.canonicalize().ok();
    let relative = path
        .strip_prefix(root)
        .ok()
        .or_else(|| path.strip_prefix(canonical.as_deref()?).ok())
        .unwrap_or(path);
    let parts: Vec<_> = relative
        .components()
        .map(|part| part.as_os_str().to_string_lossy())
        .collect();
    parts.join("/")
}

/// The agents, as a record names them and finds their prompt.
type Agents<'w, 's> = Query<
    'w,
    's,
    (
        &'static AgentId,
        Option<&'static Name>,
        &'static Conversation,
        Option<&'static SpawnedBy>,
        Option<&'static ActiveTurn>,
    ),
>;

/// Logs what a settled tool call changed, with who made it and why.
fn call_settled(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &Published, &ForAgent, &CallOf)>,
    turns: Query<&Turn>,
    agents: Agents,
    data: Option<Res<PluginData>>,
    root: Res<ProjectRoot>,
    session: Option<Res<SessionPaths>>,
) {
    let Ok((request, published, &ForAgent(agent), &CallOf(turn))) = calls.get(settled.entity)
    else {
        return;
    };
    let Ok(Some(FilesChanged(changes))) = published.0.get::<FilesChanged>() else {
        return;
    };
    let (Some(data), Ok(turn), Ok((id, name, conversation, ..))) =
        (data, turns.get(turn), agents.get(agent))
    else {
        return;
    };
    if changes.is_empty() {
        return;
    }
    let record = Record {
        session: session
            .as_ref()
            .and_then(|paths| paths.id())
            .map(|id| id.to_string())
            .unwrap_or_default(),
        agent: name.map_or_else(|| id.short().to_owned(), |name| name.as_str().to_owned()),
        agent_id: id.0.clone(),
        turn: turn.index.saturating_add(1),
        asked: conversation.request(turn.start).map(|entry| entry.key()),
        prompt: prompt(agent, turn.start, &agents, &turns),
        tool: request.call.function.name.as_str().to_owned(),
        call_id: request.call.id.to_string(),
        at_ms: now_ms(),
    };
    let changes = changes
        .into_iter()
        .map(|change| {
            let name = key(&root.0, &root.0.join(&change.path));
            if change.kind != ChangeKind::Deleted && change.after_text().is_none() {
                warn!(
                    "line provenance: {} does not say {name}'s text, so its lines are not tracked",
                    request.call.id
                );
            }
            (name, change)
        })
        .collect();
    append(&data, &Logged::Changed { record, changes });
}

/// What the user typed to start the turn of `agent` that began at
/// `first`, else to start its parent's running turn, up to the root.
fn prompt(
    mut agent: Entity,
    mut first: MessageId,
    agents: &Agents,
    turns: &Query<&Turn>,
) -> String {
    for _ in 0..64 {
        let Ok((_, _, conversation, parent, _)) = agents.get(agent) else {
            break;
        };
        let asked = conversation.request(first).map(|entry| entry.index);
        if let Some(text) = asked.and_then(|at| typed(conversation, at)) {
            return text;
        }
        let parents = parent.and_then(|&SpawnedBy(parent)| {
            let (.., active) = agents.get(parent).ok()?;
            let turn = turns.get(active?.turn()).ok()?;
            Some((parent, turn.start))
        });
        let Some(next) = parents else {
            break;
        };
        (agent, first) = next;
    }
    String::new()
}

/// What `/provenance` or `rig provenance` was asked.
#[derive(Debug, PartialEq, Eq)]
enum Asked {
    /// The lines of a file.
    Lines {
        /// The file, as typed.
        path: String,
        /// The lines, both ends included.
        range: Option<(usize, usize)>,
        /// Whether to print JSON.
        json: bool,
    },
    /// Drop a session's records.
    Forget(String),
    /// Show the turn of a record.
    Open(u64),
}

/// Reads the words after the command's name.
fn parse(words: &[String]) -> Result<Asked, String> {
    let json = words.iter().any(|word| word == "--json");
    let mut rest = words
        .iter()
        .filter(|word| *word != "--json")
        .map(String::as_str);
    match (rest.next(), rest.next(), rest.next()) {
        (Some("--forget"), Some(session), None) => Ok(Asked::Forget(session.to_owned())),
        (Some("--open"), Some(key), None) => key
            .parse()
            .map(Asked::Open)
            .map_err(|_| format!("not an entry: {key}")),
        (Some(target), None, None) if !target.starts_with("--") => {
            let (path, range) = target_of(target);
            Ok(Asked::Lines {
                path: path.to_owned(),
                range,
                json,
            })
        }
        _ => Err(format!("usage: {NAME} {USAGE}")),
    }
}

/// `file`, `file:12` or `file:12-14`.
fn target_of(target: &str) -> (&str, Option<(usize, usize)>) {
    let Some((path, lines)) = target.rsplit_once(':') else {
        return (target, None);
    };
    let range = match lines.split_once('-') {
        Some((from, to)) => from.parse().ok().zip(to.parse().ok()),
        None => lines.parse().ok().map(|line| (line, line)),
    };
    match range {
        Some(range) => (path, Some(range)),
        None => (target, None),
    }
}

/// What `/provenance` and `rig provenance` answer from: the project's
/// store, its files, and the saved sessions, for their names.
#[derive(bevy_ecs::system::SystemParam)]
struct Asking<'w> {
    data: Option<Res<'w, PluginData>>,
    root: Res<'w, ProjectRoot>,
    workspace: Res<'w, ProjectWorkspace>,
    home: Option<Res<'w, RigHome>>,
}

/// The answer to `/provenance` or `rig provenance` asked for lines.
struct Answer {
    /// The file, from the project root.
    name: String,
    /// The store, with what the user changed in the file taken in.
    store: ProvenanceStore,
    /// The lines in runs of one origin.
    spans: Vec<Span>,
    /// Each record's session as shown, by session id.
    sessions: HashMap<String, String>,
}

impl Asking<'_> {
    /// The lines asked for, after taking in what the user changed in the
    /// file since it was last seen.
    fn lines(&self, data: &PluginData, path: &str, range: Option<(usize, usize)>) -> Answer {
        let file = self.root.0.join(path);
        let name = key(&self.root.0, &file);
        let now = self.workspace.0.read_text(&file).ok();
        let mut store = load(data);
        store.user_changed(&name, now.as_deref());
        let total = now.as_deref().map_or(0, |text| text.lines().count());
        let spans = store.spans(&name, total, range);
        let sessions = spans
            .iter()
            .filter_map(|span| store.records.get(&span.origin?.record))
            .map(|record| (record.session.clone(), self.session_label(&record.session)))
            .collect();
        Answer {
            name,
            store,
            spans,
            sessions,
        }
    }

    /// A session as it is shown: the name the user gave it, else the
    /// start of its id.
    fn session_label(&self, session: &str) -> String {
        let named = self.home.as_ref().and_then(|home| {
            let id: SessionId = session.parse().ok()?;
            home.0.session(&id).name()
        });
        named.unwrap_or_else(|| session.get(..8).unwrap_or(session).to_owned())
    }

    /// Drops the records of the session `session` names or identifies.
    fn forget(&self, data: &PluginData, session: &str) -> String {
        let id = self
            .home
            .as_ref()
            .and_then(|home| home.0.find(session).ok())
            .map_or_else(|| session.to_owned(), |id| id.to_string());
        let gone = load(data).forget(&id);
        append(data, &Logged::Forgot(id));
        format!("Forgot {gone} changes of session {session}.")
    }
}

/// A record's day, in the user's time zone, such as `Oct 3`.
fn day(at_ms: u64) -> String {
    let at = i64::try_from(at_ms)
        .ok()
        .and_then(chrono::DateTime::from_timestamp_millis);
    at.map(|at| {
        at.with_timezone(&chrono::Local)
            .format("%b %-d")
            .to_string()
    })
    .unwrap_or_default()
}

impl Answer {
    /// The record that wrote a span, if an agent did.
    fn record(&self, span: &Span) -> Option<(&Record, bool)> {
        let origin = span.origin?;
        Some((
            self.store.records.get(&origin.record)?,
            origin.changed_by_you,
        ))
    }

    /// The session of a record, as shown.
    fn session<'a>(&'a self, record: &'a Record) -> &'a str {
        self.sessions
            .get(&record.session)
            .map_or(record.session.as_str(), String::as_str)
    }

    /// Who wrote a span, in a line.
    fn described(&self, span: &Span) -> String {
        let Some((record, changed_by_you)) = self.record(span) else {
            return "not written by an agent".to_owned();
        };
        let mut prompt: String = record.prompt.chars().take(80).collect();
        if prompt.len() < record.prompt.len() {
            prompt.push('…');
        }
        let wrote = format!(
            "agent `{}` in session `{}` ({}), turn {}, prompt: '{prompt}', tool: {}",
            record.agent,
            self.session(record),
            day(record.at_ms),
            record.turn,
            record.tool,
        );
        if changed_by_you {
            format!("changed by you after it was written by {wrote}")
        } else {
            wrote
        }
    }

    /// A span, in a line: `net.rs:13-15: …`.
    fn line(&self, span: &Span) -> String {
        let lines = if span.from == span.to {
            span.from.to_string()
        } else {
            format!("{}-{}", span.from, span.to)
        };
        format!("{}:{lines}: {}", self.name, self.described(span))
    }

    /// The spans as text, a line each.
    fn text(&self) -> String {
        self.spans
            .iter()
            .map(|span| format!("{}\n", self.line(span)))
            .collect()
    }

    /// The spans as JSON, for scripts.
    fn json(&self) -> serde_json::Value {
        let spans: Vec<_> = self
            .spans
            .iter()
            .map(|span| {
                let record = self.record(span);
                serde_json::json!({
                    "from": span.from,
                    "to": span.to,
                    "by_agent": record.is_some(),
                    "changed_by_you": record.is_some_and(|(_, changed)| changed),
                    "session_name": record.map(|(record, _)| self.session(record)),
                    "record": record.map(|(record, _)| record),
                })
            })
            .collect();
        serde_json::json!({ "file": self.name, "spans": spans })
    }
}

/// What `/provenance --open` reads: the saved sessions, and this one's
/// agents as they are; and the terminal view's key focus, which the
/// opened turn's panel takes.
#[derive(bevy_ecs::system::SystemParam)]
struct Opening<'w, 's> {
    stores: Option<Res<'w, StoreOpener>>,
    session: Option<Res<'w, SessionPaths>>,
    agents: Query<'w, 's, (&'static AgentId, &'static Conversation)>,
    focus: Option<ResMut<'w, KeyFocus>>,
}

/// `/provenance <file>[:<line>[-<line>]]`, `--forget <session>`, and
/// `--open <entry>`, which its picker's entries run.
fn provenance(
    In(args): In<CommandArgs>,
    asking: Asking,
    opening: Opening,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let words: Vec<String> = args.args.split_whitespace().map(str::to_owned).collect();
    let Some(data) = asking.data.as_deref() else {
        notices.write(Notice::error(args.agent, "/provenance: no project data"));
        return;
    };
    match parse(&words) {
        Err(why) => {
            notices.write(Notice::error(args.agent, format!("/{why}")));
        }
        Ok(Asked::Forget(session)) => {
            notices.write(Notice::info(args.agent, asking.forget(data, &session)));
        }
        Ok(Asked::Lines { path, range, .. }) => {
            let answer = asking.lines(data, &path, range);
            let mut picker = Picker::new("Open the turn that wrote it")
                .for_agent(args.agent)
                .body(answer.text());
            for span in &answer.spans {
                if let Some(origin) = span.origin.filter(|_| answer.record(span).is_some()) {
                    let open = format!("{NAME} --open {}", origin.record);
                    picker = picker.item(answer.line(span), open);
                }
            }
            if picker.items.is_empty() {
                notices.write(Notice::info(args.agent, answer.text()));
            } else {
                commands.spawn(picker);
            }
        }
        Ok(Asked::Open(entry)) => {
            let store = load(data);
            let Some(record) = store.records.get(&entry) else {
                notices.write(Notice::error(
                    args.agent,
                    format!("/provenance: no entry {entry}"),
                ));
                return;
            };
            match turn_of(record, &asking, &opening) {
                Ok((conversation, turn)) => {
                    let title = format!(
                        "Session `{}`, turn {}: {}",
                        asking.session_label(&record.session),
                        record.turn,
                        record.prompt
                    );
                    let panel = open_turn(&mut commands, conversation, turn, title);
                    if let Some(mut focus) = opening.focus {
                        focus.0 = Some(panel);
                    }
                }
                Err(why) => {
                    notices.write(Notice::error(args.agent, format!("/provenance: {why}")));
                }
            }
        }
    }
}

/// Shows `conversation` read-only in a pane over the screen, titled
/// `title`, with `turn` at its top; Esc in it closes it.
fn open_turn(commands: &mut Commands, conversation: Log, turn: MessageId, title: String) -> Entity {
    let over = Placement::Over {
        width: Constraint::Percentage(90),
        height: Constraint::Percentage(90),
    };
    let pane = commands
        .spawn((
            TranscriptPane::new(TranscriptSource::Value(conversation)).titled(title),
            TuiPanel::new(over),
            PaneKeys,
        ))
        .id();
    let close =
        KeyBinding::new(KeyPress::key(KeyCode::Esc), "Close").in_scope(KeyScope::Panel(pane));
    commands.spawn((close, ChildOf(pane))).observe(close_turn);
    commands.trigger(ScrollTo {
        pane,
        item: Some(TranscriptItem::Message(turn)),
        row: ItemRow::Top(0),
    });
    pane
}

/// Esc in a turn's pane: closes it.
fn close_turn(pressed: On<KeyPressed>, keys: Query<&ChildOf>, mut commands: Commands) {
    if pressed.released {
        return;
    }
    if let Ok(of) = keys.get(pressed.entity) {
        commands.entity(of.parent()).despawn();
    }
}

/// The conversation a record was made in and its turn's first message,
/// read from its session: this one's agents as they are, another's from
/// its store.
fn turn_of(
    record: &Record,
    asking: &Asking,
    opening: &Opening,
) -> Result<(Log, MessageId), String> {
    let asked = record.asked.ok_or("the turn's prompt was not recorded")?;
    let here = opening.session.as_ref().and_then(|paths| paths.id());
    let live = here
        .filter(|id| id.to_string() == record.session)
        .and_then(|_| {
            opening
                .agents
                .iter()
                .find(|(id, _)| id.0 == record.agent_id)
        })
        .map(|(_, conversation)| conversation.0.clone());
    let conversation = match live {
        Some(conversation) => conversation,
        None => {
            let id: SessionId = record.session.parse().map_err(|_| "not a session id")?;
            let (Some(home), Some(stores)) = (asking.home.as_ref(), opening.stores.as_ref()) else {
                return Err("no sessions to read".to_owned());
            };
            let store = stores
                .open(&home.0.session(&id))
                .map_err(|why| why.to_string())?;
            let session = Session::read(&*store).map_err(|why| why.to_string())?;
            session
                .agents
                .into_iter()
                .find(|agent| agent.id == record.agent_id)
                .ok_or("the agent is not in its session any more")?
                .conversation
        }
    };
    let turn = conversation
        .index_of_key(asked)
        .and_then(|at| conversation.id(at))
        .ok_or("the turn is not in its session any more (compacted or rewritten)")?;
    Ok((conversation, turn))
}

/// `rig provenance …`: prints the answer, then exits.
fn command_line(invoked: Option<Res<Invoked>>, asking: Asking, mut commands: Commands) {
    let words = invoked
        .as_ref()
        .and_then(|invoked| invoked.command())
        .map_or(&[][..], |(_, words)| words);
    let failed = |stdout: String| ProvenanceRun { stdout, status: 2 };
    let run = match (asking.data.as_deref(), parse(words)) {
        (None, _) => failed("rig provenance: no project data\n".to_owned()),
        (_, Err(why)) => failed(format!("rig {why}\n")),
        (_, Ok(Asked::Open(_))) => {
            failed("rig provenance: --open is for the agent's picker\n".to_owned())
        }
        (Some(data), Ok(Asked::Forget(session))) => ProvenanceRun {
            stdout: format!("{}\n", asking.forget(data, &session)),
            status: 0,
        },
        (Some(data), Ok(Asked::Lines { path, range, json })) => {
            let answer = asking.lines(data, &path, range);
            let stdout = if json {
                format!("{}\n", answer.json())
            } else {
                answer.text()
            };
            ProvenanceRun { stdout, status: 0 }
        }
    };
    print!("{}", run.stdout);
    std::io::stdout().flush().ok();
    let reason = match run.status {
        0 => ExitReason::Quit,
        status => ExitReason::Failed(status),
    };
    commands.insert_resource(run);
    commands.trigger(Exit(reason));
}
