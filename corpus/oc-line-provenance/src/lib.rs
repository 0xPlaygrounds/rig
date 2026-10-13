//! OC-21: which prompt wrote this line? `git blame` for agent work.
//!
//! Every file change an agent's tool call makes is recorded per project,
//! in the project's [`PluginData`] (outside the repository, kept across
//! sessions): the session, agent, turn, the prompt the user typed to start
//! the turn, the tool call and the time, and which lines it wrote. The
//! file tools publish what they changed (`FilesChanged`), and so does any
//! call that may write anywhere, such as `shell`, through the harness's
//! checkpoint around it. What a file held before a call is the text the
//! store last saw, else what was read in `Gate` as a call writing named
//! files was about to run, else the turn's checkpoint (`TurnCheckpoint`),
//! each checked against the hash the change carries.
//!
//! Each line's origin is carried forward through every later change by a
//! line diff, so it follows code that moves. A change the user made
//! between turns is found at each turn's start (and before a call writes
//! the file, and when asked): a line of an agent's they changed is marked
//! "changed by you", a line they added is no agent's.
//!
//! - `/provenance <file>[:<line>[-<line>]]` lists the lines in runs of one
//!   origin, and offers a picker of the agents' entries: choosing one shows
//!   that turn, read from its session (read-only), as a note.
//! - `/provenance --forget <session>` drops one session's records.
//! - `rig provenance <file>[:<line>[-<line>]] [--json]` prints the same from
//!   the command line, without a session; `--forget` works there too.

use std::io::Write as _;
use std::path::{Path, PathBuf};

use bevy_log::warn;
use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::Conversation as Log;
use rig_harness::rig_core::message::{AssistantContent, Message, UserContent};
use rig_harness::rig_core::tool::{FileState, FilesChanged, content_hash};
use rig_harness::rig_memory::{Session, now_ms};
use rig_sessions::SessionTitle;
use std::collections::HashMap;

mod store;
pub use store::{LineOrigin, ProvenanceStore, Record, Span, TrackedFile};

#[cfg(test)]
mod tests;

/// The slash command, the command-line command, and the author of notes.
pub const NAME: &str = "provenance";

/// The store's file in the project's plugin data.
pub const STORE: &str = "line-provenance/store.json";

/// Its usage.
const USAGE: &str = "<file>[:<line>[-<line>]] [--json] | --forget <session>";

/// Records which tool call wrote each line, and answers `/provenance` and
/// `rig provenance`.
#[derive(Default)]
pub struct LineProvenancePlugin;

impl Plugin for LineProvenancePlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<ProjectRoot>()
            .init_resource::<ProjectWorkspace>()
            .init_resource::<BeforeWrites>()
            .add_systems(AgentPass, read_before_writes.in_set(Stage::Gate))
            .add_observer(turn_started)
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

/// The project's store, as last saved; empty when there is none yet.
pub fn load(data: &PluginData) -> ProvenanceStore {
    data.project
        .read_file(STORE)
        .ok()
        .and_then(|bytes| serde_json::from_slice(&bytes).ok())
        .unwrap_or_default()
}

/// Saves the project's store.
fn save(data: &PluginData, store: &ProvenanceStore) {
    let saved = serde_json::to_vec(store)
        .map_err(std::io::Error::from)
        .and_then(|bytes| data.project.write_file(STORE, &bytes));
    if let Err(why) = saved {
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

/// The text of each file, before a call that writes it by name runs.
#[derive(Resource, Default)]
struct BeforeWrites(HashMap<PathBuf, String>);

/// On a call whose files were read before it runs.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct ReadBefore;

/// Reads each file a call names as written as it is about to run, on this
/// machine (a file read in another workspace would wait on the network).
fn read_before_writes(
    calls: Query<(Entity, &Footprint, &Decision), (Undecided, ToolCalls, Without<ReadBefore>)>,
    workspace: Res<ProjectWorkspace>,
    mut before: ResMut<BeforeWrites>,
    mut commands: Commands,
) {
    if workspace.0.host().is_some() {
        return;
    }
    for (call, footprint, decision) in &calls {
        let Touches::Writes(Scope::Paths(paths)) = &footprint.0 else {
            continue;
        };
        if decision.is_decided() {
            continue;
        }
        for path in paths {
            match workspace.0.read_text(path) {
                Ok(text) => before.0.insert(path.clone(), text),
                Err(_) => before.0.remove(path),
            };
        }
        commands.entity(call).insert(ReadBefore);
    }
}

/// At each turn's start, what the user changed since in the files agents
/// wrote.
fn turn_started(
    _: On<Add<TurnOf>>,
    data: Option<Res<PluginData>>,
    root: Res<ProjectRoot>,
    workspace: Res<ProjectWorkspace>,
) {
    let Some(data) = data else {
        return;
    };
    if workspace.0.host().is_some() {
        return;
    }
    let mut store = load(&data);
    let paths: Vec<String> = store.files.keys().cloned().collect();
    let mut changed = false;
    for path in paths {
        let now = workspace.0.read_text(&root.0.join(&path)).ok();
        changed |= store.user_changed(&path, now.as_deref());
    }
    if changed {
        save(&data, &store);
    }
}

/// Where a change is recorded: the project's data, the session, and the
/// files read before calls.
#[derive(bevy_ecs::system::SystemParam)]
struct Recording<'w> {
    data: Option<Res<'w, PluginData>>,
    root: Res<'w, ProjectRoot>,
    workspace: Res<'w, ProjectWorkspace>,
    session: Option<Res<'w, SessionPaths>>,
    title: Option<Res<'w, SessionTitle>>,
    before: ResMut<'w, BeforeWrites>,
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

/// Records what a settled tool call changed, and carries every line's
/// origin forward.
fn call_settled(
    settled: On<Add<Settled>>,
    calls: Query<(&ToolRequest, &Published, &ForAgent, &CallOf)>,
    turns: Query<(&Turn, Option<&TurnCheckpoint>)>,
    agents: Agents,
    mut recording: Recording,
) {
    let Ok((request, published, &ForAgent(agent), &CallOf(turn))) = calls.get(settled.entity)
    else {
        return;
    };
    let Ok(Some(FilesChanged(changes))) = published.0.get::<FilesChanged>() else {
        return;
    };
    let (Some(data), Ok((turn, checkpoint)), Ok((id, name, ..))) = (
        recording.data.as_deref(),
        turns.get(turn),
        agents.get(agent),
    ) else {
        return;
    };
    if recording.workspace.0.host().is_some() || changes.is_empty() {
        return;
    }
    let session = recording
        .session
        .as_ref()
        .and_then(|paths| paths.id())
        .map(|id| id.to_string())
        .unwrap_or_default();
    let record = Record {
        session,
        session_name: recording
            .title
            .as_ref()
            .and_then(|title| title.name.clone()),
        agent: name.map_or_else(|| id.short().to_owned(), |name| name.as_str().to_owned()),
        agent_id: id.0.clone(),
        turn: turn.index.saturating_add(1),
        first_message: Some(turn.first_message),
        prompt: prompt(agent, turn.first_message, &agents, &turns),
        tool: request.call.function.name.as_str().to_owned(),
        call_id: request.call.id.to_string(),
        at_ms: now_ms(),
    };
    let mut store = load(data);
    let key_of = store.add(record);
    let first = checkpoint.and_then(TurnCheckpoint::first);
    for change in changes {
        let path = PathBuf::from(&change.path);
        let path = if path.is_absolute() {
            path
        } else {
            recording.root.0.join(path)
        };
        let name = key(&recording.root.0, &path);
        let read_before = recording.before.0.remove(&path);
        let after = match &change.after {
            None => None,
            Some(state) => match recording.workspace.0.read_text(&path) {
                Ok(text) if matches(state, &text) => Some(text),
                // A later write already changed it: not this call's text.
                _ => continue,
            },
        };
        let Some(before) = change.before.as_ref().map_or(Some(String::new()), |state| {
            let tracked = store.files.get(&name).map(|file| file.text.clone());
            let kept = || {
                let bytes = first.as_ref()?.content(&path)?.ok()?;
                Some(String::from_utf8_lossy(&bytes).into_owned())
            };
            [tracked, read_before, kept()]
                .into_iter()
                .flatten()
                .find(|text| matches(state, text))
        }) else {
            warn!(
                "line provenance: {name} as it was before {} is unknown",
                request.call.id
            );
            continue;
        };
        store.agent_changed(&name, &before, after.as_deref(), key_of);
    }
    save(data, &store);
}

/// Whether `text` is the file `state` describes.
fn matches(state: &FileState, text: &str) -> bool {
    state
        .hash
        .as_ref()
        .is_none_or(|hash| *hash == content_hash(text.as_bytes()))
}

/// What the user typed to start the turn of `agent` that began at
/// `first`, else to start its parent's running turn, up to the root.
fn prompt(
    mut agent: Entity,
    mut first: MessageId,
    agents: &Agents,
    turns: &Query<(&Turn, Option<&TurnCheckpoint>)>,
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
            let (turn, _) = turns.get(active?.turn()).ok()?;
            Some((parent, turn.first_message))
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

/// The lines asked for, after taking in what the user changed in the file
/// since it was last seen.
fn answer(
    data: &PluginData,
    root: &Path,
    path: &str,
    range: Option<(usize, usize)>,
) -> (String, ProvenanceStore, Vec<Span>) {
    let file = root.join(path);
    let name = key(root, &file);
    let now = std::fs::read_to_string(&file).ok();
    let mut store = load(data);
    if store.user_changed(&name, now.as_deref()) {
        save(data, &store);
    }
    let total = now.as_deref().map_or(0, |text| text.lines().count());
    let spans = store.spans(&name, total, range);
    (name, store, spans)
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

/// Who wrote a span, in a line.
fn described(store: &ProvenanceStore, origin: Option<LineOrigin>) -> String {
    let Some((origin, record)) =
        origin.and_then(|origin| Some((origin, store.records.get(&origin.record)?)))
    else {
        return "not written by an agent".to_owned();
    };
    let mut prompt: String = record.prompt.chars().take(80).collect();
    if prompt.len() < record.prompt.len() {
        prompt.push('…');
    }
    let wrote = format!(
        "agent `{}` in session `{}` ({}), turn {}, prompt: '{prompt}', tool: {}",
        record.agent,
        record.session_label(),
        day(record.at_ms),
        record.turn,
        record.tool,
    );
    if origin.changed_by_you {
        format!("changed by you after it was written by {wrote}")
    } else {
        wrote
    }
}

/// The lines of a span, as `13` or `13-15`.
fn lines(span: &Span) -> String {
    if span.from == span.to {
        span.from.to_string()
    } else {
        format!("{}-{}", span.from, span.to)
    }
}

/// The spans as text, a line each.
fn text(name: &str, store: &ProvenanceStore, spans: &[Span]) -> String {
    spans
        .iter()
        .map(|span| {
            format!(
                "{name}:{}: {}\n",
                lines(span),
                described(store, span.origin)
            )
        })
        .collect()
}

/// The spans as JSON, for scripts.
fn json(name: &str, store: &ProvenanceStore, spans: &[Span]) -> serde_json::Value {
    let spans: Vec<_> = spans
        .iter()
        .map(|span| {
            let record = span.origin.and_then(|origin| store.records.get(&origin.record));
            serde_json::json!({
                "from": span.from,
                "to": span.to,
                "by_agent": record.is_some(),
                "changed_by_you": record.is_some() && span.origin.is_some_and(|origin| origin.changed_by_you),
                "record": record,
            })
        })
        .collect();
    serde_json::json!({ "file": name, "spans": spans })
}

/// What `/provenance` reads and changes.
#[derive(bevy_ecs::system::SystemParam)]
struct Answering<'w, 's> {
    data: Option<Res<'w, PluginData>>,
    root: Res<'w, ProjectRoot>,
    home: Option<Res<'w, RigHome>>,
    stores: Option<Res<'w, SessionStores>>,
    session: Option<Res<'w, SessionPaths>>,
    agents: Query<'w, 's, (&'static AgentId, &'static mut Conversation)>,
}

/// `/provenance <file>[:<line>[-<line>]]`, `--forget <session>`, and
/// `--open <entry>`, which its picker's entries run.
fn provenance(
    In(args): In<CommandArgs>,
    mut answering: Answering,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let words: Vec<String> = args.args.split_whitespace().map(str::to_owned).collect();
    let Some(data) = answering.data.as_deref() else {
        notices.write(Notice::error(args.agent, "/provenance: no project data"));
        return;
    };
    match parse(&words) {
        Err(why) => {
            notices.write(Notice::error(args.agent, format!("/{why}")));
        }
        Ok(Asked::Forget(session)) => {
            let mut store = load(data);
            let gone = store.forget(&session);
            save(data, &store);
            notices.write(Notice::info(
                args.agent,
                format!("Forgot {gone} changes of session {session}."),
            ));
        }
        Ok(Asked::Lines { path, range, .. }) => {
            let (name, store, spans) = answer(data, &answering.root.0, &path, range);
            notices.write(Notice::info(args.agent, text(&name, &store, &spans)));
            let mut picker = Picker::new("Open the turn that wrote it")
                .for_agent(args.agent)
                .body(text(&name, &store, &spans));
            for span in &spans {
                if let Some(origin) = span
                    .origin
                    .filter(|o| store.records.contains_key(&o.record))
                {
                    let label = format!(
                        "{name}:{}: {}",
                        lines(span),
                        described(&store, Some(origin))
                    );
                    picker = picker.item(label, format!("{NAME} --open {}", origin.record));
                }
            }
            if !picker.items.is_empty() {
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
            match turn_text(record, &mut answering) {
                Ok(shown) => {
                    let title = format!(
                        "Session `{}`, turn {}: {}",
                        record.session_label(),
                        record.turn,
                        record.prompt
                    );
                    if let Ok((_, mut conversation)) = answering.agents.get_mut(args.agent) {
                        let last = conversation.messages().len().checked_sub(1);
                        let after = last.and_then(|last| conversation.id(last));
                        let note = Note::new(title, shown).aside(Author::plugin(NAME), after);
                        conversation.add_aside(note.unsaved());
                    }
                }
                Err(why) => {
                    notices.write(Notice::error(args.agent, format!("/provenance: {why}")));
                }
            }
        }
    }
}

/// The turn a record was made in, as text: what was asked, what the agent
/// reasoned and said, and the calls it made, read from its session
/// (this one's agents as they are, another's from its store).
fn turn_text(record: &Record, answering: &mut Answering) -> Result<String, String> {
    let here = answering.session.as_ref().and_then(|paths| paths.id());
    let first = record
        .first_message
        .ok_or("the turn's start was not recorded")?;
    if here.is_some_and(|id| id.to_string() == record.session) {
        let live = answering
            .agents
            .iter()
            .find(|(id, _)| id.0 == record.agent_id);
        if let Some((_, conversation)) = live {
            return Ok(render_turn(conversation, first));
        }
    }
    let id: SessionId = record.session.parse().map_err(|_| "not a session id")?;
    let (Some(home), Some(stores)) = (answering.home.as_ref(), answering.stores.as_ref()) else {
        return Err("no sessions to read".to_owned());
    };
    let store = stores
        .open(&home.0.session(&id))
        .map_err(|why| why.to_string())?;
    let session = Session::read(&*store).map_err(|why| why.to_string())?;
    let agent = session
        .agents
        .iter()
        .find(|agent| agent.id == record.agent_id)
        .ok_or("the agent is not in its session any more")?;
    Ok(render_turn(&agent.conversation, first))
}

/// The messages of the turn that began at `first`, up to the next request.
fn render_turn(conversation: &Log, first: MessageId) -> String {
    let start = conversation.start_of(first);
    let mut out = String::new();
    for entry in conversation.entries().skip(start) {
        if entry.index > start && entry.is_request() {
            break;
        }
        match entry.message {
            Message::User { content } => {
                for part in content.iter() {
                    match part {
                        UserContent::Text(text) => out.push_str(&format!("> {}\n", text.text)),
                        UserContent::ToolResult(result) => {
                            let texts: Vec<&str> = result
                                .content
                                .iter()
                                .filter_map(|it| it.as_text())
                                .collect();
                            let text: String = texts.join(" ").chars().take(200).collect();
                            out.push_str(&format!("  ← {}: {text}\n", result.name.as_str()));
                        }
                        _ => {}
                    }
                }
            }
            Message::Assistant(reply) => {
                for part in &reply.content {
                    match part {
                        AssistantContent::Reasoning(reasoning) if !reasoning.text.is_empty() => {
                            out.push_str(&format!("(thinking) {}\n", reasoning.text));
                        }
                        AssistantContent::Text(text) => out.push_str(&format!("{}\n", text.text)),
                        AssistantContent::ToolCall(call) => out.push_str(&format!(
                            "→ {} {}\n",
                            call.function.name.as_str(),
                            serde_json::to_string(&call.function.arguments).unwrap_or_default()
                        )),
                        _ => {}
                    }
                }
            }
            _ => {}
        }
    }
    out
}

/// `rig provenance …`: prints the answer, then exits.
fn command_line(
    invoked: Option<Res<Invoked>>,
    data: Option<Res<PluginData>>,
    root: Res<ProjectRoot>,
    mut commands: Commands,
) {
    let words = invoked
        .as_ref()
        .and_then(|invoked| invoked.command())
        .map_or(&[][..], |(_, words)| words);
    let run = match (data, parse(words)) {
        (None, _) => ProvenanceRun {
            stdout: "rig provenance: no project data\n".to_owned(),
            status: 2,
        },
        (_, Err(why)) => ProvenanceRun {
            stdout: format!("rig {why}\n"),
            status: 2,
        },
        (_, Ok(Asked::Open(_))) => ProvenanceRun {
            stdout: "rig provenance: --open is for the agent's picker\n".to_owned(),
            status: 2,
        },
        (Some(data), Ok(Asked::Forget(session))) => {
            let mut store = load(&data);
            let gone = store.forget(&session);
            save(&data, &store);
            ProvenanceRun {
                stdout: format!("forgot {gone} changes of session {session}\n"),
                status: 0,
            }
        }
        (
            Some(data),
            Ok(Asked::Lines {
                path,
                range,
                json: as_json,
            }),
        ) => {
            let (name, store, spans) = answer(&data, &root.0, &path, range);
            let stdout = if as_json {
                format!("{}\n", json(&name, &store, &spans))
            } else {
                text(&name, &store, &spans)
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
