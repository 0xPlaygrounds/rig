//! Plugin corpus (held-out): H11: Recall from past sessions (spec `H11-session-recall.md`).
//!
//! [`RecallPlugin`] keeps a full-text index of the past sessions under
//! `RIG_HOME` in `RIG_HOME/recall-index`, one file per session: each
//! message of each agent's conversation (user and assistant text, tool
//! calls with their arguments, tool results cut short) as one document. The
//! index is brought up to date in the background at startup and before
//! each search, re-reading only the sessions whose logs changed; the
//! running session is never in it. `recall` ranks the messages with BM25
//! and shows each hit with ±3 messages around it, `recall_open` shows more
//! around a hit, and `/recall <query>` offers the hits in a picker, the one
//! picked going with the user's next message. What `read` returned for a
//! `.env`-like file, and such a file attached to a message, is never
//! indexed.

use std::collections::{HashMap, HashSet};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex, PoisonError};
use std::time::UNIX_EPOCH;

use bevy_log::warn;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_cassette::journal::{JournalStore, JsonlDirStore};
use rig_harness::harness_protocol::{SessionDir, SessionId};
use rig_harness::prelude::*;
use rig_harness::rig_core::message::{Message, ToolResultContent};
use rig_harness::rig_core::tool::ToolOutput;
use rig_harness::rig_ecs::journal::{SavedSession, read_session};
use rig_sessions::Meta;
use rig_tools::attach::attached_file;
use rig_tools::fs::{read_json, write_json};
use rig_tools::shorten;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

/// The index's directory under `RIG_HOME`.
const INDEX_DIR: &str = "recall-index";
/// Most characters of a tool result kept in the index.
const RESULT_CHARS: usize = 600;
/// Most characters of any other part of a message kept in the index.
const PART_CHARS: usize = 4000;
/// Most characters of a message in a hit's snippet.
const SNIPPET_CHARS: usize = 240;
/// Messages shown before and after a hit.
const AROUND: usize = 3;
/// Messages `recall_open` shows before and after a hit.
const OPEN_AROUND: usize = 10;
/// Hits unless the call asks for another number, and the most.
const DEFAULT_LIMIT: usize = 5;
const MAX_LIMIT: usize = 20;
/// BM25's parameters.
const K1: f64 = 1.2;
const B: f64 = 0.75;

const RULES: &[&str] = &[
    "Use `recall` when the user refers to earlier work (\"we fixed this last week\"): it searches \
     past sessions, in this project or all of them; `recall_open` shows more around a hit.",
];

/// The `recall` and `recall_open` tools, `/recall`, and the index.
#[derive(Default)]
pub struct RecallPlugin;

impl Plugin for RecallPlugin {
    fn build(&self, app: &mut App) {
        let options = || ToolOptions {
            rules: RULES,
            footprint: Some(Touches::Reads(rig_harness::rig_core::tool::Scope::Anything)),
        };
        app.init_resource::<RigHome>()
            .init_resource::<ProjectRoot>()
            .init_resource::<RecallIndex>()
            .add_open_tool(
                "recall",
                "Search past sessions' conversations (full text, BM25): user and assistant text, \
                 tool calls and their results. Each hit names the session, its date, its project \
                 and a reference for `recall_open`, with the messages around it.",
                options(),
                recall,
            )
            .add_open_tool(
                "recall_open",
                "Show a longer excerpt of a past session around a message `recall` found.",
                ToolOptions {
                    rules: &[],
                    footprint: Some(Touches::Reads(rig_harness::rig_core::tool::Scope::Anything)),
                },
                open,
            )
            .add_command(
                "recall",
                "Search past sessions; the excerpt you pick goes with your next message",
                recall_command,
            )
            .add_systems(Startup, warm_up)
            .add_systems(AgentPass, attach.in_set(Stage::Prepare))
            .add_observer(warmed)
            .add_observer(found);
    }
}

/// The index of past sessions, shared with the work that updates and
/// searches it off the loop.
#[derive(Resource, Clone, Default)]
pub struct RecallIndex(Arc<Mutex<Index>>);

/// Where the index looks: `RIG_HOME`, the running session, the project.
#[derive(Clone)]
struct Place {
    home: Home,
    current: Option<PathBuf>,
    project: PathBuf,
}

impl Place {
    fn of(home: &RigHome, paths: Option<&SessionPaths>, project: &ProjectRoot) -> Self {
        Self {
            home: home.0.clone(),
            current: paths.map(|paths| paths.path().to_owned()),
            project: project.0.clone(),
        }
    }
}

/// The past sessions indexed, by id.
#[derive(Default)]
struct Index {
    sessions: HashMap<String, Indexed>,
}

/// What a session's logs looked like when it was indexed: its logs'
/// count, total size and newest change.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Stamp {
    logs: usize,
    bytes: u64,
    modified_ms: u64,
}

/// A session's file in the index.
#[derive(Serialize, Deserialize, Default)]
struct Saved {
    stamp: Stamp,
    docs: Vec<Doc>,
}

/// One message of one agent's conversation, as text.
#[derive(Serialize, Deserialize, Clone)]
struct Doc {
    agent: String,
    text: String,
}

/// What a hit says about its session, read afresh at each update.
struct Info {
    name: String,
    updated_ms: u64,
    directory: Option<PathBuf>,
}

/// One indexed session: its documents and their terms.
struct Indexed {
    info: Info,
    saved: Saved,
    terms: Vec<HashMap<String, u32>>,
    lens: Vec<usize>,
}

impl Indexed {
    fn new(info: Info, saved: Saved) -> Self {
        let mut terms = Vec::with_capacity(saved.docs.len());
        let mut lens = Vec::with_capacity(saved.docs.len());
        for doc in &saved.docs {
            let mut counts: HashMap<String, u32> = HashMap::new();
            let mut len = 0;
            for token in tokens(&doc.text) {
                *counts.entry(token).or_default() += 1;
                len += 1;
            }
            terms.push(counts);
            lens.push(len);
        }
        Self {
            info,
            saved,
            terms,
            lens,
        }
    }

    /// Messages `around` either side of `at`, of the same agent's
    /// conversation, each cut to `chars` when given; the one at `at`
    /// marked.
    fn excerpt(&self, at: usize, around: usize, chars: Option<usize>) -> String {
        let Some(hit) = self.saved.docs.get(at) else {
            return String::new();
        };
        let start = at.saturating_sub(around);
        let docs = self.saved.docs.iter().enumerate().skip(start);
        docs.take(at - start + around + 1)
            .filter(|(_, doc)| doc.agent == hit.agent)
            .map(|(index, doc)| {
                let text = match chars {
                    Some(chars) => shorten(&one_line(&doc.text), chars),
                    None => doc.text.clone(),
                };
                let mark = if index == at { ">" } else { " " };
                format!("{mark} [{index}] {}", text.replace('\n', "\n      "))
            })
            .collect::<Vec<_>>()
            .join("\n")
    }

    /// The line naming the session: name, date and, with `project`, its
    /// directory.
    fn header(&self, id: &str, project: bool) -> String {
        let date = i64::try_from(self.info.updated_ms)
            .ok()
            .and_then(chrono::DateTime::from_timestamp_millis)
            .map(|date| date.format("%Y-%m-%d").to_string())
            .unwrap_or_default();
        let mut line = format!("“{}” · {date}", self.info.name);
        if project && let Some(directory) = &self.info.directory {
            line.push_str(&format!(" · {}", directory.display()));
        }
        if !self.info.name.contains(id) {
            line.push_str(&format!(" · session {id}"));
        }
        line
    }
}

/// A search hit.
struct Hit {
    /// The session's line and the reference.
    title: String,
    /// The messages around it, cut short.
    snippet: String,
    /// The messages around it, whole, as `/recall` attaches them.
    excerpt: String,
}

impl Index {
    /// Brings the index up to date with the sessions under the home,
    /// re-reading only those whose logs changed, and forgets the ones
    /// gone; the running session is left out.
    fn update(&mut self, place: &Place) {
        let dir = place.home.root().join(INDEX_DIR);
        let mut unreadable = Vec::new();
        let mut seen = HashSet::new();
        let entries = fs::read_dir(place.home.sessions()).into_iter().flatten();
        for entry in entries.filter_map(Result::ok) {
            let Some(id) = entry.file_name().to_str().and_then(|id| id.parse().ok()) else {
                continue;
            };
            let id: SessionId = id;
            let session = place.home.session(&id);
            if Some(session.path()) == place.current.as_deref() || !session.is_saved() {
                continue;
            }
            let id = id.to_string();
            let stamp = stamp(session.path());
            let info = info(&session, &id, stamp);
            seen.insert(id.clone());
            if let Some(indexed) = self.sessions.get_mut(&id)
                && indexed.saved.stamp == stamp
            {
                indexed.info = info;
                continue;
            }
            let file = dir.join(format!("{id}.json"));
            let saved = match read_json::<Saved>(&file).filter(|saved| saved.stamp == stamp) {
                Some(saved) => saved,
                None => {
                    let store = JsonlDirStore::new(session.path());
                    let docs = match read_session(&store) {
                        Ok(read) => {
                            unreadable.extend(read.unreadable.iter().map(|(agent, why)| {
                                format!("{}/{agent}.jsonl ({why})", session.path().display())
                            }));
                            docs(&read)
                        }
                        Err(why) => {
                            unreadable.push(format!("{} ({why})", session.path().display()));
                            Vec::new()
                        }
                    };
                    let saved = Saved { stamp, docs };
                    let written = fs::create_dir_all(&dir).and_then(|()| write_json(&file, &saved));
                    if let Err(why) = written {
                        warn!("recall: could not save the index of session {id}: {why}");
                    }
                    saved
                }
            };
            self.sessions.insert(id, Indexed::new(info, saved));
        }
        self.sessions.retain(|id, _| seen.contains(id));
        forget_gone(&dir, &seen);
        if !unreadable.is_empty() {
            warn!(
                "recall: skipped {} unreadable session files: {}",
                unreadable.len(),
                unreadable.join(", ")
            );
        }
    }

    /// The best `limit` hits for `query`, in the sessions of `project`
    /// or in all: messages ranked by BM25, each session's messages near a
    /// better hit left out.
    fn search(&self, query: &str, project: Option<&Path>, limit: usize) -> Vec<Hit> {
        let mut terms = tokens(query).collect::<Vec<_>>();
        terms.sort();
        terms.dedup();
        let scoped: Vec<(&String, &Indexed)> = self
            .sessions
            .iter()
            .filter(|(_, indexed)| {
                project.is_none_or(|project| {
                    indexed
                        .info
                        .directory
                        .as_deref()
                        .is_some_and(|directory| same_dir(directory, project))
                })
            })
            .collect();
        let count: usize = scoped.iter().map(|(_, indexed)| indexed.lens.len()).sum();
        if count == 0 || terms.is_empty() {
            return Vec::new();
        }
        let total: usize = scoped.iter().flat_map(|(_, indexed)| &indexed.lens).sum();
        let average = total as f64 / count as f64;
        let idf: Vec<f64> = terms
            .iter()
            .map(|term| {
                let with = scoped
                    .iter()
                    .flat_map(|(_, indexed)| &indexed.terms)
                    .filter(|counts| counts.contains_key(term))
                    .count() as f64;
                (1.0 + (count as f64 - with + 0.5) / (with + 0.5)).ln()
            })
            .collect();
        let mut scored = Vec::new();
        for (id, indexed) in &scoped {
            for (at, (counts, len)) in indexed.terms.iter().zip(&indexed.lens).enumerate() {
                let norm = K1 * (1.0 - B + B * *len as f64 / average.max(1.0));
                let score: f64 = terms
                    .iter()
                    .zip(&idf)
                    .filter_map(|(term, idf)| {
                        let tf = f64::from(*counts.get(term)?);
                        Some(idf * tf * (K1 + 1.0) / (tf + norm))
                    })
                    .sum();
                if score > 0.0 {
                    scored.push((score, indexed.info.updated_ms, id.as_str(), at));
                }
            }
        }
        scored.sort_by(|a, b| b.0.total_cmp(&a.0).then(b.1.cmp(&a.1)));
        let mut taken: Vec<(&str, usize)> = Vec::new();
        let mut hits = Vec::new();
        for (_, _, id, at) in scored {
            if hits.len() >= limit {
                break;
            }
            let near = taken
                .iter()
                .any(|&(taken, other)| taken == id && other.abs_diff(at) <= AROUND);
            let Some(indexed) = self.sessions.get(id) else {
                continue;
            };
            if near {
                continue;
            }
            taken.push((id, at));
            let reference = format!("(recall_open session \"{id}\", message {at})");
            hits.push(Hit {
                title: format!("{} {reference}", indexed.header(id, project.is_none())),
                snippet: indexed.excerpt(at, AROUND, Some(SNIPPET_CHARS)),
                excerpt: format!(
                    "From the past session {}:\n{}",
                    indexed.header(id, true),
                    indexed.excerpt(at, AROUND, None)
                ),
            });
        }
        hits
    }
}

/// Removes the index files of sessions no longer there.
fn forget_gone(dir: &Path, seen: &HashSet<String>) {
    for entry in fs::read_dir(dir)
        .into_iter()
        .flatten()
        .filter_map(Result::ok)
    {
        let path = entry.path();
        let id = path.file_stem().and_then(|stem| stem.to_str());
        if id.is_some_and(|id| !seen.contains(id)) {
            fs::remove_file(&path).ok();
        }
    }
}

/// The session's agent logs' count, size and newest change: the logs its
/// store lists, each `<agent>.jsonl` in its directory.
fn stamp(session: &Path) -> Stamp {
    let mut stamp = Stamp::default();
    let agents = JsonlDirStore::new(session).agents().unwrap_or_default();
    for agent in agents {
        let Ok(metadata) = fs::metadata(session.join(format!("{agent}.jsonl"))) else {
            continue;
        };
        let modified = metadata
            .modified()
            .ok()
            .and_then(|at| at.duration_since(UNIX_EPOCH).ok())
            .map(|since| u64::try_from(since.as_millis()).unwrap_or(u64::MAX))
            .unwrap_or_default();
        stamp.logs += 1;
        stamp.bytes += metadata.len();
        stamp.modified_ms = stamp.modified_ms.max(modified);
    }
    stamp
}

/// The session's name, date and directory, from its listing cache.
fn info(session: &SessionDir, id: &str, stamp: Stamp) -> Info {
    let meta: Option<Meta> = read_json(&session.meta());
    let name = meta.as_ref().and_then(|meta| {
        let typed = Some(meta.title.typed.clone()).filter(|typed| !typed.is_empty());
        meta.title.name.clone().or(typed)
    });
    let directory = fs::read_to_string(session.directory())
        .ok()
        .map(|text| PathBuf::from(text.trim_end_matches('\n')));
    Info {
        name: name.unwrap_or_else(|| format!("session {id}")),
        updated_ms: meta.map_or(stamp.modified_ms, |meta| meta.updated),
        directory,
    }
}

/// Whether two paths name the same directory.
fn same_dir(a: &Path, b: &Path) -> bool {
    a == b
        || matches!(
            (fs::canonicalize(a), fs::canonicalize(b)),
            (Ok(a), Ok(b)) if a == b
        )
}

/// Every message of every agent of `session`, as text: what `read`
/// returned for a `.env`-like file, and such a file attached, left out.
fn docs(session: &SavedSession) -> Vec<Doc> {
    let mut docs = Vec::new();
    for agent in &session.agents {
        let mut secret_reads = HashSet::new();
        for message in agent.conversation.messages() {
            let mut parts = Vec::new();
            match message {
                Message::User { content } => {
                    for item in content {
                        match item {
                            UserContent::Text(text) => match attached_file(&text.text) {
                                Some((label, _)) if secret(label) => {
                                    parts.push(format!("user attached {label} (not indexed)"));
                                }
                                _ => {
                                    parts.push(format!("user: {}", shorten(&text.text, PART_CHARS)))
                                }
                            },
                            UserContent::ToolResult(result) => {
                                let name = result.name.as_str();
                                if secret_reads.contains(&result.call) {
                                    parts.push(format!("{name} result: (not indexed)"));
                                    continue;
                                }
                                let text: Vec<String> = result
                                    .content
                                    .iter()
                                    .filter_map(|content| match content {
                                        ToolResultContent::Text(text) => Some(text.text.clone()),
                                        ToolResultContent::Json { value } => {
                                            Some(value.to_string())
                                        }
                                        _ => None,
                                    })
                                    .collect();
                                let text = shorten(&text.join("\n"), RESULT_CHARS);
                                parts.push(format!("{name} result: {text}"));
                            }
                            _ => {}
                        }
                    }
                }
                Message::Assistant(assistant) => {
                    for item in &assistant.content {
                        match item {
                            AssistantContent::Text(text) => {
                                parts.push(format!(
                                    "assistant: {}",
                                    shorten(&text.text, PART_CHARS)
                                ));
                            }
                            AssistantContent::ToolCall(call) => {
                                let name = call.function.name.as_str();
                                let path = call.function.arguments.get("path");
                                let path = path.and_then(|path| path.as_str()).unwrap_or_default();
                                if name == "read" && secret(path) {
                                    secret_reads.insert(call.id.clone());
                                }
                                let arguments =
                                    serde_json::Value::Object(call.function.arguments.clone());
                                parts.push(format!(
                                    "assistant called {name} {}",
                                    shorten(&arguments.to_string(), PART_CHARS)
                                ));
                            }
                            _ => {}
                        }
                    }
                }
                Message::System { .. } => {}
            }
            // Every message keeps its place, so a reference counts messages.
            docs.push(Doc {
                agent: agent.id.0.clone(),
                text: parts.join("\n"),
            });
        }
    }
    docs
}

/// Whether `path` names a `.env`-like file, whose contents are never
/// indexed.
fn secret(path: &str) -> bool {
    let name = Path::new(path)
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or_default()
        .to_lowercase();
    name == ".env" || name == ".envrc" || name.starts_with(".env.") || name.ends_with(".env")
}

/// The lowercase words and numbers of `text`.
fn tokens(text: &str) -> impl Iterator<Item = String> + '_ {
    text.split(|c: char| !c.is_alphanumeric())
        .filter(|word| !word.is_empty())
        .map(str::to_lowercase)
}

/// `text` on one line.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// The pool the index's work runs on.
fn pool() -> &'static TaskPool {
    IoTaskPool::get_or_init(TaskPool::default)
}

/// Runs `work` on the index, brought up to date first, on a thread of its
/// own.
async fn with_index<T: Send + 'static>(
    index: RecallIndex,
    place: Place,
    work: impl FnOnce(&Index, &Place) -> T + Send + 'static,
) -> Result<T, ToolExecutionError> {
    blocking(move || {
        let mut index = index.0.lock().unwrap_or_else(PoisonError::into_inner);
        index.update(&place);
        Ok(work(&index, &place))
    })
    .await
}

/// The index brought up to date at startup.
#[derive(Component)]
struct Warmed;

impl From<Result<(), ToolExecutionError>> for Warmed {
    fn from(_: Result<(), ToolExecutionError>) -> Self {
        Self
    }
}

fn warm_up(
    index: Res<RecallIndex>,
    home: Res<RigHome>,
    paths: Option<Res<SessionPaths>>,
    project: Res<ProjectRoot>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let place = Place::of(&home, paths.as_deref(), &project);
    let work = with_index(index.clone(), place, |_, _| ());
    commands.spawn((
        Name::new("recall index"),
        Running::spawn_into::<Warmed, _>(pool(), &wake, work),
    ));
}

fn warmed(warmed: On<Add<Warmed>>, mut commands: Commands) {
    commands.entity(warmed.entity).despawn();
}

/// Which sessions `recall` searches.
#[derive(Deserialize, JsonSchema, Clone, Copy, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum Scope {
    /// The sessions run in this project.
    #[default]
    Project,
    /// Every session.
    All,
}

#[derive(Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
struct RecallArgs {
    /// The words to look for, such as an error code and what it is about.
    query: String,
    /// `project` (default): the sessions run in this project; `all`: every
    /// session.
    #[serde(default)]
    scope: Scope,
    /// The most hits, 5 unless given, at most 20.
    #[serde(default)]
    limit: Option<usize>,
}

#[derive(Deserialize, JsonSchema, Clone)]
#[serde(deny_unknown_fields)]
struct OpenArgs {
    /// The session a hit names.
    session: String,
    /// The message a hit names.
    message: usize,
}

/// The work's answer as a tool result.
fn answer(result: Result<Result<String, String>, ToolExecutionError>) -> ToolResult {
    match result {
        Ok(Ok(text)) => ToolResult::success(ToolOutput::text(text)),
        Ok(Err(why)) => ToolResult::failed(ToolExecutionError::refused(why)),
        Err(error) => ToolResult::failed(error),
    }
}

fn recall(
    called: On<ToolCalled<RecallArgs>>,
    index: Res<RecallIndex>,
    home: Res<RigHome>,
    paths: Option<Res<SessionPaths>>,
    project: Res<ProjectRoot>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let args = called.args.clone();
    let place = Place::of(&home, paths.as_deref(), &project);
    let limit = args.limit.unwrap_or(DEFAULT_LIMIT).clamp(1, MAX_LIMIT);
    let work = with_index(index.clone(), place, move |index, place| {
        let project = (args.scope == Scope::Project).then_some(place.project.as_path());
        let hits = index.search(&args.query, project, limit);
        if hits.is_empty() {
            return Ok(format!("No past session matches “{}”.", args.query));
        }
        let hits: Vec<String> = hits
            .iter()
            .enumerate()
            .map(|(number, hit)| format!("{}. {}\n{}", number + 1, hit.title, hit.snippet))
            .collect();
        Ok(hits.join("\n\n"))
    });
    let work = async move { answer(work.await) };
    commands
        .entity(called.call)
        .insert(Running::answering(pool(), &wake, work));
}

fn open(
    called: On<ToolCalled<OpenArgs>>,
    index: Res<RecallIndex>,
    home: Res<RigHome>,
    paths: Option<Res<SessionPaths>>,
    project: Res<ProjectRoot>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let OpenArgs { session, message } = called.args.clone();
    let place = Place::of(&home, paths.as_deref(), &project);
    let work = with_index(index.clone(), place, move |index, _| {
        let Some(indexed) = index.sessions.get(session.trim()) else {
            return Err(format!("No past session {session}."));
        };
        if message >= indexed.saved.docs.len() {
            return Err(format!("Session {session} has no message {message}."));
        }
        Ok(format!(
            "{}\n{}",
            indexed.header(&session, true),
            indexed.excerpt(message, OPEN_AROUND, None)
        ))
    });
    let work = async move { answer(work.await) };
    commands
        .entity(called.call)
        .insert(Running::answering(pool(), &wake, work));
}

/// A `/recall` search running for an agent.
#[derive(Component)]
struct Searching {
    agent: Entity,
    query: String,
}

/// What a `/recall` search found.
#[derive(Component)]
struct Found(Result<Vec<Hit>, String>);

impl From<Result<Vec<Hit>, ToolExecutionError>> for Found {
    fn from(found: Result<Vec<Hit>, ToolExecutionError>) -> Self {
        Self(found.map_err(|error| error.to_string()))
    }
}

/// The excerpt picked with `/recall`, which goes with the agent's next
/// message from the user.
#[derive(Component)]
struct PendingRecall(String);

/// `/recall <query>`: searches every past session, then offers the hits.
fn recall_command(
    In(args): In<CommandArgs>,
    index: Res<RecallIndex>,
    home: Res<RigHome>,
    paths: Option<Res<SessionPaths>>,
    project: Res<ProjectRoot>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let query = args.args.trim().to_owned();
    if query.is_empty() {
        notices.write(Notice::error(
            args.agent,
            "/recall needs words to look for.",
        ));
        return;
    }
    let place = Place::of(&home, paths.as_deref(), &project);
    let searched = query.clone();
    let work = with_index(index.clone(), place, move |index, _| {
        index.search(&searched, None, MAX_LIMIT)
    });
    commands.spawn((
        Name::new("recall search"),
        Searching {
            agent: args.agent,
            query,
        },
        Running::spawn_into::<Found, _>(pool(), &wake, work),
    ));
}

/// Offers what a `/recall` search found in a picker.
fn found(
    add: On<Add<Found>>,
    mut searches: Query<(&Searching, &mut Found)>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let Ok((search, mut found)) = searches.get_mut(add.entity) else {
        return;
    };
    commands.entity(add.entity).despawn();
    let agent = search.agent;
    let hits = match std::mem::replace(&mut found.0, Ok(Vec::new())) {
        Ok(hits) if hits.is_empty() => {
            let text = format!("No past session matches “{}”.", search.query);
            notices.write(Notice::info(agent, text));
            return;
        }
        Ok(hits) => hits,
        Err(why) => {
            notices.write(Notice::error(agent, format!("/recall failed: {why}")));
            return;
        }
    };
    let body = format!(
        "Past sessions about “{}”: the one you pick goes with your next message.",
        search.query
    );
    let picker = hits.iter().fold(
        Picker::new("Recall").for_agent(agent).body(body),
        |picker, hit| picker.item(format!("{}  {}", hit.title, one_line(&hit.snippet)), ""),
    );
    let excerpts: Vec<String> = hits.into_iter().map(|hit| hit.excerpt).collect();
    commands.spawn(picker).observe(
        move |picked: On<Picked>, mut commands: Commands, mut notices: MessageWriter<Notice>| {
            if let Some(Chosen::Item(index)) = picked.chosen
                && let Some(excerpt) = excerpts.get(index)
            {
                commands
                    .entity(agent)
                    .insert(PendingRecall(excerpt.clone()));
                notices.write(Notice::info(
                    agent,
                    "The excerpt goes with your next message.",
                ));
            }
        },
    );
}

/// Attaches a picked excerpt to the user's next message to its agent.
fn attach(
    mut messages: Query<(&mut Incoming, &ForAgent), Added<Admitted>>,
    pending: Query<&PendingRecall>,
    mut commands: Commands,
) {
    let mut attached = HashSet::new();
    for (mut message, &ForAgent(agent)) in &mut messages {
        if message.origin.kind != OriginKind::User
            || message.command().is_some()
            || attached.contains(&agent)
        {
            continue;
        }
        if let Ok(PendingRecall(excerpt)) = pending.get(agent) {
            message.attachments.push(Attachment {
                label: "recalled session excerpt".to_owned(),
                content: UserContent::text(excerpt.clone()),
            });
            attached.insert(agent);
            commands.entity(agent).remove::<PendingRecall>();
        }
    }
}

#[cfg(test)]
mod tests;
