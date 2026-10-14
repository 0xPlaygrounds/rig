//! CT-12: session finder.
//!
//! `/find [--tools] [--all-projects] <text>` searches every saved session
//! of this project (of every project with `--all-projects`), the running
//! one included and marked: the user's messages, the replies and the tool
//! calls' arguments, and with `--tools` the tool results too. The search
//! runs on a thread of its own and sends each session's matches as it
//! reads it ([`Running::stream`]), newest session first; a new `/find`
//! drops the search before it, which cancels it. The results are a
//! [`Picker`] that fills as they come: up to three snippets per session,
//! the matched text in «», each beside the session's name, date and model,
//! and a session that does not read is skipped and counted above the list.
//!
//! Picking a snippet shows the messages around it, read only, with three
//! actions: **Resume** switches to that session (rig-sessions'
//! [`SwitchSession`]) and scrolls its transcript to the message once it is
//! restored, **Fork here** copies the session up to and with that message
//! into a new session ([`fork`]) and switches to it, and **Copy** puts the
//! message's text on the clipboard ([`CopyToClipboard`]).

use std::cmp::Reverse;
use std::fs;
use std::path::Path;
use std::sync::Arc;
use std::time::UNIX_EPOCH;

use bevy_tasks::{IoTaskPool, TaskPool};
use futures::channel::mpsc;
use regex::{Regex, RegexBuilder};
use rig_harness::harness_protocol::{SessionDir, SessionId};
use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::Conversation as Log;
use rig_harness::rig_core::message::{Message, ToolResultContent};
use rig_harness::rig_ecs::journal::{SessionRestored, SessionStore};
use rig_harness::rig_memory::SessionMeta;
use rig_sessions::{SessionTitle, SwitchSession};
use rig_tui::{CopyToClipboard, ItemRow, ScrollTo, TranscriptItem};
use serde::{Deserialize, Serialize};

/// Snippets shown per session.
const SNIPPETS: usize = 3;

/// Characters of a snippet on each side of its match.
const AROUND: usize = 40;

/// Messages shown on each side of a hit when it is opened.
const CONTEXT: usize = 2;

/// Characters of each message shown around a hit.
const CONTEXT_CHARS: usize = 400;

/// The file, in a session's plugin files, naming the message its
/// transcript scrolls to once the session is restored.
const JUMP: &str = "session-finder/jump.json";

/// `/find`, its results and what is done with a hit.
#[derive(Default)]
pub struct SessionFinderPlugin;

impl Plugin for SessionFinderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RigHome>()
            .init_resource::<SessionStores>()
            .init_resource::<ProjectRoot>()
            .add_command(
                "find",
                "Search the saved sessions: /find [--tools] [--all-projects] <text>",
                find,
            )
            .add_message::<ForkedHere>()
            .add_observer(arrived)
            .add_systems(Update, jump.run_if(resource_added::<SessionRestored>));
    }
}

/// What `/find` was asked.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FindQuery {
    /// The text looked for, ignoring case.
    pub text: String,
    /// Whether tool results are searched too.
    pub tools: bool,
    /// Whether every project's sessions are searched.
    pub all_projects: bool,
}

impl FindQuery {
    /// `/find`'s arguments: the flags anywhere, the rest the text.
    pub fn parse(args: &str) -> Self {
        let mut query = Self::default();
        let mut words = Vec::new();
        for word in args.split_whitespace() {
            match word {
                "--tools" => query.tools = true,
                "--all-projects" => query.all_projects = true,
                word => words.push(word),
            }
        }
        query.text = words.join(" ");
        query
    }
}

/// A message that matched.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Hit {
    /// The agent whose conversation it is in: its id, its log's name.
    pub agent: String,
    /// The message.
    pub message: MessageId,
    /// What matched in it: `user`, `assistant`, `call <tool>` or
    /// `<tool> result`.
    pub kind: String,
    /// Where it matched, on one line, the matches in «».
    pub snippet: String,
    /// The message's text, as searched.
    pub text: String,
    /// The messages around it, the matches in «».
    pub context: String,
    /// The message after it, where a fork stops, with what the user typed
    /// in it; `None` when it is the last.
    pub next: Option<(MessageId, Option<String>)>,
}

/// A session with matches.
#[derive(Clone, Debug)]
pub struct Found {
    /// Its id.
    pub id: SessionId,
    /// Its name, or the start of what was first typed in it.
    pub name: String,
    /// When it last changed, in milliseconds since the Unix epoch.
    pub updated_ms: u64,
    /// Its main agent's model.
    pub model: Option<String>,
    /// Whether it is the running session.
    pub current: bool,
    /// Its first matches, at most three.
    pub hits: Vec<Hit>,
    /// How many messages matched in all.
    pub matched: usize,
}

impl Found {
    /// The session's line: name, date, model, and whether it runs.
    pub fn header(&self) -> String {
        let date =
            chrono::DateTime::from_timestamp_millis(i64::try_from(self.updated_ms).unwrap_or(0))
                .map(|at| at.format("%Y-%m-%d %H:%M").to_string())
                .unwrap_or_default();
        let mut header = format!("{} · {date}", self.name);
        if let Some(model) = &self.model {
            header.push_str(&format!(" · {model}"));
        }
        if self.current {
            header.push_str(" · this session");
        }
        header
    }
}

/// What the search sends, session by session.
#[derive(Debug)]
pub enum Searched {
    /// A session with matches.
    Found(Box<Found>),
    /// A session read with no match.
    Nothing,
    /// A session that did not read, and why.
    Skipped(SessionId, String),
    /// The search is over.
    Finished,
}

/// A running or finished `/find`, on the entity of its results' picker.
#[derive(Component, Debug)]
pub struct Search {
    /// What it looks for.
    pub query: FindQuery,
    /// The agent it was typed for.
    pub agent: Entity,
    /// The sessions with matches, newest first.
    pub found: Vec<Found>,
    /// The sessions read.
    pub read: usize,
    /// The sessions that did not read.
    pub skipped: Vec<(SessionId, String)>,
    /// Whether every session was read.
    pub finished: bool,
}

impl Search {
    /// What the picker shows above the list: the query, how far it got,
    /// and the sessions skipped.
    pub fn summary(&self) -> String {
        let scope = if self.query.all_projects {
            "every project"
        } else {
            "this project"
        };
        let state = if self.finished {
            ""
        } else {
            " · searching…"
        };
        let mut summary = format!(
            "“{}” in {scope}: {} of {} sessions match{state}",
            self.query.text,
            self.found.len(),
            self.read + self.skipped.len(),
        );
        if !self.skipped.is_empty() {
            summary.push_str(&format!(
                "\n{} skipped: they did not read",
                self.skipped.len()
            ));
        }
        summary
    }

    /// The hit of each picker item, in list order.
    fn items(&self) -> impl Iterator<Item = (&Found, &Hit)> {
        self.found
            .iter()
            .flat_map(|found| found.hits.iter().map(move |hit| (found, hit)))
    }

    /// The results as a picker.
    fn picker(&self) -> Picker {
        let picker = Picker::new("Find")
            .for_agent(self.agent)
            .body(self.summary());
        let mut picker = self.items().fold(picker, |picker, (_, hit)| {
            picker.item(format!("{}: {}", hit.kind, hit.snippet), "")
        });
        for (item, (found, _)) in picker.items.iter_mut().zip(self.items()) {
            item.detail = Some(found.header());
        }
        picker
    }
}

/// An opened hit, on the entity of its actions' picker.
#[derive(Component, Debug)]
pub struct Opened {
    /// The session it is in.
    pub session: Found,
    /// The hit.
    pub hit: Hit,
}

/// The actions on an opened hit, in the picker's order.
const ACTIONS: [&str; 3] = ["Resume", "Fork here", "Copy"];

/// A session forked by Fork here.
#[derive(Message, Clone, Debug, PartialEq, Eq)]
pub struct ForkedHere {
    /// The session forked.
    pub from: SessionId,
    /// The new session.
    pub to: SessionId,
}

/// Where a session's transcript scrolls once restored.
#[derive(Serialize, Deserialize)]
struct Jump {
    agent: String,
    message: MessageId,
}

/// What the search thread needs.
struct Place {
    home: Home,
    stores: SessionStores,
    project: std::path::PathBuf,
    current: Option<(SessionDir, Arc<dyn Store>)>,
}

/// `/find [--tools] [--all-projects] <text>`: drops the search before
/// it, starts this one off the main thread and shows its picker.
fn find(
    In(args): In<CommandArgs>,
    (home, stores, root): (Res<RigHome>, Res<SessionStores>, Res<ProjectRoot>),
    (paths, store): (Option<Res<SessionPaths>>, Option<Res<SessionStore>>),
    searches: Query<Entity, With<Search>>,
    wake: Res<Wake>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    let query = FindQuery::parse(&args.args);
    let Some(regex) = matcher(&query.text) else {
        notices.write(Notice::error(args.agent, "/find needs a text to look for."));
        return;
    };
    for search in &searches {
        commands.entity(search).despawn();
    }
    let place = Place {
        home: home.0.clone(),
        stores: stores.clone(),
        project: root.0.clone(),
        current: paths
            .zip(store)
            .map(|(paths, store)| (paths.0.clone(), store.0.clone())),
    };
    let (sender, received) = mpsc::unbounded();
    let (tools, all) = (query.tools, query.all_projects);
    let started = std::thread::Builder::new()
        .name("session-finder".to_owned())
        .spawn(move || search(&place, &regex, (tools, all), &sender));
    if let Err(failure) = started {
        notices.write(Notice::error(
            args.agent,
            format!("/find could not start: {failure}"),
        ));
        return;
    }
    let search = Search {
        query,
        agent: args.agent,
        found: Vec::new(),
        read: 0,
        skipped: Vec::new(),
        finished: false,
    };
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    commands
        .spawn((
            search.picker(),
            search,
            Running::stream(pool, &wake, received),
        ))
        .observe(picked_hit);
}

/// The query as a regular expression matching it as typed, ignoring case.
fn matcher(text: &str) -> Option<Regex> {
    let text = text.trim();
    if text.is_empty() {
        return None;
    }
    RegexBuilder::new(&regex::escape(text))
        .case_insensitive(true)
        .build()
        .ok()
}

/// Reads each session, newest first, and sends what it found in it; stops
/// once nobody receives.
fn search(
    place: &Place,
    regex: &Regex,
    (tools, all): (bool, bool),
    sender: &mpsc::UnboundedSender<Searched>,
) {
    for (dir, updated_ms) in sessions(place, all) {
        let searched = search_one(place, &dir, updated_ms, regex, tools);
        if sender.unbounded_send(searched).is_err() {
            return;
        }
    }
    sender.unbounded_send(Searched::Finished).ok();
}

/// The saved sessions to search, newest first, with when each changed:
/// this project's, or every one, and the running one always.
fn sessions(place: &Place, all: bool) -> Vec<(SessionDir, u64)> {
    let current = place.current.as_ref().map(|(dir, _)| dir.path());
    let entries = fs::read_dir(place.home.sessions()).into_iter().flatten();
    let mut sessions: Vec<(SessionDir, u64)> = entries
        .filter_map(Result::ok)
        .filter_map(|entry| {
            let id: SessionId = entry.file_name().to_str()?.parse().ok()?;
            let dir = place.home.session(&id);
            let running = Some(dir.path()) == current;
            if !running && (!dir.is_saved() || dir.is_deleting()) {
                return None;
            }
            let here = dir
                .working_directory()
                .is_some_and(|directory| same_dir(&directory, &place.project));
            if !(running || all || here) {
                return None;
            }
            let meta = place
                .stores
                .open(&dir)
                .ok()
                .and_then(|store| SessionMeta::read(&*store));
            let updated = match meta {
                Some(meta) => meta.updated,
                None => modified_ms(dir.path()),
            };
            Some((dir, updated))
        })
        .collect();
    sessions.sort_by_key(|(_, updated)| Reverse(*updated));
    sessions
}

/// When `path` was last modified, in milliseconds since the Unix epoch.
fn modified_ms(path: &Path) -> u64 {
    let modified = fs::metadata(path).and_then(|meta| meta.modified()).ok();
    let since = modified.and_then(|at| at.duration_since(UNIX_EPOCH).ok());
    since.map_or(0, |since| {
        u64::try_from(since.as_millis()).unwrap_or(u64::MAX)
    })
}

/// Whether two paths name the same directory.
fn same_dir(a: &Path, b: &Path) -> bool {
    a == b || matches!((fs::canonicalize(a), fs::canonicalize(b)), (Ok(a), Ok(b)) if a == b)
}

/// The matches in the session in `dir`, or why it did not read.
fn search_one(
    place: &Place,
    dir: &SessionDir,
    updated_ms: u64,
    regex: &Regex,
    tools: bool,
) -> Searched {
    let Some(id) = dir.id() else {
        return Searched::Nothing;
    };
    let current = place
        .current
        .as_ref()
        .filter(|(running, _)| running.path() == dir.path());
    let store = match current {
        Some((_, store)) => Ok(store.clone()),
        None => place.stores.open(dir),
    };
    let read = store.and_then(|store| Ok((Session::read_lenient(&*store)?, store)));
    let ((session, unreadable), store) = match read {
        Ok(read) => read,
        Err(failure) => return Searched::Skipped(id, failure.to_string()),
    };
    if session.agents.is_empty() && !unreadable.is_empty() {
        let why = unreadable
            .iter()
            .map(|log| format!("{}: {}", log.log, log.error));
        return Searched::Skipped(id, why.collect::<Vec<_>>().join("; "));
    }
    let mut hits = Vec::new();
    for agent in &session.agents {
        hits.extend(matches(&agent.id, &agent.conversation, regex, tools));
    }
    if hits.is_empty() {
        return Searched::Nothing;
    }
    let title = SessionTitle::read(dir, &*store);
    let name = title
        .as_ref()
        .map(SessionTitle::shown)
        .filter(|name| !name.is_empty());
    let matched = hits.len();
    hits.truncate(SNIPPETS);
    Searched::Found(Box::new(Found {
        name: name.map_or_else(|| format!("session {id}"), str::to_owned),
        id,
        updated_ms,
        model: session.main_agent().and_then(SavedAgent::model),
        current: current.is_some(),
        hits,
        matched,
    }))
}

/// The searchable parts of `message`: what each is and its text.
fn parts(message: &Message, tools: bool) -> Vec<(String, String)> {
    let mut parts = Vec::new();
    match message {
        Message::User { content } => {
            for item in content {
                match item {
                    UserContent::Text(text) => parts.push(("user".to_owned(), text.text.clone())),
                    UserContent::ToolResult(result) if tools => {
                        let text: Vec<String> = result
                            .content
                            .iter()
                            .filter_map(|content| match content {
                                ToolResultContent::Text(text) => Some(text.text.clone()),
                                ToolResultContent::Json { value } => Some(value.to_string()),
                                ToolResultContent::Image(_) => None,
                            })
                            .collect();
                        parts.push((format!("{} result", result.name.as_str()), text.join("\n")));
                    }
                    _ => {}
                }
            }
        }
        Message::Assistant(assistant) => {
            for item in &assistant.content {
                match item {
                    AssistantContent::Text(text) => {
                        parts.push(("assistant".to_owned(), text.text.clone()));
                    }
                    AssistantContent::ToolCall(call) => {
                        let arguments = serde_json::Value::Object(call.function.arguments.clone());
                        let name = call.function.name.as_str();
                        parts.push((format!("call {name}"), arguments.to_string()));
                    }
                    _ => {}
                }
            }
        }
        Message::System { .. } => {}
    }
    parts
}

/// Every message of `conversation` with a match, in order.
fn matches(agent: &str, conversation: &Log, regex: &Regex, tools: bool) -> Vec<Hit> {
    let entries: Vec<EntryRef> = conversation.entries().collect();
    let mut hits = Vec::new();
    for (at, entry) in entries.iter().enumerate() {
        let searched = parts(entry.message, tools);
        let Some((kind, text)) = searched.iter().find(|(_, text)| regex.is_match(text)) else {
            continue;
        };
        let next = entries
            .get(at + 1)
            .map(|next| (next.id, typed(conversation, at + 1)));
        let from = at.saturating_sub(CONTEXT);
        let around = entries.iter().skip(from).take(at + CONTEXT + 1 - from);
        let context = around.map(|near| {
            let text: Vec<String> = parts(near.message, true)
                .into_iter()
                .map(|(kind, text)| format!("{kind}: {}", cut(&text, CONTEXT_CHARS)))
                .collect();
            let mark = if near.index == entry.index {
                "▶ "
            } else {
                "  "
            };
            format!("{mark}{}", mark_matches(regex, &text.join("\n  ")))
        });
        hits.push(Hit {
            agent: agent.to_owned(),
            message: entry.id,
            kind: kind.clone(),
            snippet: snippet(regex, text),
            text: parts(entry.message, true)
                .into_iter()
                .map(|(_, text)| text)
                .collect::<Vec<_>>()
                .join("\n"),
            context: context.collect::<Vec<_>>().join("\n"),
            next,
        });
    }
    hits
}

/// The first match of `regex` in `text` with up to [`AROUND`] characters
/// on each side, on one line, the matches in «».
fn snippet(regex: &Regex, text: &str) -> String {
    let Some(first) = regex.find(text) else {
        return String::new();
    };
    let before = text.get(..first.start()).unwrap_or_default();
    let start = before
        .char_indices()
        .rev()
        .nth(AROUND - 1)
        .map_or(0, |(at, _)| at);
    let after = text.get(first.end()..).unwrap_or_default();
    let end = first.end()
        + after
            .char_indices()
            .nth(AROUND)
            .map_or(after.len(), |(at, _)| at);
    let window = text.get(start..end).unwrap_or_default();
    let mut snippet = one_line(&mark_matches(regex, window));
    if start > 0 {
        snippet.insert(0, '…');
    }
    if end < text.len() {
        snippet.push('…');
    }
    snippet
}

/// `text` with every match of `regex` in «».
fn mark_matches(regex: &Regex, text: &str) -> String {
    regex.replace_all(text, "«$0»").into_owned()
}

/// `text` on one line.
fn one_line(text: &str) -> String {
    text.split_whitespace().collect::<Vec<_>>().join(" ")
}

/// At most `chars` characters of `text`, an ellipsis when cut.
fn cut(text: &str, chars: usize) -> String {
    match text.char_indices().nth(chars) {
        None => text.to_owned(),
        Some((at, _)) => format!("{}…", text.get(..at).unwrap_or_default()),
    }
}

/// Fills the results' picker with what the search sent.
fn arrived(
    sent: On<Insert<Done<Searched>>>,
    mut searches: Query<(&mut Search, &mut Done<Searched>)>,
    mut commands: Commands,
) {
    let Ok((mut search, mut done)) = searches.get_mut(sent.entity) else {
        return;
    };
    match std::mem::replace(&mut done.0, Searched::Nothing) {
        Searched::Found(found) => {
            search.read += 1;
            let at = search
                .found
                .partition_point(|other| other.updated_ms >= found.updated_ms);
            search.found.insert(at, *found);
        }
        Searched::Nothing => search.read += 1,
        Searched::Skipped(id, why) => search.skipped.push((id, why)),
        Searched::Finished => search.finished = true,
    }
    commands.entity(sent.entity).insert(search.picker());
}

/// Opens the hit picked: the messages around it, and what to do with it.
fn picked_hit(picked: On<Picked>, searches: Query<&Search>, mut commands: Commands) {
    let Ok(search) = searches.get(picked.entity) else {
        return;
    };
    let chosen = picked
        .chosen
        .as_ref()
        .and_then(|chosen| chosen.items.first());
    let Some((found, hit)) = chosen.and_then(|at| search.items().nth(*at)) else {
        return;
    };
    let body = format!("{}\n\n{}", found.header(), hit.context);
    let picker = ACTIONS.iter().fold(
        Picker::new("Found").for_agent(search.agent).body(body),
        |picker, action| picker.item(*action, ""),
    );
    let opened = Opened {
        session: found.clone(),
        hit: hit.clone(),
    };
    commands.spawn((picker, opened)).observe(acted);
}

/// What an opened hit's actions need.
#[derive(bevy_ecs::system::SystemParam)]
struct Sessions<'w> {
    home: Res<'w, RigHome>,
    stores: Res<'w, SessionStores>,
    paths: Option<Res<'w, SessionPaths>>,
    store: Option<Res<'w, SessionStore>>,
}

impl Sessions<'_> {
    /// The store of the session `id`, the running one's if it is that.
    fn store(&self, id: &SessionId) -> Result<(SessionDir, Arc<dyn Store>), String> {
        let dir = self.home.0.session(id);
        let running = self
            .paths
            .as_ref()
            .is_some_and(|paths| paths.path() == dir.path());
        match (&self.store, running) {
            (Some(store), true) => Ok((dir, store.0.clone())),
            _ => {
                let store = self
                    .stores
                    .open(&dir)
                    .map_err(|failure| failure.to_string())?;
                Ok((dir, store))
            }
        }
    }
}

/// Carries out the action picked on an opened hit.
fn acted(
    picked: On<Picked>,
    opened: Query<&Opened>,
    sessions: Sessions,
    agents: Query<(Entity, &AgentId)>,
    mut copies: Option<ResMut<bevy_ecs::message::Messages<CopyToClipboard>>>,
    mut forked: MessageWriter<ForkedHere>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok(Opened { session, hit }) = opened.get(picked.entity) else {
        return;
    };
    let chosen = picked
        .chosen
        .as_ref()
        .and_then(|chosen| chosen.items.first());
    let action = chosen.and_then(|at| ACTIONS.get(*at)).copied();
    let result = match action {
        Some("Resume") if session.current => {
            let agent = agents.iter().find(|(_, id)| id.0 == hit.agent);
            if let Some((agent, _)) = agent {
                let item = Some(TranscriptItem::Message(hit.message));
                commands.trigger(ScrollTo {
                    agent,
                    item,
                    row: ItemRow::Reveal,
                });
            }
            Ok(())
        }
        Some("Resume") => sessions.store(&session.id).and_then(|(dir, store)| {
            if !dir.runs_elsewhere() {
                write_jump(&*store, hit)?;
            }
            commands.trigger(SwitchSession {
                session: Some(session.id.to_string()),
            });
            Ok(())
        }),
        Some("Fork here") => fork_here(&sessions, session, hit).map(|to| {
            forked.write(ForkedHere {
                from: session.id.clone(),
                to: to.clone(),
            });
            notices.write(Notice::info(
                None,
                format!("Forked {} into session {to}.", session.name),
            ));
            commands.trigger(SwitchSession {
                session: Some(to.to_string()),
            });
        }),
        Some("Copy") => match copies.as_mut() {
            Some(copies) => {
                copies.write(CopyToClipboard(hit.text.clone()));
                notices.write(Notice::info(None, "Copied the message."));
                Ok(())
            }
            None => Err("there is no clipboard in this front".to_owned()),
        },
        _ => Ok(()),
    };
    if let Err(why) = result {
        notices.write(Notice::error(None, format!("/find: {why}")));
    }
}

/// Names `hit`'s message as where the session in `store` scrolls once
/// restored.
fn write_jump(store: &dyn Store, hit: &Hit) -> Result<(), String> {
    let jump = Jump {
        agent: hit.agent.clone(),
        message: hit.message,
    };
    let bytes = serde_json::to_vec(&jump).map_err(|failure| failure.to_string())?;
    // A session's plugin files are `plugins/…` in its store (`PluginData`).
    store
        .write_file(&format!("plugins/{JUMP}"), &bytes)
        .map_err(|failure| failure.to_string())
}

/// Copies the session up to and with `hit`'s message into a new session,
/// which names where it came from, works in the same directory and
/// scrolls to the message.
fn fork_here(sessions: &Sessions, session: &Found, hit: &Hit) -> Result<SessionId, String> {
    let (from, store) = sessions.store(&session.id)?;
    let (to, target) = sessions
        .home
        .0
        .new_session()
        .map_err(|failure| failure.to_string())?;
    let before = hit
        .next
        .as_ref()
        .map(|(next, _)| (hit.agent.as_str(), *next));
    let parent = ForkedFrom {
        session: session.id.to_string(),
        before: before.map(|(agent, next)| (agent.to_owned(), next)),
        text: hit.next.as_ref().and_then(|(_, text)| text.clone()),
        ..ForkedFrom::default()
    };
    let copied = sessions
        .stores
        .open(&target)
        .and_then(|copy| fork(&*store, &*copy, before, &Keep).map(|_| copy))
        .map_err(|failure| failure.to_string())
        .and_then(|copy| {
            parent
                .write(&*copy)
                .map_err(|failure| failure.to_string())?;
            write_jump(&*copy, hit)
        })
        .and_then(|()| match from.working_directory() {
            Some(directory) => {
                fs::write(target.directory(), directory.to_string_lossy().as_bytes())
                    .map_err(|failure| failure.to_string())
            }
            None => Ok(()),
        });
    if let Err(why) = copied {
        fs::remove_dir_all(target.path()).ok();
        return Err(why);
    }
    Ok(to)
}

/// Once the session is restored, scrolls its transcript to the message a
/// `/find` named, then forgets it.
fn jump(data: Option<Res<PluginData>>, agents: Query<(Entity, &AgentId)>, mut commands: Commands) {
    let Some(data) = data else {
        return;
    };
    let Ok(bytes) = data.session.read_file(JUMP) else {
        return;
    };
    data.session.remove_file(JUMP).ok();
    let Ok(jump) = serde_json::from_slice::<Jump>(&bytes) else {
        return;
    };
    if let Some((agent, _)) = agents.iter().find(|(_, id)| id.0 == jump.agent) {
        let item = Some(TranscriptItem::Message(jump.message));
        commands.trigger(ScrollTo {
            agent,
            item,
            row: ItemRow::Reveal,
        });
    }
}

#[cfg(test)]
mod tests;
