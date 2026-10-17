//! CT-12: session finder.
//!
//! `/find [--tools] [--all-projects] <text>` searches every saved session
//! of this project (of every project with `--all-projects`), the running
//! one included and marked: the user's messages, the replies and the tool
//! calls' arguments, and with `--tools` the tool results too. The search
//! is a task that lists the sessions ([`Home::saved_sessions`],
//! [`SessionEntry`]) and sends each session's matches as it reads it
//! ([`Running::stream`]); a new `/find` drops the search before it, which
//! cancels it. The results are a [`Picker`] that fills in place as they
//! come, newest session first: up to three snippets per session, the
//! matched text in «», each beside the session's label, age and model,
//! with the highlighted hit's conversation previewed at it
//! ([`PickPreview`]). A session that does not read is skipped and counted
//! above the list.
//!
//! Picking a snippet shows its conversation at that message, read only,
//! with two actions: **Resume** switches to that session (rig-sessions'
//! [`SwitchSession`]) and scrolls its transcript to the message once it is
//! restored, and **Copy** puts the message's text on the clipboard
//! ([`CopyToClipboard`]).

use std::cmp::Reverse;
use std::sync::Arc;

use bevy_tasks::{IoTaskPool, TaskPool};
use regex::{Regex, RegexBuilder};
use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::Conversation as Log;
use rig_harness::rig_core::message::{Message, ToolResultContent};
use rig_harness::rig_ecs::session_log::{SessionRestored, SessionStore};
use rig_sessions::SwitchSession;
use rig_tui::{CopyToClipboard, ItemRow, MainPane, ScrollTo, ShowAgent, TranscriptItem};
use serde::{Deserialize, Serialize};

/// Snippets shown per session.
const SNIPPETS: usize = 3;

/// Characters of a snippet on each side of its match.
const AROUND: usize = 40;

/// The file, in a session's plugin files, naming the message its
/// transcript scrolls to once the session is restored.
const JUMP: &str = "session-finder/jump.json";

/// `/find`, its results and what is done with a hit.
#[derive(Default)]
pub struct SessionFinderPlugin;

impl Plugin for SessionFinderPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<RigHome>()
            .init_resource::<StoreOpener>()
            .init_resource::<ProjectRoot>()
            .init_resource::<Resuming>()
            .add_command(
                "find",
                "Search the saved sessions: /find [--tools] [--all-projects] <text>",
                find,
            )
            .add_observer(arrived)
            .add_observer(previewed)
            .add_systems(Update, jump.run_if(resource_added::<SessionRestored>))
            .add_systems(Last, leaving.run_if(resource_added::<ExitReason>));
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
#[derive(Clone, Debug, PartialEq)]
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
    /// The agent's conversation, shown at the message.
    pub conversation: Arc<Log>,
}

impl Hit {
    /// Its conversation, shown at the message.
    fn preview(&self) -> PickPreview {
        PickPreview::Conversation {
            conversation: (*self.conversation).clone(),
            at: Some(self.message),
        }
    }
}

/// A session with matches.
#[derive(Clone, Debug)]
pub struct Found {
    /// The session as a listing shows it.
    pub session: SessionEntry,
    /// Whether it is the running session.
    pub current: bool,
    /// Its first matches, at most three.
    pub hits: Vec<Hit>,
    /// How many messages matched in all.
    pub matched: usize,
}

impl Found {
    /// The session's line: label, age, model, and whether it runs.
    pub fn header(&self) -> String {
        let session = &self.session;
        let mut header = format!("{} · {}", session.label(), session.detail());
        if !session.meta.model.is_empty() {
            header.push_str(&format!(" · {}", session.meta.model));
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
    fn hits(&self) -> impl Iterator<Item = (&Found, &Hit)> {
        self.found
            .iter()
            .flat_map(|found| found.hits.iter().map(move |hit| (found, hit)))
    }

    /// The picker's items: a hit's snippet, its session's line beside it.
    fn items(&self) -> Vec<PickItem> {
        let item = |(found, hit): (&Found, &Hit)| PickItem {
            label: format!("{}: {}", hit.kind, hit.snippet),
            detail: Some(found.header()),
            ..PickItem::default()
        };
        self.hits().map(item).collect()
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
const ACTIONS: [&str; 2] = ["Resume", "Copy"];

/// The session Resume switches to, and where its transcript scrolls,
/// until the switch is made.
#[derive(Resource, Default)]
struct Resuming(Option<(SessionId, Jump)>);

/// Where a session's transcript scrolls once restored.
#[derive(Clone, Serialize, Deserialize)]
struct Jump {
    agent: String,
    message: MessageId,
}

/// What the search task needs.
struct Place {
    home: Home,
    stores: StoreOpener,
    project: std::path::PathBuf,
    current: Option<(SessionId, Arc<dyn Store>)>,
}

/// `/find [--tools] [--all-projects] <text>`: drops the search before
/// it, starts this one off the main thread and shows its picker.
fn find(
    In(args): In<CommandArgs>,
    (home, stores, root): (Res<RigHome>, Res<StoreOpener>, Res<ProjectRoot>),
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
            .and_then(|paths| paths.0.id())
            .zip(store.map(|store| store.0.clone())),
    };
    let (tools, all) = (query.tools, query.all_projects);
    let sent = std::iter::once((place, regex))
        .flat_map(move |(place, regex)| {
            sessions(&place, all)
                .into_iter()
                .map(move |session| search_one(&place, session, &regex, tools))
        })
        .chain([Searched::Finished]);
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let search = Search {
        query,
        found: Vec::new(),
        read: 0,
        skipped: Vec::new(),
        finished: false,
    };
    commands
        .spawn((
            Picker::new("Find")
                .for_agent(args.agent)
                .body(search.summary()),
            search,
            Running::stream(pool, &wake, futures::stream::iter(sent)),
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

/// The saved sessions to search, newest first: this project's, or every
/// one, and the running one always.
fn sessions(place: &Place, all: bool) -> Vec<SessionEntry> {
    let running = place.current.as_ref().map(|(id, _)| id);
    let mut sessions: Vec<SessionEntry> = (place.home.saved_sessions().iter())
        .filter_map(|(_, dir)| SessionEntry::read(dir, &place.stores))
        .filter(|session| {
            all || Some(&session.id) == running
                || session.directory.as_ref() == Some(&place.project)
        })
        .collect();
    sessions.sort_by_key(|session| Reverse(session.updated));
    sessions
}

/// The matches in `session`, or why it did not read.
fn search_one(place: &Place, session: SessionEntry, regex: &Regex, tools: bool) -> Searched {
    let current = place
        .current
        .as_ref()
        .filter(|(running, _)| *running == session.id);
    let store = match current {
        Some((_, store)) => Ok(store.clone()),
        None => place.stores.open(&place.home.session(&session.id)),
    };
    let (saved, unreadable) = match store.and_then(|store| Session::read_lenient(&*store)) {
        Ok(read) => read,
        Err(failure) => return Searched::Skipped(session.id, failure.to_string()),
    };
    if saved.agents.is_empty() && !unreadable.is_empty() {
        let why = unreadable
            .iter()
            .map(|log| format!("{}: {}", log.log, log.error));
        return Searched::Skipped(session.id, why.collect::<Vec<_>>().join("; "));
    }
    let hits: Vec<Hit> = (saved.agents.into_iter())
        .flat_map(|agent| matches(&agent.id, Arc::new(agent.conversation), regex, tools))
        .collect();
    if hits.is_empty() {
        return Searched::Nothing;
    }
    Searched::Found(Box::new(Found {
        session,
        current: current.is_some(),
        matched: hits.len(),
        hits: hits.into_iter().take(SNIPPETS).collect(),
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

/// Every message of `agent`'s `conversation` with a match, in order.
fn matches(agent: &str, conversation: Arc<Log>, regex: &Regex, tools: bool) -> Vec<Hit> {
    let mut hits = Vec::new();
    for entry in conversation.entries() {
        let searched = parts(entry.message, tools);
        let Some((kind, text)) = searched.iter().find(|(_, text)| regex.is_match(text)) else {
            continue;
        };
        let all = parts(entry.message, true).into_iter().map(|(_, text)| text);
        hits.push(Hit {
            agent: agent.to_owned(),
            message: entry.id,
            kind: kind.clone(),
            snippet: snippet(regex, text),
            text: all.collect::<Vec<_>>().join("\n"),
            conversation: conversation.clone(),
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
    let marked = regex.replace_all(window, "«$0»");
    let mut snippet = marked.split_whitespace().collect::<Vec<_>>().join(" ");
    if start > 0 {
        snippet.insert(0, '…');
    }
    if end < text.len() {
        snippet.push('…');
    }
    snippet
}

/// Fills the results' picker in place with what the search sent.
fn arrived(
    sent: On<Insert<Done<Searched>>>,
    mut searches: Query<(&mut Search, &mut Done<Searched>, &mut Picker)>,
) {
    let Ok((mut search, mut done, mut picker)) = searches.get_mut(sent.entity) else {
        return;
    };
    match std::mem::replace(&mut done.0, Searched::Nothing) {
        Searched::Found(found) => {
            search.read += 1;
            let at = search
                .found
                .partition_point(|other| other.session.updated >= found.session.updated);
            search.found.insert(at, *found);
            picker.items = search.items();
        }
        Searched::Nothing => search.read += 1,
        Searched::Skipped(id, why) => search.skipped.push((id, why)),
        Searched::Finished => search.finished = true,
    }
    picker.body = search.summary();
}

/// Previews the hit a front highlights in the results.
fn previewed(
    highlighted: On<Insert<Highlighted>>,
    searches: Query<(&Search, &Highlighted)>,
    mut commands: Commands,
) {
    let Ok((search, at)) = searches.get(highlighted.entity) else {
        return;
    };
    if let Some((_, hit)) = search.hits().nth(at.0) {
        commands.entity(highlighted.entity).insert(hit.preview());
    }
}

/// Opens the hit picked: its conversation at the message, and what to do
/// with it.
fn picked_hit(picked: On<Picked>, searches: Query<(&Search, &Picker)>, mut commands: Commands) {
    let Ok((search, results)) = searches.get(picked.entity) else {
        return;
    };
    let chosen = picked
        .chosen
        .as_ref()
        .and_then(|chosen| chosen.items.first());
    let Some((found, hit)) = chosen.and_then(|at| search.hits().nth(*at)) else {
        return;
    };
    let mut picker = ACTIONS.iter().fold(
        Picker::new("Found").body(found.header()),
        |picker, action| picker.item(*action, ""),
    );
    picker.agent = results.agent;
    let opened = Opened {
        session: found.clone(),
        hit: hit.clone(),
    };
    commands
        .spawn((picker, hit.preview(), opened))
        .observe(acted);
}

/// Carries out the action picked on an opened hit.
fn acted(
    picked: On<Picked>,
    opened: Query<&Opened>,
    mut resuming: ResMut<Resuming>,
    agents: Query<(Entity, &AgentId)>,
    main: Option<Single<Entity, With<MainPane>>>,
    mut copies: Option<ResMut<bevy_ecs::message::Messages<CopyToClipboard>>>,
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
    let id = &session.session.id;
    match action {
        Some("Resume") if session.current => {
            let agent = agents.iter().find(|(_, id)| id.0 == hit.agent);
            if let (Some((agent, _)), Some(pane)) = (agent, main.as_deref()) {
                scroll(&mut commands, agent, *pane, hit.message);
            }
        }
        Some("Resume") => {
            let jump = Jump {
                agent: hit.agent.clone(),
                message: hit.message,
            };
            resuming.0 = Some((id.clone(), jump));
            commands.trigger(SwitchSession {
                session: Some(id.to_string()),
            });
        }
        Some("Copy") => match copies.as_mut() {
            Some(copies) => {
                copies.write(CopyToClipboard(hit.text.clone()));
                notices.write(Notice::info(None, "Copied the message."));
            }
            None => {
                notices.write(Notice::error(
                    None,
                    "/find: there is no clipboard in this front",
                ));
            }
        },
        _ => {}
    }
}

/// Once the switch Resume asked for is made, names the message where the
/// session's transcript scrolls once restored, in its plugin files.
fn leaving(
    reason: Res<ExitReason>,
    mut resuming: ResMut<Resuming>,
    home: Res<RigHome>,
    stores: Res<StoreOpener>,
    mut notices: MessageWriter<Notice>,
) {
    let Some((id, jump)) = resuming.0.take() else {
        return;
    };
    let dir = home.0.session(&id);
    if *reason != ExitReason::Switch(Some(id)) || dir.runs_elsewhere() {
        return;
    }
    let written = serde_json::to_vec(&jump)
        .map_err(|failure| failure.to_string())
        .and_then(|bytes| {
            let files = PluginData::of(&stores, &dir);
            let files = files.map_err(|failure| failure.to_string())?;
            files
                .write_file(JUMP, &bytes)
                .map_err(|failure| failure.to_string())
        });
    if let Err(why) = written {
        notices.write(Notice::warn(
            None,
            format!("/find: the resumed session will not scroll to the hit: {why}"),
        ));
    }
}

/// Focuses `agent` and scrolls `pane` to `message`.
fn scroll(commands: &mut Commands, agent: Entity, pane: Entity, message: MessageId) {
    commands.trigger(ShowAgent { entity: agent });
    commands.trigger(ScrollTo {
        pane,
        item: Some(TranscriptItem::Message(message)),
        row: ItemRow::Reveal,
    });
}

/// Once the session is restored, scrolls its transcript to the message a
/// `/find` named, then forgets it.
fn jump(
    data: Option<Res<PluginData>>,
    agents: Query<(Entity, &AgentId)>,
    main: Option<Single<Entity, With<MainPane>>>,
    mut commands: Commands,
) {
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
    let agent = agents.iter().find(|(_, id)| id.0 == jump.agent);
    if let (Some((agent, _)), Some(pane)) = (agent, main.as_deref()) {
        scroll(&mut commands, agent, *pane, jump.message);
    }
}

#[cfg(test)]
mod tests;
