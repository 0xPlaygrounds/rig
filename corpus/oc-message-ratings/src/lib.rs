//! OC-13: rate answers and keep the feedback (one-conversation fresh
//! held-out spec `13-message-ratings.md`).
//!
//! `/good [reason]` and `/bad [reason]` (Alt+Up and Alt+Down from the
//! terminal view's input) rate the agent's last reply. A rating is two
//! saved asides after that reply in the agent's conversation: a
//! [`Rating`] (the data, kind [`Rating::KIND`]) and a [`Note`] the fronts
//! draw as a marker whose expanded text is the reason. Asides never reach
//! a model and are saved, restored and forked with the conversation, so a
//! rating stays with its reply. `/ratings` lists the session's ratings and
//! jumps to one; `/ratings --all` reads every saved session of the project
//! and tallies good, bad and unrated replies per model.

use std::collections::BTreeMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::path::PathBuf;

use rig_harness::harness_protocol::SessionId;
use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::Conversation as Saved;
use rig_harness::rig_core::message::{AssistantContent, Message};
use rig_harness::rig_ecs::journal::{SessionRestored, SessionStore};
use rig_harness::rig_memory::now_ms;
use rig_tui::crossterm::event::{KeyCode, KeyModifiers};
use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::Line;
use rig_tui::{
    AppKeyBindingsExt, AppToolRenderersExt, EntryView, InputAction, ItemRow, KeyBinding, KeyPress,
    KeyPressed, KeyScope, ScrollTo, TranscriptItem, TranscriptView, TuiPlugin, run_command,
};
use serde::{Deserialize, Serialize};

/// The plugin name the ratings' asides are written by.
pub const AUTHOR: &str = "message-ratings";

/// How many sessions deep `/ratings --all` follows a fork to its root.
const FORK_DEPTH: usize = 32;

/// Good or bad.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Verdict {
    /// The reply went well.
    Good,
    /// The reply went badly.
    Bad,
}

impl Verdict {
    /// `good` or `bad`.
    pub fn word(self) -> &'static str {
        match self {
            Self::Good => "good",
            Self::Bad => "bad",
        }
    }
}

/// A reply's rating, kept as an aside after the reply.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct Rating {
    /// Good or bad.
    pub verdict: Verdict,
    /// Why, if the user said.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// The model that wrote the reply, as the reply's origin names it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model: Option<String>,
    /// When it was rated, in milliseconds since the Unix epoch.
    pub at_ms: u64,
}

impl Rating {
    /// The kind of the asides that hold a rating.
    pub const KIND: &'static str = "rating";

    /// The rating `aside` holds, when it is one of this plugin's.
    pub fn of(aside: &Aside) -> Option<Self> {
        let ours = aside.kind == Self::KIND && aside.author == Author::plugin(AUTHOR);
        ours.then(|| Self::deserialize(&aside.value).ok()).flatten()
    }

    /// The marker the fronts draw for it.
    fn note(&self) -> Note {
        let title = match &self.reason {
            Some(reason) => format!("{}: {reason}", self.verdict.word()),
            None => self.verdict.word().to_owned(),
        };
        let model = self.model.as_deref().unwrap_or("unknown model");
        let text = format!(
            "{} ({model})",
            self.reason.as_deref().unwrap_or("no reason given")
        );
        Note::new(title, text)
    }
}

/// `/good`, `/bad`, `/ratings`, the keys and the markers.
#[derive(Default)]
pub struct MessageRatingsPlugin;

impl Plugin for MessageRatingsPlugin {
    fn build(&self, app: &mut App) {
        app.add_message::<SessionForked>()
            .add_command(
                "good",
                "Rate the last reply good: /good [reason]",
                rate_good,
            )
            .add_command("bad", "Rate the last reply bad: /bad [reason]", rate_bad)
            .add_command(
                "ratings",
                "This session's ratings; --all: the project's, by model",
                list,
            )
            .add_entry_renderer(AUTHOR, marker)
            .add_systems(
                Update,
                carry_into_fork.run_if(resource_added::<SessionRestored>),
            );
    }

    fn finish(&self, app: &mut App) {
        if app.is_plugin_added::<TuiPlugin>() {
            bind_keys(app);
        }
    }
}

/// Alt+Up and Alt+Down rate from the input, in place of the view's own
/// "select the item above/below", which they still do while the transcript
/// is being read.
fn bind_keys(app: &mut App) {
    let world = app.world_mut();
    let mut built_in = world.query::<(Entity, &KeyBinding, &InputAction)>();
    let taken: Vec<Entity> = built_in
        .iter(world)
        .filter(|(_, binding, action)| {
            binding.scope == KeyScope::Input
                && matches!(action, InputAction::SelectUp | InputAction::SelectDown)
        })
        .map(|(entity, ..)| entity)
        .collect();
    for entity in taken {
        world.entity_mut(entity).insert(Disabled);
    }
    for (code, line, select) in [
        (KeyCode::Up, "good", InputAction::SelectUp),
        (KeyCode::Down, "bad", InputAction::SelectDown),
    ] {
        let press = KeyPress::new(code, KeyModifiers::ALT);
        let rate = KeyBinding::new(press, format!("Rate the last reply {line}"));
        app.add_key_binding(rate.in_scope(KeyScope::Input), run_command(line));
        let reading = KeyBinding::new(press, "Select another item").in_scope(KeyScope::Transcript);
        if let Some(bound) = app.add_key_binding(reading, |_: On<KeyPressed>| {}) {
            app.world_mut().entity_mut(bound).insert(select);
        }
    }
}

/// The marker's look: the verdict and reason, then the model when
/// expanded.
fn marker(entry: &EntryView<'_>) -> Vec<Line<'static>> {
    let style = if entry.title.starts_with("good") {
        Style::new().green()
    } else {
        Style::new().red()
    };
    let mut lines = vec![Line::styled(format!("◈ {}", entry.title), style)];
    if entry.expanded {
        lines.extend(
            entry
                .text
                .lines()
                .map(|line| Line::from(line.to_owned()).dim()),
        );
    }
    lines
}

fn rate_good(
    In(args): In<CommandArgs>,
    agents: Query<&mut Conversation>,
    notices: MessageWriter<Notice>,
) {
    rate(Verdict::Good, &args, agents, notices);
}

fn rate_bad(
    In(args): In<CommandArgs>,
    agents: Query<&mut Conversation>,
    notices: MessageWriter<Notice>,
) {
    rate(Verdict::Bad, &args, agents, notices);
}

/// The model's last reply in `conversation`: its id and message.
fn last_reply(conversation: &Saved) -> Option<(MessageId, &Message)> {
    let mut entries = conversation.entries().rev();
    let reply = entries.find(|entry| is_reply(entry.message))?;
    Some((reply.id, reply.message))
}

/// Whether `message` is a reply of the model's with text.
fn is_reply(message: &Message) -> bool {
    matches!(message, Message::Assistant(reply)
        if reply.content.iter().any(|part| matches!(part, AssistantContent::Text(_))))
}

/// The model that wrote `message`, as its origin names it.
fn model_of(message: &Message) -> Option<String> {
    match message {
        Message::Assistant(reply) => reply.origin.as_ref().map(|origin| origin.model.clone()),
        _ => None,
    }
}

/// The first line of `message`'s text.
fn first_line(message: &Message) -> String {
    let Message::Assistant(reply) = message else {
        return String::new();
    };
    let text = reply.content.iter().find_map(|part| match part {
        AssistantContent::Text(text) => Some(text.text.as_str()),
        _ => None,
    });
    text.and_then(|text| text.trim().lines().next())
        .unwrap_or_default()
        .to_owned()
}

/// This plugin's asides after `reply`: its rating and its marker.
fn ours_after(conversation: &Saved, reply: MessageId) -> Vec<AsideId> {
    let author = Author::plugin(AUTHOR);
    conversation
        .asides()
        .filter(|(_, aside)| aside.author == author && aside.after == Some(reply))
        .map(|(id, _)| id)
        .collect()
}

/// Puts `rating` on `reply`, in place of the rating it had.
fn put(conversation: &mut Conversation, reply: MessageId, rating: &Rating) {
    for id in ours_after(conversation, reply) {
        conversation.remove_aside(id);
    }
    let value = serde_json::to_value(rating).unwrap_or_default();
    let by = Author::plugin(AUTHOR);
    let data = Aside::new(Rating::KIND, by.clone(), value).after(Some(reply));
    conversation.add_aside(data);
    conversation.add_aside(rating.note().aside(by, Some(reply)));
}

/// `/good` or `/bad`: rates the agent's last reply.
fn rate(
    verdict: Verdict,
    args: &CommandArgs,
    mut agents: Query<&mut Conversation>,
    mut notices: MessageWriter<Notice>,
) {
    let Ok(mut conversation) = agents.get_mut(args.agent) else {
        return;
    };
    let Some((reply, message)) = last_reply(&conversation) else {
        notices.write(Notice::error(args.agent, "There is no reply to rate yet."));
        return;
    };
    let reason = args.args.trim();
    let rating = Rating {
        verdict,
        reason: (!reason.is_empty()).then(|| reason.to_owned()),
        model: model_of(message),
        at_ms: now_ms(),
    };
    put(&mut conversation, reply, &rating);
    notices.write(Notice::info(
        args.agent,
        format!("Rated the last reply {}.", verdict.word()),
    ));
}

/// On a `/ratings` picker: what each item opens, an agent's marker.
#[derive(Component)]
struct Opens(Vec<(Entity, AsideId)>);

/// A rated reply found in a session.
struct Rated {
    rating: Rating,
    first_line: String,
}

/// The ratings in `conversation`, oldest first, each with the marker
/// drawn for it.
fn ratings_in(conversation: &Saved) -> Vec<(Rated, Option<AsideId>)> {
    let mut found: Vec<(Rated, Option<AsideId>)> = conversation
        .asides()
        .filter_map(|(_, aside)| {
            let rating = Rating::of(aside)?;
            let reply = aside.after?;
            let message = conversation.messages().get(conversation.index_of(reply)?)?;
            let marker = ours_after(conversation, reply).into_iter().find(|id| {
                conversation
                    .aside(*id)
                    .is_some_and(|it| it.kind == Note::KIND)
            });
            Some((
                Rated {
                    first_line: first_line(message),
                    rating,
                },
                marker,
            ))
        })
        .collect();
    found.sort_by_key(|(rated, _)| rated.rating.at_ms);
    found
}

/// One line of a listed rating.
fn item_of(rated: &Rated) -> PickItem {
    let rating = &rated.rating;
    let model = rating.model.as_deref().unwrap_or("unknown model");
    let reason = rating
        .reason
        .as_deref()
        .map_or_else(String::new, |reason| format!(" · {reason}"));
    PickItem {
        label: format!("{} · {model}{reason}", rating.verdict.word()),
        command: String::new(),
        detail: Some(format!("{} · {}", when(rating.at_ms), rated.first_line)),
        checked: false,
    }
}

/// `at_ms` as a local date and time, to the minute.
fn when(at_ms: u64) -> String {
    let at = i64::try_from(at_ms)
        .ok()
        .and_then(chrono::DateTime::from_timestamp_millis);
    at.map(|at| {
        at.with_timezone(&chrono::Local)
            .format("%Y-%m-%d %H:%M")
            .to_string()
    })
    .unwrap_or_default()
}

/// What `/ratings --all` reads of the sessions besides this one.
#[derive(bevy_ecs::system::SystemParam)]
struct Project<'w> {
    paths: Option<Res<'w, SessionPaths>>,
    store: Option<Res<'w, SessionStore>>,
    stores: Option<Res<'w, SessionStores>>,
    home: Option<Res<'w, RigHome>>,
}

impl Project<'_> {
    /// `RIG_HOME`, and how its sessions' stores open.
    fn home_and_stores(&self) -> (Home, SessionStores) {
        let home = self
            .home
            .as_ref()
            .map_or_else(Home::from_env, |home| home.0.clone());
        let stores = self.stores.as_deref().cloned().unwrap_or_default();
        (home, stores)
    }
}

/// `/ratings` and `/ratings --all`.
fn list(
    In(args): In<CommandArgs>,
    agents: Query<(Entity, &AgentId, &Conversation)>,
    project: Project,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if args.args.trim() == "--all" {
        let picker = across_project(&agents, &project).for_agent(args.agent);
        commands.spawn(picker);
        return;
    }
    let mut picker = Picker::new("Ratings").for_agent(args.agent);
    let mut opens = Vec::new();
    let mut found: Vec<(Entity, Rated, Option<AsideId>)> = agents
        .iter()
        .flat_map(|(agent, _, conversation)| {
            ratings_in(conversation)
                .into_iter()
                .map(move |(rated, marker)| (agent, rated, marker))
        })
        .collect();
    found.sort_by_key(|(_, rated, _)| rated.rating.at_ms);
    for (agent, rated, marker) in found {
        if let Some(marker) = marker {
            picker.items.push(item_of(&rated));
            opens.push((agent, marker));
        }
    }
    if picker.items.is_empty() {
        notices.write(Notice::info(args.agent, "No ratings yet: /good or /bad."));
        return;
    }
    commands.spawn((picker, Opens(opens))).observe(open);
}

/// Scrolls to the rating chosen in a `/ratings` picker and shows it
/// expanded and selected.
fn open(
    picked: On<Picked>,
    opens: Query<&Opens>,
    view: Option<ResMut<TranscriptView>>,
    mut commands: Commands,
) {
    let Some(Chosen::Item(chosen)) = &picked.chosen else {
        return;
    };
    let Some(&(agent, marker)) = opens
        .get(picked.entity)
        .ok()
        .and_then(|opens| opens.0.get(*chosen))
    else {
        return;
    };
    let item = TranscriptItem::Note(marker);
    if let Some(mut view) = view {
        view.expanded.insert(item.clone());
        view.selected = Some(item.clone());
    }
    commands.trigger(ScrollTo {
        agent,
        item: Some(item),
        row: ItemRow::Reveal,
    });
}

/// A reply's identity across sessions: the session a chain of forks
/// began at, the agent, the message's id and its content, so a fork's copy
/// of a reply is the same reply and a reply made after it is not.
type ReplyKey = (String, String, MessageId, u64);

/// What `/ratings --all` gathers: each reply once, with its model and its
/// newest rating.
#[derive(Default)]
struct Gathered {
    replies: BTreeMap<ReplyKey, (String, Option<(Rated, String)>)>,
}

impl Gathered {
    /// Adds the replies of `agent`'s `conversation` in session `session`,
    /// whose forks began at `root`, listed as `label`.
    fn add(&mut self, (root, label): (&str, &str), agent: &str, conversation: &Saved) {
        let ratings: BTreeMap<MessageId, Rating> = conversation
            .asides()
            .filter_map(|(_, aside)| Some((aside.after?, Rating::of(aside)?)))
            .collect();
        for entry in conversation
            .entries()
            .filter(|entry| is_reply(entry.message))
        {
            let mut hasher = DefaultHasher::new();
            serde_json::to_string(entry.message)
                .unwrap_or_default()
                .hash(&mut hasher);
            let key = (root.to_owned(), agent.to_owned(), entry.id, hasher.finish());
            let model = model_of(entry.message).unwrap_or_else(|| "unknown model".to_owned());
            let rated = ratings.get(&entry.id).map(|rating| {
                let rated = Rated {
                    rating: rating.clone(),
                    first_line: first_line(entry.message),
                };
                (rated, label.to_owned())
            });
            let kept = self.replies.entry(key).or_insert((model, None));
            let newer = |(new, _): &(Rated, String)| {
                kept.1
                    .as_ref()
                    .is_none_or(|(old, _)| new.rating.at_ms > old.rating.at_ms)
            };
            if let Some(rated) = rated.filter(newer) {
                kept.1 = Some(rated);
            }
        }
    }

    /// The picker: the tally by model in its body, each rating an item.
    fn picker(self) -> Picker {
        let mut tally: BTreeMap<String, [usize; 3]> = BTreeMap::new();
        let mut rated = Vec::new();
        for (model, rating) in self.replies.into_values() {
            let counts = tally.entry(model).or_default();
            let slot = match &rating {
                Some((rated, _)) if rated.rating.verdict == Verdict::Good => counts.get_mut(0),
                Some(_) => counts.get_mut(1),
                None => counts.get_mut(2),
            };
            if let Some(slot) = slot {
                *slot += 1;
            }
            rated.extend(rating);
        }
        let lines: Vec<String> = tally
            .into_iter()
            .map(|(model, counts)| {
                let parts = counts.iter().zip(["good", "bad", "unrated"]);
                let parts = parts
                    .filter(|(count, _)| **count > 0)
                    .map(|(count, word)| format!("{count} {word}"));
                format!("{model}: {}", parts.collect::<Vec<_>>().join(", "))
            })
            .collect();
        rated.sort_by_key(|(rated, _)| rated.rating.at_ms);
        let mut picker = Picker::new("Ratings in this project").body(lines.join("\n"));
        for (rated, session) in rated {
            let mut item = item_of(&rated);
            item.detail = item.detail.map(|detail| format!("{session} · {detail}"));
            picker.items.push(item);
        }
        picker
    }
}

/// The session a chain of forks that `id` ends began at.
fn root_of(id: &SessionId, home: &Home, stores: &SessionStores) -> String {
    let mut root = id.clone();
    for _ in 0..FORK_DEPTH {
        let parent = stores.open(&home.session(&root)).ok();
        match parent.as_deref().and_then(ForkedFrom::of) {
            Some(parent) => root = parent.session,
            None => break,
        }
    }
    root.to_string()
}

/// Every saved session of the project, this one as it stands now, its
/// replies tallied and its ratings listed.
fn across_project(agents: &Query<(Entity, &AgentId, &Conversation)>, project: &Project) -> Picker {
    let (home, stores) = project.home_and_stores();
    let current = project
        .paths
        .as_ref()
        .map(|paths| (paths.path().to_path_buf(), paths.id()));
    let mut gathered = Gathered::default();
    let here = current.as_ref().and_then(|(_, id)| id.clone());
    let this_root = match (&here, &project.store) {
        (Some(id), Some(store)) => match ForkedFrom::of(&*store.0) {
            Some(parent) => root_of(&parent.session, &home, &stores),
            None => id.to_string(),
        },
        _ => "this session".to_owned(),
    };
    for (_, id, conversation) in agents {
        gathered.add((&this_root, "this session"), &id.0, conversation);
    }
    let directory = project
        .paths
        .as_ref()
        .and_then(|paths| paths.working_directory());
    let path = current.map_or_else(PathBuf::new, |(path, _)| path);
    for session in rig_sessions::list(&home, &stores, &path) {
        if session.directory != directory {
            continue;
        }
        let store = stores.open(&home.session(&session.id));
        let Some((saved, _)) = store
            .ok()
            .and_then(|store| Session::read_lenient(&*store).ok())
        else {
            continue;
        };
        let root = root_of(&session.id, &home, &stores);
        let label = session.label();
        for agent in &saved.agents {
            gathered.add((&root, &label), &agent.id, &agent.conversation);
        }
    }
    gathered.picker()
}

/// In a fork's first start, the parent's newest rating of each reply the
/// fork copied: the copy stops at a point in time, so a reply rated or
/// rated again after it would lose its rating otherwise.
fn carry_into_fork(
    first: Option<Res<FirstStart>>,
    project: Project,
    mut agents: Query<(&AgentId, &mut Conversation)>,
) {
    let (Some(_), Some(store)) = (first, &project.store) else {
        return;
    };
    let Some(parent) = ForkedFrom::of(&*store.0) else {
        return;
    };
    let (home, stores) = project.home_and_stores();
    let parent_store = stores.open(&home.session(&parent.session));
    let Some((saved, _)) = parent_store
        .ok()
        .and_then(|store| Session::read_lenient(&*store).ok())
    else {
        return;
    };
    for (id, mut conversation) in &mut agents {
        let Some(then) = saved.agents.iter().find(|agent| agent.id == id.0) else {
            continue;
        };
        let then = &then.conversation;
        let carried: Vec<(MessageId, Rating)> = then
            .asides()
            .filter_map(|(_, aside)| Some((aside.after?, Rating::of(aside)?)))
            .filter(|(reply, rating)| {
                let message = |c: &Saved| {
                    c.index_of(*reply)
                        .and_then(|at| c.messages().get(at).cloned())
                };
                let copied = message(&conversation).is_some_and(|here| Some(here) == message(then));
                let ours = ratings_of(&conversation, *reply);
                copied && ours.is_none_or(|ours| ours.at_ms < rating.at_ms)
            })
            .collect();
        for (reply, rating) in carried {
            put(&mut conversation, reply, &rating);
        }
    }
}

/// The rating of `reply` in `conversation`, if it has one.
fn ratings_of(conversation: &Saved, reply: MessageId) -> Option<Rating> {
    let mut asides = conversation
        .asides()
        .filter(|(_, aside)| aside.after == Some(reply));
    asides.find_map(|(_, aside)| Rating::of(aside))
}

#[cfg(test)]
mod tests;
