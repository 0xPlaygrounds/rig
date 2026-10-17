//! SP-15: a start screen of the project's recent sessions.
//!
//! A new session started at the terminal with no prompt (no `--print`, no
//! command, `AppStart` `New`, an empty conversation and a front that
//! answers) reads the saved sessions' listings off the main thread
//! ([`SessionEntry::read`]: no log is folded). When some ran in this
//! project, a [`Picker`] lists the 8 newest: name or first prompt, age,
//! turns and last model, a session whose store or listing does not read as
//! `unreadable`, and one a launcher runs now as `open elsewhere`. Turns
//! come from [`SessionShape`], a value this plugin lists with each session
//! it runs (`saved_as(..).listed()`).
//!
//! The terminal view's keys: Up and Down move, Enter resumes (`rig_basics::sessions`'
//! [`SwitchSession`]), `n` or
//! Esc closes it for the new session, `/` filters by name or first prompt
//! over every session in scope, and `a` toggles every project's sessions.
//! Enter on an unreadable session says why and stays; on one open
//! elsewhere it offers a read-only view of its conversation.
//! Nothing is written until the user picks: the new session stays unsaved.
//! `start_screen = false` in `[recent_sessions]` turns it off.

use std::cmp::Reverse;
use std::io;
use std::path::PathBuf;

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_basics::session_picker::SessionEntry;
use rig_basics::sessions::SwitchSession;
use rig_harness::harness_protocol::{SessionDir, SessionId};
use rig_harness::prelude::*;
use rig_harness::rig_core::conversation::Conversation as Log;
use rig_harness::rig_ecs::session_log::{ConversationChanged, SessionRestored};
use rig_harness::rig_memory::{META, SessionMeta, read_conversation};
use rig_harness::rig_tools::ago;
use rig_tui::crossterm::event::{Event, KeyCode, KeyEventKind, KeyModifiers};
use rig_tui::ratatui::layout::Constraint;
use rig_tui::{
    InputContext, InputSystems, KeyFocus, KeyInput, PaneKeys, PendingInput, Placement,
    TranscriptPane, TranscriptSource, TuiKeys, TuiPanel,
};
use serde::{Deserialize, Serialize};

/// How many sessions the screen lists.
pub const SHOWN: usize = 8;

/// The key [`SessionShape`] is saved and listed under.
pub const SHAPE: &str = "recent_sessions.shape";

/// The keys, above the list.
const HELP: &str = "↑↓ move · Enter resume · n or Esc new session · / filter · a all projects";

/// The start screen, and the turns each session lists.
#[derive(Default)]
pub struct RecentSessionsPlugin;

impl Plugin for RecentSessionsPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<StartScreenSettings>("recent_sessions")
            .init_resource::<RigHome>()
            .init_resource::<StoreOpener>()
            .init_resource::<ProjectRoot>()
            .init_resource::<KeyFocus>()
            .saved_as::<SessionShape>(SHAPE)
            .listed();
        app.add_systems(Update, start.run_if(resource_added::<SessionRestored>))
            .add_systems(
                Update,
                (
                    track_shape.run_if(on_message::<ConversationChanged>),
                    read_only_closed.run_if(resource_changed::<KeyFocus>),
                ),
            )
            .add_systems(TuiKeys, screen_keys.in_set(InputSystems::Filter))
            .add_observer(listed)
            .add_observer(picked)
            .add_observer(chose_elsewhere)
            .add_observer(line_read);
    }
}

/// The `[recent_sessions]` settings table.
#[derive(Resource, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(default, deny_unknown_fields)]
pub struct StartScreenSettings {
    /// Whether a new session started with no prompt shows the start
    /// screen; on by default.
    pub start_screen: bool,
}

impl Default for StartScreenSettings {
    fn default() -> Self {
        Self { start_screen: true }
    }
}

/// How many turns the main agent took: saved with the session and copied
/// into its listing, so the start screen shows them without reading a
/// log. Inserted once the first turn began.
#[derive(Resource, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
#[reflect(opaque, Resource, Saved, Default, Clone, Serialize, Deserialize)]
pub struct SessionShape {
    /// The turns.
    pub turns: usize,
}

/// A saved session as the start screen lists it.
#[derive(Clone, Debug)]
pub struct RecentSession {
    /// The session as a listing reads it.
    pub entry: SessionEntry,
    /// Its turns, when the rig that ran it listed them.
    pub shape: Option<SessionShape>,
    /// Why its store or listing does not read, when it does not.
    pub unreadable: Option<String>,
}

impl RecentSession {
    /// The saved session in `dir`, from its listing alone; `None` when it
    /// is not a saved session.
    pub fn read(dir: &SessionDir, stores: &StoreOpener) -> Option<Self> {
        let entry = SessionEntry::read(dir, stores)?;
        let unreadable = match stores.open(dir) {
            Err(failure) => Some(failure.to_string()),
            Ok(store) => match store.read_file(META) {
                Ok(bytes) => serde_json::from_slice::<SessionMeta>(&bytes)
                    .err()
                    .map(|failure| format!("its listing does not read: {failure}")),
                Err(failure) if failure.kind() == io::ErrorKind::NotFound => None,
                Err(failure) => Some(failure.to_string()),
            },
        };
        let shape = entry.meta.listed.get(SHAPE);
        let shape = shape.and_then(|saved| serde_json::from_value(saved.clone()).ok());
        Some(Self {
            entry,
            shape,
            unreadable,
        })
    }

    /// Its name, else the first prompt; empty when it has neither.
    pub fn title(&self) -> &str {
        let meta = &self.entry.meta;
        self.entry.name.as_deref().unwrap_or(&meta.title)
    }

    /// Whether the filter `text` is in its name or first prompt, ignoring
    /// case.
    pub fn matches(&self, text: &str) -> bool {
        let text = text.to_lowercase();
        let name = self.entry.name.as_deref().unwrap_or_default();
        [name, &self.entry.meta.title]
            .iter()
            .any(|part| part.to_lowercase().contains(&text))
    }

    /// Its row: the label and the detail beside it.
    pub fn row(&self) -> PickItem {
        let id = &self.entry.id;
        let age = ago(self.entry.updated.elapsed().unwrap_or_default());
        let (label, detail) = match &self.unreadable {
            Some(_) => (
                format!("unreadable · session {id}"),
                format!("{age} · unreadable"),
            ),
            None => {
                let label = match self.title() {
                    "" => format!("session {id}"),
                    title => title.to_owned(),
                };
                let meta = &self.entry.meta;
                let mut parts = vec![age];
                parts.push(match self.shape {
                    Some(shape) => plural(shape.turns, "turn"),
                    None => plural(meta.messages, "message"),
                });
                if !meta.model.is_empty() {
                    parts.push(meta.model.clone());
                }
                if self.entry.running {
                    parts.push("open elsewhere".to_owned());
                }
                (label, parts.join(" · "))
            }
        };
        PickItem {
            label,
            detail: Some(detail),
            ..PickItem::default()
        }
    }
}

/// `count` `noun`s.
fn plural(count: usize, noun: &str) -> String {
    match count {
        1 => format!("1 {noun}"),
        _ => format!("{count} {noun}s"),
    }
}

/// The start screen's state, on its [`Picker`]'s entity.
#[derive(Component, Clone, Debug, Default)]
pub struct StartScreen {
    /// Every saved session read, newest first.
    pub sessions: Vec<RecentSession>,
    /// The project's directory.
    pub project: PathBuf,
    /// Whether every project's sessions are listed.
    pub all_projects: bool,
    /// Whether `/` was pressed: keys type the filter.
    pub filtering: bool,
    /// The filter typed.
    pub filter: String,
    /// The sessions listed, as positions in `sessions`, in the picker's
    /// order.
    pub shown: Vec<usize>,
    /// What went wrong last, shown above the list.
    pub error: Option<String>,
}

impl StartScreen {
    /// The session listed at `item`.
    pub fn listed(&self, item: usize) -> Option<&RecentSession> {
        self.sessions.get(*self.shown.get(item)?)
    }

    /// Lists the 8 newest sessions in scope that pass the filter into
    /// `picker`, with the keys and the filter above them.
    fn fill(&mut self, picker: &mut Picker) {
        let in_scope = |session: &&RecentSession| {
            self.all_projects || session.entry.directory.as_ref() == Some(&self.project)
        };
        self.shown = (self.sessions.iter().enumerate())
            .filter(|(_, session)| in_scope(session) && session.matches(&self.filter))
            .take(SHOWN)
            .map(|(at, _)| at)
            .collect();
        let items: Vec<PickItem> = (self.shown.iter())
            .filter_map(|at| Some(self.sessions.get(*at)?.row()))
            .collect();
        let scope = match self.all_projects {
            true => "every project",
            false => "this project",
        };
        let mut body = format!("{HELP}\nSessions of {scope}");
        if self.filtering {
            body.push_str(&format!(" · filter: {}▏", self.filter));
        }
        if let Some(error) = &self.error {
            body.push_str(&format!("\n{error}"));
        }
        if picker.items != items {
            picker.items = items;
        }
        if picker.body != body {
            picker.body = body;
        }
    }

    /// Its picker and itself, filled, `selected` highlighted.
    fn spawn(mut self, selected: usize, commands: &mut Commands) {
        let mut picker = Picker::new("Recent sessions");
        self.fill(&mut picker);
        picker.selected = selected.min(self.shown.len().saturating_sub(1));
        commands.spawn((picker, self));
    }
}

/// The task reading the saved sessions for the start screen.
#[derive(Component)]
struct Reading;

/// What [`Reading`] read: every saved session but this one, newest first.
#[derive(Component)]
struct ReadSessions(Vec<RecentSession>);

impl From<Vec<RecentSession>> for ReadSessions {
    fn from(sessions: Vec<RecentSession>) -> Self {
        Self(sessions)
    }
}

/// Starts reading the saved sessions once the session is restored, when
/// this run starts a new session with no prompt and someone answers.
fn start(
    settings: Res<StartScreenSettings>,
    (start, invoked): (Option<Res<AppStart>>, Option<Res<Invoked>>),
    answerers: Answerers,
    conversations: Query<&Conversation, With<Agent>>,
    (home, stores, paths): (Res<RigHome>, Res<StoreOpener>, Option<Res<SessionPaths>>),
    wake: Res<Wake>,
    mut commands: Commands,
) {
    let new = start.is_some_and(|start| start.0 == StartReason::New);
    let bare =
        invoked.is_none_or(|invoked| invoked.args.print.is_none() && invoked.command().is_none());
    let empty = (conversations.iter()).all(|conversation| conversation.messages().is_empty());
    if !(settings.start_screen && new && bare && empty && answerers.can_answer(None)) {
        return;
    }
    let (home, stores) = (home.0.clone(), stores.clone());
    let current = paths.and_then(|paths| paths.id());
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    let read = async move {
        let saved = home.saved_sessions().into_iter();
        let others = saved.filter(|(id, _)| Some(id) != current.as_ref());
        let mut read: Vec<RecentSession> = others
            .filter_map(|(_, dir)| RecentSession::read(&dir, &stores))
            .collect();
        read.sort_by_key(|session| Reverse(session.entry.updated));
        read
    };
    let running = Running::spawn_into::<ReadSessions, _>(pool, &wake, read);
    commands.spawn((Name::new("recent sessions"), Reading, running));
}

/// Opens the start screen once the sessions are read, when some ran in
/// this project and the user has not started typing to the agent.
fn listed(
    read: On<Insert<ReadSessions>>,
    mut tasks: Query<&mut ReadSessions, With<Reading>>,
    conversations: Query<&Conversation, With<Agent>>,
    root: Res<ProjectRoot>,
    mut commands: Commands,
) {
    let Ok(mut sessions) = tasks.get_mut(read.entity) else {
        return;
    };
    let sessions = std::mem::take(&mut sessions.0);
    commands.entity(read.entity).despawn();
    let here = |session: &RecentSession| session.entry.directory.as_ref() == Some(&root.0);
    let empty = (conversations.iter()).all(|conversation| conversation.messages().is_empty());
    if empty && sessions.iter().any(here) {
        let screen = StartScreen {
            sessions,
            project: root.0.clone(),
            ..StartScreen::default()
        };
        screen.spawn(0, &mut commands);
    }
}

/// The start screen's own keys, before the picker sees them: `/` starts
/// the filter, which then takes the characters typed and Backspace (on an
/// empty filter, Backspace leaves it); `a`
/// toggles every project, `n` closes it; other characters do nothing, so
/// one filter applies. Enter on an unreadable session says why and stays.
fn screen_keys(
    mut events: ResMut<PendingInput>,
    state: Res<InputContext>,
    mut screens: Query<(Entity, &mut StartScreen, &mut Picker)>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Some((entity, mut screen, mut picker)) =
        state.picking().and_then(|at| screens.get_mut(at).ok())
    else {
        return;
    };
    let Some(KeyInput::Event(Event::Key(key))) = events.0.first() else {
        return;
    };
    let plain = (key.modifiers - KeyModifiers::SHIFT).is_empty();
    if key.kind != KeyEventKind::Press || !plain {
        return;
    }
    match (key.code, screen.filtering) {
        (KeyCode::Enter, _) => {
            let at = state.selection().unwrap_or_default();
            let Some(why) = screen
                .listed(at)
                .and_then(|session| session.unreadable.clone())
            else {
                return;
            };
            let error = format!("Cannot resume that session: {why}");
            notices.write(Notice::error(None, error.clone()));
            screen.error = Some(error);
        }
        (KeyCode::Char(typed), true) => screen.filter.push(typed),
        (KeyCode::Backspace, true) => {
            // Backspace on an empty filter leaves it, for `a` and `n`.
            screen.filtering = screen.filter.pop().is_some();
        }
        (KeyCode::Char('/'), false) => screen.filtering = true,
        (KeyCode::Char('a'), false) => screen.all_projects = !screen.all_projects,
        (KeyCode::Char('n'), false) => {
            commands.entity(entity).despawn();
        }
        (KeyCode::Char(_), false) => {}
        _ => return,
    }
    events.0.remove(0);
    let screen = screen.into_inner();
    screen.fill(&mut picker);
}

/// Resumes the session picked; an unreadable one says why and reopens the
/// screen, one open elsewhere asks what to do. Closed, the new session
/// goes on.
fn picked(
    picked: On<Picked>,
    screens: Query<&StartScreen>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok(screen) = screens.get(picked.entity) else {
        return;
    };
    let Some(&at) = picked
        .chosen
        .as_ref()
        .and_then(|chosen| chosen.items.first())
    else {
        return;
    };
    let Some(session) = screen.listed(at) else {
        return;
    };
    let id = session.entry.id.clone();
    if let Some(why) = &session.unreadable {
        let error = format!("Cannot resume that session: {why}");
        notices.write(Notice::error(None, error.clone()));
        let screen = StartScreen {
            error: Some(error),
            ..screen.clone()
        };
        screen.spawn(at, &mut commands);
    } else if session.entry.running {
        let title = format!("“{}” is open in another rig", session.title());
        let picker = Picker::new(title)
            .item("Open it read-only", "")
            .item("Back", "");
        let elsewhere = Elsewhere {
            screen: screen.clone(),
            at,
            id,
        };
        commands.spawn((picker, elsewhere));
    } else {
        commands.trigger(SwitchSession {
            session: Some(id.to_string()),
        });
    }
}

/// The question about a session open elsewhere, on its picker's entity.
#[derive(Component)]
struct Elsewhere {
    /// The screen it came from, to go back to.
    screen: StartScreen,
    /// The session's position in the screen.
    at: usize,
    /// The session.
    id: SessionId,
}

/// Reads a session's conversation for the read-only view; [`line_read`]
/// shows it.
#[derive(Component)]
struct LineRead(Result<Log, String>);

impl From<Result<Log, String>> for LineRead {
    fn from(read: Result<Log, String>) -> Self {
        Self(read)
    }
}

/// The screen to go back to from the read-only view or its reading task.
#[derive(Component)]
struct ReadOnly {
    screen: StartScreen,
    at: usize,
    title: String,
}

/// Opens a session open elsewhere read-only, or goes back.
fn chose_elsewhere(
    picked: On<Picked>,
    asked: Query<&Elsewhere>,
    (home, stores, wake): (Res<RigHome>, Res<StoreOpener>, Res<Wake>),
    mut commands: Commands,
) {
    let Ok(elsewhere) = asked.get(picked.entity) else {
        return;
    };
    let (screen, at) = (elsewhere.screen.clone(), elsewhere.at);
    match picked
        .chosen
        .as_ref()
        .and_then(|chosen| chosen.items.first())
    {
        Some(0) => {
            let (dir, stores) = (home.0.session(&elsewhere.id), stores.clone());
            let read = async move {
                let store = stores.open(&dir).map_err(|failure| failure.to_string())?;
                let read = read_conversation(&*store, None).map_err(|failure| failure.to_string());
                read.map(|(_, conversation)| conversation)
            };
            let pool = IoTaskPool::get_or_init(TaskPool::default);
            let running = Running::spawn_into::<LineRead, _>(pool, &wake, read);
            let title = screen.listed(at).map(|session| session.title().to_owned());
            let title = format!("{} (read-only, Esc to go back)", title.unwrap_or_default());
            commands.spawn((
                Name::new("recent session"),
                ReadOnly { screen, at, title },
                running,
            ));
        }
        _ => screen.spawn(at, &mut commands),
    }
}

/// Shows a conversation read for the read-only view, over the screen with the
/// keys, or says why it did not read and goes back.
fn line_read(
    read: On<Insert<LineRead>>,
    mut reads: Query<(&mut LineRead, &ReadOnly)>,
    mut focus: ResMut<KeyFocus>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok((mut line, view)) = reads.get_mut(read.entity) else {
        return;
    };
    match std::mem::replace(&mut line.0, Err(String::new())) {
        Ok(line) => {
            let pane =
                TranscriptPane::new(TranscriptSource::Value(line)).titled(view.title.clone());
            let size = Constraint::Percentage(90);
            let panel = TuiPanel::new(Placement::Over {
                width: size,
                height: size,
            });
            commands.entity(read.entity).insert((pane, panel, PaneKeys));
            focus.0 = Some(read.entity);
        }
        Err(why) => {
            notices.write(Notice::error(
                None,
                format!("Cannot read that session: {why}"),
            ));
            view.screen.clone().spawn(view.at, &mut commands);
            commands.entity(read.entity).despawn();
        }
    }
}

/// Goes back to the screen once the read-only view lost the keys (Esc).
fn read_only_closed(
    focus: Res<KeyFocus>,
    views: Query<(Entity, &ReadOnly), With<TuiPanel>>,
    mut commands: Commands,
) {
    for (entity, view) in &views {
        if focus.0 != Some(entity) {
            view.screen.clone().spawn(view.at, &mut commands);
            commands.entity(entity).despawn();
        }
    }
}

/// Keeps [`SessionShape`] as the main agent's conversation has it, from
/// its first turn on: nothing is saved before.
fn track_shape(
    agents: Query<(&AgentId, &Conversation, Has<SpawnedBy>), With<Agent>>,
    shape: Option<ResMut<SessionShape>>,
    mut commands: Commands,
) {
    let main = agents.iter().min_by(|a, b| {
        rig_harness::rig_memory::main_order(a.2, &a.0.0)
            .cmp(&rig_harness::rig_memory::main_order(b.2, &b.0.0))
    });
    let Some((_, conversation, _)) = main else {
        return;
    };
    let now = SessionShape {
        turns: conversation.turns().count(),
    };
    match shape {
        Some(mut shape) => {
            shape.set_if_neq(now);
        }
        None if now.turns > 0 => commands.insert_resource(now),
        None => {}
    }
}

#[cfg(test)]
mod tests;
