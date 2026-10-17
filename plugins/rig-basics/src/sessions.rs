//! Sessions beyond the one running: `/new`, `/resume` and `/name`, and
//! [`NameSession`] for a plugin naming any session.
//!
//! The session's listing (rig-memory's [`SessionMeta`]) is rewritten on
//! every change of a conversation, the [`SessionTitle`], a [`ResumeLabel`],
//! the main agent's model or a value plugins list with rig-harness's
//! `saved_as(..).listed()`. `/resume` lists sessions from it, and a resumed
//! session reads its title back from it.
//! `/resume` with no argument opens the harness's [`SessionPicker`]. With
//! the `tui` feature, the status line shows the session's name.
//!
//! The agent runs one session per process: switching names the next one in
//! [`SessionDir::switch`] and exits with the reload code, and the launcher
//! starts it in its own directory.

use std::any::TypeId;
use std::collections::BTreeMap;

use bevy_app::prelude::*;
use bevy_ecs::change_detection::Tick;
use bevy_ecs::prelude::*;
use bevy_ecs::reflect::{AppTypeRegistry, ReflectComponent};
use bevy_log::error;
use bevy_reflect::prelude::*;
use bevy_reflect::serde::TypedReflectSerializer;
use rig_core::serde_json;
use rig_harness::harness_protocol::{Home, SessionDir, SessionId};
use rig_harness::rig_memory::{SessionMeta, main_order, title_of};

use crate::session_picker::{SessionPicker, SessionPickerPlugin};
use rig_ecs::WriteLogs;
use rig_ecs::agent::{Agent, AgentId, Conversation, Notice, SpawnedBy, TurnOf};
use rig_ecs::model::ModelChoice;
use rig_ecs::prelude::{SavedKeys, now_ms};
use rig_ecs::session_log::{ConversationChanged, SessionStore};
use rig_harness::commands::{AppCommandsExt, CommandArgs};
use rig_harness::pick::Picker;
use rig_harness::prelude::{Exit, ExitReason, Files, RigHome, SessionPaths, first_typed};

mod agents;
#[cfg(feature = "tui")]
mod tui;

pub use agents::{AgentColumn, AgentColumns, AgentsPlugin, ColumnOf, ShownId};

/// `/new`, `/resume` and `/name`, and the session's listing
/// ([`SessionMeta`]), rewritten whenever a conversation, the title, a
/// [`ResumeLabel`], a model or a listed value changes.
#[derive(Default)]
pub struct SessionsPlugin;

impl Plugin for SessionsPlugin {
    fn build(&self, app: &mut App) {
        if !app.is_plugin_added::<SessionPickerPlugin>() {
            app.add_plugins(SessionPickerPlugin);
        }
        app.init_resource::<SessionTitle>()
            .init_resource::<Listed>()
            .init_resource::<ListedValues>()
            .add_command("new", "Start a new session; this one is kept", new)
            .add_command(
                "resume",
                "Pick an earlier session to resume, or /resume <id or name>",
                resume,
            )
            .add_command(
                "name",
                "Name this session for /resume, or show its name",
                name,
            )
            .add_observer(on_switch_session)
            .add_observer(name_session)
            .add_systems(PreStartup, restore_title)
            .add_systems(
                Last,
                (
                    read_listed,
                    update_meta.run_if(
                        on_message::<ConversationChanged>
                            .or_eager(resource_changed::<SessionTitle>)
                            .or_eager(labels_changed)
                            .or_eager(models_changed)
                            .or_eager(resource_changed::<ListedValues>),
                    ),
                )
                    .chain()
                    .after(WriteLogs),
            )
            .add_systems(
                Last,
                write_name
                    .after(update_meta)
                    .run_if(resource_changed::<SessionTitle>),
            );
        #[cfg(feature = "tui")]
        tui::add(app);
    }
}

/// What the session is listed as: the name `/name` gave it, else the start
/// of the first message the user typed, kept from the first turn on.
#[derive(Resource, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Resource, Clone, Debug, Default, PartialEq)]
pub struct SessionTitle {
    /// The name set with `/name`, kept in the session's directory
    /// ([`SessionDir::name`]), where `rig --resume <name>` finds it.
    pub name: Option<String>,
    /// The start of the first message typed ([`SessionMeta::title`]);
    /// empty until a turn ended.
    pub typed: String,
}

impl SessionTitle {
    /// What the saved session in `dir`, kept in `store`, is listed as: its
    /// name ([`SessionDir::name`]) and the title of its listing
    /// ([`SessionMeta`]); `None` when it has neither. Generic, as
    /// [`SessionMeta::read`] is.
    pub fn read(dir: &SessionDir, store: &(impl Files + ?Sized)) -> Option<Self> {
        let name = dir.name();
        let meta = SessionMeta::read(store);
        (name.is_some() || meta.is_some()).then(|| Self {
            name,
            typed: meta.map(|meta| meta.title).unwrap_or_default(),
        })
    }

    /// The name, else the start of the first message typed (empty when
    /// neither is known).
    pub fn shown(&self) -> &str {
        self.name.as_deref().unwrap_or(&self.typed)
    }
}

/// A part of this session's line in `/resume`, such as its goal: spawn
/// one and keep its `text` current. It is written with the session's
/// listing ([`SessionMeta::labels`]), so `/resume` shows it for sessions
/// that are not running, after the title, by `tag`; an empty `text`, or
/// Bevy's `Disabled` on the entity, leaves it out.
#[derive(Component, Reflect, Clone, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Clone, Debug, Default)]
pub struct ResumeLabel {
    /// What it is, and what sorts it, such as `goal`.
    pub tag: String,
    /// What `/resume` shows.
    pub text: String,
}

impl ResumeLabel {
    /// `text` in this session's line, sorted by `tag`.
    pub fn new(tag: impl Into<String>, text: impl Into<String>) -> Self {
        Self {
            tag: tag.into(),
            text: text.into(),
        }
    }
}

/// Run another session: the one named, or a new one. Refused while a turn
/// runs, and without the launcher.
#[derive(Event, Reflect, Clone, Debug)]
#[reflect(Event, Clone, Debug)]
pub struct SwitchSession {
    /// The id or name of the session to resume ([`Home::find`]), or
    /// `None` for a new session.
    pub session: Option<String>,
}

/// Names a session ([`SessionDir::set_name`]): its title in `/resume`,
/// and nothing of its listing, such as its labels. Naming this session is
/// `/name`'s.
#[derive(Event, Clone, Debug)]
pub struct NameSession {
    /// The session.
    pub session: SessionId,
    /// Its name.
    pub name: String,
}

/// What the session's listing holds, as last written or read back,
/// its time left out: a title, label, model or listed value set to what it
/// already says is no change, so resuming a session leaves its place in
/// `/resume`.
#[derive(Resource, Default, PartialEq)]
struct Listed(SessionTitle, SessionMeta);

/// What the session in `paths`, kept in `store`, is listed as, and its
/// listing.
fn listed(paths: &SessionPaths, store: &SessionStore) -> Option<Listed> {
    let title = SessionTitle::read(paths, &*store.0)?;
    let meta = SessionMeta::read(&*store.0).unwrap_or_default();
    Some(Listed(title, SessionMeta { updated: 0, ..meta }))
}

/// The values of the session plugins list (rig-harness's
/// `saved_as(..).listed()`), by key, as last read: copied into the
/// listing whenever one changes.
#[derive(Resource, Default, PartialEq)]
struct ListedValues(BTreeMap<String, serde_json::Value>);

/// A listed value's type: its key and its reflection.
struct ListedType {
    key: String,
    id: TypeId,
    reflect: ReflectComponent,
}

/// Reads each listed value that changed since it was last read: a saved
/// resource's, else the main agent's. Every saved value is listed or not
/// once the app is built, so the listed types are found once.
fn read_listed(
    world: &mut World,
    mut types: Local<Option<Vec<ListedType>>>,
    mut seen: Local<BTreeMap<String, Tick>>,
) {
    let types = types.get_or_insert_with(|| listed_types(world));
    if types.is_empty() {
        return;
    }
    let mut agents = world.query_filtered::<(Entity, &AgentId, Has<SpawnedBy>), With<Agent>>();
    let mut agents: Vec<_> = agents
        .iter(world)
        .map(|(entity, id, spawned)| (entity, id.0.clone(), spawned))
        .collect();
    agents.sort_by(|a, b| main_order(a.2, &a.1).cmp(&main_order(b.2, &b.1)));
    let mut values = world.resource::<ListedValues>().0.clone();
    let registry = world.resource::<AppTypeRegistry>().clone();
    let registry = registry.read();
    for listed in types.iter() {
        let component = world.components().get_id(listed.id);
        let holder = component.and_then(|component| {
            let resource = world.resource_entities().get(component);
            let agent = agents.iter().map(|(entity, ..)| *entity);
            let mut holders = resource.into_iter().chain(agent);
            let holder = holders.find(|entity| world.entity(*entity).contains_id(component))?;
            let ticks = world.entity(holder).get_change_ticks_by_id(component)?;
            Some((holder, ticks.changed))
        });
        let Some((holder, changed)) = holder else {
            values.remove(&listed.key);
            seen.remove(&listed.key);
            continue;
        };
        if seen.get(&listed.key) == Some(&changed) {
            continue;
        }
        seen.insert(listed.key.clone(), changed);
        let Some(reflect) = listed.reflect.reflect(world.entity(holder)) else {
            continue;
        };
        let value = TypedReflectSerializer::new(reflect.as_partial_reflect(), &registry);
        match serde_json::to_value(value) {
            Ok(value) => {
                values.insert(listed.key.clone(), value);
            }
            Err(failure) => error!("not listing `{}`: {failure}", listed.key),
        }
    }
    drop(registry);
    let mut listed = world.resource_mut::<ListedValues>();
    if listed.0 != values {
        listed.0 = values;
    }
}

/// The types of the values listed, of every type registered.
fn listed_types(world: &World) -> Vec<ListedType> {
    let (Some(keys), Some(registry)) = (
        world.get_resource::<SavedKeys>(),
        world.get_resource::<AppTypeRegistry>(),
    ) else {
        return Vec::new();
    };
    let registry = registry.read();
    registry
        .iter()
        .filter_map(|registration| {
            let path = registration.type_info().type_path();
            let key = keys.key(path).filter(|key| keys.is_listed(key))?;
            Some(ListedType {
                key: key.to_owned(),
                id: registration.type_id(),
                reflect: registration.data::<ReflectComponent>()?.clone(),
            })
        })
        .collect()
}

/// Reads the session's name from its directory and its title from its
/// listing.
fn restore_title(
    (paths, store): (Option<Res<SessionPaths>>, Option<Res<SessionStore>>),
    mut title: ResMut<SessionTitle>,
    mut listed_now: ResMut<Listed>,
) {
    let (Some(paths), Some(store)) = (paths, store) else {
        return;
    };
    if let Some(listed) = listed(&paths, &store) {
        // Not a change: nothing to write back.
        *title.bypass_change_detection() = listed.0.clone();
        *listed_now = listed;
    }
}

/// Whether a [`ResumeLabel`] changed, came or went.
fn labels_changed(
    labels: Query<(), Changed<ResumeLabel>>,
    mut removed: RemovedComponents<ResumeLabel>,
) -> bool {
    removed.read().count() > 0 || !labels.is_empty()
}

/// Whether an agent's model changed.
fn models_changed(models: Query<(), (With<Agent>, Changed<ModelChoice>)>) -> bool {
    !models.is_empty()
}

/// Rewrites the session's listing when a conversation changed, and
/// on a new name, label, model or listed value, once some agent logged a
/// message; the first time, the title takes the start of the first message
/// typed. The main agent's model and how many messages its line holds are
/// listed too.
fn update_meta(
    (paths, store): (Option<Res<SessionPaths>>, Option<Res<SessionStore>>),
    (mut title, mut listed): (ResMut<SessionTitle>, ResMut<Listed>),
    mut committed: MessageReader<ConversationChanged>,
    agents: Query<
        (
            &AgentId,
            &Conversation,
            Has<SpawnedBy>,
            Option<&ModelChoice>,
        ),
        With<Agent>,
    >,
    (labels, values): (Query<&ResumeLabel>, Res<ListedValues>),
) {
    let changed = committed.read().count() > 0;
    let (Some(_), Some(store)) = (paths.filter(|paths| paths.is_saved()), store) else {
        return;
    };
    // The agents the user started come first: the title is theirs, not a
    // spawned agent's.
    let mut agents: Vec<_> = agents.iter().collect();
    agents.sort_by(|a, b| main_order(a.2, &a.0.0).cmp(&main_order(b.2, &b.0.0)));
    if title.typed.is_empty()
        && let Some(typed) = agents
            .iter()
            .find_map(|(_, conversation, ..)| first_typed(conversation))
    {
        title.bypass_change_detection().typed = title_of(&typed);
    }
    let labels = labels
        .iter()
        .filter(|label| !label.text.is_empty())
        .map(|label| (label.tag.clone(), title_of(&label.text)))
        .collect();
    let main = agents.first();
    let now = Listed(
        title.clone(),
        SessionMeta {
            title: title.typed.clone(),
            labels,
            updated: 0,
            model: main
                .and_then(|(.., model)| model.map(|model| model.0.clone()))
                .unwrap_or_default(),
            messages: main.map_or(0, |(_, conversation, ..)| conversation.messages().len()),
            listed: values.0.clone(),
        },
    );
    if !changed && *listed == now {
        return;
    }
    let meta = SessionMeta {
        updated: now_ms(),
        ..now.1.clone()
    };
    match meta.write(&*store.0) {
        Ok(()) => *listed = now,
        Err(failure) => error!("could not write the session's meta.json: {failure}"),
    }
}

fn new(
    In(args): In<CommandArgs>,
    conversations: Query<&Conversation, With<Agent>>,
    mut commands: Commands,
    mut notices: MessageWriter<Notice>,
) {
    if conversations
        .iter()
        .all(|conversation| conversation.messages().is_empty())
    {
        notices.write(Notice::info(args.agent, "This session is new already."));
        return;
    }
    commands.trigger(SwitchSession { session: None });
}

/// `/resume <id or name>` runs that session; without arguments, the user
/// picks one of the saved sessions in the native session picker.
fn resume(
    In(args): In<CommandArgs>,
    (home, paths): (Res<RigHome>, Option<Res<SessionPaths>>),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    if !args.args.is_empty() {
        commands.trigger(SwitchSession {
            session: Some(args.args),
        });
        return;
    }
    let Some(paths) = paths else {
        return;
    };
    let saved = home.0.saved_sessions();
    if saved.iter().all(|(_, dir)| dir.path() == paths.path()) {
        notices.write(Notice::info(args.agent, "No earlier session to resume."));
        return;
    }
    commands.spawn((
        Picker::new("Resume a session").for_agent(args.agent),
        SessionPicker {
            command: "resume {session}".to_owned(),
            ..SessionPicker::default()
        },
    ));
}

fn name(
    In(args): In<CommandArgs>,
    mut title: ResMut<SessionTitle>,
    mut notices: MessageWriter<Notice>,
) {
    if args.args.is_empty() {
        let text = match &title.name {
            Some(current) => format!("This session is named “{current}”."),
            None => "This session has no name; /name <title> gives it one.".to_owned(),
        };
        notices.write(Notice::info(args.agent, text));
        return;
    }
    let name = title_of(&args.args);
    notices.write(Notice::info(
        args.agent,
        format!("This session is now named “{name}”."),
    ));
    title.name = Some(name);
}

/// Keeps the session's name in its directory ([`SessionDir::set_name`]),
/// where `rig --resume <name>` and `/resume <name>` find it, as its
/// [`SessionTitle`] has it.
fn write_name(paths: Option<Res<SessionPaths>>, title: Res<SessionTitle>) {
    let Some(paths) = paths else {
        return;
    };
    if paths.name() != title.name
        && let Err(failure) = paths.set_name(title.name.as_deref())
    {
        error!("could not write the session's name: {failure}");
    }
}

/// Names the session asked for: this one's [`SessionTitle`], another's
/// name file ([`SessionDir::set_name`]).
fn name_session(
    named: On<NameSession>,
    paths: Option<Res<SessionPaths>>,
    home: Res<RigHome>,
    mut title: ResMut<SessionTitle>,
    mut notices: MessageWriter<Notice>,
) {
    let name = title_of(&named.name);
    if paths.and_then(|paths| paths.id()).as_ref() == Some(&named.session) {
        title.name = Some(name);
        return;
    }
    let dir = home.0.session(&named.session);
    let written = if dir.path().is_dir() {
        dir.set_name(Some(&name))
            .map_err(|failure| failure.to_string())
    } else {
        Err("there is no such session".to_owned())
    };
    if let Err(why) = written {
        let why = format!("Could not name session {}: {why}.", named.session);
        notices.write(Notice::error(None, why));
    }
}

/// Exits to run the session asked for next.
fn on_switch_session(
    switch: On<SwitchSession>,
    turns: Query<(), With<TurnOf>>,
    paths: Option<Res<SessionPaths>>,
    home: Res<RigHome>,
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    if !turns.is_empty() {
        notices.write(Notice::info(
            None,
            "A turn is running. Press Esc to stop it first.",
        ));
        return;
    }
    let Some(paths) = paths else {
        return;
    };
    let target = match &switch.session {
        None => None,
        Some(id) => match checked(id, &home.0, &paths.0) {
            Ok(id) => Some(id),
            Err(why) => {
                notices.write(Notice::error(None, why));
                return;
            }
        },
    };
    commands.trigger(Exit(ExitReason::Switch(target)));
}

/// The saved session other than the running one that `name_or_id` names.
fn checked(name_or_id: &str, home: &Home, current: &SessionDir) -> Result<SessionId, String> {
    let id = home.find(name_or_id).map_err(|why| format!("{why}."))?;
    if home.session(&id).path() == current.path() {
        return Err("That is this session.".to_owned());
    }
    Ok(id)
}

#[cfg(test)]
mod tests;
