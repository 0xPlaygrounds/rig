//! The view: the terminal front as a client of `rigd`, with a mirror world
//! of the kernel's own components built from a session's events, so the
//! TUI's queries are unchanged.
//!
//! [`view_app`] is the app of the `view` role: [`HeadlessPlugins`], the
//! harness services a front reads (pickers, questions, file watches), and
//! the [`MirrorPlugin`]. The mirror connects to `rigd` on a thread of its
//! own, starts a session or attaches to the one `RIG_SESSION` names, and
//! follows its events from the first ([`mirror`]); each frame applies
//! those that came. What a front asks of an agent goes the other way:
//! [`Deliver`], [`StopTurn`] and [`RunCommand`] for a mirrored agent, and
//! a [`Pick`] on a mirrored question, become the session's commands
//! ([`SendCommand`] and the rest of [`Outbound`]). A view's own commands,
//! such as `/quit`, which detaches and leaves the session running, are
//! [`AppViewCommandsExt::add_view_command`]s.

pub mod mirror;

use std::path::PathBuf;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

use bevy_app::prelude::*;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemId;
use bevy_log::warn;
use crossbeam_channel::{Receiver, Sender};
use rig_ecs::agent::{Notice, StopTurn};
use rig_ecs::inbox::{Deliver, DeliveryMode};
use rig_ecs::stage::EffectRecorded;
use rig_ecs::tasks::Wake;
use rig_protocol::event::Event;
use rig_protocol::methods::{self, AttachRole, SendMode, StopScope};
use rig_protocol::{ClientId, EventRecord, Key, Seq, SessionId};

use crate::ask::Ask;
use crate::commands::{CommandArgs, RunCommand, SlashCommand};
use crate::host::exit::{Exit, ExitReason};
use crate::main_agent::ShownAgent;
use crate::pick::Pick;

pub use mirror::{Applied, Mirror, Mirrored, MirroredAsk};
/// The wire this view speaks, for a view plugin's own requests and events.
pub use rig_protocol as protocol;

/// The environment variable that says how a view attaches to the session
/// `RIG_SESSION` names: `viewer`, `driver` (the default) or `takeover`.
pub const ATTACH: &str = "RIG_ATTACH";

/// The app of a view before its view plugins are loaded.
pub fn view_app() -> App {
    let mut app = App::new();
    // The view's log and pasted images go in a place of its own, never in
    // a session's directory, which is its worker's.
    let home = crate::harness_protocol::Home::from_env();
    let place = home
        .root()
        .join("views")
        .join(std::process::id().to_string());
    if let Err(why) = std::fs::create_dir_all(&place) {
        warn!("the view has no place for its log: {why}");
    }
    let paths = crate::harness_protocol::SessionDir::at(place.clone());
    app.insert_resource(crate::host::session::SessionPaths(paths))
        .insert_resource(crate::host::session::Scratch(Some(place)))
        .insert_resource(crate::services::RigHome(home))
        .set_error_handler(crate::error::warn)
        .add_plugins((crate::HeadlessPlugins, ViewHarnessPlugin, MirrorPlugin));
    app
}

/// The harness services a front reads, without the agent: the paths and
/// processes, the picker and question types, file watches.
struct ViewHarnessPlugin;

impl Plugin for ViewHarnessPlugin {
    fn build(&self, app: &mut App) {
        if !app
            .world()
            .contains_resource::<crate::host::launcher::Invoked>()
        {
            app.insert_resource(crate::host::launcher::Invoked::default());
        }
        app.add_plugins((
            crate::services::ServicesPlugin,
            crate::pick::PickPlugin,
            crate::watch::WatchPlugin,
        ));
    }
}

/// Follows one session's events into the mirror and sends what the view
/// asks of it as the session's commands.
pub struct MirrorPlugin;

impl Plugin for MirrorPlugin {
    fn build(&self, app: &mut App) {
        app.init_resource::<Mirror>()
            .init_resource::<ShownAgent>()
            .init_resource::<Wake>()
            .init_resource::<rig_ecs::effects::Effects>()
            // What a front reads of the kernel, which a view has none of.
            .add_message::<Notice>()
            .add_message::<rig_ecs::inbox::Recalled>()
            .add_message::<rig_ecs::turn::StreamDelta>()
            .add_message::<rig_ecs::session_log::ConversationChanged>()
            .add_message::<EffectRecorded>()
            .add_message::<Applied>()
            .add_systems(PreUpdate, receive.run_if(resource_exists::<ViewLink>))
            .add_systems(Startup, connect.run_if(not(resource_exists::<ViewLink>)))
            .add_observer(deliver)
            .add_observer(stop_turn)
            .add_observer(run_command)
            .add_observer(send_command)
            .add_observer(answer)
            .add_observer(exit);
        app.add_view_command("quit", "Leave the view; the session goes on", quit)
            .add_view_command("detach", "Leave the view; the session goes on", quit)
            .add_view_command("takeover", "Drive the session from this view", takeover);
    }
}

/// The view's link to `rigd`: the session it shows, whether it drives it,
/// and the queue its requests go out on.
#[derive(Resource)]
pub struct ViewClient {
    /// The session shown, once started or attached.
    pub session: Option<SessionId>,
    /// This view's client id, once connected.
    pub me: Option<ClientId>,
    /// Whether this view holds the session's driver lease.
    pub driver: bool,
    out: Sender<Outbound>,
}

impl ViewClient {
    /// Sends `request` to the session; its failure comes back as a notice.
    pub fn send(&self, request: Outbound) {
        if self.out.send(request).is_err() {
            warn!("the view's link to rigd is closed");
        }
    }
}

/// What a view asks of its session.
#[derive(Debug, Clone)]
pub enum Outbound {
    /// A message or a `/` line for the agent.
    Send(methods::Send),
    /// Stop the running turn, or the whole session.
    Stop(StopScope),
    /// Answer an open question.
    Answer(methods::Answer),
    /// Attach as `role`, taking the lease over when `takeover`.
    Attach { role: AttachRole, takeover: bool },
    /// Leave the session running and stop viewing it.
    Detach,
}

/// What comes back from `rigd`.
#[derive(Debug)]
pub enum Inbound {
    /// An event of the session.
    Event(EventRecord),
    /// The session this view shows and how it attached.
    Attached {
        session: SessionId,
        me: Option<ClientId>,
        driver: bool,
    },
    /// Something for the user to read.
    Notice(Notice),
}

/// The receiving end of the link, and the link's own end of what goes out.
#[derive(Resource)]
pub struct ViewLink {
    inbound: Receiver<Inbound>,
}

impl ViewLink {
    /// A link that is not connected to `rigd`: what the view sends is
    /// returned through the receiver, and what it should receive is sent
    /// through the sender. For tests, and for a front that carries the
    /// protocol itself.
    pub fn detached() -> (ViewLink, ViewClient, Sender<Inbound>, Receiver<Outbound>) {
        let (inbound_tx, inbound) = crossbeam_channel::unbounded();
        let (out, outbound) = crossbeam_channel::unbounded();
        let client = ViewClient {
            session: None,
            me: None,
            driver: true,
            out,
        };
        (ViewLink { inbound }, client, inbound_tx, outbound)
    }
}

/// Applies the events that came since the last frame.
fn receive(world: &mut World) {
    let items: Vec<Inbound> = world.resource::<ViewLink>().inbound.try_iter().collect();
    let mut events = Vec::new();
    for item in items {
        match item {
            Inbound::Event(record) => events.push(record),
            Inbound::Attached {
                session,
                me,
                driver,
            } => {
                if let Some(mut client) = world.get_resource_mut::<ViewClient>() {
                    client.session = Some(session);
                    client.me = me;
                    client.driver = driver;
                }
            }
            Inbound::Notice(notice) => {
                world.write_message(notice);
            }
        }
    }
    if events.is_empty() {
        return;
    }
    mirror::apply(world, events);
}

/// Where the view connects and what it shows, from the environment.
#[derive(Debug, Clone)]
struct Target {
    rig_home: PathBuf,
    session: Option<SessionId>,
    role: AttachRole,
    takeover: bool,
    cwd: PathBuf,
}

impl Target {
    fn from_env() -> Target {
        let home = crate::harness_protocol::Home::from_env();
        let session = std::env::var(crate::harness_protocol::env::SESSION)
            .ok()
            .filter(|id| !id.is_empty())
            .map(SessionId);
        let attach = std::env::var(ATTACH).unwrap_or_default();
        let (role, takeover) = match attach.as_str() {
            "viewer" => (AttachRole::Viewer, false),
            "takeover" => (AttachRole::Driver, true),
            _ => (AttachRole::Driver, false),
        };
        Target {
            rig_home: home.root().to_owned(),
            session,
            role,
            takeover,
            cwd: std::env::current_dir().unwrap_or_default(),
        }
    }
}

/// Connects to `rigd` on threads of the view's own, which wake the loop
/// with each item.
fn connect(mut commands: Commands, wake: Option<Res<Wake>>) {
    let wake = wake.map(|wake| wake.clone()).unwrap_or_default();
    let (inbound_tx, inbound) = crossbeam_channel::unbounded();
    let (out, outbound) = crossbeam_channel::unbounded();
    commands.insert_resource(ViewLink { inbound });
    commands.insert_resource(ViewClient {
        session: None,
        me: None,
        driver: false,
        out,
    });
    let target = Target::from_env();
    let spawned = std::thread::Builder::new()
        .name("rig-view-link".into())
        .spawn(move || link(target, inbound_tx, outbound, wake));
    if let Err(why) = spawned {
        warn!("the view could not start its link to rigd: {why}");
    }
}

/// A fresh idempotency key.
pub fn key() -> Key {
    static NEXT: AtomicU64 = AtomicU64::new(0);
    let n = NEXT.fetch_add(1, Ordering::Relaxed);
    let nanos = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or(0, |since| since.as_nanos());
    Key(format!("view-{}-{nanos}-{n}", std::process::id()))
}

/// The link: connects, starts or attaches, then follows the session's
/// events on this thread and sends the view's requests on another.
fn link(target: Target, inbound: Sender<Inbound>, outbound: Receiver<Outbound>, wake: Wake) {
    let tell = |item: Inbound| {
        let sent = inbound.send(item).is_ok();
        wake.wake();
        sent
    };
    let failed = |what: &str, why: &rig_client::ClientError| {
        let next = why
            .next()
            .map(|next| format!(" Next: {next}"))
            .unwrap_or_default();
        Inbound::Notice(Notice::error(None, format!("{what}: {why}.{next}")))
    };
    let local = rig_client::Target::Local {
        rig_home: target.rig_home.clone(),
    };
    let client = match rig_client::Client::connect(&local, "rig-tui") {
        Ok(client) => Arc::new(client),
        Err(why) => {
            tell(failed("Could not reach the rig server", &why));
            return;
        }
    };
    let session = match &target.session {
        Some(session) => session.clone(),
        None => {
            let start = methods::Start {
                cwd: target.cwd.clone(),
                prompt: None,
                name: None,
                host: None,
                build: None,
                grant: None,
                policy: None,
                budget: None,
                context: None,
                unload: None,
                view: true,
                key: key(),
            };
            match client.call::<methods::Start>(&start) {
                Ok(started) => started.session_id,
                Err(why) => {
                    tell(failed("Could not start a session", &why));
                    return;
                }
            }
        }
    };
    let attach = |role: AttachRole, takeover: bool| methods::Attach {
        session_id: session.clone(),
        role,
        takeover,
    };
    let attached = match client.call::<methods::Attach>(&attach(target.role, target.takeover)) {
        // Another view drives it: watch, and take over when asked.
        Err(_) if target.role == AttachRole::Driver && !target.takeover => {
            client.call::<methods::Attach>(&attach(AttachRole::Viewer, false))
        }
        attached => attached,
    };
    match attached {
        Ok(attached) => {
            tell(attached_as(&client, &session, attached.driver));
        }
        Err(why) => {
            tell(failed("Could not attach to the session", &why));
            return;
        }
    }
    let requests = {
        let client = client.clone();
        let session = session.clone();
        let inbound = inbound.clone();
        let wake = wake.clone();
        std::thread::Builder::new()
            .name("rig-view-requests".into())
            .spawn(move || {
                for request in outbound {
                    match request_once(&client, &session, request) {
                        Ok(Some(item)) => {
                            let _ = inbound.send(item);
                            wake.wake();
                        }
                        Ok(None) => {}
                        Err(why) => {
                            let next = why
                                .next()
                                .map(|next| format!(" Next: {next}"))
                                .unwrap_or_default();
                            let text = format!("The session refused: {why}.{next}");
                            let _ = inbound.send(Inbound::Notice(Notice::error(None, text)));
                            wake.wake();
                        }
                    }
                }
            })
    };
    if let Err(why) = requests {
        tell(Inbound::Notice(Notice::error(
            None,
            format!("No request thread: {why}"),
        )));
        return;
    }
    let events = match client.subscribe(&session, Seq(0)) {
        Ok(events) => events,
        Err(why) => {
            tell(failed("Could not follow the session", &why));
            return;
        }
    };
    for event in events {
        let item = match event {
            Ok(record) => {
                if let Some(Event::DriverChanged(changed)) = Event::decode(&record) {
                    tell(attached_as(&client, &session, changed.driver));
                }
                Inbound::Event(record)
            }
            Err(why) => {
                tell(failed("The session's events stopped", &why));
                return;
            }
        };
        if !tell(item) {
            return;
        }
    }
}

/// The session this view shows, and whether `driver`, the session's
/// driver now, is this view.
fn attached_as(
    client: &rig_client::Client,
    session: &SessionId,
    driver: Option<ClientId>,
) -> Inbound {
    let me = client.id();
    Inbound::Attached {
        session: session.clone(),
        driver: driver.is_some() && driver == me,
        me,
    }
}

/// Sends one request of the view; what it changed for the view, if
/// anything, comes back.
fn request_once(
    client: &rig_client::Client,
    session: &SessionId,
    request: Outbound,
) -> Result<Option<Inbound>, rig_client::ClientError> {
    match request {
        Outbound::Send(send) => client.call::<methods::Send>(&send).map(|_| None),
        Outbound::Stop(scope) => {
            let stop = methods::Stop {
                session_id: session.clone(),
                scope,
                key: key(),
            };
            client.call::<methods::Stop>(&stop).map(|_| None)
        }
        Outbound::Answer(answer) => client.call::<methods::Answer>(&answer).map(|_| None),
        Outbound::Attach { role, takeover } => {
            let attach = methods::Attach {
                session_id: session.clone(),
                role,
                takeover,
            };
            let attached = client.call::<methods::Attach>(&attach)?;
            Ok(Some(attached_as(client, session, attached.driver)))
        }
        Outbound::Detach => {
            let detach = methods::Detach {
                session_id: session.clone(),
            };
            client.call::<methods::Detach>(&detach).map(|_| None)
        }
    }
}

/// What the view's input triggers in place of delivering to a local
/// agent: the session's `rig/session/send`.
#[derive(Event, Debug, Clone)]
pub struct SendCommand(pub methods::Send);

fn send_command(send: On<SendCommand>, client: Option<Res<ViewClient>>) {
    let Some(client) = client else {
        return;
    };
    client.send(Outbound::Send(send.0.clone()));
}

/// The session's `send` of `text`, sent as `mode`.
fn send_of(client: &ViewClient, text: String, mode: DeliveryMode) -> Option<methods::Send> {
    let mode = match mode {
        DeliveryMode::Queue | DeliveryMode::Inform => SendMode::Queue,
        _ => SendMode::Steer,
    };
    Some(methods::Send {
        session_id: client.session.clone()?,
        text: Some(text),
        content: None,
        mode: Some(mode),
        key: key(),
    })
}

/// A message for a mirrored agent goes to the session; a `/` line that
/// names a view command runs here.
fn deliver(
    deliver: On<Deliver>,
    mirrored: Query<(), With<Mirrored>>,
    view_commands: Query<(&Name, &ViewCommand)>,
    client: Option<Res<ViewClient>>,
    mut commands: Commands,
) {
    if !mirrored.contains(deliver.entity) {
        return;
    }
    let text = deliver.message.text().to_owned();
    if let Some(line) = deliver.message.command()
        && let Some(system) = view_command(&view_commands, line)
    {
        let args = CommandArgs {
            agent: deliver.entity,
            args: arguments(line),
            mode: deliver.message.mode,
        };
        commands.run_system_with(system, args);
        return;
    }
    let Some(client) = client else {
        return;
    };
    match send_of(&client, text, deliver.message.mode) {
        Some(send) => commands.trigger(SendCommand(send)),
        None => {
            let text = "Not attached to a session yet.";
            commands.write_message(Notice::warn(deliver.entity, text));
        }
    }
}

/// A key bound to a command, or a picker's: a view command runs here, any
/// other goes to the session as its `/` line.
fn run_command(
    run: On<RunCommand>,
    mirrored: Query<(), With<Mirrored>>,
    view_commands: Query<(&Name, &ViewCommand)>,
    client: Option<Res<ViewClient>>,
    mut commands: Commands,
) {
    if let Some(system) = view_command(&view_commands, &run.line) {
        let args = CommandArgs {
            agent: run.entity,
            args: arguments(&run.line),
            mode: run.mode,
        };
        commands.run_system_with(system, args);
        return;
    }
    if !mirrored.contains(run.entity) {
        return;
    }
    let Some(client) = client else {
        return;
    };
    if let Some(send) = send_of(&client, format!("/{}", run.line), run.mode) {
        commands.trigger(SendCommand(send));
    }
}

/// An answer to a mirrored question goes to the session, which settles it
/// for every view (`ask.resolved`); closing it denies it.
fn answer(
    pick: On<Pick>,
    asks: Query<(&MirroredAsk, &Ask)>,
    client: Option<Res<ViewClient>>,
    mut commands: Commands,
) {
    let (Ok((asked, ask)), Some(client)) = (asks.get(pick.entity), client) else {
        return;
    };
    let Some(session_id) = client.session.clone() else {
        return;
    };
    let chosen = pick.chosen.as_ref();
    // Closed: the option that refuses, when the question offers one.
    let index = match chosen {
        Some(chosen) => chosen.items.first().copied(),
        None => asked
            .kinds
            .iter()
            .position(|kind| kind.starts_with("reject")),
    };
    let option = index.and_then(|index| ask.choices.get(index));
    let answer = methods::Answer {
        session_id,
        ask_id: asked.id.clone(),
        option_id: option.map(|choice| choice.id.clone()),
        text: chosen.and_then(|chosen| chosen.text.clone()),
        key: key(),
    };
    commands.queue(move |world: &mut World| {
        if let Some(client) = world.get_resource::<ViewClient>() {
            client.send(Outbound::Answer(answer));
        }
    });
}

/// Queued as a command, so it goes out after the messages sent before it.
fn stop_turn(stop: On<StopTurn>, mirrored: Query<(), With<Mirrored>>, mut commands: Commands) {
    if mirrored.contains(stop.entity) {
        commands.queue(|world: &mut World| {
            if let Some(client) = world.get_resource::<ViewClient>() {
                client.send(Outbound::Stop(StopScope::Turn));
            }
        });
    }
}

/// An exit asked of the view leaves it, and leaves the session running.
fn exit(exit: On<Exit>, mut commands: Commands) {
    commands.queue(|world: &mut World| {
        if let Some(client) = world.get_resource::<ViewClient>() {
            client.send(Outbound::Detach);
        }
    });
    let reason = exit.0.clone();
    let code = match &reason {
        ExitReason::Failed(code) => AppExit::from_code(*code),
        _ => AppExit::Success,
    };
    commands.insert_resource(reason);
    commands.write_message(code);
}

fn quit(_: In<CommandArgs>, mut commands: Commands) {
    commands.trigger(Exit(ExitReason::Quit));
}

fn takeover(_: In<CommandArgs>, client: Option<Res<ViewClient>>, mut commands: Commands) {
    match client {
        Some(client) => client.send(Outbound::Attach {
            role: AttachRole::Driver,
            takeover: true,
        }),
        None => {
            commands.write_message(Notice::warn(None, "Not attached to a session."));
        }
    }
}

/// On a slash command's entity: it runs in the view, not in the agent.
#[derive(Component, Clone, Copy, Debug)]
pub struct ViewCommand(SystemId<In<CommandArgs>>);

/// Registers commands that run in the view rather than the agent.
pub trait AppViewCommandsExt {
    /// Register `/name`, described by `help`, that runs `system` in the
    /// view. The terminal view completes it as it completes the agent's.
    fn add_view_command<M>(
        &mut self,
        name: &str,
        help: &str,
        system: impl IntoSystem<In<CommandArgs>, (), M> + 'static,
    ) -> &mut Self;
}

impl AppViewCommandsExt for App {
    fn add_view_command<M>(
        &mut self,
        name: &str,
        help: &str,
        system: impl IntoSystem<In<CommandArgs>, (), M> + 'static,
    ) -> &mut Self {
        let id = self.register_system(system);
        let command = (
            Name::new(format!("/{name}")),
            SlashCommand {
                help: help.to_owned(),
            },
            ViewCommand(id),
        );
        self.world_mut().entity_mut(id.entity()).insert(command);
        self
    }
}

/// The view command `line` (without its `/`) names.
fn view_command(
    commands: &Query<(&Name, &ViewCommand)>,
    line: &str,
) -> Option<SystemId<In<CommandArgs>>> {
    let name = line.split_whitespace().next()?;
    let found = commands
        .iter()
        .find(|(command, _)| command.as_str().strip_prefix('/') == Some(name));
    found.map(|(_, ViewCommand(id))| *id)
}

/// The words of `line` after its command's name.
fn arguments(line: &str) -> String {
    let line = line.trim();
    let args = line
        .split_once(char::is_whitespace)
        .map_or("", |(_, args)| args);
    args.trim().to_owned()
}

#[cfg(test)]
mod tests;
