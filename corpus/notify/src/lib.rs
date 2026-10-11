//! Plugin corpus: 50. Notifications (spec `50-notify.md`).
//!
//! A desktop notification, or the terminal's OSC 9 / OSC 777 escape, and
//! the bell, when a top-level agent's long turn ends or a picker waits on
//! the user. `/notify on|off` turns them on and off. Not added in a
//! `--print` run.

use std::fs::OpenOptions;
use std::io::Write;
use std::process::{Command, Stdio};
use std::sync::Arc;

use rig_harness::prelude::*;
use rig_tui::PickRequest;

/// How long a picker waits on the user before it notifies.
pub const PATIENCE: Duration = Duration::from_secs(2);

/// Sends a notification.
pub trait Notifier: Send + Sync + 'static {
    /// Shows `text`, such as `rig: done — All done.`.
    fn notify(&self, text: &str);
}

/// What notifies, and when. Inserted before the plugin, it replaces the
/// default: the system's notifier, for turns of 20 s or more.
#[derive(Resource, Clone)]
pub struct Notifications {
    /// What sends them.
    pub notifier: Arc<dyn Notifier>,
    /// The shortest turn whose end notifies.
    pub min_turn: Duration,
    /// Whether they are sent (`/notify on|off`).
    pub on: bool,
}

impl Default for Notifications {
    fn default() -> Self {
        Self {
            notifier: Arc::new(SystemNotifier::from_env()),
            min_turn: Duration::from_secs(20),
            on: true,
        }
    }
}

impl Notifications {
    fn send(&self, text: &str) {
        if self.on {
            self.notifier.notify(text);
        }
    }
}

/// How [`SystemNotifier`] notifies, from `RIG_NOTIFY`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// `notify-send` or `osascript` (the default).
    Desktop,
    /// The terminal's OSC 9 escape (`RIG_NOTIFY=osc9`), over SSH too.
    Osc9,
    /// The terminal's OSC 777 escape (`RIG_NOTIFY=osc777`).
    Osc777,
    /// The bell alone (`RIG_NOTIFY=bell`).
    Bell,
}

/// Notifies by [`Method`], and rings the terminal's bell.
pub struct SystemNotifier(pub Method);

impl SystemNotifier {
    /// The method `RIG_NOTIFY` names.
    pub fn from_env() -> Self {
        let method = match std::env::var("RIG_NOTIFY").as_deref() {
            Ok("osc9") => Method::Osc9,
            Ok("osc777") => Method::Osc777,
            Ok("bell") => Method::Bell,
            _ => Method::Desktop,
        };
        Self(method)
    }
}

impl Notifier for SystemNotifier {
    fn notify(&self, text: &str) {
        // An escape must not end early on a control character of the text.
        let clean: String = text.chars().filter(|c| !c.is_control()).collect();
        let escape = match self.0 {
            Method::Desktop if desktop(&clean) => String::new(),
            Method::Osc9 => format!("\x1b]9;{clean}\x07"),
            Method::Osc777 => format!("\x1b]777;notify;rig;{clean}\x07"),
            Method::Desktop | Method::Bell => String::new(),
        };
        to_terminal(&format!("{escape}\x07"));
    }
}

/// Shows `text` with the desktop's notifier; whether one started.
fn desktop(text: &str) -> bool {
    let mut command = if cfg!(target_os = "macos") {
        let quoted = text.replace('\\', "\\\\").replace('"', "\\\"");
        let mut command = Command::new("osascript");
        command.arg("-e").arg(format!(
            "display notification \"{quoted}\" with title \"rig\""
        ));
        command
    } else {
        let mut command = Command::new("notify-send");
        command.arg("rig").arg(text);
        command
    };
    let started = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    match started {
        Ok(mut child) => {
            // Reaped on a thread of its own, not waited on in a frame.
            std::thread::spawn(move || child.wait());
            true
        }
        Err(_) => false,
    }
}

/// Writes `bytes` to the controlling terminal, beside the terminal view's
/// frames, if there is one.
fn to_terminal(bytes: &str) {
    if let Ok(mut tty) = OpenOptions::new().write(true).open("/dev/tty") {
        tty.write_all(bytes.as_bytes()).ok();
        tty.flush().ok();
    }
}

/// When the agent's running turn started, on the real clock.
#[derive(Component)]
struct TurnStarted(Duration);

/// A picker open for `agent`, asking `question`.
#[derive(Component)]
struct Asking {
    agent: Entity,
    question: String,
}

/// The picker on this entity waited [`PATIENCE`].
#[derive(Event)]
struct Overdue(Entity);

/// Notifies when a long turn ends or a picker waits on the user.
#[derive(Default)]
pub struct NotifyPlugin;

impl Plugin for NotifyPlugin {
    fn build(&self, app: &mut App) {
        let print = app
            .world()
            .get_resource::<Invoked>()
            .is_some_and(|invoked| invoked.args.print.is_some());
        if print {
            return;
        }
        if !app.world().contains_resource::<Notifications>() {
            app.init_resource::<Notifications>();
        }
        app.add_message::<PickRequest>()
            .add_command("notify", "Desktop notifications: /notify on|off", toggle)
            .add_observer(started)
            .add_observer(ended)
            .add_observer(answered)
            .add_observer(overdue)
            .add_systems(Update, asked);
    }
}

fn started(
    turn: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    time: Res<Time<Real>>,
    mut commands: Commands,
) {
    if let Ok(TurnOf(agent)) = turns.get(turn.entity) {
        commands
            .entity(*agent)
            .try_insert(TurnStarted(time.elapsed()));
    }
}

/// A top-level agent's turn of at least [`Notifications::min_turn`] ended.
fn ended(
    ended: On<TurnEnded>,
    agents: Query<(&TurnStarted, Has<SpawnedBy>)>,
    time: Res<Time<Real>>,
    notifications: Res<Notifications>,
) {
    let Ok((TurnStarted(since), sub)) = agents.get(ended.entity) else {
        return;
    };
    if sub || time.elapsed().saturating_sub(*since) < notifications.min_turn {
        return;
    }
    let first_line = |text: &str| text.lines().next().unwrap_or_default().trim().to_owned();
    let text = match &ended.outcome {
        TurnOutcome::Answered(message) => match final_answer(message) {
            Some(answer) => format!("rig: done — {}", first_line(&answer)),
            None => "rig: done".to_owned(),
        },
        TurnOutcome::Failed(reason) => format!("rig: failed — {}", first_line(reason)),
        TurnOutcome::Stopped => "rig: interrupted".to_owned(),
    };
    notifications.send(&text);
}

/// A picker opened: it notifies unless answered within [`PATIENCE`]. The
/// view shows one at a time, so a new one replaces any open.
fn asked(
    mut requests: MessageReader<PickRequest>,
    open: Query<Entity, With<Asking>>,
    mut commands: Commands,
) {
    for request in requests.read() {
        for asking in &open {
            commands.entity(asking).despawn();
        }
        let asking = Asking {
            agent: request.agent,
            question: request.title.clone(),
        };
        let asking = commands.spawn(asking).id();
        commands
            .delayed()
            .duration(PATIENCE)
            .trigger(Overdue(asking));
    }
}

/// The view runs the picked item as a command for the agent.
fn answered(run: On<RunCommand>, open: Query<(Entity, &Asking)>, mut commands: Commands) {
    for (entity, asking) in &open {
        if asking.agent == run.entity {
            commands.entity(entity).despawn();
        }
    }
}

fn overdue(
    overdue: On<Overdue>,
    open: Query<&Asking>,
    notifications: Res<Notifications>,
    mut commands: Commands,
) {
    if let Ok(asking) = open.get(overdue.0) {
        notifications.send(&format!("rig needs you: {}", asking.question));
        commands.entity(overdue.0).despawn();
    }
}

/// `/notify on|off`.
fn toggle(
    In(args): In<CommandArgs>,
    mut notifications: ResMut<Notifications>,
    mut notices: MessageWriter<Notice>,
) {
    match args.args.as_str() {
        "on" => notifications.on = true,
        "off" => notifications.on = false,
        "" => {}
        _ => {
            notices.write(Notice::error(args.agent, "/notify takes on or off"));
            return;
        }
    }
    let state = if notifications.on { "on" } else { "off" };
    notices.write(Notice::info(
        args.agent,
        format!("Notifications are {state}."),
    ));
}

#[cfg(test)]
mod tests;
