//! Plugin corpus: 50. Notifications (spec `50-notify.md`).
//!
//! A desktop notification, or the terminal's OSC 9 / OSC 777 escape, and
//! the bell, when a top-level agent's long turn ends or a picker or
//! question waits on the user. Nothing is sent while the terminal reports
//! that it has focus, nor in a `--print` run. `/notify on|off`.
//!
//! The `[notify]` table of the settings: `min_seconds = 20`,
//! `via = "desktop" | "osc9" | "osc777" | "bell"`.

use std::fs::OpenOptions;
use std::io::Write;
use std::process::Stdio;
use std::sync::Arc;

use bevy_ecs::system::SystemParam;
use rig_harness::prelude::*;
use rig_tui::crossterm::event::Event;
use rig_tui::{Picker, TerminalEvent};
use serde::Deserialize;

/// How long a picker or question waits on the user before it notifies.
pub const PATIENCE: Duration = Duration::from_secs(2);

/// Sends a notification.
pub trait Notify: Send + Sync + 'static {
    /// Shows `text`, such as `rig: done — All done.`.
    fn notify(&self, text: &str);
}

/// What sends the notifications. Inserted before the app is finished, it
/// replaces the [`SystemNotifier`].
#[derive(Resource, Clone)]
pub struct Notifier(pub Arc<dyn Notify>);

/// The `[notify]` settings; `on` is also what `/notify on|off` sets.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct NotifySettings {
    /// The shortest turn, in seconds, whose end notifies.
    pub min_seconds: u64,
    /// How the [`SystemNotifier`] notifies.
    pub via: Via,
    /// Whether notifications are sent.
    pub on: bool,
}

impl Default for NotifySettings {
    fn default() -> Self {
        Self {
            min_seconds: 20,
            via: Via::Desktop,
            on: true,
        }
    }
}

/// How the [`SystemNotifier`] notifies; it rings the bell besides.
#[derive(Deserialize, Clone, Copy, Debug, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
pub enum Via {
    /// `notify-send` or `osascript`.
    Desktop,
    /// The terminal's OSC 9 escape, which works over SSH.
    Osc9,
    /// The terminal's OSC 777 escape.
    Osc777,
    /// The bell alone.
    Bell,
}

/// Notifies [`Via`] the desktop or the terminal, and rings the bell.
pub struct SystemNotifier {
    /// How.
    pub via: Via,
    /// How a desktop notifier's process starts.
    pub processes: Processes,
}

impl Notify for SystemNotifier {
    fn notify(&self, text: &str) {
        // An escape must not end early on a control character of the text.
        let clean: String = text.chars().filter(|c| !c.is_control()).collect();
        let escape = match self.via {
            Via::Desktop => {
                desktop(&self.processes, &clean);
                String::new()
            }
            Via::Osc9 => format!("\x1b]9;{clean}\x07"),
            Via::Osc777 => format!("\x1b]777;notify;rig;{clean}\x07"),
            Via::Bell => String::new(),
        };
        to_terminal(&format!("{escape}\x07"));
    }
}

/// Shows `text` with the desktop's notifier, if it starts.
fn desktop(processes: &Processes, text: &str) {
    let launcher = processes.launcher();
    let mut command = if cfg!(target_os = "macos") {
        let quoted = text.replace('\\', "\\\\").replace('"', "\\\"");
        let mut command = launcher.command("osascript");
        let script = format!("display notification \"{quoted}\" with title \"rig\"");
        command.arg("-e").arg(script);
        command
    } else {
        let mut command = launcher.command("notify-send");
        command.arg("rig").arg(text);
        command
    };
    let started = command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn();
    if let Ok(mut child) = started {
        // Reaped on a thread of its own, not waited on in a frame.
        std::thread::spawn(move || child.wait());
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

/// Whether the terminal last reported that it has focus.
#[derive(Resource, Default)]
struct TerminalFocused(bool);

/// While it lives, the terminal reports focus changes (`CSI ? 1004 h`).
#[derive(Resource)]
struct FocusReports;

impl FocusReports {
    fn start() -> Self {
        to_terminal("\x1b[?1004h");
        Self
    }
}

impl Drop for FocusReports {
    fn drop(&mut self) {
        to_terminal("\x1b[?1004l");
    }
}

/// When the turn on this entity started, on the app's clock.
#[derive(Component)]
struct TurnClock(Duration);

/// The picker or question on this entity waited [`PATIENCE`].
#[derive(Event)]
struct Overdue(Entity);

/// Sends a notification unless they are off or the terminal has focus.
#[derive(SystemParam)]
struct Sender<'w> {
    settings: Res<'w, NotifySettings>,
    focused: Res<'w, TerminalFocused>,
    notifier: Res<'w, Notifier>,
}

impl Sender<'_> {
    fn send(&self, text: &str) {
        if self.settings.on && !self.focused.0 {
            self.notifier.0.notify(text);
        }
    }
}

/// Notifies when a long turn ends or a picker or question waits on the
/// user.
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
        app.add_setting::<NotifySettings>("notify")
            .init_resource::<TerminalFocused>()
            .add_message::<TerminalEvent>()
            .add_command("notify", "Desktop notifications: /notify on|off", toggle)
            .add_observer(started)
            .add_observer(ended)
            .add_observer(picker_opened)
            .add_observer(asked)
            .add_observer(overdue)
            .add_systems(Update, track_focus);
    }

    fn finish(&self, app: &mut App) {
        let world = app.world_mut();
        if !world.contains_resource::<NotifySettings>() {
            return;
        }
        if !world.contains_resource::<Notifier>() {
            let via = world.resource::<NotifySettings>().via;
            let processes = world.get_resource_or_init::<Processes>().clone();
            world.insert_resource(Notifier(Arc::new(SystemNotifier { via, processes })));
        }
        if app.runs_front(rig_tui::FRONT) {
            app.insert_resource(FocusReports::start());
        }
    }
}

fn started(turn: On<Add<TurnOf>>, time: Res<Time<Real>>, mut commands: Commands) {
    commands
        .entity(turn.entity)
        .try_insert(TurnClock(time.elapsed()));
}

/// A top-level agent's turn of at least `min_seconds` ended.
fn ended(
    settled: On<Add<Settled>>,
    turns: Query<(&TurnOutcome, &ForAgent, &TurnClock)>,
    subagents: Query<(), With<SpawnedBy>>,
    time: Res<Time<Real>>,
    sender: Sender,
) {
    let Ok((outcome, &ForAgent(agent), TurnClock(since))) = turns.get(settled.entity) else {
        return;
    };
    let min = Duration::from_secs(sender.settings.min_seconds);
    if subagents.contains(agent) || time.elapsed().saturating_sub(*since) < min {
        return;
    }
    let first_line = |text: &str| text.lines().next().unwrap_or_default().trim().to_owned();
    let text = match outcome {
        TurnOutcome::Answered(message) => match final_answer(message) {
            Some(answer) => format!("rig: done — {}", first_line(&answer)),
            None => "rig: done".to_owned(),
        },
        TurnOutcome::Failed(reason) => format!("rig: failed — {}", first_line(reason)),
        TurnOutcome::Stopped(None) => "rig: interrupted".to_owned(),
        TurnOutcome::Stopped(Some(reason)) => format!("rig: stopped — {}", first_line(reason)),
    };
    sender.send(&text);
}

fn picker_opened(picker: On<Add<Picker>>, mut commands: Commands) {
    commands
        .delayed()
        .duration(PATIENCE)
        .trigger(Overdue(picker.entity));
}

fn asked(ask: On<Add<Ask>>, mut commands: Commands) {
    commands
        .delayed()
        .duration(PATIENCE)
        .trigger(Overdue(ask.entity));
}

/// A picker still open, or a question still unanswered, after
/// [`PATIENCE`].
fn overdue(
    overdue: On<Overdue>,
    pickers: Query<&Picker>,
    asks: Query<&Ask, Without<Outcome>>,
    sender: Sender,
) {
    let question = match (pickers.get(overdue.0), asks.get(overdue.0)) {
        (Ok(picker), _) => &picker.title,
        (_, Ok(ask)) => &ask.title,
        _ => return,
    };
    sender.send(&format!("rig needs you: {question}"));
}

/// Keeps what the terminal reports about its focus.
fn track_focus(mut events: MessageReader<TerminalEvent>, mut focused: ResMut<TerminalFocused>) {
    for TerminalEvent(event) in events.read() {
        match event {
            Event::FocusGained => focused.0 = true,
            Event::FocusLost => focused.0 = false,
            _ => {}
        }
    }
}

/// `/notify on|off`.
fn toggle(
    In(args): In<CommandArgs>,
    mut settings: ResMut<NotifySettings>,
    mut notices: MessageWriter<Notice>,
) {
    match args.args.as_str() {
        "on" => settings.on = true,
        "off" => settings.on = false,
        "" => {}
        _ => {
            notices.write(Notice::error(args.agent, "/notify takes on or off"));
            return;
        }
    }
    let state = if settings.on { "on" } else { "off" };
    notices.write(Notice::info(
        args.agent,
        format!("Notifications are {state}."),
    ));
}

#[cfg(test)]
mod tests;
