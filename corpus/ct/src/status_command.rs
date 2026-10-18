//! CT-19: a status line item from a user script (conversation-over-time
//! fresh held-out spec `19-status-command.md`).
//!
//! `status_command = "~/.config/rig/status.sh"` in `RIG_HOME/settings.toml`
//! (or a `[status_command]` table with `command`, `every_ms` and
//! `timeout_ms`) makes the terminal view's status line show the first line
//! the command prints. The command runs under `sh -c` in the project, as
//! every child process starts ([`Processes`]), with the [`StatusInput`] as
//! JSON on stdin. It runs again when that input changes, at most once every
//! `every_ms` (300 ms), one copy at a time: a change while it runs, or
//! within the pace, runs it once more afterwards. A run past `timeout_ms`
//! (2 s) is killed and the previous line stays, dimmed with a marker; a
//! failing run says its last error line once, as a warning, and keeps the
//! previous line. Control characters are stripped; the status item holds
//! one tone, so ANSI colours become the tone of the first colour named.
//! The terminal view cuts the item to the width it has. An edit of
//! `settings.toml` takes effect at once, and the script is read anew on
//! each run. Without the terminal view the plugin does nothing.

use std::io;
use std::path::PathBuf;

use bevy_tasks::{IoTaskPool, TaskPool};
use rig_basics::sessions::SessionTitle;
use rig_basics::telemetry::usage::SessionUsage;
use rig_harness::prelude::*;
use rig_harness::rig_tools::process::{Launcher, Spawn};
use rig_harness::rig_tools::workspace::Listing;
use rig_tui::{AppStatus, FRONT, Side, StatusItem, StatusSystems, Tone};
use serde::{Deserialize, Serialize};

/// The settings key (or table) the command is read from.
pub const SECTION: &str = "status_command";

/// The item on the status line: after the turn's spending, among the first
/// to go when the line is narrow, cut to 12 columns before it goes.
pub const ITEM: StatusItem = StatusItem::new("status-command", Side::Left, 85, 9).shrinks_to(12);

/// What the item adds to a line kept from before a run that was killed.
pub const STALE_MARKER: &str = "⧗";

/// The most characters of the line the item keeps.
const MAX_CHARS: usize = 256;

/// Shows what the user's status command prints on the status line.
#[derive(Default)]
pub struct StatusCommandPlugin;

impl Plugin for StatusCommandPlugin {
    fn build(&self, _app: &mut App) {}

    /// Once every plugin offered its front, so the run's is known.
    fn finish(&self, app: &mut App) {
        if !app.runs_front(FRONT) {
            return;
        }
        app.add_setting::<StatusCommand>(SECTION)
            .init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .init_resource::<ShownAgent>()
            .init_resource::<AppStatus>()
            .init_resource::<StatusInput>()
            .init_resource::<Runs>()
            .init_resource::<Line>()
            .add_systems(Startup, watch_settings)
            .add_systems(
                PostUpdate,
                (
                    (reload_settings, observe, start).chain(),
                    show.in_set(StatusSystems::Show)
                        .run_if(resource_changed::<Line>),
                ),
            )
            .add_observer(paced)
            .add_observer(ran);
    }
}

/// The `status_command` setting: a command (`status_command = "…"`), or a
/// table with `command` and the timings.
#[derive(Resource, Deserialize, Clone, Debug, PartialEq, Eq)]
#[serde(from = "Written")]
pub struct StatusCommand {
    /// The command, run with `sh -c`; `None` shows nothing.
    pub command: Option<String>,
    /// The shortest time between the starts of two runs.
    pub every: Duration,
    /// How long a run may take before it is killed.
    pub timeout: Duration,
}

impl Default for StatusCommand {
    /// No command, every 300 ms at most, killed after 2 s.
    fn default() -> Self {
        Table::default().into()
    }
}

/// The setting as written.
#[derive(Deserialize)]
#[serde(untagged)]
enum Written {
    /// `status_command = "…"`.
    Command(String),
    /// `[status_command]`.
    Table(Table),
}

/// The `[status_command]` table.
#[derive(Deserialize)]
#[serde(default, deny_unknown_fields)]
struct Table {
    command: Option<String>,
    every_ms: u64,
    timeout_ms: u64,
}

impl Default for Table {
    fn default() -> Self {
        Self {
            command: None,
            every_ms: 300,
            timeout_ms: 2000,
        }
    }
}

impl From<Table> for StatusCommand {
    fn from(table: Table) -> Self {
        Self {
            command: table.command.filter(|command| !command.trim().is_empty()),
            every: Duration::from_millis(table.every_ms),
            timeout: Duration::from_millis(table.timeout_ms),
        }
    }
}

impl From<Written> for StatusCommand {
    fn from(written: Written) -> Self {
        match written {
            Written::Command(command) => Table {
                command: Some(command),
                ..Table::default()
            }
            .into(),
            Written::Table(table) => table.into(),
        }
    }
}

/// What the command reads on stdin, as JSON; a change runs it again.
#[derive(Resource, Serialize, Clone, Debug, Default, PartialEq)]
pub struct StatusInput {
    /// The session's id.
    pub session_id: Option<String>,
    /// The session's name (`/name`), else the start of its first message.
    pub session_name: Option<String>,
    /// The project's directory.
    pub project_dir: PathBuf,
    /// The name of the agent the user looks at.
    pub agent: Option<String>,
    /// Its model, as `vendor/model`.
    pub model: Option<String>,
    /// Whether a turn of it runs.
    pub busy: bool,
    /// `busy` or `idle`.
    pub status: &'static str,
    /// The tokens its next request sends again, when its provider said.
    pub context_tokens: Option<u64>,
    /// Those tokens as a percent of its model's window, when both are known.
    pub context_percent: Option<u64>,
    /// The input tokens of the session's model calls.
    pub session_input_tokens: u64,
    /// The output tokens of the session's model calls.
    pub session_output_tokens: u64,
    /// What the session's model calls cost, in USD, as far as known.
    pub session_cost_usd: f64,
    /// The number of its newest turn, from 1; 0 before its first.
    pub turn: u64,
}

/// Whether a run is wanted, the pace since the last start runs, and a run
/// runs.
#[derive(Resource, Default)]
struct Runs {
    wanted: bool,
    paced: bool,
    running: bool,
}

/// The line shown: its text and tone, whether it is kept from before a run
/// that was killed, and whether the current streak of failures was told.
#[derive(Resource, Default, PartialEq)]
struct Line {
    text: String,
    tone: Tone,
    stale: bool,
    told: bool,
}

/// The entity whose [`WakeAt`] ends the pace.
#[derive(Component)]
struct Pace;

/// The entity of the run that runs.
#[derive(Component)]
struct Run;

/// The watch of `RIG_HOME`, and whether its first listing came.
#[derive(Component)]
struct SettingsWatch {
    armed: bool,
}

/// How a run ended.
enum Ran {
    /// It printed this first line.
    Printed(String),
    /// It ran past its timeout and was killed.
    TimedOut,
    /// It failed: its last error line.
    Failed(String),
}

/// Watches the settings file's directory: a single-file watch loses an
/// editor's save by rename.
fn watch_settings(home: Option<Res<RigHome>>, mut commands: Commands) {
    let Some(home) = home else {
        return;
    };
    let watch = FileWatch::new(home.0.root()).listing(Listing::All);
    commands.spawn((
        Name::new("status command settings"),
        SettingsWatch { armed: false },
        watch,
    ));
}

/// Reads the setting again when `settings.toml` changed, and runs the new
/// command.
fn reload_settings(
    mut changes: MessageReader<FileChanges>,
    mut watches: Query<&mut SettingsWatch>,
    home: Option<Res<RigHome>>,
    mut setting: ResMut<StatusCommand>,
    mut runs: ResMut<Runs>,
    mut notices: MessageWriter<Notice>,
) {
    let Some(home) = home else {
        return;
    };
    let file = home.0.settings();
    for change in changes.read() {
        let Ok(mut watch) = watches.get_mut(change.watch) else {
            continue;
        };
        if !std::mem::replace(&mut watch.armed, true) {
            continue;
        }
        if !change
            .paths
            .iter()
            .any(|path| path.file_name() == file.file_name())
        {
            continue;
        }
        match Settings::read(&file).get::<StatusCommand>(SECTION) {
            Ok(read) => {
                if setting.set_if_neq(read) {
                    runs.wanted = true;
                }
            }
            Err(why) => {
                notices.write(Notice::warn(None, why));
            }
        }
    }
}

/// The agents as [`observe`] reads them.
type Observed<'a> = (
    &'a Name,
    &'a Conversation,
    &'a LastUsage,
    Option<&'a ModelChoice>,
    Option<&'a Connection>,
    Has<ActiveTurn>,
);

/// Brings the [`StatusInput`] up to date, and wants a run when it changed.
fn observe(
    (shown, root, paths, title, spending): (
        Res<ShownAgent>,
        Res<ProjectRoot>,
        Option<Res<SessionPaths>>,
        Option<Res<SessionTitle>>,
        Option<Res<SessionUsage>>,
    ),
    agents: Query<Observed, With<Agent>>,
    mut input: ResMut<StatusInput>,
    mut runs: ResMut<Runs>,
) {
    let agent = shown.0.and_then(|agent| agents.get(agent).ok());
    let total = spending
        .map(|spending| spending.0.total)
        .unwrap_or_default();
    let busy = agent.is_some_and(|(.., busy)| busy);
    let context_tokens = agent.and_then(|(_, _, last, ..)| last.context());
    let window = agent
        .and_then(|(.., connection, _)| connection)
        .and_then(|connection| connection.spec.context_window);
    let now = StatusInput {
        session_id: paths.and_then(|paths| paths.id()).map(|id| id.to_string()),
        session_name: title.and_then(|title| {
            title
                .name
                .clone()
                .or_else(|| (!title.typed.is_empty()).then(|| title.typed.clone()))
        }),
        project_dir: root.0.clone(),
        agent: agent.map(|(name, ..)| name.to_string()),
        model: agent.and_then(|(_, _, _, choice, connection, _)| {
            choice
                .map(|choice| choice.0.clone())
                .or_else(|| connection.map(|connection| connection.spec.reference()))
        }),
        busy,
        status: if busy { "busy" } else { "idle" },
        context_tokens,
        context_percent: context_tokens
            .zip(window)
            .and_then(|(tokens, window)| (tokens * 100).checked_div(u64::from(window))),
        session_input_tokens: total.tokens.input_tokens.unwrap_or_default(),
        session_output_tokens: total.tokens.output_tokens.unwrap_or_default(),
        session_cost_usd: total.cost,
        turn: agent
            .and_then(|(_, conversation, ..)| conversation.turns().next_back())
            .map_or(0, |(_, record)| record.index.saturating_add(1)),
    };
    if input.set_if_neq(now) {
        runs.wanted = true;
    }
}

/// Starts a wanted run unless one runs or the pace since the last start
/// has not ended.
fn start(
    mut runs: ResMut<Runs>,
    (input, setting): (Res<StatusInput>, Res<StatusCommand>),
    (processes, root, wake): (Res<Processes>, Res<ProjectRoot>, Res<Wake>),
    mut commands: Commands,
) {
    if !runs.wanted || runs.paced || runs.running {
        return;
    }
    runs.wanted = false;
    let Some(command) = setting.command.clone() else {
        return;
    };
    let Ok(json) = serde_json::to_vec(&*input) else {
        return;
    };
    runs.paced = true;
    runs.running = true;
    commands.spawn((
        Name::new("status command pace"),
        Pace,
        WakeAt::after(setting.every),
    ));
    let (launcher, timeout) = (processes.launcher().in_dir(&root.0), setting.timeout);
    let work = async move {
        blocking(move || Ok(run(&launcher, &command, json, timeout)))
            .await
            .unwrap_or_else(|failure| Ran::Failed(failure.to_string()))
    };
    let pool = IoTaskPool::get_or_init(TaskPool::default);
    commands.spawn((
        Name::new("status command"),
        Run,
        Running::spawn(pool, &wake, work),
    ));
}

/// Runs `command` with `input` on stdin, killed after `timeout`.
fn run(launcher: &Launcher, command: &str, input: Vec<u8>, timeout: Duration) -> Ran {
    let how = Spawn {
        stdin: Some(input),
        merge_stderr: false,
    };
    let process = match launcher.spawn("sh", ["-c", command], how) {
        Ok(process) => process,
        Err(failure) => return Ran::Failed(format!("could not start: {failure}")),
    };
    match process.output(Some(timeout)) {
        Err(failure) if failure.kind() == io::ErrorKind::TimedOut => Ran::TimedOut,
        Err(failure) => Ran::Failed(failure.to_string()),
        Ok(output) if output.status.success() => {
            let stdout = String::from_utf8_lossy(&output.stdout);
            Ran::Printed(stdout.lines().next().unwrap_or_default().to_owned())
        }
        Ok(output) => {
            let stderr = String::from_utf8_lossy(&output.stderr);
            let last = stderr.lines().rev().find(|line| !line.trim().is_empty());
            Ran::Failed(last.map_or_else(|| output.status.to_string(), str::to_owned))
        }
    }
}

/// The pace ended: a wanted run may start.
fn paced(
    due: On<Add<Due>>,
    paces: Query<(), With<Pace>>,
    mut runs: ResMut<Runs>,
    mut commands: Commands,
) {
    if paces.contains(due.entity) {
        runs.paced = false;
        commands.entity(due.entity).despawn();
    }
}

/// A run ended: the line shows what it printed, or keeps the last one.
fn ran(
    done: On<Add<Done<Ran>>>,
    mut runs_done: Query<&mut Done<Ran>, With<Run>>,
    (mut line, mut runs): (ResMut<Line>, ResMut<Runs>),
    mut notices: MessageWriter<Notice>,
    mut commands: Commands,
) {
    let Ok(mut finished) = runs_done.get_mut(done.entity) else {
        return;
    };
    match std::mem::replace(&mut finished.0, Ran::TimedOut) {
        Ran::Printed(printed) => {
            let (text, tone) = clean(&printed);
            line.set_if_neq(Line {
                text,
                tone,
                stale: false,
                told: false,
            });
        }
        Ran::TimedOut => line.stale = true,
        Ran::Failed(why) if !line.told => {
            line.told = true;
            notices.write(Notice::warn(
                None,
                format!("The status command failed: {why}"),
            ));
        }
        Ran::Failed(_) => {}
    }
    runs.running = false;
    commands.entity(done.entity).despawn();
}

fn show(line: Res<Line>, mut status: ResMut<AppStatus>) {
    status.0.show(match (line.stale, line.text.is_empty()) {
        (_, true) => ITEM,
        (true, false) => ITEM.says(format!("{} {STALE_MARKER}", line.text), Tone::Dim),
        (false, false) => ITEM.says(line.text.as_str(), line.tone),
    });
}

/// `line` without control characters and escape sequences, at most
/// [`MAX_CHARS`], and the tone of the first SGR sequence that names one.
fn clean(line: &str) -> (String, Tone) {
    let (mut text, mut tone) = (String::new(), None);
    let mut chars = line.chars().peekable();
    while let Some(character) = chars.next() {
        match character {
            '\u{1b}' => match chars.next() {
                // CSI: parameters up to a final byte.
                Some('[') => {
                    let mut params = String::new();
                    for next in chars.by_ref() {
                        if ('\u{40}'..='\u{7e}').contains(&next) {
                            if next == 'm' && tone.is_none() {
                                tone = params.split(';').find_map(sgr_tone);
                            }
                            break;
                        }
                        params.push(next);
                    }
                }
                // OSC: up to BEL or ST.
                Some(']') => {
                    while let Some(next) = chars.next() {
                        if next == '\u{7}'
                            || (next == '\u{1b}' && chars.next_if_eq(&'\\').is_some())
                        {
                            break;
                        }
                    }
                }
                _ => {}
            },
            '\t' => text.push(' '),
            other if other.is_control() => {}
            other => text.push(other),
        }
    }
    let text = text.trim().chars().take(MAX_CHARS).collect();
    (text, tone.unwrap_or(Tone::Plain))
}

/// The tone an SGR parameter names, if any.
fn sgr_tone(param: &str) -> Option<Tone> {
    Some(match param {
        "1" => Tone::Bold,
        "2" | "90" => Tone::Dim,
        "31" | "91" => Tone::Red,
        "32" | "92" => Tone::Green,
        "33" | "93" => Tone::Yellow,
        "34" | "36" | "94" | "96" => Tone::Cyan,
        "35" | "95" => Tone::Magenta,
        _ => return None,
    })
}

#[cfg(test)]
mod tests;
