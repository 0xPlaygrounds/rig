//! Inline shell expansion (held-out corpus spec `H04-inline-shell.md`).
//!
//! In a message the user typed, each `!{cmd}` runs in the project directory
//! (the agent's [`WorkDir`] when it has one) and is replaced by a fenced
//! block of its output. A Prepare system finds the commands in a new
//! message and holds it with one entity per command, each running its
//! command off the main thread (rig-tools' `Shell`, raced against the
//! timeout). Once every command finished, a new message with
//! the expanded text supersedes the held one, so it goes through Prepare
//! again (attachments, redaction) and the conversation keeps the expanded
//! text. A held message is on its way to its agent, so a turn it steers
//! does not end without it, and if the turn ends anyway it starts the next.

use std::collections::HashMap;
use std::time::Duration;

use bevy_tasks::futures_lite::FutureExt as _;
use bevy_tasks::{IoTaskPool, TaskPool};
use rig_harness::prelude::*;
use rig_harness::rig_core::tool::{ExitStatus, ToolExecutionError};
use rig_tools::{Shell, ShellArgs, ShellRun};
use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};
use serde::Deserialize;

#[cfg(test)]
mod tests;

/// What each command's output is cut to, in bytes.
pub const OUTPUT_LIMIT: usize = 4 * 1024;

/// The status item: after the agent's status.
const EXPANDING: StatusItem = StatusItem::new("ho-inline-shell", Side::Left, 52, 10);

/// The `[inline_shell]` table: `enabled = false` turns expansion off;
/// `timeout_secs` is how long each command may run.
#[derive(Resource, Deserialize, Clone, Debug)]
#[serde(default, deny_unknown_fields)]
pub struct InlineShell {
    /// Whether `!{cmd}` is expanded.
    pub enabled: bool,
    /// Seconds each command may run before it reads "(timed out)".
    pub timeout_secs: f64,
}

impl Default for InlineShell {
    fn default() -> Self {
        Self {
            enabled: true,
            timeout_secs: 10.0,
        }
    }
}

impl InlineShell {
    /// The timeout, as a duration; a nonsensical one is the default.
    fn timeout(&self) -> Duration {
        Duration::try_from_secs_f64(self.timeout_secs).unwrap_or(Duration::from_secs(10))
    }
}

/// The Prepare system that finds a message's commands and holds it while
/// they run, for others to order against. It runs before the files a
/// message names are attached, so they are attached to the text the model
/// reads.
#[derive(SystemSet, Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct InlineShellSystems;

#[derive(Default)]
pub struct InlineShellPlugin;

impl Plugin for InlineShellPlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<InlineShell>("inline_shell")
            .init_resource::<ProjectRoot>()
            .init_resource::<Processes>()
            .configure_sets(AgentPass, InlineShellSystems.in_set(PrepareSystems::Input))
            .add_systems(
                AgentPass,
                expand
                    .in_set(InlineShellSystems)
                    .run_if(|settings: Res<InlineShell>| settings.enabled),
            )
            .add_systems(PostUpdate, show_expanding.in_set(StatusSystems::Show))
            .add_observer(finished);
    }
}

/// A piece of a message: its own text, or a command whose output goes in
/// its place.
#[derive(Clone, Debug, PartialEq, Eq)]
enum Piece {
    Text(String),
    Shell(String),
}

/// Splits `text` into its pieces: `!{cmd}` is a command (braces inside it
/// nest), `\!{` is a literal `!{`, and an unclosed or empty `!{` is text.
fn pieces(text: &str) -> Vec<Piece> {
    let mut pieces = Vec::new();
    let mut plain = String::new();
    let mut rest = text;
    while let Some(at) = rest.find("!{") {
        let (before, after) = rest.split_at(at);
        let after = after.get(2..).unwrap_or_default();
        if let Some(escaped) = before.strip_suffix('\\') {
            plain.push_str(escaped);
            plain.push_str("!{");
            rest = after;
            continue;
        }
        plain.push_str(before);
        let mut depth = 0usize;
        let end = after.char_indices().find_map(|(index, c)| match c {
            '{' => {
                depth += 1;
                None
            }
            '}' if depth == 0 => Some(index),
            '}' => {
                depth -= 1;
                None
            }
            _ => None,
        });
        match end.map(|end| after.split_at(end)) {
            Some((command, closed)) if !command.trim().is_empty() => {
                pieces.push(Piece::Text(std::mem::take(&mut plain)));
                pieces.push(Piece::Shell(command.trim().to_owned()));
                rest = closed.get(1..).unwrap_or_default();
            }
            _ => {
                plain.push_str("!{");
                rest = after;
            }
        }
    }
    plain.push_str(rest);
    pieces.push(Piece::Text(plain));
    pieces
}

/// On a message being expanded: its pieces, until its commands finished.
#[derive(Component)]
struct Expansion(Vec<Piece>);

/// On the message that replaced an expanded one: not expanded again.
#[derive(Component)]
struct Expanded;

/// A command of a message, running: it holds the message ([`HoldOf`]).
#[derive(Component)]
struct Job {
    /// The message it is in.
    message: Entity,
    /// The agent the message goes to.
    agent: Entity,
    /// Its place among the message's pieces.
    slot: usize,
}

/// A finished command's block: what goes in its place.
struct Block(String);

/// Finds the commands in each message the user typed as it is admitted,
/// and holds the message with one running job per command. A message with
/// only escapes gets them resolved at once.
fn expand(
    mut messages: Query<
        (Entity, &mut Incoming, &ForAgent),
        (Added<Admitted>, IncomingMessages, Without<Expanded>),
    >,
    dirs: Query<&WorkDir>,
    settings: Res<InlineShell>,
    processes: Res<Processes>,
    root: Res<ProjectRoot>,
    wake: Res<Wake>,
    mut commands: Commands,
) {
    for (message, mut incoming, &ForAgent(agent)) in &mut messages {
        let typed = incoming.author.kind == AuthorKind::User;
        if !typed || incoming.literal || incoming.command().is_some() {
            continue;
        }
        let pieces = pieces(incoming.text());
        let shells: Vec<(usize, String)> = pieces
            .iter()
            .enumerate()
            .filter_map(|(slot, piece)| match piece {
                Piece::Shell(command) => Some((slot, command.clone())),
                Piece::Text(_) => None,
            })
            .collect();
        if shells.is_empty() {
            let text = compose(&pieces, &HashMap::new());
            if text != incoming.text() {
                *incoming.text_mut() = text;
            }
            continue;
        }
        // The agent's own directory, else where every process starts, else
        // the project.
        let launcher = processes.launcher();
        let launcher = match (dirs.get(agent), &launcher.cwd) {
            (Ok(WorkDir(dir)), _) => launcher.in_dir(dir),
            (Err(_), Some(_)) => launcher,
            (Err(_), None) => launcher.in_dir(&root.0),
        };
        let shell = Shell {
            launcher,
            spill: None,
            confine: false,
            cut: None,
            ..Shell::default()
        };
        let pool = IoTaskPool::get_or_init(TaskPool::default);
        for (slot, command) in shells {
            let work = run(shell.clone(), command, settings.timeout());
            commands.spawn((
                Name::new("inline shell"),
                Job {
                    message,
                    agent,
                    slot,
                },
                HoldOf(message),
                Running::spawn(pool, &wake, work),
            ));
        }
        commands.entity(message).insert(Expansion(pieces));
    }
}

/// Runs `command` and makes its block, or "(timed out)" after `timeout`;
/// dropping the run kills the command.
async fn run(shell: Shell, command: String, timeout: Duration) -> Block {
    // The shell's own limit, in whole seconds, only backs ours up.
    let backup = timeout.as_secs().saturating_add(2);
    let args = ShellArgs::new(command).timeout_secs(backup);
    let ran = async { Some(shell.run(args).await) };
    let timed_out = async {
        futures_timer::Delay::new(timeout).await;
        None
    };
    Block(fenced(&match ran.or(timed_out).await {
        None => "(timed out)".to_owned(),
        Some(ran) => output(ran),
    }))
}

/// What a finished command printed, cut to [`OUTPUT_LIMIT`], and how it
/// ended unless it exited with 0.
fn output(ran: Result<ShellRun, ToolExecutionError>) -> String {
    let ShellRun { output, status } = match ran {
        Ok(ran) => ran,
        Err(error) => return format!("(could not run: {error})"),
    };
    // The run's output ends in a line saying how it ended; `status` says
    // that as data.
    let printed = output.rsplit_once('\n').map_or("", |(printed, _)| printed);
    let mut text = cut(printed);
    let ended = match status {
        ExitStatus::Exited(0) => None,
        ExitStatus::Exited(code) => Some(format!("(exit code {code})")),
        ExitStatus::Signaled => Some("(killed by a signal)".to_owned()),
        ExitStatus::TimedOut => Some("(timed out)".to_owned()),
        ExitStatus::Killed => Some("(killed)".to_owned()),
    };
    if let Some(ended) = ended {
        if !text.is_empty() {
            text.push('\n');
        }
        text.push_str(&ended);
    }
    text
}

/// `text` cut to [`OUTPUT_LIMIT`] bytes, saying how many were cut.
fn cut(text: &str) -> String {
    if text.len() <= OUTPUT_LIMIT {
        return text.to_owned();
    }
    let end = (0..=OUTPUT_LIMIT)
        .rev()
        .find(|&end| text.is_char_boundary(end))
        .unwrap_or_default();
    let kept = text.get(..end).unwrap_or_default();
    format!("{kept}\n({} bytes cut)", text.len() - end)
}

/// `text` in a fenced block, its fence longer than any run of backticks in
/// it.
fn fenced(text: &str) -> String {
    let longest = text
        .split(|c| c != '`')
        .map(str::len)
        .max()
        .unwrap_or_default();
    let fence = "`".repeat(longest.max(2) + 1);
    format!("\n{fence}\n{text}\n{fence}\n")
}

/// The message's text with each command's block in its place.
fn compose(pieces: &[Piece], blocks: &HashMap<usize, String>) -> String {
    pieces
        .iter()
        .enumerate()
        .map(|(slot, piece)| match piece {
            Piece::Text(text) => text.as_str(),
            Piece::Shell(_) => blocks.get(&slot).map_or("", String::as_str),
        })
        .collect()
}

/// Once every command of a message finished, sends the expanded message
/// in its place: a new message superseding it, so the held one never
/// reaches the agent, and the jobs that held it go.
fn finished(
    done: On<Add<Done<Block>>>,
    jobs: Query<(&Job, Option<&Done<Block>>)>,
    mut messages: Query<(&Incoming, &Effect, &Holds, &mut Expansion)>,
    mut commands: Commands,
) {
    let Ok((job, _)) = jobs.get(done.entity) else {
        return;
    };
    let message = job.message;
    let Ok((incoming, effect, holds, mut expansion)) = messages.get_mut(message) else {
        return;
    };
    let mine: Vec<_> = holds
        .iter()
        .filter_map(|hold| Some((hold, jobs.get(hold).ok()?)))
        .collect();
    if mine.iter().any(|(_, (_, done))| done.is_none()) {
        return;
    }
    let blocks: HashMap<usize, String> = mine
        .iter()
        .filter_map(|(_, (job, done))| Some((job.slot, done.as_ref()?.0.0.clone())))
        .collect();
    let pieces = std::mem::take(&mut expansion.0);
    if pieces.is_empty() {
        return;
    }
    let text = compose(&pieces, &blocks);
    let expanded = Incoming::new(text, incoming.mode).with_author(incoming.author.clone());
    commands
        .spawn((
            expanded,
            Expanded,
            ForAgent(job.agent),
            Effect::with_parent_id(effect.parent),
        ))
        .add_related::<SupersededBy>(&[message]);
    for (hold, _) in mine {
        commands.entity(hold).despawn();
    }
}

/// "expanding 2 commands…" on each agent while commands of its messages
/// run, whenever that changed.
fn show_expanding(
    jobs: Query<&Job, Without<Done<Block>>>,
    changed: Query<(), Or<(Added<Job>, Added<Done<Block>>)>>,
    mut gone: RemovedComponents<Job>,
    mut agents: Query<(Entity, &mut StatusItems), With<Agent>>,
) {
    if changed.is_empty() && gone.read().count() == 0 {
        return;
    }
    let mut running: HashMap<Entity, usize> = HashMap::new();
    for job in &jobs {
        *running.entry(job.agent).or_default() += 1;
    }
    for (agent, mut items) in &mut agents {
        let item = match running.get(&agent) {
            Some(1) => EXPANDING.says("expanding 1 command…", Tone::Dim),
            Some(&count) => EXPANDING.says(format!("expanding {count} commands…"), Tone::Dim),
            None => EXPANDING,
        };
        items.show(item);
    }
}
