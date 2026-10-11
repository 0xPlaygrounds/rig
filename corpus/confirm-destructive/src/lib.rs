//! Plugin corpus: 01. Confirm destructive shell commands (spec
//! `01-confirm-destructive.md`).
//!
//! A Gate system sees each `shell` call when it is about to run. A command
//! that matches no [`DestructivePatterns`], or a pattern the user allowed
//! for the session ([`SessionAllowances`]), runs as the `shell` tool's
//! call. One that matches is held by an [`AwaitingApproval`] entity until
//! the user answers in a picker (`/confirm`), and is refused at once when
//! nobody sits at the terminal.

use std::fs;
use std::io::ErrorKind;

use bevy_log::warn;
use regex::Regex;
use rig_harness::prelude::*;
use rig_tui::{PickItem, PickRequest};
use serde::Deserialize;

/// The tool whose calls are checked.
pub const SHELL: &str = "shell";

/// The commands asked about unless `RIG_HOME/confirm.toml` adds more.
pub const BUILT_IN: &[&str] = &[
    r"\brm\s+(?:-\S*\s+)*-[a-zA-Z]*[rRf]",
    r"\brm\b[^;&|]*\s--(?:recursive|force)\b",
    r"\bgit\s+push\b[^;&|]*\s(?:--force\S*|-f\b)",
    r"\bgit\s+reset\b[^;&|]*\s--hard\b",
    r"\bgit\s+clean\b[^;&|]*\s-[a-zA-Z]*f",
    r"\bdd\s",
    r"\bmkfs\b",
    r"(?i)\bdrop\s+(?:table|database|schema)\b",
];

/// What the model reads for a call the user denied.
pub const DENIED: &str = "The user denied this command";

/// What the model reads for a matching call with nobody at the terminal.
pub const HEADLESS: &str = "needs approval; not available non-interactively";

/// The patterns a `shell` command is checked against.
#[derive(Resource, Clone, Debug)]
pub struct DestructivePatterns(pub Vec<Regex>);

impl DestructivePatterns {
    /// The patterns of `sources`; one that is not a valid regex is left out
    /// with a warning.
    pub fn new(sources: impl IntoIterator<Item = String>) -> Self {
        let compile = |source: String| {
            Regex::new(&source)
                .inspect_err(|error| warn!("confirm: pattern `{source}` left out: {error}"))
                .ok()
        };
        Self(sources.into_iter().filter_map(compile).collect())
    }

    /// The built-in patterns and those of `RIG_HOME/confirm.toml`:
    /// `patterns = ['\bterraform\s+destroy\b']`.
    fn from_home() -> Self {
        let path = Home::from_env().root().join("confirm.toml");
        let extra = match fs::read_to_string(&path) {
            Ok(text) => toml::from_str::<ConfirmFile>(&text)
                .inspect_err(|error| warn!("confirm: {} not read: {error}", path.display()))
                .map(|file| file.patterns)
                .unwrap_or_default(),
            Err(error) if error.kind() == ErrorKind::NotFound => Vec::new(),
            Err(error) => {
                warn!("confirm: {} not read: {error}", path.display());
                Vec::new()
            }
        };
        Self::new(BUILT_IN.iter().copied().map(str::to_owned).chain(extra))
    }

    /// The first pattern `command` matches.
    pub fn matching(&self, command: &str) -> Option<&Regex> {
        self.0.iter().find(|pattern| pattern.is_match(command))
    }
}

impl Default for DestructivePatterns {
    fn default() -> Self {
        Self::new(BUILT_IN.iter().copied().map(str::to_owned))
    }
}

/// `RIG_HOME/confirm.toml`.
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct ConfirmFile {
    /// Regexes added to the built-in ones.
    #[serde(default)]
    patterns: Vec<String>,
}

/// The patterns the user allowed for the rest of the session, kept with
/// it across `/reload` and `/resume`.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved, Default)]
pub struct SessionAllowances(pub Vec<String>);

/// A hold on a `shell` call waiting for the user's answer: an entity
/// [`HoldOf`] the call. Interrupting or stopping the turn despawns this
/// hold, and its question with it.
#[derive(Component, Clone, Debug)]
pub struct AwaitingApproval {
    /// The calling agent.
    pub agent: Entity,
    /// The command.
    pub command: String,
    /// The pattern it matched.
    pub pattern: String,
    /// Whether the user chose "Deny with reason": the next message they
    /// type to the agent is the reason.
    pub reason_asked: bool,
}

impl AwaitingApproval {
    /// The picker asking about the hold `hold`, with "Deny" selected.
    fn picker(&self, hold: Entity) -> PickRequest {
        let key = hold.to_bits();
        let item = |label: &str, answer: &str| PickItem {
            label: label.to_owned(),
            command: format!("confirm {key} {answer}"),
        };
        PickRequest {
            agent: self.agent,
            title: format!("Allow `{}`?", self.command),
            items: vec![
                item("Allow once", "allow"),
                item("Allow for this session", "session"),
                item("Deny", "deny"),
                item("Deny with reason", "reason"),
            ],
            selected: 2,
        }
    }
}

/// On a `shell` call the user allowed: the gate lets it run.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Approved;

/// Asks before a `shell` command matching a [`DestructivePatterns`] runs.
#[derive(Default)]
pub struct ConfirmDestructivePlugin;

impl Plugin for ConfirmDestructivePlugin {
    fn build(&self, app: &mut App) {
        if !app.world().contains_resource::<DestructivePatterns>() {
            app.insert_resource(DestructivePatterns::from_home());
        }
        app.add_message::<PickRequest>()
            .init_resource::<SessionAllowances>()
            .add_systems(AgentPass, (gate, take_reason).in_set(Stage::Gate))
            .add_command(
                "confirm",
                "Answer a shell command waiting for approval; alone, asks again",
                confirm,
            )
            .add_observer(ask_next);
    }
}

/// Holds each destructive `shell` call about to run for the user, or
/// refuses it with nobody to ask; any other runs.
fn gate(
    calls: Query<(Entity, &ToolRequest, &ForAgent), (Undecided, Without<Approved>)>,
    patterns: Res<DestructivePatterns>,
    allowed: Res<SessionAllowances>,
    invoked: Option<Res<Invoked>>,
    waiting: Query<(), With<AwaitingApproval>>,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
) {
    let interactive = invoked.is_some_and(|invoked| invoked.interactive());
    // One question at a time: the next is asked once this one is answered.
    let mut asking = !waiting.is_empty();
    for (call, ToolRequest { call: request }, &ForAgent(agent)) in &calls {
        if request.function.name.as_str() != SHELL {
            continue;
        }
        let command = request
            .function
            .arguments
            .get("command")
            .and_then(serde_json::Value::as_str)
            .unwrap_or_default();
        let pattern = patterns
            .matching(command)
            .map(|pattern| pattern.as_str().to_owned())
            .filter(|pattern| !allowed.0.contains(pattern));
        let Some(pattern) = pattern else {
            continue;
        };
        if !interactive {
            let refused = Outcome::refused(format!("`{command}` {HEADLESS}"));
            commands.entity(call).insert_if_new(refused);
            continue;
        }
        let awaiting = AwaitingApproval {
            agent,
            command: command.to_owned(),
            pattern,
            reason_asked: false,
        };
        let hold = commands.spawn((awaiting.clone(), HoldOf(call))).id();
        if !asking {
            asking = true;
            picks.write(awaiting.picker(hold));
        }
    }
}

/// Refuses the held call of `hold` as denied, for `reason` if given, and
/// releases it.
fn deny(commands: &mut Commands, (hold, call): (Entity, Entity), reason: Option<&str>) {
    let why = match reason.map(str::trim).filter(|reason| !reason.is_empty()) {
        Some(reason) => format!("{DENIED}: {reason}"),
        None => format!("{DENIED}."),
    };
    commands.entity(call).insert_if_new(Outcome::refused(why));
    commands.entity(hold).despawn();
}

/// `/confirm <hold> allow|session|deny [reason]|reason`, as the picker
/// runs it; `/confirm` alone asks about the first waiting call again, such
/// as after the picker was closed.
fn confirm(
    In(args): In<CommandArgs>,
    mut waiting: Query<(Entity, &HoldOf, &mut AwaitingApproval)>,
    mut allowed: ResMut<SessionAllowances>,
    mut commands: Commands,
    mut picks: MessageWriter<PickRequest>,
    mut notices: MessageWriter<Notice>,
) {
    let mut words = args.args.splitn(3, char::is_whitespace);
    let (key, answer, reason) = (words.next(), words.next(), words.next());
    let hold = key
        .filter(|key| !key.is_empty())
        .and_then(|key| key.parse().ok())
        .and_then(Entity::try_from_bits);
    let Some(hold) = hold else {
        match waiting.iter().next() {
            Some((hold, _, awaiting)) => {
                picks.write(awaiting.picker(hold));
            }
            None => {
                notices.write(Notice::info(args.agent, "No shell command waits."));
            }
        }
        return;
    };
    let Ok((hold, &HoldOf(call), mut awaiting)) = waiting.get_mut(hold) else {
        notices.write(Notice::error(
            args.agent,
            "That shell command no longer waits.",
        ));
        return;
    };
    match answer.unwrap_or_default() {
        "allow" | "session" => {
            if answer == Some("session") && !allowed.0.contains(&awaiting.pattern) {
                allowed.0.push(awaiting.pattern.clone());
            }
            commands.entity(call).insert(Approved);
            commands.entity(hold).despawn();
        }
        "deny" => deny(&mut commands, (hold, call), reason),
        "reason" => {
            awaiting.reason_asked = true;
            notices.write(Notice::info(
                awaiting.agent,
                format!(
                    "Type why `{}` is denied: your next message is the reason.",
                    awaiting.command
                ),
            ));
        }
        other => {
            notices.write(Notice::error(
                args.agent,
                format!("/confirm: `{other}` is not allow, session, deny or reason."),
            ));
        }
    }
}

/// The user's next message after "Deny with reason" is the reason: this
/// Gate system takes it over, so it denies the call and goes nowhere else.
fn take_reason(
    typed: Query<(Entity, &Incoming, &ForAgent), Undecided>,
    waiting: Query<(Entity, &HoldOf, &AwaitingApproval)>,
    mut commands: Commands,
) {
    let mut denied = Vec::new();
    for (message, typed, &ForAgent(agent)) in &typed {
        if typed.origin.kind != OriginKind::User || typed.command().is_some() {
            continue;
        }
        let asked = waiting.iter().find(|(hold, _, awaiting)| {
            awaiting.agent == agent && awaiting.reason_asked && !denied.contains(hold)
        });
        if let Some((hold, &HoldOf(call), _)) = asked {
            denied.push(hold);
            let taken = Outcome::custom("the reason for a denial");
            commands.entity(message).insert_if_new(taken);
            deny(&mut commands, (hold, call), Some(typed.text.trim()));
        }
    }
}

/// Once a question is answered, or its call is gone, asks the next one.
fn ask_next(
    answered: On<Remove<AwaitingApproval>>,
    waiting: Query<(Entity, &AwaitingApproval)>,
    mut picks: MessageWriter<PickRequest>,
) {
    let next = waiting
        .iter()
        .find(|(hold, awaiting)| *hold != answered.entity && !awaiting.reason_asked);
    if let Some((hold, awaiting)) = next {
        picks.write(awaiting.picker(hold));
    }
}

#[cfg(test)]
mod tests;
