//! Plugin corpus: 01. Confirm destructive shell commands (spec
//! `01-confirm-destructive.md`).
//!
//! A Gate system sees each `shell` call when it is about to run. A command
//! that matches no [`DestructivePatterns`], or a pattern the user allowed
//! for the session ([`SessionAllowances`]), runs as the `shell` tool's
//! call. One that matches is held by an [`Ask`] that any front answers:
//! allow it once, allow its pattern for the session, or deny it (a line of
//! text denies it with that reason). With nobody to answer, it is refused
//! at once. The user's own patterns are the `[confirm]` table of
//! `settings.toml`: `patterns = ['\bterraform\s+destroy\b']`.

use bevy_log::warn;
use regex::Regex;
use rig_harness::prelude::*;
use serde::Deserialize;

/// The tool whose calls are checked.
pub const SHELL: &str = "shell";

/// The commands asked about, besides the user's own patterns.
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

/// What the model reads for a matching call with nobody to answer.
pub const HEADLESS: &str = "needs approval; not available non-interactively";

/// Refuses the call; offered first, as the default answer.
pub const DENY: &str = "deny";
/// Runs the call.
pub const ALLOW: &str = "allow";
/// Runs the call, and any later one matching its pattern this session.
pub const SESSION: &str = "session";

/// The `[confirm]` table of `settings.toml`.
#[derive(Resource, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub struct ConfirmSettings {
    /// Regexes added to the built-in ones.
    pub patterns: Vec<String>,
}

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

/// The patterns the user allowed for the rest of the session, kept with
/// it across `/reload` and `/resume`.
#[derive(Resource, Reflect, Clone, Debug, Default)]
#[reflect(Resource, Saved, Default)]
pub struct SessionAllowances(pub Vec<String>);

/// On an [`Ask`] about a `shell` call: the pattern its command matched.
#[derive(Component, Clone, Debug)]
pub struct Matched(pub String);

/// On a `shell` call the user allowed: the gate lets it run.
#[derive(Component)]
#[component(storage = "SparseSet")]
struct Approved;

/// Asks before a `shell` command matching a [`DestructivePatterns`] runs.
#[derive(Default)]
pub struct ConfirmDestructivePlugin;

impl Plugin for ConfirmDestructivePlugin {
    fn build(&self, app: &mut App) {
        app.add_setting::<ConfirmSettings>("confirm");
        if !app.world().contains_resource::<DestructivePatterns>() {
            let own = app.world().resource::<ConfirmSettings>().patterns.clone();
            let built_in = BUILT_IN.iter().copied().map(str::to_owned);
            app.insert_resource(DestructivePatterns::new(built_in.chain(own)));
        }
        app.init_resource::<SessionAllowances>()
            .add_systems(AgentPass, (answered, gate).chain().in_set(Stage::Gate));
    }
}

/// Asks about each destructive `shell` call about to run, holding it, or
/// refuses it with nobody to ask; any other runs.
fn gate(
    calls: Query<(Entity, &ToolRequest, &ForAgent), (Undecided, Without<Approved>)>,
    patterns: Res<DestructivePatterns>,
    allowed: Res<SessionAllowances>,
    answerers: Answerers,
    mut commands: Commands,
) {
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
        if !answerers.can_answer(Some(agent)) {
            let refused = Outcome::refused(format!("`{command}` {HEADLESS}"));
            commands.entity(call).insert_if_new(refused);
            continue;
        }
        let ask = Ask::new(format!("Allow `{command}`?"))
            .body(format!(
                "It matches `{pattern}`. Type a reason to deny it with."
            ))
            .choice(DENY, "Deny")
            .choice(ALLOW, "Allow once")
            .choice(SESSION, "Allow for this session")
            .free_text();
        commands.spawn((ask, Matched(pattern), ForAgent(agent), HoldOf(call)));
    }
}

/// Carries out each answer: the call runs, or is refused with the reason
/// typed, and the question lets go of it.
fn answered(
    asks: Query<(Entity, &Outcome, &HoldOf, &Matched), Added<Outcome>>,
    mut allowed: ResMut<SessionAllowances>,
    mut commands: Commands,
) {
    for (ask, outcome, &HoldOf(call), Matched(pattern)) in &asks {
        match Answer::of(outcome) {
            Some(Answer::Choice(choice)) if choice == ALLOW || choice == SESSION => {
                if choice == SESSION && !allowed.0.contains(pattern) {
                    allowed.0.push(pattern.clone());
                }
                commands.entity(call).insert(Approved);
            }
            Some(Answer::Text(reason)) if !reason.trim().is_empty() => {
                let why = format!("{DENIED}: {}", reason.trim());
                commands.entity(call).insert_if_new(Outcome::refused(why));
            }
            _ => {
                let why = format!("{DENIED}.");
                commands.entity(call).insert_if_new(Outcome::refused(why));
            }
        }
        commands.entity(ask).remove::<HoldOf>();
    }
}

#[cfg(test)]
mod tests;
