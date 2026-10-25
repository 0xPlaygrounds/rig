//! What the terminal is told of the turns of the agents nothing spawned,
//! so a terminal or a multiplexer driving rig can tell a turn running from
//! rig waiting for the user. Around each turn go the semantic prompt marks
//! (OSC 133) a shell writes around a command: `C` as it starts, `D;0` as it
//! answers or `D;1` as it fails or stops, then `A` and `B` once no such
//! turn runs, as a shell draws its prompt. Each is followed by a state
//! report (`133;S;activity=busy` or `reading`), and the terminal's title
//! says `rig · working`, `rig · needs you` while a picker or a question is
//! open, `rig · error` after a failed turn, or `rig · idle`.
//!
//! A subagent's turns change none of it. The terminal's own title is
//! pushed on its title stack before the first and popped as the app exits.
//! The `[tui]` setting `mark_nonce_env` names an environment variable
//! holding the nonce of the shell's marks (such as fux's `FUX_MARK_NONCE`),
//! so the marks count for the command that started rig.

use bevy_ecs::prelude::*;
use rig_ecs::agent::{SpawnedBy, TurnOf, TurnOutcome};
use rig_ecs::stage::{ForAgent, Outcome, Settled};
use rig_harness::ask::Ask;
use rig_harness::pick::Picker;
use serde::Deserialize;

use super::terminal::TerminalWrite;

/// Saves the terminal's title on its title stack (xterm's `CSI 22 ; 0 t`).
const PUSH_TITLE: &[u8] = b"\x1b[22;0t";

/// Restores the title saved last (`CSI 23 ; 0 t`).
const POP_TITLE: &[u8] = b"\x1b[23;0t";

/// `[tui]` in `settings.toml`.
#[derive(Resource, Deserialize, Default)]
#[serde(default, deny_unknown_fields)]
pub(crate) struct TuiSettings {
    /// The environment variable holding the nonce each mark carries, if
    /// the terminal's shell integration sets one.
    mark_nonce_env: Option<String>,
}

/// What the terminal was told last.
#[derive(Resource)]
pub(crate) struct Told {
    /// Whether a turn runs: `C` was written and its `D` not yet.
    running: Option<bool>,
    /// The title's state, once one was written.
    title: Option<&'static str>,
    /// Whether the last turn that ended failed or was stopped.
    failed: bool,
    /// The options each mark ends with: `;nonce=…` when
    /// [`TuiSettings::mark_nonce_env`] names a set variable.
    options: String,
}

impl FromWorld for Told {
    fn from_world(world: &mut World) -> Self {
        let nonce = world
            .get_resource::<TuiSettings>()
            .and_then(|settings| settings.mark_nonce_env.as_deref())
            .and_then(|name| std::env::var(name).ok());
        Self {
            running: None,
            title: None,
            failed: false,
            options: nonce
                .map(|nonce| format!(";nonce={nonce}"))
                .unwrap_or_default(),
        }
    }
}

impl Told {
    /// The mark `133;mark`, with the options.
    fn mark(&self, mark: &str) -> String {
        format!("\x1b]133;{mark}{}\x1b\\", self.options)
    }

    /// A turn starts.
    fn start(&mut self) -> String {
        self.running = Some(true);
        self.mark("C") + &self.mark("S;activity=busy")
    }

    /// The turn ended. A terminal moves a cursor not at a line's start to
    /// the next line for `A`, which may scroll the screen, so `A` goes at
    /// the start of the cursor's line and the cursor goes back after.
    fn idle(&mut self) -> String {
        let end = match self.running {
            Some(true) => self.mark(if self.failed { "D;1" } else { "D;0" }),
            Some(false) | None => String::new(),
        };
        self.running = Some(false);
        let prompt = self.mark("A") + &self.mark("B");
        end + "\x1b7\r" + &prompt + "\x1b8" + &self.mark("S;activity=reading")
    }
}

/// A turn of an agent nothing spawned starts: `C`.
pub(crate) fn turn_started(
    turn: On<Add<TurnOf>>,
    turns: Query<&TurnOf>,
    subagents: Query<(), With<SpawnedBy>>,
    mut told: ResMut<Told>,
    mut out: MessageWriter<TerminalWrite>,
) {
    let Ok(&TurnOf(agent)) = turns.get(turn.entity) else {
        return;
    };
    if !subagents.contains(agent) && told.running != Some(true) {
        out.write(TerminalWrite(told.start().into_bytes()));
    }
}

/// Such a turn ended: `D`, then the prompt.
pub(crate) fn turn_ended(
    settled: On<Add<Settled>>,
    turns: Query<(&ForAgent, &TurnOutcome)>,
    subagents: Query<(), With<SpawnedBy>>,
    mut told: ResMut<Told>,
    mut out: MessageWriter<TerminalWrite>,
) {
    let Ok((&ForAgent(agent), outcome)) = turns.get(settled.entity) else {
        return;
    };
    if !subagents.contains(agent) && told.running == Some(true) {
        told.failed = !matches!(outcome, TurnOutcome::Answered(_));
        out.write(TerminalWrite(told.idle().into_bytes()));
    }
}

/// Draws the first prompt, ends a turn gone without an outcome (with its
/// agent: stopped), and keeps the title, once per frame.
pub(crate) fn tell_state(
    turns: Query<&TurnOf, Without<Settled>>,
    subagents: Query<(), With<SpawnedBy>>,
    pickers: Query<(), With<Picker>>,
    asks: Query<(), (With<Ask>, Without<Outcome>)>,
    mut told: ResMut<Told>,
    mut out: MessageWriter<TerminalWrite>,
) {
    let running = turns.iter().any(|turn| !subagents.contains(turn.0));
    if told.running.is_none() || told.running == Some(true) && !running {
        if told.running.is_some() {
            told.failed = true;
        }
        out.write(TerminalWrite(told.idle().into_bytes()));
    }
    let state = if !pickers.is_empty() || !asks.is_empty() {
        "needs you"
    } else if running {
        "working"
    } else if told.failed {
        "error"
    } else {
        "idle"
    };
    if told.title == Some(state) {
        return;
    }
    let mut bytes = match told.title {
        None => PUSH_TITLE.to_vec(),
        Some(_) => Vec::new(),
    };
    bytes.extend_from_slice(format!("\x1b]2;rig · {state}\x1b\\").as_bytes());
    out.write(TerminalWrite(bytes));
    told.title = Some(state);
}

/// Gives the terminal its title back as the app exits.
pub(crate) fn restore_title(mut told: ResMut<Told>, mut out: MessageWriter<TerminalWrite>) {
    if told.title.take().is_some() {
        out.write(TerminalWrite(POP_TITLE.to_vec()));
    }
}

#[cfg(test)]
mod tests;
