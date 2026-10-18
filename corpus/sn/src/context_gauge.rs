//! SN-19: a context window gauge. Each agent's status line item reads
//! `ctx 43% · 86k/200k`: the tokens its latest model reply read and wrote
//! (the kernel's [`LastUsage`]) against the window of its current model
//! (its [`Connection`]'s catalog spec), dimmed below 70%, yellow to 90%
//! and red above. Without a known window it reads `ctx 86k`, before the
//! agent's first reply `ctx —`. The terminal view shows the focused
//! agent's items, so focusing a subagent shows its own gauge.
//!
//! The first time an agent goes above 90% in a session, a notice suggests
//! condensing or handing off; that it was given is saved with the agent.
//!
//! The gauge replaces `rig_basics::telemetry`'s item named `context` (right, 40) in
//! `StatusSystems::Edit`, so the line shows one context item, not two.

use rig_core::completion::{ContextUse, tokens_label};
use rig_harness::prelude::*;
use rig_tui::{Side, StatusItem, StatusItems, StatusSystems, Tone};

/// The default meter's context item, which the gauge replaces by name: it
/// stays longer than the cache and the cost when the line is too narrow.
pub const GAUGE: StatusItem = StatusItem::new("context", Side::Right, 40, 5);

/// Shows each agent's context gauge and warns once when it is nearly
/// full. Needs the terminal view's plugin (`rig_tui::TuiPlugin`), which
/// gives every agent its [`StatusItems`].
#[derive(Default)]
pub struct ContextGaugePlugin;

impl Plugin for ContextGaugePlugin {
    fn build(&self, app: &mut App) {
        app.register_required_components::<Agent, NearlyFullWarned>()
            .add_systems(
                PostUpdate,
                (
                    warn_once,
                    // After `rig_basics::telemetry` shows its meter.
                    show.in_set(StatusSystems::Edit),
                ),
            );
    }
}

/// Whether the agent was told its context is nearly full, once a session.
/// Saved with the session.
#[derive(Component, Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
#[reflect(Component, Saved, Default)]
pub struct NearlyFullWarned(pub bool);

/// How full an agent's context is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Fill {
    /// No reply reported its tokens yet.
    Unknown,
    /// Below 70% of the window, or the window is not known.
    Normal,
    /// 70% to 90%.
    Warning,
    /// Above 90%.
    Alert,
}

impl Fill {
    /// How full `context` is.
    pub fn of(context: Option<ContextUse>) -> Self {
        let Some(context) = context else {
            return Self::Unknown;
        };
        let Some(window) = context.window.filter(|window| *window > 0) else {
            return Self::Normal;
        };
        let used = u128::from(context.tokens) * 100;
        let window = u128::from(window);
        if used > window * 90 {
            Self::Alert
        } else if used >= window * 70 {
            Self::Warning
        } else {
            Self::Normal
        }
    }

    fn tone(self) -> Tone {
        match self {
            Self::Unknown | Self::Normal => Tone::Dim,
            Self::Warning => Tone::Yellow,
            Self::Alert => Tone::Red,
        }
    }
}

/// The context an agent sends again with its next request: its latest
/// reply's tokens (`last`), against the window of its model
/// (`connection`).
pub fn context_of(last: &LastUsage, connection: Option<&Connection>) -> Option<ContextUse> {
    let window = connection.and_then(|connection| connection.spec.context_window);
    last.context().map(|tokens| ContextUse { tokens, window })
}

/// The gauge's text: `ctx 75% · 150k/200k`, `ctx 150k`, or `ctx —`.
pub fn label(context: Option<ContextUse>) -> String {
    let Some(context) = context else {
        return "ctx —".to_owned();
    };
    let tokens = tokens_label(context.tokens);
    match (context.window, context.percent()) {
        (Some(window), Some(percent)) => {
            let window = tokens_label(u64::from(window));
            format!("ctx {percent}% · {tokens}/{window}")
        }
        _ => format!("ctx {tokens}"),
    }
}

/// Each agent's gauge, whenever its tokens, its model or its line changed
/// (another plugin may have put its item at the same place). Writes only
/// when the item differs, so its own write does not make it run again.
fn show(
    mut agents: Query<
        (&LastUsage, Option<&Connection>, &mut StatusItems),
        (
            With<Agent>,
            Or<(
                Changed<StatusItems>,
                Changed<LastUsage>,
                Changed<Connection>,
            )>,
        ),
    >,
) {
    for (last, connection, mut items) in &mut agents {
        let context = context_of(last, connection);
        let item = GAUGE.says(label(context), Fill::of(context).tone());
        if !items.iter().any(|shown| *shown == item) {
            items.show(item);
        }
    }
}

/// The first time an agent's context goes above 90%, after a reply or a
/// change of model, a notice suggests condensing or handing off.
fn warn_once(
    mut agents: Query<
        (
            Entity,
            &LastUsage,
            Option<&Connection>,
            &mut NearlyFullWarned,
        ),
        (With<Agent>, Or<(Changed<LastUsage>, Changed<Connection>)>),
    >,
    mut notices: MessageWriter<Notice>,
) {
    for (agent, last, connection, mut warned) in &mut agents {
        let context = context_of(last, connection);
        if warned.0 || Fill::of(context) != Fill::Alert {
            continue;
        }
        warned.0 = true;
        let text = format!(
            "The context is nearly full ({}). Condense it with /compact or hand off \
             to a fresh agent before quality drops or a request fails.",
            label(context).trim_start_matches("ctx ")
        );
        notices.write(Notice::warn(agent, text));
    }
}

#[cfg(test)]
mod tests;
