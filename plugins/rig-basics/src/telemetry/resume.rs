//! What the session spent and its main agent's context, in its line in
//! `/resume`: [`ResumeLabel`]s like any plugin's, kept current from
//! [`SessionUsage`] and the main agent's [`LastUsage`].

use crate::sessions::ResumeLabel;
use rig_core::completion::tokens_label;
use rig_ecs::{StopTurnsOnExit, WriteLogs};
use rig_harness::prelude::*;

use super::usage::SessionUsage;

/// The tag of the session's spending among its [`ResumeLabel`]s, and so
/// in its listing's `labels`.
const SPENT: &str = "spent";
/// The tag of the main agent's context among the [`ResumeLabel`]s.
const CONTEXT: &str = "ctx";

/// On the [`ResumeLabel`] that says what the session spent.
#[derive(Component)]
struct Spent;

/// On the [`ResumeLabel`] that says the main agent's context.
#[derive(Component)]
struct Context;

pub(super) fn add(app: &mut App) {
    let world = app.world_mut();
    world.spawn((
        Name::new("resume label: spent"),
        Spent,
        ResumeLabel::new(SPENT, ""),
    ));
    world.spawn((
        Name::new("resume label: context"),
        Context,
        ResumeLabel::new(CONTEXT, ""),
    ));
    app.add_systems(
        Last,
        (
            label_usage.run_if(resource_changed::<SessionUsage>),
            label_context.after(CurrentAgentSystems),
        )
            .after(StopTurnsOnExit)
            .before(WriteLogs),
    );
}

/// Says what the session's model calls cost, or their tokens when none
/// was priced; nothing before the first call.
fn label_usage(spent: Res<SessionUsage>, mut labels: Query<&mut ResumeLabel, With<Spent>>) {
    let total = &spent.0.total;
    let text = if total.calls > 0 {
        format!("{} spent", total.cost_or_tokens())
    } else {
        String::new()
    };
    for mut label in &mut labels {
        label.set_if_neq(ResumeLabel::new(SPENT, text.clone()));
    }
}

/// Says the main agent's context, as its last reply left it, or `ctx ?`
/// while a compaction left that figure stale ([`ContextStale`]); every
/// pass, so it follows a new main agent and the marker coming and going.
/// Not the shown agent's: a label that changes rewrites the session's
/// listing, so showing another agent would move the session to the top of
/// `/resume`.
fn label_context(
    agents: MainAgentQuery,
    usage: Query<(&LastUsage, Has<ContextStale>)>,
    mut labels: Query<&mut ResumeLabel, With<Context>>,
) {
    let last = main_agent(&agents).and_then(|agent| usage.get(agent).ok());
    let stale = last.is_some_and(|(_, stale)| stale);
    let text = context_label(last.and_then(|(usage, _)| usage.context()), stale);
    for mut label in &mut labels {
        label.set_if_neq(ResumeLabel::new(CONTEXT, text.clone()));
    }
}

/// The context label for `tokens` of context, as `ctx 4.7k`; `ctx ?` when
/// the figure is `stale`; empty when no reply reported it.
fn context_label(tokens: Option<u64>, stale: bool) -> String {
    match (tokens, stale) {
        (_, true) => "ctx ?".to_owned(),
        (Some(tokens), false) => format!("ctx {}", tokens_label(tokens)),
        (None, false) => String::new(),
    }
}

#[cfg(test)]
mod tests;
