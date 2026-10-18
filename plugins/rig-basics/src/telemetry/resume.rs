//! What the session spent, in its line in `/resume`: a [`ResumeLabel`]
//! like any plugin's, kept current from [`SessionUsage`].

use crate::sessions::ResumeLabel;
use rig_ecs::{StopTurnsOnExit, WriteLogs};
use rig_harness::prelude::*;

use super::usage::SessionUsage;

/// The tag of the session's spending among its [`ResumeLabel`]s, and so
/// in its listing's `labels`.
const SPENT: &str = "spent";

/// On the [`ResumeLabel`] that says what the session spent.
#[derive(Component)]
struct Spent;

pub(super) fn add(app: &mut App) {
    let label = ResumeLabel::new(SPENT, "");
    app.world_mut()
        .spawn((Name::new("resume label: spent"), Spent, label));
    app.add_systems(
        Last,
        label_usage
            .after(StopTurnsOnExit)
            .before(WriteLogs)
            .run_if(resource_changed::<SessionUsage>),
    );
}

/// Says what the session's model calls cost, or their tokens when none
/// was priced; nothing before the first call.
fn label_usage(spent: Res<SessionUsage>, mut labels: Query<&mut ResumeLabel, With<Spent>>) {
    let total = &spent.0.total;
    let text = if total.calls > 0 {
        total.cost_or_tokens()
    } else {
        String::new()
    };
    for mut label in &mut labels {
        label.set_if_neq(ResumeLabel::new(SPENT, text.clone()));
    }
}

#[cfg(test)]
mod tests;
