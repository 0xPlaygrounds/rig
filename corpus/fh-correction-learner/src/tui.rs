//! The waiting proposal under the transcript, and its keys: Alt+S saves
//! it, Alt+E puts `/rules save <rule>` in the input to edit, Alt+X
//! dismisses it. The input keeps working meanwhile.

use rig_harness::prelude::*;
use rig_tui::ratatui::layout::Constraint;
use rig_tui::ratatui::text::Line;
use rig_tui::ratatui::widgets::{Block, Paragraph, Wrap};
use rig_tui::{
    AppKeyBindingsExt, KeyBinding, KeyPress, KeyPressed, PanelCanvas, Placement, TuiInput,
    TuiPanel, TuiSystems, run_command,
};

use super::Proposal;

/// Adds the panel and the keys.
pub(super) fn add(app: &mut App) {
    app.add_observer(show)
        .add_observer(hide)
        .add_systems(PostUpdate, draw.in_set(TuiSystems::Draw));
    let save = KeyBinding::new(KeyPress::alt('s'), "Save the proposed rule");
    app.add_key_binding(save, run_command("rules save"));
    let dismiss = KeyBinding::new(KeyPress::alt('x'), "Dismiss the proposed rule");
    app.add_key_binding(dismiss, run_command("rules dismiss"));
    let edit = KeyBinding::new(KeyPress::alt('e'), "Edit the proposed rule");
    app.add_key_binding(edit, edit_rule);
}

/// A proposal takes rows under the transcript.
fn show(added: On<Add<Proposal>>, mut commands: Commands) {
    let panel = TuiPanel::new(Placement::Bottom(Constraint::Length(4)));
    commands.entity(added.entity).insert(panel);
}

/// An answered proposal gives its rows back.
fn hide(answered: On<Add<Outcome>>, proposals: Query<(), With<Proposal>>, mut commands: Commands) {
    if proposals.contains(answered.entity) {
        commands.entity(answered.entity).try_remove::<TuiPanel>();
    }
}

fn draw(mut panels: Query<(&Proposal, &mut PanelCanvas), Without<Outcome>>) {
    for (proposal, mut canvas) in &mut panels {
        let lines = vec![
            Line::from(format!("`{}`", proposal.rule)),
            Line::from("alt+s save · alt+e edit · alt+x dismiss"),
        ];
        let block = Block::bordered().title("Remember for this project?");
        canvas.render(Paragraph::new(lines).wrap(Wrap { trim: true }).block(block));
    }
}

/// Puts `/rules save <rule>` in the input, to edit and send.
fn edit_rule(
    _: On<KeyPressed>,
    proposals: Query<&Proposal, Without<Outcome>>,
    mut input: ResMut<TuiInput>,
) {
    if let Some(proposal) = proposals.iter().next() {
        input.set(format!("/rules save {}", proposal.rule));
    }
}
