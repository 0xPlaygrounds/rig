//! How the terminal view (rig-tui) draws the subagent tools' calls: the
//! task's title or the agent a call names, then what the tool answered.

use rig_harness::prelude::*;
use rig_tui::ratatui::text::Line;
use rig_tui::{RenderBatch, RenderSystems, ToolCallView, TranscriptRender};

use super::{MESSAGE, TASK, WAIT};

pub(super) fn add(app: &mut App) {
    app.add_systems(TranscriptRender, draw.in_set(RenderSystems::Render));
}

/// Draws the subagent tools' calls.
fn draw(mut batch: ResMut<RenderBatch>) {
    for (tool, shown) in [(TASK, "description"), (MESSAGE, "agent"), (WAIT, "agent")] {
        for (call, output) in batch.tools_mut(tool) {
            output.lines = look(&call.view(), shown);
        }
    }
}

/// The tool's name and its argument `shown`, then the result.
fn look(view: &ToolCallView<'_>, shown: &str) -> Vec<Line<'static>> {
    let title = format!(
        "{} {}",
        view.name(),
        view.argument(shown).unwrap_or_default()
    );
    let mut lines = vec![view.header(title, "")];
    lines.extend(view.result_lines());
    lines
}
