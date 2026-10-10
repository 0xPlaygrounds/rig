//! How the terminal view (rig-tui) draws the subagent tools' calls: the
//! task's title or the agent a call names, then what the tool answered.

use rig_harness::prelude::*;
use rig_tui::ratatui::text::Line;
use rig_tui::{AppToolRenderersExt, RESULT_LINES, ToolCallView};

use super::{MESSAGE, TASK, WAIT};

pub(super) fn add(app: &mut App) {
    app.add_tool_renderer(TASK, |view| look(view, "description"))
        .add_tool_renderer(MESSAGE, |view| look(view, "agent"))
        .add_tool_renderer(WAIT, |view| look(view, "agent"));
}

/// The tool's name and its argument `shown`, then the start of the result.
fn look(view: &ToolCallView<'_>, shown: &str) -> Vec<Line<'static>> {
    let title = format!(
        "{} {}",
        view.name(),
        view.argument(shown).unwrap_or_default()
    );
    let mut lines = vec![view.header(title, "")];
    lines.extend(view.result_lines(RESULT_LINES));
    lines
}
