//! How the terminal view draws a `consult` call: the question, then each
//! consultant's model and tool calls nested under it, then the answers.

use rig_harness::prelude::*;
use rig_tui::ratatui::style::{Style, Stylize};
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::{CallState, ChildCall, LookLine, RenderBatch, ToolCallView};

use super::CONSULT;

/// Draws the `consult` calls.
pub(crate) fn draw(mut batch: ResMut<RenderBatch>) {
    for (call, output) in batch.tools_mut(CONSULT) {
        output.lines = consult(&call.view());
    }
}

/// A `consult` call.
pub(crate) fn consult(view: &ToolCallView) -> Vec<LookLine> {
    let question = view.argument("question").unwrap_or_default();
    let mut lines = vec![LookLine::from(view.header("Consult", question))];
    lines.extend(view.children.iter().flat_map(child).map(LookLine::from));
    lines.extend(view.result_lines());
    lines
}

/// One of the consultants' calls: a model call names its model, and the
/// tool calls under it their tool and arguments, and how they ended, with
/// the whole answer under a tool call's line.
fn child(child: &ChildCall) -> Vec<Line<'static>> {
    let indent = "  ".repeat(child.depth + 1);
    let what = match &child.call {
        Some(call) => {
            let arguments = serde_json::Value::Object(call.arguments.clone());
            format!("{} {arguments}", child.name)
        }
        None => child.name.clone(),
    };
    let mut spans = vec![Span::from(format!("{indent}↳ {what}")).dim()];
    let ended = child
        .answer
        .as_deref()
        .filter(|_| child.state == CallState::Done);
    let (said, style) = match ended {
        Some(why) if child.failed => (format!("failed: {why}"), Style::new().red()),
        Some(answer) if child.call.is_some() => (answer.to_owned(), Style::new().dim()),
        _ if child.state != CallState::Done => (String::new(), Style::new()),
        _ => return vec![Line::from(spans)],
    };
    if child.state != CallState::Done {
        spans.push(Span::from(" …").yellow());
        return vec![Line::from(spans)];
    }
    let mut said = said.lines();
    spans.push(Span::styled(
        format!(" · {}", said.next().unwrap_or_default()),
        style,
    ));
    let rest = said.map(|line| Line::styled(format!("{indent}    {line}"), style));
    std::iter::once(Line::from(spans)).chain(rest).collect()
}
