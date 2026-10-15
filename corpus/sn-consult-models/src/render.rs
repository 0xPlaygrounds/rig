//! How the terminal view draws a `consult` call: the question, then each
//! consultant's model and tool calls nested under it, then the answers.

use rig_tui::ratatui::style::Stylize;
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::{CallState, ChildCall, ToolCallView};

/// A `consult` call.
pub(crate) fn consult(view: &ToolCallView) -> Vec<Line<'static>> {
    let question = view.argument("question").unwrap_or_default();
    let mut lines = vec![view.header("Consult", question)];
    lines.extend(view.children.iter().map(child));
    lines.extend(view.result_lines());
    lines
}

/// One of the consultants' calls: a model call names its model, and the
/// tool calls under it their tool and arguments, and how they ended.
fn child(child: &ChildCall) -> Line<'static> {
    let indent = "  ".repeat(child.depth + 1);
    let what = match &child.call {
        Some(call) => {
            let arguments = serde_json::Value::Object(call.function.arguments.clone());
            format!("{} {arguments}", child.name)
        }
        None => child.name.clone(),
    };
    let mut spans = vec![Span::from(format!("{indent}↳ {what}")).dim()];
    let ended = child
        .line
        .as_deref()
        .filter(|_| child.state == CallState::Done);
    match ended {
        Some(why) if child.failed => spans.push(Span::from(format!(" · failed: {why}")).red()),
        Some(line) if child.call.is_some() => spans.push(Span::from(format!(" · {line}")).dim()),
        _ if child.state != CallState::Done => spans.push(Span::from(" …").yellow()),
        _ => {}
    }
    Line::from(spans)
}
