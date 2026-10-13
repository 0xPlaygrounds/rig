//! How the terminal view draws a `consult` call: the question, then each
//! consultant's model and tool calls nested under it, then the answers.

use rig_tui::ratatui::style::Stylize;
use rig_tui::ratatui::text::{Line, Span};
use rig_tui::{CallState, ChildCall, RESULT_LINES, ToolCallView};

/// The most characters of a consultant's tool call's arguments shown.
const ARGUMENTS: usize = 60;

/// A `consult` call.
pub(crate) fn consult(view: &ToolCallView) -> Vec<Line<'static>> {
    let question = view.argument("question").unwrap_or_default();
    let mut lines = vec![view.header("Consult", question)];
    lines.extend(view.children.iter().map(child));
    lines.extend(view.result_lines(view.limit(RESULT_LINES)));
    lines
}

/// One of the consultants' calls: a model call names its model, a tool
/// call its consultant, tool and arguments, and how it ended.
fn child(child: &ChildCall) -> Line<'static> {
    let indent = "  ".repeat(child.depth + 1);
    let what = match &child.call {
        Some(call) => {
            let arguments = serde_json::Value::Object(call.function.arguments.clone());
            let mut arguments = arguments.to_string();
            if let Some((at, _)) = arguments.char_indices().nth(ARGUMENTS) {
                arguments.truncate(at);
                arguments.push('…');
            }
            let agent = child.agent.as_deref().unwrap_or_default();
            format!("{agent}: {} {arguments}", child.name)
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
