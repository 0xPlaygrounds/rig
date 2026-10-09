//! How tool calls look in the transcript. A plugin that adds a tool can
//! add its look with [`AppToolRenderersExt::add_tool_renderer`]: like a
//! tool or a command, a renderer is a component on an entity of its own.
//! A call whose tool has none is drawn by [`ToolCallView::default_lines`].

use std::sync::Arc;

use bevy_app::App;
use bevy_ecs::prelude::*;
use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};
use rig_core::message::{ToolCall, ToolResult};
use rig_tools::shorten;

use super::diff;
use rig_ecs::subagents::{MESSAGE, TASK};

/// Characters of a call's arguments shown on its header line.
const ARGUMENT_CHARS: usize = 160;
/// Lines of a result the default look shows.
pub const RESULT_LINES: usize = 4;
/// Lines of a diff shown.
const DIFF_LINES: usize = 40;

/// A tool call and its result, if it has one yet, as a renderer sees them.
pub struct ToolCallView<'a> {
    /// The call.
    pub call: &'a ToolCall,
    /// Its result, once the tool answered.
    pub result: Option<&'a ToolResult>,
}

impl ToolCallView<'_> {
    /// The tool's name.
    pub fn name(&self) -> &str {
        &self.call.function.name
    }

    /// The string argument `key`, if the call has one.
    pub fn argument(&self, key: &str) -> Option<&str> {
        self.call.function.arguments.get(key)?.as_str()
    }

    /// The result's text, joined, if the tool answered.
    pub fn result_text(&self) -> Option<String> {
        self.result.map(|result| {
            result
                .content
                .iter()
                .filter_map(|content| content.as_text())
                .collect::<Vec<_>>()
                .join("\n")
        })
    }

    /// Whether the tool answered with an error.
    pub fn failed(&self) -> bool {
        self.result.is_some_and(|result| result.is_error)
    }

    /// The header line: a dot coloured by state (yellow running, green
    /// done, red failed), `title` in bold and `detail` dimmed.
    pub fn header(&self, title: impl Into<String>, detail: impl Into<String>) -> Line<'static> {
        let dot = match self.result {
            None => Span::from("● ").yellow(),
            Some(result) if result.is_error => Span::from("● ").red(),
            Some(_) => Span::from("● ").green(),
        };
        let detail = detail.into();
        let mut spans = vec![dot, Span::from(title.into()).bold()];
        if !detail.is_empty() {
            spans.push(Span::from(format!(" {detail}")).dim());
        }
        Line::from(spans)
    }

    /// Up to `limit` lines of the result under a `⎿`, red for an error,
    /// and how many more there are. A unified diff is coloured.
    pub fn result_lines(&self, limit: usize) -> Vec<Line<'static>> {
        let Some(text) = self.result_text() else {
            return Vec::new();
        };
        if !self.failed()
            && let Some(lines) = diff::unified_lines(&text, DIFF_LINES)
        {
            return lines;
        }
        let style = if self.failed() {
            Style::new().red()
        } else {
            Style::new().dim()
        };
        excerpt(&text, limit, style)
    }

    /// The look of a tool with no renderer: the name and arguments, then
    /// the first lines of the result.
    pub fn default_lines(&self) -> Vec<Line<'static>> {
        let arguments = serde_json::Value::Object(self.call.function.arguments.clone()).to_string();
        let mut lines =
            vec![self.header(self.name().to_owned(), shorten(&arguments, ARGUMENT_CHARS))];
        lines.extend(self.result_lines(RESULT_LINES));
        lines
    }
}

/// Up to `limit` lines of `text` in `style`, the first under a `⎿`, and a
/// count of the rest.
pub fn excerpt(text: &str, limit: usize, style: Style) -> Vec<Line<'static>> {
    let total = text.lines().count();
    let mut lines: Vec<Line<'static>> = text
        .lines()
        .take(limit)
        .enumerate()
        .map(|(index, line)| {
            let prefix = if index == 0 { "  ⎿ " } else { "    " };
            Line::styled(format!("{prefix}{}", line.replace('\t', "    ")), style)
        })
        .collect();
    if total > limit {
        lines.push(Line::styled(
            format!("    … {} more lines", total - limit),
            style,
        ));
    }
    lines
}

/// Draws a tool call as transcript lines, not yet wrapped.
pub type RenderToolCall = dyn Fn(&ToolCallView<'_>) -> Vec<Line<'static>> + Send + Sync;

/// The look of the tool named `tool`, on an entity of its own.
#[derive(Component, Clone)]
pub struct ToolRenderer {
    /// The tool's name.
    pub tool: String,
    /// Draws one of its calls.
    pub render: Arc<RenderToolCall>,
}

/// Registers tool renderers on an [`App`].
pub trait AppToolRenderersExt {
    /// Draw the calls of `tool` with `render`. It replaces the renderer the
    /// tool had, so a plugin can restyle a built-in tool.
    fn add_tool_renderer(
        &mut self,
        tool: &str,
        render: impl Fn(&ToolCallView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self;
}

impl AppToolRenderersExt for App {
    fn add_tool_renderer(
        &mut self,
        tool: &str,
        render: impl Fn(&ToolCallView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self {
        let world = self.world_mut();
        let old: Vec<Entity> = world
            .query::<(Entity, &ToolRenderer)>()
            .iter(world)
            .filter(|(_, renderer)| renderer.tool == tool)
            .map(|(entity, _)| entity)
            .collect();
        for entity in old {
            world.despawn(entity);
        }
        world.spawn((
            Name::new(format!("renderer:{tool}")),
            ToolRenderer {
                tool: tool.to_owned(),
                render: Arc::new(render),
            },
        ));
        self
    }
}

/// The looks of the built-in tools: `read` and `search` summarize what
/// they found, `edit` shows its change as a diff, `write` the start of the
/// new file, `shell` the command and the end of its output, and the
/// subagents' `task` and `message` whom they went to.
pub(crate) fn add_builtin_renderers(app: &mut App) {
    app.add_tool_renderer("read", read)
        .add_tool_renderer("edit", edit)
        .add_tool_renderer("write", write)
        .add_tool_renderer("shell", shell)
        .add_tool_renderer("search", search)
        .add_tool_renderer(TASK, task)
        .add_tool_renderer(MESSAGE, message);
}

fn read(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let range = match (
        view.call
            .function
            .arguments
            .get("offset")
            .and_then(|v| v.as_u64()),
        view.call
            .function
            .arguments
            .get("limit")
            .and_then(|v| v.as_u64()),
    ) {
        (Some(offset), Some(limit)) => format!("lines {offset}..{}", offset + limit),
        (Some(offset), None) => format!("from line {offset}"),
        (None, Some(limit)) => format!("first {limit} lines"),
        (None, None) => String::new(),
    };
    let mut lines = vec![view.header(format!("read {path}"), range)];
    match view.result_text() {
        Some(text) if view.failed() => {
            lines.extend(excerpt(&text, RESULT_LINES, Style::new().red()))
        }
        Some(text) => lines.push(Line::from(format!("  ⎿ {} lines", text.lines().count())).dim()),
        None => {}
    }
    lines
}

fn edit(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let mut lines = vec![view.header(format!("edit {path}"), "")];
    if view.failed() {
        lines.extend(view.result_lines(RESULT_LINES));
        return lines;
    }
    // A tool that returns its own diff is shown by it; otherwise the
    // change asked for.
    if let Some(text) = view.result_text()
        && let Some(diff) = diff::unified_lines(&text, DIFF_LINES)
    {
        lines.extend(diff);
        return lines;
    }
    // Before the result: each replacement asked for, on its own.
    let edits = view
        .call
        .function
        .arguments
        .get("edits")
        .and_then(|edits| edits.as_array());
    let mut asked = Vec::new();
    for edit in edits.into_iter().flatten() {
        let text = |key: &str| edit.get(key).and_then(|text| text.as_str());
        if let (Some(old), Some(new)) = (text("old_text"), text("new_text")) {
            if !asked.is_empty() {
                asked.push(Line::from("    ⋯").dark_gray());
            }
            asked.extend(diff::diff_lines(old, new, DIFF_LINES));
        }
    }
    if asked.is_empty() {
        lines.extend(view.result_lines(RESULT_LINES));
    } else {
        asked.truncate(DIFF_LINES);
        lines.extend(asked);
    }
    lines
}

fn write(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let path = view.argument("path").unwrap_or_default().to_owned();
    let content = view.argument("content").unwrap_or_default();
    let mut lines = vec![view.header(
        format!("write {path}"),
        format!("{} lines", content.lines().count()),
    )];
    if view.failed() {
        lines.extend(view.result_lines(RESULT_LINES));
    } else {
        lines.extend(excerpt(content, RESULT_LINES, Style::new().dim()));
    }
    lines
}

fn shell(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let command = view.argument("command").unwrap_or_default();
    let mut lines = Vec::new();
    for (index, line) in command.lines().enumerate() {
        if index == 0 {
            lines.push(view.header(format!("$ {line}"), ""));
        } else {
            lines.push(Line::from(format!("    {line}")).bold());
        }
    }
    if lines.is_empty() {
        lines.push(view.header("$", ""));
    }
    // The end of a command's output is where its errors and summary are.
    if let Some(text) = view.result_text() {
        let style = if view.failed() {
            Style::new().red()
        } else {
            Style::new().dim()
        };
        let total = text.lines().count();
        let skipped = total.saturating_sub(RESULT_LINES + 2);
        if skipped > 0 {
            lines.push(Line::styled(
                format!("  ⎿ … {skipped} earlier lines"),
                style,
            ));
        }
        let tail: Vec<&str> = text.lines().skip(skipped).collect();
        lines.extend(tail.iter().enumerate().map(|(index, line)| {
            let prefix = if index == 0 && skipped == 0 {
                "  ⎿ "
            } else {
                "    "
            };
            Line::styled(format!("{prefix}{}", line.replace('\t', "    ")), style)
        }));
    }
    lines
}

fn search(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let pattern = view.argument("pattern").unwrap_or_default();
    let mut detail = String::new();
    if let Some(path) = view.argument("path") {
        detail.push_str(&format!("in {path}"));
    }
    if let Some(glob) = view.argument("glob") {
        detail.push_str(&format!(" ({glob})"));
    }
    let mut lines = vec![view.header(format!("search {pattern}"), detail.trim().to_owned())];
    lines.extend(view.result_lines(RESULT_LINES));
    lines
}

/// A subagent request's result, or that it is still being sent.
fn sent(view: &ToolCallView<'_>, lines: &mut Vec<Line<'static>>) {
    match view.result_text() {
        Some(text) => {
            let style = if view.failed() {
                Style::new().red()
            } else {
                Style::new().dim()
            };
            lines.extend(excerpt(&text, RESULT_LINES, style));
        }
        None => lines.push(Line::from("  ⎿ sending…").dim()),
    }
}

/// The subagent's title and model, then whether it started.
fn task(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let title = view.argument("description").unwrap_or("task").to_owned();
    let detail = view
        .argument("model")
        .map(|model| format!("on {model}"))
        .unwrap_or_default();
    let mut lines = vec![view.header(format!("task {title}"), detail)];
    sent(view, &mut lines);
    lines
}

/// The subagent written to, then whether the request went out.
fn message(view: &ToolCallView<'_>) -> Vec<Line<'static>> {
    let agent = view.argument("agent").unwrap_or("?").to_owned();
    let mut lines = vec![view.header(format!("message {agent}"), String::new())];
    sent(view, &mut lines);
    lines
}
