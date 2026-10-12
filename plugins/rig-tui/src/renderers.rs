//! How tool calls and entries look in the transcript. A plugin that adds a
//! tool can add its look with [`AppToolRenderersExt::add_tool_renderer`]:
//! like a tool or a command, a renderer is a component on an entity of its
//! own, named `renderer:<tool>`, and the first one of a tool stays; Bevy's
//! `Disabled` on it frees the tool for another. A call whose tool has none
//! is drawn by [`ToolCallView::default_lines`]. A call is drawn while the
//! model still streams its arguments too ([`CallState::Streaming`]), from
//! what has come of them.
//!
//! The same goes for an origin's look with
//! [`AppToolRenderersExt::add_entry_renderer`] (`entry:<origin>`): it draws
//! a plugin's [`TranscriptEntry`](rig_harness::transcript::TranscriptEntry)
//! of that origin, and text the conversation holds from a plugin of that
//! name, such as a compaction's summary. One without draws as
//! [`EntryView::default_lines`]. A change to a renderer draws the
//! transcript again.

use std::sync::Arc;
use std::time::Duration;

use bevy_app::App;
use bevy_ecs::prelude::*;
use bevy_log::warn;
use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};
use rig_core::message::{ToolCall, ToolResult};
use rig_core::tool::ToolResultContext;
use rig_tools::shorten;

use super::diff;

/// Characters of a call's arguments shown on its header line.
const ARGUMENT_CHARS: usize = 160;
/// Lines of a result the default look shows.
pub const RESULT_LINES: usize = 4;
/// Lines of a diff shown.
const DIFF_LINES: usize = 40;
/// Characters of one result line shown: a longer line, such as a whole
/// JSON answer, ends in `…`.
const LINE_CHARS: usize = 240;

/// A tool call and its result, if it has one yet, as a renderer sees them.
pub struct ToolCallView<'a> {
    /// The call. While it streams, its arguments are what has come of them,
    /// and its id is not the final one.
    pub call: &'a ToolCall,
    /// Its result, once it is in the conversation.
    pub result: Option<&'a ToolResult>,
    /// Where the call stands.
    pub state: CallState,
    /// Whether it is drawn whole: the user expanded it.
    pub expanded: bool,
    /// What the call published, such as rig-core's `ExitStatus` and
    /// `FilesChanged`, once it settled in this run; not kept with the
    /// session.
    pub facts: Option<&'a ToolResultContext>,
    /// How long it ran, once it settled in this run; not kept with the
    /// session.
    pub took: Option<Duration>,
    /// The output so far of a running call whose tool reports it as it
    /// runs ([`CallOutput`]), such as a command's: cut as tool output is
    /// cut, its last lines after a line that counts the earlier ones.
    /// `None` once the call answered, and before it printed anything.
    ///
    /// [`CallOutput`]: rig_harness::prelude::CallOutput
    pub output: Option<&'a str>,
}

/// Where a tool call stands. A call of an earlier turn, or one whose
/// effect is gone, is `Done` when it has a result and `Running` when not.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum CallState {
    /// The model is still streaming its arguments.
    Streaming,
    /// Not started: it waits for earlier calls of its reply it must not
    /// run beside, or to be let through.
    Waiting,
    /// Held from starting, or from settling once answered: the holds'
    /// [`Name`]s, which say why, such as `waiting for lease on src/a.rs`.
    Held(Vec<String>),
    /// Running, with its [`ToolProgress`] line once its owner keeps one.
    ///
    /// [`ToolProgress`]: rig_harness::prelude::ToolProgress
    Running(Option<String>),
    /// Answered.
    Done,
}

/// What a hold without a [`Name`] is called.
pub(crate) const UNNAMED_HOLD: &str = "unnamed hold";

impl<'a> ToolCallView<'a> {
    /// `call`, answered by `result` if it was, standing as `state`,
    /// collapsed, with no facts or output known.
    pub fn new(call: &'a ToolCall, result: Option<&'a ToolResult>, state: CallState) -> Self {
        Self {
            call,
            result,
            state,
            expanded: false,
            facts: None,
            took: None,
            output: None,
        }
    }

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
        self.result.map(result_text)
    }

    /// Whether the tool answered with an error.
    pub fn failed(&self) -> bool {
        self.result.is_some_and(|result| result.is_error)
    }

    /// The header line: a dot coloured by state (dim waiting, yellow
    /// running or held, green done, red failed), `title` in bold, `detail`
    /// dimmed, and what holds a held call.
    pub fn header(&self, title: impl Into<String>, detail: impl Into<String>) -> Line<'static> {
        let dot = match (self.result, &self.state) {
            (Some(result), _) if result.is_error => Span::from("● ").red(),
            (Some(_), _) => Span::from("● ").green(),
            (None, CallState::Waiting | CallState::Streaming) => Span::from("○ ").dim(),
            (None, _) => Span::from("● ").yellow(),
        };
        let detail = detail.into();
        let mut spans = vec![dot, Span::from(title.into()).bold()];
        if !detail.is_empty() {
            spans.push(Span::from(format!(" {detail}")).dim());
        }
        if let CallState::Held(holds) = &self.state {
            spans.push(Span::from(format!(" · held: {}", holds.join(", "))).yellow());
        }
        Line::from(spans)
    }

    /// Up to `limit` lines of the result under a `⎿`, red for an error,
    /// and how many more there are. A unified diff is coloured. Before the
    /// result, a running call's progress line.
    pub fn result_lines(&self, limit: usize) -> Vec<Line<'static>> {
        let Some(text) = self.result_text() else {
            return self.progress_lines();
        };
        if !self.failed()
            && let Some(lines) = diff::unified_lines(&text, DIFF_LINES)
        {
            return lines;
        }
        excerpt(&text, limit, result_style(self.failed()))
    }

    /// The last `limit` lines of the result, where a command's errors and
    /// summary are, after a count of the earlier ones; a cut marker before
    /// them names the file with all of it. Before the result, a running
    /// call's progress line.
    pub fn tail_lines(&self, limit: usize) -> Vec<Line<'static>> {
        let Some(text) = self.result_text() else {
            return self.progress_lines();
        };
        let mut output = text.lines().peekable();
        let marker = output.next_if(|first| is_cut_marker(first));
        let output: Vec<&str> = output.collect();
        let skipped = output.len().saturating_sub(limit);
        let earlier = format!("… {skipped} earlier lines");
        let earlier = (skipped > 0).then_some(earlier.as_str());
        let shown = marker.into_iter().chain(earlier);
        let shown = shown.chain(output.into_iter().skip(skipped));
        let style = result_style(self.failed());
        shown
            .enumerate()
            .map(|(index, line)| result_line(index == 0, line, style))
            .collect()
    }

    /// A running call's progress line (its first line) under a `⎿`,
    /// where its result goes once it answers; none without one.
    pub fn progress_lines(&self) -> Vec<Line<'static>> {
        let CallState::Running(Some(progress)) = &self.state else {
            return Vec::new();
        };
        let line = progress.lines().next().unwrap_or_default();
        vec![result_line(true, line, Style::new().yellow())]
    }

    /// The look of a tool with no renderer: the name and arguments, then
    /// the first lines of the result, or all of it when expanded.
    pub fn default_lines(&self) -> Vec<Line<'static>> {
        let arguments = serde_json::Value::Object(self.call.function.arguments.clone()).to_string();
        let mut lines =
            vec![self.header(self.name().to_owned(), shorten(&arguments, ARGUMENT_CHARS))];
        lines.extend(self.result_lines(self.limit(RESULT_LINES)));
        lines
    }

    /// `limit` lines, or every line when the call is expanded: what a
    /// renderer shows of its result.
    pub fn limit(&self, limit: usize) -> usize {
        if self.expanded { usize::MAX } else { limit }
    }
}

/// An entry or a delivered text, as an entry renderer sees it: a plugin's
/// [`TranscriptEntry`](rig_harness::transcript::TranscriptEntry), or a
/// content item the conversation holds from a plugin, such as a compaction
/// summary.
pub struct EntryView<'a> {
    /// The origin whose look draws it: the entry's, or the plugin's name.
    pub origin: &'a str,
    /// What a collapsed one shows in a line: the entry's title, or the
    /// origin's label.
    pub title: &'a str,
    /// What it holds; a delivered text without the header the model reads.
    pub text: &'a str,
    /// Whether it is drawn whole: the user expanded it.
    pub expanded: bool,
}

impl EntryView<'_> {
    /// The look of an origin with no renderer: `⤶` and the title, then
    /// the first lines of the text, or all of it when expanded.
    pub fn default_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![Line::styled(
            format!("⤶ {}", self.title),
            Style::new().magenta().bold(),
        )];
        let limit = if self.expanded {
            usize::MAX
        } else {
            RESULT_LINES
        };
        lines.extend(excerpt(self.text, limit, Style::new().dim()));
        lines
    }
}

/// A tool result's text, joined.
pub(crate) fn result_text(result: &ToolResult) -> String {
    result
        .content
        .iter()
        .filter_map(|content| content.as_text())
        .collect::<Vec<_>>()
        .join("\n")
}

/// The style of a tool result's lines: red for an error, else dimmed.
pub(crate) fn result_style(failed: bool) -> Style {
    if failed {
        Style::new().red()
    } else {
        Style::new().dim()
    }
}

/// Up to `limit` lines of `text` in `style`, the first under a `⎿`, a
/// count of the rest, and the cut marker that ends a cut result.
pub fn excerpt(text: &str, limit: usize, style: Style) -> Vec<Line<'static>> {
    let mut all: Vec<&str> = text.lines().collect();
    let marker = all.pop_if(|last| is_cut_marker(last));
    let more = format!("… {} more lines", all.len().saturating_sub(limit));
    let more = (all.len() > limit).then_some(more.as_str());
    let shown = all.into_iter().take(limit).chain(more).chain(marker);
    shown
        .enumerate()
        .map(|(index, line)| result_line(index == 0, line, style))
        .collect()
}

/// Whether `line` is a tool's cut marker, which says what was cut and
/// which file has all of it, such as `[Cut at 16384 of 113000 bytes; all
/// of it is in …]`.
fn is_cut_marker(line: &str) -> bool {
    line.starts_with('[') && line.ends_with(']') && line.to_lowercase().contains("cut")
}

/// A line of a result in `style`, under a `⎿` when `first`; a long one is
/// shortened, but a cut marker is shown whole.
fn result_line(first: bool, line: &str, style: Style) -> Line<'static> {
    let prefix = if first { "  ⎿ " } else { "    " };
    let line = line.replace('\t', "    ");
    let line = if is_cut_marker(&line) {
        line
    } else {
        shorten(&line, LINE_CHARS)
    };
    Line::styled(format!("{prefix}{line}"), style)
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

/// Draws an entry or a delivered text as transcript lines, not yet wrapped.
pub type RenderEntry = dyn Fn(&EntryView<'_>) -> Vec<Line<'static>> + Send + Sync;

/// The look of the entries and delivered texts of `origin`, on an entity
/// of its own.
#[derive(Component, Clone)]
pub struct EntryRenderer {
    /// The origin, such as `btw` or `compaction`.
    pub origin: String,
    /// Draws one of its entries.
    pub render: Arc<RenderEntry>,
}

/// Registers tool and entry renderers on an [`App`].
pub trait AppToolRenderersExt {
    /// Draw the calls of `tool` with `render`, unless the tool has a
    /// renderer already: that one stays, with a warning. To restyle
    /// another plugin's tool, insert `Disabled` on its renderer first (in
    /// `Plugin::finish`, once every plugin's `build` ran).
    fn add_tool_renderer(
        &mut self,
        tool: &str,
        render: impl Fn(&ToolCallView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self;

    /// Draw the entries and delivered texts of `origin` with `render`,
    /// unless the origin has a renderer already: that one stays, with a
    /// warning, and `Disabled` on it frees the origin, as for tools.
    fn add_entry_renderer(
        &mut self,
        origin: &str,
        render: impl Fn(&EntryView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self;
}

impl AppToolRenderersExt for App {
    fn add_tool_renderer(
        &mut self,
        tool: &str,
        render: impl Fn(&ToolCallView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self {
        let world = self.world_mut();
        // A query skips a disabled renderer.
        let mut renderers = world.query::<&ToolRenderer>();
        if renderers.iter(world).any(|renderer| renderer.tool == tool) {
            warn!("renderer not added: `{tool}` has one; insert `Disabled` on it to replace it");
            return self;
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

    fn add_entry_renderer(
        &mut self,
        origin: &str,
        render: impl Fn(&EntryView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self {
        let world = self.world_mut();
        let mut renderers = world.query::<&EntryRenderer>();
        if renderers
            .iter(world)
            .any(|renderer| renderer.origin == origin)
        {
            warn!(
                "renderer not added: origin `{origin}` has one; insert `Disabled` on it to replace it"
            );
            return self;
        }
        world.spawn((
            Name::new(format!("entry:{origin}")),
            EntryRenderer {
                origin: origin.to_owned(),
                render: Arc::new(render),
            },
        ));
        self
    }
}

#[cfg(test)]
mod tests;
