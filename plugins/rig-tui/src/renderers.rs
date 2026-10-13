//! How tool calls and entries look in the transcript. A plugin that adds a
//! tool can add its look with [`AppToolRenderersExt::add_tool_renderer`]:
//! like a tool or a command, a renderer is a component on an entity of its
//! own, named `renderer:<tool>`, and the first one of a tool stays; Bevy's
//! `Disabled` on it frees the tool for another. A call whose tool has none
//! is drawn by [`ToolCallView::default_lines`]. A call is drawn while the
//! model still streams its arguments too ([`CallState::Streaming`]), from
//! what has come of them.
//!
//! The same goes for a plugin's look with
//! [`AppToolRenderersExt::add_entry_renderer`] (`entry:<plugin>`): it draws
//! the [`Note`](rig_harness::note::Note)s that plugin wrote, and the
//! entries of the conversation it wrote, such as a compaction's summary. One without draws as
//! [`EntryView::default_lines`]. And for a kind of content block in a
//! message, an image, audio, a video or a document, with
//! [`AppToolRenderersExt::add_content_renderer`] (`content:<kind>`): one
//! without draws as [`ContentView::default_lines`]. A plugin that adds
//! rows under every call, whatever draws it, adds a decorator
//! ([`AppToolRenderersExt::add_call_decorator`], `decorator:<name>`):
//! decorators add up, each drawing after the call's look. A change to a
//! renderer or a decorator draws the transcript again.
//!
//! A renderer also sees the calls its call started, such as a subagent's
//! model and tool calls, through the spawn and the brief between them
//! ([`ToolCallView::children`]), and draws them nested where it likes:
//!
//! ```no_run
//! use rig_harness::prelude::*;
//! use rig_tui::AppToolRenderersExt;
//! use rig_tui::ratatui::text::Line;
//!
//! fn build(app: &mut App) {
//!     app.add_tool_renderer("consult", |view| {
//!         let mut lines = view.default_lines();
//!         for child in view.children {
//!             let indent = "  ".repeat(child.depth + 2);
//!             lines.push(Line::from(format!("{indent}↳ {} {:?}", child.name, child.state)));
//!         }
//!         lines
//!     });
//! }
//! ```

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use bevy_app::App;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use bevy_log::warn;
use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};
use rig_core::completion::AssistantContent;
use rig_core::message::{Audio, Document, Image, ToolCall, ToolResult, UserContent, Video};
use rig_core::tool::{KeptEnd, OutputCut, ToolResultContext};
use rig_ecs::prelude::MessageId;
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
    /// `FilesChanged`, once it settled: in this run, or in an earlier one
    /// of a resumed session whose effect log was kept.
    pub facts: Option<&'a ToolResultContext>,
    /// How long it ran, once it settled, as for [`facts`](Self::facts).
    pub took: Option<Duration>,
    /// The output so far of a running call whose tool reports it as it
    /// runs ([`CallOutput`]), such as a command's: cut as tool output is
    /// cut, its last lines after a line that counts the earlier ones.
    /// `None` once the call answered, and before it printed anything.
    ///
    /// [`CallOutput`]: rig_harness::prelude::CallOutput
    pub output: Option<&'a str>,
    /// The tool and model calls the call started, and those they started,
    /// in order, each after the one that started it, through any effect
    /// between that is no call: such as a subagent's calls under the call
    /// that spawned it, through the spawn and the brief it was sent. Also
    /// after a resume, from the session's effect log
    /// ([`PastEffects`](crate::PastEffects)).
    pub children: &'a [ChildCall],
}

/// A tool or model call that a tool call started, or one of those
/// started, as a renderer sees it ([`ToolCallView::children`]).
#[derive(Clone, Debug, PartialEq)]
pub struct ChildCall {
    /// How far below the call it is: 0 for one the call started itself.
    pub depth: usize,
    /// The tool call, or `None` for a model call.
    pub call: Option<ToolCall>,
    /// The tool's name, or the model's catalog reference (`vendor/model`),
    /// or `model` before it is known.
    pub name: String,
    /// The name of the agent it is for, such as a subagent's.
    pub agent: Option<String>,
    /// Where it stands; a model call is [`CallState::Streaming`] while its
    /// reply streams in.
    pub state: CallState,
    /// Whether it failed: a tool's error result, or a model call's error.
    pub failed: bool,
    /// The first line of its answer, once it has one: a tool's result, a
    /// model's reply, or why it failed.
    pub line: Option<String>,
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
            children: &[],
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
    /// and how many more there are. A unified diff is coloured. The line
    /// that says the tool cut its output ([`OutputCut`]) is shown whole,
    /// on its side. Before the result, a running call's progress line.
    pub fn result_lines(&self, limit: usize) -> Vec<Line<'static>> {
        let Some(text) = self.result_text() else {
            return self.progress_lines();
        };
        if !self.failed()
            && let Some(lines) = diff::unified_lines(&text, DIFF_LINES)
        {
            return lines;
        }
        let (output, marker) = self.cut_marker(&text);
        let more = format!("… {} more lines", output.len().saturating_sub(limit));
        let more = (output.len() > limit).then_some(more.as_str());
        let shown = output.iter().copied().take(limit).chain(more);
        result_rows(shown, marker, result_style(self.failed()))
    }

    /// The last `limit` lines of the result, where a command's errors and
    /// summary are, after a count of the earlier ones; the line that says
    /// the tool cut its output ([`OutputCut`]), naming the file with all
    /// of it, is shown whole, on its side. Before the result, a running
    /// call's progress line.
    pub fn tail_lines(&self, limit: usize) -> Vec<Line<'static>> {
        let Some(text) = self.result_text() else {
            return self.progress_lines();
        };
        let (output, marker) = self.cut_marker(&text);
        let skipped = output.len().saturating_sub(limit);
        let earlier = format!("… {skipped} earlier lines");
        let earlier = (skipped > 0).then_some(earlier.as_str());
        let shown = earlier.into_iter().chain(output.into_iter().skip(skipped));
        result_rows(shown, marker, result_style(self.failed()))
    }

    /// The lines of `text`, the result, without the line that says the
    /// tool cut its output, and that line with the end it is on, when the
    /// call published [`OutputCut`] (a call of an earlier run has no facts,
    /// so its marker is a line like any other).
    fn cut_marker<'t>(&self, text: &'t str) -> (Vec<&'t str>, Option<(&'t str, KeptEnd)>) {
        let mut lines: Vec<&str> = text.lines().collect();
        let cut = self
            .facts
            .and_then(|facts| facts.get::<OutputCut>().ok().flatten());
        let marker = cut.and_then(|cut| {
            let line = match cut.kept {
                // The output's end is kept: the marker is its first line.
                KeptEnd::End if lines.is_empty() => None,
                KeptEnd::End => Some(lines.remove(0)),
                KeptEnd::Start => lines.pop(),
            };
            Some((line?, cut.kept))
        });
        (lines, marker)
    }

    /// A running call's progress line (its first line) under a `⎿`,
    /// where its result goes once it answers; none without one.
    pub fn progress_lines(&self) -> Vec<Line<'static>> {
        let CallState::Running(Some(progress)) = &self.state else {
            return Vec::new();
        };
        let line = progress.lines().next().unwrap_or_default();
        result_rows(std::iter::once(line), None, Style::new().yellow())
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

/// A note or a delivered text, as an entry renderer sees it: a plugin's
/// [`Note`](rig_harness::note::Note), or a text of an entry of the
/// conversation another author wrote, such as a compaction summary.
pub struct EntryView<'a> {
    /// The plugin whose look draws it: the name of the plugin that wrote
    /// it, or empty when no plugin did (an agent, rig).
    pub plugin: &'a str,
    /// What a collapsed one shows in a line: the note's title, or the
    /// author's label.
    pub title: &'a str,
    /// What it holds; a delivered text without the header the model reads.
    pub text: &'a str,
    /// Whether it is drawn whole: the user expanded it.
    pub expanded: bool,
}

impl EntryView<'_> {
    /// The look of a plugin with no renderer: `⤶` and the title, then
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

/// A kind of content block a [`ContentRenderer`] draws.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum ContentKind {
    /// An image, the user's or one the model made.
    Image,
    /// Audio.
    Audio,
    /// A video.
    Video,
    /// A document, such as a PDF.
    Document,
}

impl ContentKind {
    /// Its name: `image`, `audio`, `video` or `document`.
    pub fn name(self) -> &'static str {
        match self {
            Self::Image => "image",
            Self::Audio => "audio",
            Self::Video => "video",
            Self::Document => "document",
        }
    }
}

/// A content block of a message that is not text, a tool call or a
/// result.
#[derive(Clone, Copy, Debug)]
pub enum ContentBlock<'a> {
    /// An image, the user's or one the model made.
    Image(&'a Image),
    /// Audio.
    Audio(&'a Audio),
    /// A video.
    Video(&'a Video),
    /// A document.
    Document(&'a Document),
}

impl<'a> ContentBlock<'a> {
    /// A user's block, if it is one.
    pub(crate) fn of_user(content: &'a UserContent) -> Option<Self> {
        match content {
            UserContent::Image(image) => Some(Self::Image(image)),
            UserContent::Audio(audio) => Some(Self::Audio(audio)),
            UserContent::Video(video) => Some(Self::Video(video)),
            UserContent::Document(document) => Some(Self::Document(document)),
            UserContent::Text(_) | UserContent::ToolResult(_) => None,
        }
    }

    /// A reply's block, if it is one.
    pub(crate) fn of_assistant(content: &'a AssistantContent) -> Option<Self> {
        match content {
            AssistantContent::Image(image) => Some(Self::Image(image)),
            _ => None,
        }
    }

    /// Its kind.
    pub fn kind(self) -> ContentKind {
        match self {
            Self::Image(_) => ContentKind::Image,
            Self::Audio(_) => ContentKind::Audio,
            Self::Video(_) => ContentKind::Video,
            Self::Document(_) => ContentKind::Document,
        }
    }
}

/// A content block of a message, as a content renderer sees it: an
/// image, audio, a video or a document, the user's or the model's. Its
/// rows are the transcript item [`TranscriptItem::Content`], which a
/// plugin finds in the [`DrawnTranscript`], such as to draw an image over
/// the rows its renderer left blank.
///
/// [`TranscriptItem::Content`]: crate::TranscriptItem::Content
/// [`DrawnTranscript`]: crate::DrawnTranscript
pub struct ContentView<'a> {
    /// The message it is in, if the message has an id.
    pub message: Option<MessageId>,
    /// Its place among the message's blocks.
    pub index: usize,
    /// The block.
    pub block: ContentBlock<'a>,
    /// Whether the message was written by the model.
    pub assistant: bool,
    /// Whether it is drawn whole: the user expanded it.
    pub expanded: bool,
}

impl ContentView<'_> {
    /// The look of a kind with no renderer: its name in brackets, such as
    /// `[image]`.
    pub fn default_lines(&self) -> Vec<Line<'static>> {
        let name = self.block.kind().name();
        vec![Line::from(format!("  [{name}]")).dim()]
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

/// Up to `limit` lines of `text` in `style`, the first under a `⎿`, and
/// a count of the rest.
pub fn excerpt(text: &str, limit: usize, style: Style) -> Vec<Line<'static>> {
    let all: Vec<&str> = text.lines().collect();
    let more = format!("… {} more lines", all.len().saturating_sub(limit));
    let more = (all.len() > limit).then_some(more.as_str());
    result_rows(all.into_iter().take(limit).chain(more), None, style)
}

/// `lines` of a result in `style`, the first under a `⎿`, each shortened,
/// with `marker` whole before them when the output's end was kept and
/// after them when its start was.
fn result_rows<'t>(
    lines: impl Iterator<Item = &'t str>,
    marker: Option<(&'t str, KeptEnd)>,
    style: Style,
) -> Vec<Line<'static>> {
    let (before, after) = match marker {
        Some((marker, KeptEnd::End)) => (Some(marker), None),
        Some((marker, KeptEnd::Start)) => (None, Some(marker)),
        None => (None, None),
    };
    let lines = lines.map(|line| (line, false));
    let rows = before.map(|marker| (marker, true)).into_iter().chain(lines);
    let rows = rows.chain(after.map(|marker| (marker, true)));
    rows.enumerate()
        .map(|(index, (line, whole))| {
            let prefix = if index == 0 { "  ⎿ " } else { "    " };
            let line = line.replace('\t', "    ");
            let line = if whole {
                line
            } else {
                shorten(&line, LINE_CHARS)
            };
            Line::styled(format!("{prefix}{line}"), style)
        })
        .collect()
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

/// Draws a note or a delivered text as transcript lines, not yet wrapped.
pub type RenderEntry = dyn Fn(&EntryView<'_>) -> Vec<Line<'static>> + Send + Sync;

/// The look of the notes and delivered texts of the plugin `plugin`, on an
/// entity of its own.
#[derive(Component, Clone)]
pub struct EntryRenderer {
    /// The plugin's name, as its [`Author::plugin`](rig_core::conversation::Author::plugin)
    /// says, such as `btw` or `compaction`.
    pub plugin: String,
    /// Draws one of its notes or texts.
    pub render: Arc<RenderEntry>,
}

/// Draws a content block as transcript lines, not yet wrapped.
pub type RenderContent = dyn Fn(&ContentView<'_>) -> Vec<Line<'static>> + Send + Sync;

/// The look of the content blocks of a kind, on an entity of its own.
#[derive(Component, Clone)]
pub struct ContentRenderer {
    /// The kind it draws.
    pub kind: ContentKind,
    /// Draws one block.
    pub render: Arc<RenderContent>,
}

/// Lines drawn under every tool call, after its look, on an entity of
/// its own.
#[derive(Component, Clone)]
pub struct CallDecorator {
    /// What it is called, such as the plugin's name.
    pub name: String,
    /// Draws its lines under a call; none leaves the call as it is.
    pub render: Arc<RenderToolCall>,
}

/// Registers tool, entry and content renderers and call decorators on an
/// [`App`].
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

    /// Draw the notes and delivered texts of the plugin `plugin` with
    /// `render`, unless the plugin has a renderer already: that one stays,
    /// with a warning, and `Disabled` on it frees the plugin, as for tools.
    fn add_entry_renderer(
        &mut self,
        plugin: &str,
        render: impl Fn(&EntryView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self;

    /// Draw the content blocks of `kind` with `render`, unless the kind
    /// has a renderer already: that one stays, with a warning, and
    /// `Disabled` on it frees the kind, as for tools.
    fn add_content_renderer(
        &mut self,
        kind: ContentKind,
        render: impl Fn(&ContentView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self;

    /// Draw `render`'s lines under every tool call, after the call's own
    /// look, whichever draws it. Decorators add up, in the order of their
    /// entities; `Disabled` on one (`decorator:<name>`) leaves it out.
    fn add_call_decorator(
        &mut self,
        name: &str,
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
        plugin: &str,
        render: impl Fn(&EntryView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self {
        let world = self.world_mut();
        let mut renderers = world.query::<&EntryRenderer>();
        if renderers
            .iter(world)
            .any(|renderer| renderer.plugin == plugin)
        {
            warn!(
                "renderer not added: plugin `{plugin}` has one; insert `Disabled` on it to replace it"
            );
            return self;
        }
        world.spawn((
            Name::new(format!("entry:{plugin}")),
            EntryRenderer {
                plugin: plugin.to_owned(),
                render: Arc::new(render),
            },
        ));
        self
    }

    fn add_content_renderer(
        &mut self,
        kind: ContentKind,
        render: impl Fn(&ContentView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self {
        let world = self.world_mut();
        let mut renderers = world.query::<&ContentRenderer>();
        let name = kind.name();
        if renderers.iter(world).any(|renderer| renderer.kind == kind) {
            warn!("renderer not added: `{name}` has one; insert `Disabled` on it to replace it");
            return self;
        }
        let render = Arc::new(render);
        world.spawn((
            Name::new(format!("content:{name}")),
            ContentRenderer { kind, render },
        ));
        self
    }

    fn add_call_decorator(
        &mut self,
        name: &str,
        render: impl Fn(&ToolCallView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self {
        self.world_mut().spawn((
            Name::new(format!("decorator:{name}")),
            CallDecorator {
                name: name.to_owned(),
                render: Arc::new(render),
            },
        ));
        self
    }
}

/// The looks plugins added, as a frame draws with them.
#[derive(SystemParam)]
pub(crate) struct Looks<'w, 's> {
    tools: Query<'w, 's, Ref<'static, ToolRenderer>>,
    entries: Query<'w, 's, Ref<'static, EntryRenderer>>,
    contents: Query<'w, 's, Ref<'static, ContentRenderer>>,
    decorators: Query<'w, 's, (Entity, Ref<'static, CallDecorator>)>,
}

impl Looks<'_, '_> {
    /// Each look by what it draws.
    pub(crate) fn by_kind(&self) -> Renderers<'_> {
        let tools = self.tools.iter().map(|renderer| {
            let renderer = renderer.into_inner();
            (renderer.tool.as_str(), &renderer.render)
        });
        let entries = self.entries.iter().map(|renderer| {
            let renderer = renderer.into_inner();
            (renderer.plugin.as_str(), &renderer.render)
        });
        let contents = self.contents.iter().map(|renderer| {
            let renderer = renderer.into_inner();
            (renderer.kind, &renderer.render)
        });
        let mut decorators: Vec<(Entity, &Arc<RenderToolCall>)> = (self.decorators.iter())
            .map(|(entity, decorator)| (entity, &decorator.into_inner().render))
            .collect();
        decorators.sort_by_key(|(entity, _)| *entity);
        Renderers {
            tools: tools.collect(),
            entries: entries.collect(),
            contents: contents.collect(),
            decorators: decorators.into_iter().map(|(_, render)| render).collect(),
        }
    }

    /// Whether a look changed or was added since the last frame.
    pub(crate) fn changed(&self) -> bool {
        self.tools.iter().any(|look| look.is_changed())
            || self.entries.iter().any(|look| look.is_changed())
            || self.contents.iter().any(|look| look.is_changed())
            || self.decorators.iter().any(|(_, look)| look.is_changed())
    }
}

/// The looks plugins added, by what each draws.
#[derive(Default)]
pub(crate) struct Renderers<'a> {
    /// The tool renderers by tool name.
    pub(crate) tools: HashMap<&'a str, &'a Arc<RenderToolCall>>,
    /// The entry renderers by plugin name.
    pub(crate) entries: HashMap<&'a str, &'a Arc<RenderEntry>>,
    /// The content renderers by kind.
    pub(crate) contents: HashMap<ContentKind, &'a Arc<RenderContent>>,
    /// The call decorators, in order.
    pub(crate) decorators: Vec<&'a Arc<RenderToolCall>>,
}

impl Renderers<'_> {
    /// A tool call's lines, by its tool's renderer or the default look,
    /// then every decorator's.
    pub(crate) fn call(&self, view: &ToolCallView<'_>) -> Vec<Line<'static>> {
        let mut lines = match self.tools.get(view.name()) {
            Some(render) => render(view),
            None => view.default_lines(),
        };
        for decorate in &self.decorators {
            lines.extend(decorate(view));
        }
        lines
    }

    /// A note's or a delivered text's lines, by its plugin's renderer or
    /// the default look.
    pub(crate) fn entry(&self, view: &EntryView<'_>) -> Vec<Line<'static>> {
        match self.entries.get(view.plugin) {
            Some(render) => render(view),
            None => view.default_lines(),
        }
    }

    /// A content block's lines, by its kind's renderer or the default
    /// look.
    pub(crate) fn content(&self, view: &ContentView<'_>) -> Vec<Line<'static>> {
        match self.contents.get(&view.block.kind()) {
            Some(render) => render(view),
            None => view.default_lines(),
        }
    }
}

#[cfg(test)]
mod tests;
