//! Looks: how the transcript draws calls, notes, content, asides and text.

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Duration;

use bevy_app::App;
use bevy_ecs::prelude::*;
use bevy_ecs::system::SystemParam;
use bevy_log::warn;
use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};
use rig_core::completion::{AssistantContent, Message};
use rig_core::message::{Audio, Document, Image, ToolCall, ToolResult, UserContent, Video};
use rig_core::tool::{ExitStatus, KeptEnd, OutputCut, ToolResultContext};
use rig_ecs::prelude::{Aside, AsideId, Author, AuthorKind, MessageId};
use rig_harness::rig_core::conversation::Note;

use super::{diff, markdown};
use crate::Theme;

/// A tool call and its result, if it has one yet, as a renderer sees them.
pub struct ToolCallView<'a> {
    /// The call. While it streams, its arguments are what has come of them,
    /// and its id is not the final one.
    pub call: &'a ToolCall,
    /// Its result, once it is in the conversation.
    pub result: Option<&'a ToolResult>,
    /// Where the call stands.
    pub state: CallState,
    /// The transcript's colors.
    pub theme: Theme,
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
    /// Its answer, once it has one: a tool's result, a model's reply, or
    /// why it failed, whole.
    pub answer: Option<String>,
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
    /// `call`, answered by `result` if it was, standing as `state`, with
    /// no facts or output known, in the default colors.
    pub fn new(call: &'a ToolCall, result: Option<&'a ToolResult>, state: CallState) -> Self {
        Self {
            call,
            result,
            state,
            theme: Theme::default(),
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

    /// Whether the call failed: the tool answered with an error, or the
    /// process it ran did not exit with 0 ([`ExitStatus`]).
    pub fn failed(&self) -> bool {
        self.result.is_some_and(|result| result.is_error)
            || self.exit().is_some_and(|exit| !exit.success())
    }

    /// How the process the call ran ended, if the tool said.
    fn exit(&self) -> Option<ExitStatus> {
        self.facts?.get::<ExitStatus>().ok().flatten()
    }

    /// The header line: `title` in bold and `detail` dimmed on a band
    /// tinted by where the call stands ([`Theme`]'s `call_*` colors), with
    /// the state in words too when it is running, held or failed (how its
    /// process ended, when it ran one).
    pub fn header(&self, title: impl Into<String>, detail: impl Into<String>) -> Line<'static> {
        let detail = detail.into();
        let detail = (!detail.is_empty()).then(|| Span::from(detail).dim());
        self.header_spans(title, detail.into_iter().collect())
    }

    /// [`header`](Self::header) with `detail` in spans styled by the look,
    /// such as a highlighted command; the state's words still end it.
    pub fn header_spans(
        &self,
        title: impl Into<String>,
        detail: Vec<Span<'static>>,
    ) -> Line<'static> {
        let failure = || match self.exit() {
            Some(ExitStatus::Exited(code)) if code != 0 => format!("exit {code}"),
            Some(ExitStatus::Signaled) => "killed by a signal".to_owned(),
            Some(ExitStatus::TimedOut) => "timed out".to_owned(),
            Some(ExitStatus::Killed) => "killed".to_owned(),
            _ => "failed".to_owned(),
        };
        let (band, state) = match (self.result, &self.state) {
            (Some(_), _) if self.failed() => (self.theme.call_failed, failure()),
            (_, CallState::Held(holds)) => (
                self.theme.call_waiting,
                format!("held: {}", holds.join(", ")),
            ),
            (Some(_), _) => (self.theme.call_done, String::new()),
            (None, CallState::Running(_)) => (self.theme.call_running, "running".to_owned()),
            (None, _) => (self.theme.call_waiting, String::new()),
        };
        let mut spans = vec![Span::from(" "), Span::from(title.into()).bold()];
        if !detail.is_empty() {
            spans.push(Span::from(" "));
            spans.extend(detail);
        }
        if !state.is_empty() {
            spans.push(Span::from(format!(" · {state}")).italic());
        }
        Line::from(spans).bg(band)
    }

    /// The whole result, indented under the header, red for an error. A unified diff is
    /// coloured. The line that says the tool cut its output
    /// ([`OutputCut`]), naming the file with all of it, is on its side.
    /// Before the result, a running call's progress.
    pub fn result_lines(&self) -> Vec<Line<'static>> {
        let Some(text) = self.result_text() else {
            return self.progress_lines();
        };
        if !self.failed()
            && let Some(lines) = diff::unified_lines(&text, &self.theme)
        {
            return lines;
        }
        let (output, marker) = self.cut_marker(&text);
        result_rows(output.into_iter(), marker, result_style(self.failed()))
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
                // The output's ends are kept: the marker is its first line.
                KeptEnd::Ends if lines.is_empty() => None,
                KeptEnd::Ends => Some(lines.remove(0)),
                KeptEnd::Start => lines.pop(),
            };
            Some((line?, cut.kept))
        });
        (lines, marker)
    }

    /// A running call's progress, indented under the header, where its
    /// result goes once
    /// it answers; none without one.
    pub fn progress_lines(&self) -> Vec<Line<'static>> {
        let CallState::Running(Some(progress)) = &self.state else {
            return Vec::new();
        };
        result_rows(progress.lines(), None, Style::new().yellow())
    }

    /// The look of a tool with no renderer: the name, the arguments and
    /// the result.
    pub fn default_lines(&self) -> Vec<Line<'static>> {
        let arguments = serde_json::Value::Object(self.call.function.arguments.clone()).to_string();
        let mut lines = vec![self.header(self.name().to_owned(), arguments)];
        lines.extend(self.result_lines());
        lines
    }
}

/// A note or a delivered text, as an entry renderer sees it: a plugin's
/// [`Note`](rig_harness::rig_core::conversation::Note), or a text of an entry of the
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
}

impl EntryView<'_> {
    /// The look of a plugin with no renderer: `⤶` and the title, then
    /// the text.
    pub fn default_lines(&self) -> Vec<Line<'static>> {
        let mut lines = vec![Line::styled(
            format!("⤶ {}", self.title),
            Style::new().magenta().bold(),
        )];
        lines.extend(text_rows(self.text, Style::new().dim()));
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
}

impl ContentView<'_> {
    /// The look of a kind with no renderer: its name in brackets, such as
    /// `[image]`.
    pub fn default_lines(&self) -> Vec<Line<'static>> {
        let name = self.block.kind().name();
        vec![Line::from(format!("  [{name}]")).dim()]
    }
}

/// The name of the plugin `author` is, or empty for any other author.
pub(crate) fn plugin_name(author: &Author) -> &str {
    match &author.kind {
        AuthorKind::Plugin(name) => name,
        _ => "",
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

/// The indent of a tool's output and of the text under an entry's title,
/// for a look that draws its own rows.
pub const OUTPUT_INDENT: &str = "   ";

/// The lines of `text` in `style`, each indented by [`OUTPUT_INDENT`].
pub fn text_rows(text: &str, style: Style) -> Vec<Line<'static>> {
    result_rows(text.lines(), None, style)
}

/// `lines` of a result in `style`, indented, with `marker`
/// before them when the output's ends were kept and after them when its
/// start was.
fn result_rows<'t>(
    lines: impl Iterator<Item = &'t str>,
    marker: Option<(&'t str, KeptEnd)>,
    style: Style,
) -> Vec<Line<'static>> {
    let (before, after) = match marker {
        Some((marker, KeptEnd::Ends)) => (Some(marker), None),
        Some((marker, KeptEnd::Start)) => (None, Some(marker)),
        None => (None, None),
    };
    let rows = before.into_iter().chain(lines).chain(after);
    rows.map(|line| {
        let line = line.replace('\t', "    ");
        Line::styled(format!("{OUTPUT_INDENT}{line}"), style)
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

impl ToolRenderer {
    /// The look `render` of the tool `tool`.
    pub fn new(
        tool: &str,
        render: impl Fn(&ToolCallView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> Self {
        let (tool, render) = (tool.to_owned(), Arc::new(render));
        Self { tool, render }
    }
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

impl EntryRenderer {
    /// The look `render` of the plugin `plugin`'s notes and texts.
    pub fn new(
        plugin: &str,
        render: impl Fn(&EntryView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> Self {
        let (plugin, render) = (plugin.to_owned(), Arc::new(render));
        Self { plugin, render }
    }
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

impl ContentRenderer {
    /// The look `render` of the blocks of `kind`.
    pub fn new(
        kind: ContentKind,
        render: impl Fn(&ContentView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> Self {
        let render = Arc::new(render);
        Self { kind, render }
    }
}

/// Draws an aside as transcript lines, not yet wrapped.
pub type RenderAside = dyn Fn(&AsideView<'_>) -> Vec<Line<'static>> + Send + Sync;

/// The look of the asides of a kind, on an entity of its own.
#[derive(Component, Clone)]
pub struct AsideRenderer {
    /// The kind it draws, such as `note`.
    pub kind: String,
    /// Draws one aside.
    pub render: Arc<RenderAside>,
}

impl AsideRenderer {
    /// The look `render` of the asides of `kind`.
    pub fn new(
        kind: &str,
        render: impl Fn(&AsideView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> Self {
        let (kind, render) = (kind.to_owned(), Arc::new(render));
        Self { kind, render }
    }
}

/// The decorators that drew lines under a call or a message, each by name
/// with the lines it drew among its own.
pub(crate) type Decorated = Vec<(String, std::ops::Range<usize>)>;

/// Lines drawn under every tool call, after its look, on an entity of
/// its own.
#[derive(Component, Clone)]
pub struct CallDecorator {
    /// What it is called, such as the plugin's name.
    pub name: String,
    /// Draws its lines under a call; none leaves the call as it is.
    pub render: Arc<RenderToolCall>,
}

impl CallDecorator {
    /// The decorator `name` drawing `render`'s lines.
    pub fn new(
        name: &str,
        render: impl Fn(&ToolCallView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> Self {
        let (name, render) = (name.to_owned(), Arc::new(render));
        Self { name, render }
    }
}

/// Draws lines under a message.
pub type DecorateMessage = dyn Fn(&DecoratedMessage<'_>) -> Vec<Line<'static>> + Send + Sync;

/// Lines drawn under every message the user or the model wrote, after its
/// own, on an entity of its own.
#[derive(Component, Clone)]
pub struct MessageDecorator {
    /// What it is called, such as the plugin's name.
    pub name: String,
    /// Draws its lines under a message; none leaves it as it is.
    pub render: Arc<DecorateMessage>,
}

impl MessageDecorator {
    /// The decorator `name` drawing `render`'s lines.
    pub fn new(
        name: &str,
        render: impl Fn(&DecoratedMessage<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> Self {
        let (name, render) = (name.to_owned(), Arc::new(render));
        Self { name, render }
    }
}

/// An aside of a conversation, as an aside renderer sees it.
pub struct AsideView<'a> {
    /// Its id.
    pub id: AsideId,
    /// The aside.
    pub aside: &'a Aside,
}

/// Whose text a [`MessageView`] is.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum Role {
    /// What the user typed.
    User,
    /// A text block of a reply, or the reply streaming in.
    Assistant,
    /// A reply's reasoning.
    Reasoning,
}

impl Role {
    /// Its name: `user`, `assistant` or `reasoning`.
    pub fn name(self) -> &'static str {
        match self {
            Self::User => "user",
            Self::Assistant => "assistant",
            Self::Reasoning => "reasoning",
        }
    }
}

/// A text of a message, as a message renderer sees it.
pub struct MessageView<'a> {
    /// Whose text it is.
    pub role: Role,
    /// The message it is in; `None` for the reply streaming in.
    pub message: Option<MessageId>,
    /// The text's place among the message's content.
    pub block: usize,
    /// The text.
    pub text: &'a str,
    /// The labels drawn inside it ([`TextMark`](crate::TextMark)), each
    /// before its byte.
    pub marks: &'a [(usize, Span<'static>)],
    /// The width its lines are wrapped to, which a reply's tables are fit
    /// to.
    pub width: usize,
}

impl MessageView<'_> {
    /// The default look: the user's text after `›`, a reply's as markdown
    /// with its marks, reasoning dimmed.
    pub fn default_lines(&self) -> Vec<Line<'static>> {
        match self.role {
            Role::User => {
                let lines = self.text.lines().enumerate();
                let rows = lines.map(|(index, line)| {
                    let prefix = if index == 0 { "› " } else { "  " };
                    let text = line.replace('\t', "    ");
                    Line::from(vec![Span::from(prefix).dim(), Span::from(text).bold()])
                });
                rows.collect()
            }
            Role::Assistant => markdown::render_marked(self.text, self.marks, Some(self.width)),
            Role::Reasoning => {
                let lines = self.text.lines();
                let style = Style::new().dim().italic();
                let rows = lines.map(|line| Line::styled(line.replace('\t', "    "), style));
                rows.collect()
            }
        }
    }
}

/// A message the user or the model wrote, as a message decorator sees it.
pub struct DecoratedMessage<'a> {
    /// The message.
    pub message: &'a Message,
    /// Its id, when it has one.
    pub id: Option<MessageId>,
    /// Its author, when it has one.
    pub author: Option<&'a Author>,
}

/// Registers transcript looks: how tool calls, notes, content blocks,
/// asides and message text are drawn, and lines under every call or
/// message.
///
/// Each look is a component on an entity of its own, named by what it
/// draws: `renderer:<tool>`, `entry:<plugin>`, `content:<kind>`,
/// `aside:<kind>`, `message:<role>`, `decorator:<name>`,
/// `message-decorator:<name>`. The first look of a name stays (a later one
/// is refused with a warning); `Disabled` on it, in `Plugin::finish`, frees
/// it. A look spawned as a child of a [`TranscriptPane`](crate::TranscriptPane)
/// applies in that pane only, before the global one. A change to a look
/// draws the transcript again. Without a look, a call draws as
/// [`ToolCallView::default_lines`], also while its arguments stream
/// ([`CallState::Streaming`]). A renderer sees the calls its call started,
/// through any effect in between, as [`ToolCallView::children`]:
///
/// ```no_run
/// use rig_harness::prelude::*;
/// use rig_tui::AppToolRenderersExt;
/// use rig_tui::ratatui::text::Line;
///
/// fn build(app: &mut App) {
///     app.add_tool_renderer("consult", |view| {
///         let mut lines = view.default_lines();
///         for child in view.children {
///             let indent = "  ".repeat(child.depth + 2);
///             lines.push(Line::from(format!("{indent}↳ {} {:?}", child.name, child.state)));
///         }
///         lines
///     });
/// }
/// ```
pub trait AppToolRenderersExt {
    /// Draw the calls of `tool` with `render`.
    fn add_tool_renderer(
        &mut self,
        tool: &str,
        render: impl Fn(&ToolCallView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self;

    /// Draw the notes and the entries the plugin `plugin` wrote, such as a
    /// compaction's summary, with `render`; without one they draw as
    /// [`EntryView::default_lines`].
    fn add_entry_renderer(
        &mut self,
        plugin: &str,
        render: impl Fn(&EntryView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self;

    /// Draw the content blocks of `kind` (an image, audio, a video or a
    /// document) with `render`; without one they draw as
    /// [`ContentView::default_lines`].
    fn add_content_renderer(
        &mut self,
        kind: ContentKind,
        render: impl Fn(&ContentView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self;

    /// Draw the asides of `kind` with `render` after the entry they
    /// follow, as the transcript item
    /// [`TranscriptItem::Aside`](crate::TranscriptItem::Aside). An aside of a
    /// kind with no renderer is not drawn; a note (`note`) is drawn by its
    /// plugin's entry renderer.
    fn add_aside_renderer(
        &mut self,
        kind: &str,
        render: impl Fn(&AsideView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self;

    /// Draw `render`'s lines under every tool call, after the call's own
    /// look, whichever draws it. Decorators add up, in the order of their
    /// entities. Its rows under a call are the transcript item
    /// [`TranscriptItem::Decoration`](crate::TranscriptItem::Decoration), by
    /// the call's id and `name`.
    fn add_call_decorator(
        &mut self,
        name: &str,
        render: impl Fn(&ToolCallView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self;
}

/// Spawns the global look `look` named `name`, unless a global one of the
/// same key (`same`) is there: a disabled one, which a query skips, frees
/// its key.
fn add_look<T: Component>(app: &mut App, name: String, look: T, same: impl Fn(&T) -> bool) {
    let world = app.world_mut();
    let mut looks = world.query_filtered::<&T, Without<ChildOf>>();
    if looks.iter(world).any(same) {
        warn!("look not added: `{name}` has one; insert `Disabled` on it to replace it");
        return;
    }
    world.spawn((Name::new(name), look));
}

impl AppToolRenderersExt for App {
    fn add_tool_renderer(
        &mut self,
        tool: &str,
        render: impl Fn(&ToolCallView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self {
        let look = ToolRenderer::new(tool, render);
        add_look(self, format!("renderer:{tool}"), look, |other| {
            other.tool == tool
        });
        self
    }

    fn add_entry_renderer(
        &mut self,
        plugin: &str,
        render: impl Fn(&EntryView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self {
        let look = EntryRenderer::new(plugin, render);
        add_look(self, format!("entry:{plugin}"), look, |other| {
            other.plugin == plugin
        });
        self
    }

    fn add_content_renderer(
        &mut self,
        kind: ContentKind,
        render: impl Fn(&ContentView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self {
        let look = ContentRenderer::new(kind, render);
        let name = format!("content:{}", kind.name());
        add_look(self, name, look, |other| other.kind == kind);
        self
    }

    fn add_aside_renderer(
        &mut self,
        kind: &str,
        render: impl Fn(&AsideView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self {
        let look = AsideRenderer::new(kind, render);
        add_look(self, format!("aside:{kind}"), look, |other| {
            other.kind == kind
        });
        self
    }

    fn add_call_decorator(
        &mut self,
        name: &str,
        render: impl Fn(&ToolCallView<'_>) -> Vec<Line<'static>> + Send + Sync + 'static,
    ) -> &mut Self {
        let look = CallDecorator::new(name, render);
        add_look(self, format!("decorator:{name}"), look, |other| {
            other.name == name
        });
        self
    }
}

/// A look as [`Looks`] reads it: changed since the last frame or not, and
/// the pane it applies in, if only one.
type Scoped<T> = (Entity, Ref<'static, T>, Option<&'static ChildOf>);

/// The looks plugins added, as a frame draws with them.
#[derive(SystemParam)]
pub(crate) struct Looks<'w, 's> {
    tools: Query<'w, 's, Scoped<ToolRenderer>>,
    entries: Query<'w, 's, Scoped<EntryRenderer>>,
    contents: Query<'w, 's, Scoped<ContentRenderer>>,
    asides: Query<'w, 's, Scoped<AsideRenderer>>,
    decorators: Query<'w, 's, Scoped<CallDecorator>>,
    message_decorators: Query<'w, 's, Scoped<MessageDecorator>>,
}

/// The looks of `looks` that apply in `pane` (`None`: in no pane of its
/// own, such as a layout for an export), by key: those of the pane first,
/// then the global ones, each in entity order.
fn scoped<'a, T: Component, K, V>(
    looks: impl Iterator<Item = (Entity, Ref<'a, T>, Option<&'a ChildOf>)>,
    pane: Option<Entity>,
    key: impl Fn(&'a T) -> (K, V),
) -> Vec<(K, V)> {
    let mut found: Vec<(bool, Entity, &'a T)> = looks
        .filter_map(|(entity, look, parent)| match parent.map(ChildOf::parent) {
            None => Some((true, entity, look.into_inner())),
            Some(of) if Some(of) == pane => Some((false, entity, look.into_inner())),
            Some(_) => None,
        })
        .collect();
    found.sort_by_key(|(global, entity, _)| (*global, *entity));
    found.into_iter().map(|(.., look)| key(look)).collect()
}

/// `looks` by key, the first of each key kept.
fn first_of<K: Eq + std::hash::Hash, V>(looks: Vec<(K, V)>) -> HashMap<K, V> {
    let mut map = HashMap::new();
    for (key, look) in looks {
        map.entry(key).or_insert(look);
    }
    map
}

impl Looks<'_, '_> {
    /// Each look that applies in `pane` by what it draws: a pane's own
    /// before the global one of the same key, and its decorators before the
    /// global ones.
    pub(crate) fn by_kind(&self, pane: Option<Entity>) -> Renderers<'_> {
        let tools = scoped(self.tools.iter(), pane, |look| {
            (look.tool.as_str(), &look.render)
        });
        let entries = scoped(self.entries.iter(), pane, |look| {
            (look.plugin.as_str(), &look.render)
        });
        let contents = scoped(self.contents.iter(), pane, |look| (look.kind, &look.render));
        let asides = scoped(self.asides.iter(), pane, |look| {
            (look.kind.as_str(), &look.render)
        });
        let decorators = scoped(self.decorators.iter(), pane, |look| {
            (look.name.as_str(), &look.render)
        });
        let message_decorators = scoped(self.message_decorators.iter(), pane, |look| {
            (look.name.as_str(), &look.render)
        });
        Renderers {
            tools: first_of(tools),
            entries: first_of(entries),
            contents: first_of(contents),
            asides: first_of(asides),
            decorators,
            message_decorators,
        }
    }

    /// Whether a look changed or was added since the last frame.
    pub(crate) fn changed(&self) -> bool {
        self.tools.iter().any(|(_, look, _)| look.is_changed())
            || self.entries.iter().any(|(_, look, _)| look.is_changed())
            || self.contents.iter().any(|(_, look, _)| look.is_changed())
            || self.asides.iter().any(|(_, look, _)| look.is_changed())
            || self.decorators.iter().any(|(_, look, _)| look.is_changed())
            || (self.message_decorators.iter()).any(|(_, look, _)| look.is_changed())
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
    /// The aside renderers by kind.
    pub(crate) asides: HashMap<&'a str, &'a Arc<RenderAside>>,
    /// The call decorators, by name, in order.
    pub(crate) decorators: Vec<(&'a str, &'a Arc<RenderToolCall>)>,
    /// The message decorators, by name, in order.
    pub(crate) message_decorators: Vec<(&'a str, &'a Arc<DecorateMessage>)>,
}

impl Renderers<'_> {
    /// A tool call's lines, by its tool's renderer or the default look,
    /// then every decorator's.
    pub(crate) fn call(&self, view: &ToolCallView<'_>) -> Vec<Line<'static>> {
        self.call_decorated(view).0
    }

    /// A tool call's lines, as [`call`](Self::call) draws them, and each
    /// decorator that drew any with the lines it drew.
    pub(crate) fn call_decorated(
        &self,
        view: &ToolCallView<'_>,
    ) -> (Vec<Line<'static>>, Decorated) {
        let mut lines = match self.tools.get(view.name()) {
            Some(render) => render(view),
            None => view.default_lines(),
        };
        let mut decorated = Vec::new();
        for (name, decorate) in &self.decorators {
            let start = lines.len();
            lines.extend(decorate(view));
            if lines.len() > start {
                decorated.push(((*name).to_owned(), start..lines.len()));
            }
        }
        (lines, decorated)
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

    /// An aside's lines: a note's by its plugin's entry renderer, any
    /// other's by its kind's renderer; `None` for a kind nothing draws.
    pub(crate) fn aside(&self, view: &AsideView<'_>) -> Option<Vec<Line<'static>>> {
        if let Some(note) = Note::of(view.aside) {
            let entry = EntryView {
                plugin: plugin_name(&view.aside.author),
                title: &note.title,
                text: &note.text,
            };
            return Some(self.entry(&entry));
        }
        let render = self.asides.get(view.aside.kind.as_str())?;
        Some(render(view))
    }

    /// Whether an aside of `aside`'s kind is drawn.
    pub(crate) fn draws(&self, aside: &Aside) -> bool {
        aside.kind == Note::KIND || self.asides.contains_key(aside.kind.as_str())
    }

    /// Every message decorator's lines under `message`, and each that drew
    /// any with the lines it drew.
    pub(crate) fn message_decorations(
        &self,
        message: &DecoratedMessage<'_>,
    ) -> (Vec<Line<'static>>, Decorated) {
        let (mut lines, mut decorated) = (Vec::new(), Vec::new());
        for (name, decorate) in &self.message_decorators {
            let start = lines.len();
            lines.extend(decorate(message));
            if lines.len() > start {
                decorated.push(((*name).to_owned(), start..lines.len()));
            }
        }
        (lines, decorated)
    }
}

#[cfg(test)]
mod tests;
