//! Looks: how the transcript draws calls, notes, content, asides and text.

use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::collections::HashSet;
use std::time::Duration;

use bevy_app::App;
use bevy_ecs::prelude::*;
use bevy_ecs::schedule::ScheduleLabel;
use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};
use rig_core::completion::{AssistantContent, Message};
use rig_core::message::{
    Audio, Document, Image, ToolCall, ToolFunction, ToolResult, UserContent, Video,
};
use rig_core::tool::{ExitStatus, KeptEnd, OutputCut, ToolResultContext};
use rig_ecs::prelude::{Aside, AsideId, Author, AuthorKind, MessageId};
use rig_harness::rig_core::conversation::Note;

use super::{diff, markdown};
use crate::Theme;
use crate::highlight::Code;

/// A tool call and its result, if it has one yet, as a render system sees
/// them through [`ToolRender::view`].
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
    /// `FilesChanged`, as the session keeps it with the call's result:
    /// once the result is in the conversation, in this run or an earlier
    /// one alike.
    pub facts: Option<&'a ToolResultContext>,
    /// How long it ran, once it settled: in this run, or in an earlier one
    /// of a resumed session whose effect log was kept.
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
/// started, as a render system sees it ([`ToolCallView::children`]).
#[derive(Clone, Debug, PartialEq)]
pub struct ChildCall {
    /// How far below the call it is: 0 for one the call started itself.
    pub depth: usize,
    /// The tool and arguments it ran with, or `None` for a model call.
    pub call: Option<ToolFunction>,
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
/// effect is gone, is `Stopped` when a stop answered it, else `Done` when
/// it has a result and `Running` when not.
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
    /// Answered by a stop that cut its turn off.
    Stopped,
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
    /// process it ran did not exit with 0 ([`ExitStatus`]). A stopped
    /// call's result is an error too, which a look draws as its result.
    pub fn failed(&self) -> bool {
        self.result.is_some_and(|result| result.is_error)
            || self.exit().is_some_and(|exit| !exit.success())
    }

    /// Whether a stop answered the call ([`CallState::Stopped`]).
    pub fn stopped(&self) -> bool {
        self.state == CallState::Stopped
    }

    /// How the process the call ran ended, if the tool said.
    fn exit(&self) -> Option<ExitStatus> {
        self.facts?.get::<ExitStatus>().ok().flatten()
    }

    /// The header line: `title` in bold and `detail` dimmed on a band
    /// tinted by where the call stands ([`Theme`]'s `call_*` colors), with
    /// the state in words too when it is running, held, stopped or failed
    /// (how its process ended, when it ran one).
    pub fn header(&self, title: impl Into<String>, detail: impl Into<String>) -> Line<'static> {
        let detail = detail.into();
        let detail = (!detail.is_empty()).then(|| Span::from(detail).dim());
        self.header_at(title, detail.into_iter().collect()).0
    }

    /// [`header`](Self::header) with line `index` of `code` as its detail,
    /// such as a command, colored when drawn.
    pub fn header_code(&self, title: impl Into<String>, code: &Code, index: usize) -> LookLine {
        let detail = code.line(index, Vec::new());
        let (line, at) = self.header_at(title, detail.line.spans);
        let code = detail.code.map(|code| CodeLine { at, ..code });
        LookLine { line, code }
    }

    /// The header line with `detail` in spans, the state's words after
    /// them, and the byte of its text where `detail` starts.
    fn header_at(
        &self,
        title: impl Into<String>,
        detail: Vec<Span<'static>>,
    ) -> (Line<'static>, usize) {
        let failure = || match self.exit() {
            Some(ExitStatus::Exited(code)) if code != 0 => format!("exit {code}"),
            Some(ExitStatus::Signaled) => "killed by a signal".to_owned(),
            Some(ExitStatus::TimedOut) => "timed out".to_owned(),
            Some(ExitStatus::Killed) => "killed".to_owned(),
            _ => "failed".to_owned(),
        };
        // Only a call of a reply the stop cut off has no result: one whose
        // arguments were still streaming keeps them as they came.
        let cut = match self.call.function.invalid_arguments {
            Some(_) => "stopped while it streamed; nothing ran",
            None => "stopped before it ran",
        };
        let (band, state) = match (self.result, &self.state) {
            (None, CallState::Stopped) => (self.theme.call_waiting, cut.to_owned()),
            (_, CallState::Stopped) => (self.theme.call_waiting, "stopped".to_owned()),
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
        }
        let at = spans.iter().map(|span| span.content.len()).sum();
        spans.extend(detail);
        if !state.is_empty() {
            spans.push(Span::from(format!(" · {state}")).italic());
        }
        (Line::from(spans).bg(band), at)
    }

    /// The whole result, indented under the header, red for an error but
    /// a stop's. A unified diff is
    /// coloured. The line that says the tool cut its output
    /// ([`OutputCut`]), naming the file with all of it, is on its side.
    /// Before the result, a running call's progress.
    pub fn result_lines(&self) -> Vec<LookLine> {
        let Some(text) = self.result_text() else {
            return self
                .progress_lines()
                .into_iter()
                .map(LookLine::from)
                .collect();
        };
        if !self.failed()
            && let Some(lines) = diff::unified_lines(&text, &self.theme)
        {
            return lines;
        }
        let (output, marker) = self.cut_marker(&text);
        let failed = self.failed() && !self.stopped();
        let rows = result_rows(output.into_iter(), marker, result_style(failed));
        rows.into_iter().map(LookLine::from).collect()
    }

    /// The lines of `text`, the result, without the line that says the
    /// tool cut its output, and that line with the end it is on, when the
    /// call published [`OutputCut`] (a call without it, such as one saved
    /// before calls kept their facts, has its marker as a line like any
    /// other).
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

    /// The look of a tool no render system draws: the name, the arguments and
    /// the result.
    pub fn default_lines(&self) -> Vec<LookLine> {
        let arguments = serde_json::Value::Object(self.call.function.arguments.clone()).to_string();
        let mut lines = vec![self.header(self.name().to_owned(), arguments).into()];
        lines.extend(self.result_lines());
        lines
    }
}

/// A note or a delivered text, as a render system sees it: a plugin's
/// [`Note`](rig_harness::rig_core::conversation::Note), or a text of an entry of the
/// conversation another author wrote, such as a compaction summary.
pub struct EntryView<'a> {
    /// The conversation entry, when this is delivered text.
    pub id: Option<MessageId>,
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
    /// The look of a plugin's entry no render system draws: `⤶` and the title, then
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

/// A kind of non-text content block.
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

/// A content block of a message, as a render system sees it: an
/// image, audio, a video or a document, the user's or the model's. Its
/// rows are the transcript item [`TranscriptItem::Content`], which a
/// plugin finds in the [`DrawnTranscript`], such as to draw an image over
/// the rows its render system left blank.
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
    /// The look of a kind no render system draws: its name in brackets, such as
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

/// A line a look draws: styled text, from a [`Line`], or a line of
/// [`Code`] ([`Code::line`], [`ToolCallView::header_code`]) laid out plain
/// and colored when the line is drawn.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct LookLine {
    /// The line as laid out, any code in it plain.
    pub(crate) line: Line<'static>,
    /// The line of code in it, if any.
    pub(crate) code: Option<CodeLine>,
}

/// Where a line of [`Code`] is in a [`LookLine`].
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct CodeLine {
    pub(crate) code: Code,
    /// The line's place in it.
    pub(crate) index: usize,
    /// The byte of the look line's text where the code line's text
    /// ([`Code::line`]) starts.
    pub(crate) at: usize,
}

impl From<Line<'static>> for LookLine {
    fn from(line: Line<'static>) -> Self {
        Self { line, code: None }
    }
}

/// Its text, as drawn.
impl std::fmt::Display for LookLine {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.line.fmt(formatter)
    }
}

impl LookLine {
    /// The line as drawn: its code's text in the code's colors, each
    /// colored piece with the style of the laid-out text there laid over
    /// it, such as a diff's shade on the words a line changed. The text is
    /// the laid-out text, and plain if the code has no colors for it.
    pub fn colored(&self) -> Line<'static> {
        let Some(CodeLine { code, index, at }) = &self.code else {
            return self.line.clone();
        };
        let Some(colors) = code.spans(*index) else {
            return self.line.clone();
        };
        let laid: String = self.line.spans.iter().map(|span| &*span.content).collect();
        let text: String = colors.iter().map(|span| &*span.content).collect();
        if laid.get(*at..*at + text.len()) != Some(text.as_str()) {
            return self.line.clone();
        }
        let mut runs = Vec::with_capacity(colors.len());
        let mut end = *at;
        for span in colors {
            let start = end;
            end += span.content.len();
            runs.push((start..end, span.style));
        }
        // The laid-out spans and the colors' runs both run end to end over
        // the line, so one pass cuts each span where a run starts or ends.
        let mut runs = runs.into_iter().peekable();
        let mut spans = Vec::new();
        let mut start = 0;
        for span in &self.line.spans {
            let end = start + span.content.len();
            let mut from = start;
            while from < end {
                while runs.next_if(|(run, _)| run.end <= from).is_some() {}
                let (to, style) = match runs.peek() {
                    Some((run, color)) if run.start <= from => {
                        (run.end.min(end), color.patch(span.style))
                    }
                    Some((run, _)) => (run.start.min(end), span.style),
                    None => (end, span.style),
                };
                if let Some(text) = span.content.get(from - start..to - start) {
                    spans.push(Span::styled(text.to_owned(), style));
                }
                from = to;
            }
            start = end;
        }
        Line {
            spans,
            style: self.line.style,
            alignment: self.line.alignment,
        }
    }

    /// The line as laid out, any code in it plain: the text it draws.
    pub fn line(&self) -> &Line<'static> {
        &self.line
    }

    /// The line with `style` laid over its own, as [`Line::patch_style`].
    pub fn patch_style(mut self, style: Style) -> Self {
        self.line = self.line.patch_style(style);
        self
    }
}

/// Named line ranges within a rendered item.
pub(crate) type Decorated = Vec<(String, std::ops::Range<usize>)>;

/// An aside of a conversation, as a render system sees it.
pub struct AsideView<'a> {
    /// Its id.
    pub id: AsideId,
    /// The aside.
    pub aside: &'a Aside,
    /// The colors it is drawn in.
    pub theme: Theme,
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

/// A text of a message, as the transcript draws it.
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
    /// The colors a reply's code, alerts and tables are drawn in.
    pub theme: Theme,
}

impl MessageView<'_> {
    /// The default look: the user's text after `›`, a reply's as markdown
    /// with its marks, reasoning as dimmed markdown.
    pub fn default_lines(&self) -> Vec<LookLine> {
        match self.role {
            Role::User => {
                let lines = self.text.lines().enumerate();
                let rows = lines.map(|(index, line)| {
                    let prefix = if index == 0 { "› " } else { "  " };
                    let text = line.replace('\t', "    ");
                    Line::from(vec![Span::from(prefix).dim(), Span::from(text).bold()]).into()
                });
                rows.collect()
            }
            Role::Assistant => {
                markdown::render_marked(self.text, self.marks, &self.theme, Some(self.width))
            }
            Role::Reasoning => {
                let lines = markdown::render(self.text, &self.theme, Some(self.width));
                let style = Style::new().dim().italic();
                let lines = lines.into_iter();
                lines.map(|line| line.patch_style(style)).collect()
            }
        }
    }
}

/// A message the user or the model wrote, as a render system decorating it
/// sees it.
pub struct DecoratedMessage<'a> {
    /// The message.
    pub message: &'a Message,
    /// Its id, when it has one.
    pub id: Option<MessageId>,
    /// Its author, when it has one.
    pub author: Option<&'a Author>,
}

/// Runs presentation systems for the changed items of one pane or export.
/// Systems may read agent data but write only presentation data. Advance
/// [`RenderRevision`] when anything else a render system reads changes.
#[derive(ScheduleLabel, Clone, Debug, PartialEq, Eq, Hash)]
pub struct TranscriptRender;

/// Ordered phases of [`TranscriptRender`]. Order competing systems within a
/// phase using their own public sets.
#[derive(SystemSet, Clone, Debug, PartialEq, Eq, Hash)]
pub enum RenderSystems {
    /// Replace an item's default lines.
    Render,
    /// Add named decorations after its lines.
    Decorate,
    /// Edit the completed presentation.
    Edit,
}

/// Increment to invalidate cached presentation after state a render system
/// reads outside its [`RenderBatch`] inputs changes, or after render systems
/// were added or removed. Advancing it also requests a terminal frame.
#[derive(Resource, Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct RenderRevision(pub u64);

/// Lines of an item and named contributions after them. Contributions are
/// ordered and names identify row ranges. An item with no lines and no
/// contributions is hidden; contributions are drawn even when `lines` is
/// empty. A message's output only adds to it ([`RenderBatch::messages_mut`]).
#[derive(Clone, Debug, Default, PartialEq)]
pub struct RenderOutput {
    /// The item's unwrapped lines.
    pub lines: Vec<LookLine>,
    /// Named lines appended after the item's lines.
    pub decorations: Vec<(String, Vec<Line<'static>>)>,
}

impl RenderOutput {
    /// Append a named contribution. Systems may edit or remove contributions
    /// directly before the layout consumes them. Reusing a name replaces its
    /// lines in place.
    pub fn decorate(&mut self, name: impl Into<String>, lines: Vec<Line<'static>>) {
        let name = name.into();
        if let Some((_, current)) = self.decorations.iter_mut().find(|(key, _)| *key == name) {
            *current = lines;
        } else {
            self.decorations.push((name, lines));
        }
    }

    fn drawn(&self) -> (Vec<LookLine>, Decorated) {
        let mut lines = self.lines.clone();
        let mut ranges = Vec::new();
        for (name, more) in &self.decorations {
            let start = lines.len();
            lines.extend(more.iter().cloned().map(LookLine::from));
            if start != lines.len() {
                ranges.push((name.clone(), start..lines.len()));
            }
        }
        (lines, ranges)
    }
}

/// Owned tool input, including synthetic streaming calls and saved calls.
#[derive(Clone, PartialEq)]
pub struct ToolRender {
    /// The tool call.
    pub call: ToolCall,
    /// Its result, when available.
    pub result: Option<ToolResult>,
    /// Current execution state.
    pub state: CallState,
    /// Presentation colors.
    pub theme: Theme,
    /// Published tool facts.
    pub facts: Option<ToolResultContext>,
    /// Execution duration.
    pub took: Option<Duration>,
    /// Live output.
    pub output: Option<String>,
    /// Descendant calls.
    pub children: Vec<ChildCall>,
}

impl ToolRender {
    /// Borrow this input for the formatting helpers.
    pub fn view(&self) -> ToolCallView<'_> {
        ToolCallView {
            call: &self.call,
            result: self.result.as_ref(),
            state: self.state.clone(),
            theme: self.theme,
            facts: self.facts.as_ref(),
            took: self.took,
            output: self.output.as_deref(),
            children: &self.children,
        }
    }
}

/// Owned note or delivered text.
#[derive(Clone, PartialEq)]
pub struct EntryRender {
    /// The conversation entry, when this is delivered text.
    pub id: Option<MessageId>,
    /// Author plugin.
    pub plugin: String,
    /// Display title.
    pub title: String,
    /// Body text.
    pub text: String,
}

impl EntryRender {
    /// Borrow this input for the formatting helpers.
    pub fn view(&self) -> EntryView<'_> {
        EntryView {
            id: self.id,
            plugin: &self.plugin,
            title: &self.title,
            text: &self.text,
        }
    }
}

/// An owned non-text content block.
#[derive(Clone, PartialEq)]
pub enum RenderContent {
    /// An image.
    Image(Image),
    /// Audio.
    Audio(Audio),
    /// Video.
    Video(Video),
    /// A document.
    Document(Document),
}

impl RenderContent {
    /// Borrow the content block.
    pub fn block(&self) -> ContentBlock<'_> {
        match self {
            Self::Image(v) => ContentBlock::Image(v),
            Self::Audio(v) => ContentBlock::Audio(v),
            Self::Video(v) => ContentBlock::Video(v),
            Self::Document(v) => ContentBlock::Document(v),
        }
    }
}

/// Owned content input.
#[derive(Clone, PartialEq)]
pub struct ContentRender {
    /// Message identity, when available.
    pub message: Option<MessageId>,
    /// Block index.
    pub index: usize,
    /// Content to draw.
    pub content: RenderContent,
    /// Whether the model wrote the message.
    pub assistant: bool,
}

impl ContentRender {
    /// Borrow this input for the formatting helpers.
    pub fn view(&self) -> ContentView<'_> {
        ContentView {
            message: self.message,
            index: self.index,
            block: self.content.block(),
            assistant: self.assistant,
        }
    }
}

/// Owned aside input.
#[derive(Clone, PartialEq)]
pub struct AsideRender {
    /// Aside identity.
    pub id: AsideId,
    /// Aside value and author.
    pub aside: Aside,
    /// Presentation colors.
    pub theme: Theme,
}

impl AsideRender {
    /// Borrow this input for the formatting helpers.
    pub fn view(&self) -> AsideView<'_> {
        AsideView {
            id: self.id,
            aside: &self.aside,
            theme: self.theme,
        }
    }
}

/// Owned message input for message decorations.
#[derive(Clone, PartialEq)]
pub struct MessageRender {
    /// Message value.
    pub message: Message,
    /// Message identity, when available.
    pub id: Option<MessageId>,
    /// Message author.
    pub author: Option<Author>,
}

impl MessageRender {
    /// Borrow this input for the formatting helpers.
    pub fn view(&self) -> DecoratedMessage<'_> {
        DecoratedMessage {
            message: &self.message,
            id: self.id,
            author: self.author.as_ref(),
        }
    }
}

/// The input of one render request.
#[derive(Clone, PartialEq)]
pub enum RenderInput {
    /// Tool call and its execution data.
    Tool(ToolRender),
    /// Note or delivered text.
    Entry(EntryRender),
    /// Non-text content.
    Content(ContentRender),
    /// Non-note aside.
    Aside(AsideRender),
    /// Message decorations.
    Message(MessageRender),
}

/// An item's identity within one batch. Message ordinals distinguish values
/// without persisted ids. These are presentation keys, not effect entities.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub enum RenderKey {
    /// Message ordinal and block index of a tool call.
    Tool(usize, usize),
    /// A call streaming in, by its position among the calls of the reply
    /// streaming in, as its id may be synthetic or repeated.
    Streaming(usize),
    /// An aside id, including notes.
    Aside(AsideId),
    /// Message ordinal and block index.
    Content(usize, usize),
    /// Message ordinal and delivered text block index.
    Entry(usize, usize),
    /// Message ordinal.
    Message(usize),
}

/// Public input and output of an item that needs rendering.
#[derive(Clone, PartialEq)]
pub struct RenderRequest {
    /// Owned input, independent of live effect entities.
    pub input: RenderInput,
    /// Mutable presentation.
    pub output: RenderOutput,
}

/// Changed items of one layout. Present during [`TranscriptRender`], including
/// pane-less exports. Typed iterators select requests without matching enums.
/// Cached messages are absent until their inputs or [`RenderRevision`] change.
#[derive(Resource, Default)]
pub struct RenderBatch {
    /// Pane being drawn; `None` for an export.
    pub pane: Option<Entity>,
    /// Live source agent; `None` for a saved conversation.
    pub agent: Option<Entity>,
    /// Available columns, specific to this pane or export.
    pub width: u16,
    /// Requests keyed within this source.
    pub requests: HashMap<RenderKey, RenderRequest>,
}

impl RenderBatch {
    /// Tool calls with the exact tool name.
    pub fn tools_mut<'a>(
        &'a mut self,
        name: &'a str,
    ) -> impl Iterator<Item = (&'a ToolRender, &'a mut RenderOutput)> {
        self.requests
            .values_mut()
            .filter_map(move |r| match &r.input {
                RenderInput::Tool(v) if v.call.function.name.as_str() == name => {
                    Some((v, &mut r.output))
                }
                _ => None,
            })
    }
    /// All tool calls, for decorations applying to every tool.
    pub fn calls_mut(&mut self) -> impl Iterator<Item = (&ToolRender, &mut RenderOutput)> {
        self.requests.values_mut().filter_map(|r| match &r.input {
            RenderInput::Tool(v) => Some((v, &mut r.output)),
            _ => None,
        })
    }
    /// Notes and delivered text by author plugin.
    pub fn entries_mut<'a>(
        &'a mut self,
        plugin: &'a str,
    ) -> impl Iterator<Item = (&'a EntryRender, &'a mut RenderOutput)> {
        self.requests
            .values_mut()
            .filter_map(move |r| match &r.input {
                RenderInput::Entry(v) if v.plugin == plugin => Some((v, &mut r.output)),
                _ => None,
            })
    }
    /// Content blocks of a kind.
    pub fn contents_mut(
        &mut self,
        kind: ContentKind,
    ) -> impl Iterator<Item = (&ContentRender, &mut RenderOutput)> {
        self.requests
            .values_mut()
            .filter_map(move |r| match &r.input {
                RenderInput::Content(v) if v.content.block().kind() == kind => {
                    Some((v, &mut r.output))
                }
                _ => None,
            })
    }
    /// Asides of a kind.
    pub fn asides_mut<'a>(
        &'a mut self,
        kind: &'a str,
    ) -> impl Iterator<Item = (&'a AsideRender, &'a mut RenderOutput)> {
        self.requests
            .values_mut()
            .filter_map(move |r| match &r.input {
                RenderInput::Aside(v) if v.aside.kind == kind => Some((v, &mut r.output)),
                _ => None,
            })
    }
    /// Messages that can receive decorations. A message request's `lines`
    /// start empty and are drawn after the message, as an unnamed
    /// decoration; they cannot hide the message.
    pub fn messages_mut(&mut self) -> impl Iterator<Item = (&MessageRender, &mut RenderOutput)> {
        self.requests.values_mut().filter_map(|r| match &r.input {
            RenderInput::Message(v) => Some((v, &mut r.output)),
            _ => None,
        })
    }
}

pub(crate) fn add(app: &mut App) {
    app.init_resource::<RenderRevision>().configure_sets(
        TranscriptRender,
        (
            RenderSystems::Render,
            RenderSystems::Decorate,
            RenderSystems::Edit,
        )
            .chain(),
    );
}

/// Traversal bridge between the transcript layout and [`TranscriptRender`].
/// The public batch owns every input and output; this records where each
/// request was made, so only the messages whose presentation a system
/// changed are laid out again.
#[derive(Default)]
pub(crate) struct Renderers {
    pub(crate) batch: RefCell<RenderBatch>,
    pub(crate) pass: Pass,
    /// The messages to lay out again in [`Pass::Finish`].
    pub(crate) messages: RefCell<HashSet<usize>>,
    /// Outputs of items outside any message, kept between frames.
    pub(crate) cached: RefCell<HashMap<RenderKey, RenderRequest>>,
    /// Where each request of this batch was made.
    made: RefCell<HashMap<RenderKey, Place>>,
    place: Cell<Place>,
}

/// Which traversal of a layout runs.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) enum Pass {
    /// Draw default lines and collect the requests of the items not cached.
    #[default]
    Collect,
    /// Lay out again the messages whose requests a system changed, drawing
    /// the batch's outputs.
    Finish,
}

/// Where the traversal is.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Place {
    /// Block `block` of message `index`, whose rows the transcript caches.
    Message { index: usize, block: usize },
    /// Outside any message: asides, the reply streaming in, and the call at
    /// this position among those it streams.
    Outside(usize),
}

impl Default for Place {
    fn default() -> Self {
        Self::Outside(0)
    }
}

impl Renderers {
    /// Renderers for one layout of `pane`, or of an export, for `agent`'s
    /// transcript at `width`, reusing the outputs `cached` outside messages.
    pub(crate) fn new(
        (pane, agent, width): (Option<Entity>, Option<Entity>, u16),
        cached: HashMap<RenderKey, RenderRequest>,
    ) -> Self {
        let batch = RenderBatch {
            pane,
            agent,
            width,
            requests: HashMap::new(),
        };
        Self {
            batch: RefCell::new(batch),
            cached: RefCell::new(cached),
            ..Default::default()
        }
    }

    pub(crate) fn at(&self, place: Place) {
        self.place.set(place);
    }

    /// The message and block of the item drawn now, inside a message.
    fn in_message(&self) -> Option<(usize, usize)> {
        match self.place.get() {
            Place::Message { index, block } => Some((index, block)),
            Place::Outside(_) => None,
        }
    }

    /// Runs [`TranscriptRender`] over the collected requests and switches to
    /// [`Pass::Finish`]. Returns whether a system changed any output, so the
    /// layout must run again; otherwise the collected layout stands.
    pub(crate) fn run(&mut self, world: &mut World) -> bool {
        let made = self.made.get_mut();
        let (cached, messages) = (self.cached.get_mut(), self.messages.get_mut());
        cached.retain(|key, _| matches!(made.get(key), Some(Place::Outside(_))));
        messages.clear();
        self.pass = Pass::Finish;
        let batch = std::mem::take(self.batch.get_mut());
        if batch.requests.is_empty() {
            return false;
        }
        let defaults: HashMap<RenderKey, RenderOutput> = batch
            .requests
            .iter()
            .map(|(key, request)| (key.clone(), request.output.clone()))
            .collect();
        let previous = world.remove_resource::<RenderBatch>();
        world.insert_resource(batch);
        world.try_run_schedule(TranscriptRender).ok();
        let batch = world.remove_resource::<RenderBatch>().unwrap_or_default();
        if let Some(previous) = previous {
            world.insert_resource(previous);
        }
        let made = self.made.get_mut();
        let (cached, messages) = (self.cached.get_mut(), self.messages.get_mut());
        let mut changed = false;
        for (key, request) in &batch.requests {
            let place = made.get(key).copied().unwrap_or_default();
            if matches!(place, Place::Outside(_)) {
                cached.insert(key.clone(), request.clone());
            }
            if defaults.get(key) != Some(&request.output) {
                changed = true;
                if let Place::Message { index, .. } = place {
                    messages.insert(index);
                }
            }
        }
        *self.batch.get_mut() = batch;
        changed
    }

    fn output(
        &self,
        key: RenderKey,
        input: RenderInput,
        lines: Vec<LookLine>,
    ) -> (Vec<LookLine>, Decorated) {
        match self.pass {
            Pass::Collect => {
                let place = self.place.get();
                if matches!(place, Place::Outside(_))
                    && let Some(previous) = self
                        .cached
                        .borrow()
                        .get(&key)
                        .filter(|previous| previous.input == input)
                {
                    self.made.borrow_mut().insert(key, place);
                    return previous.output.drawn();
                }
                self.made.borrow_mut().insert(key.clone(), place);
                let output = RenderOutput {
                    lines: lines.clone(),
                    decorations: Vec::new(),
                };
                let request = RenderRequest { input, output };
                self.batch.borrow_mut().requests.insert(key, request);
                (lines, Vec::new())
            }
            Pass::Finish => {
                let batch = self.batch.borrow();
                let cached = self.cached.borrow();
                let request = batch.requests.get(&key).or_else(|| cached.get(&key));
                request.map_or((lines, Vec::new()), |request| request.output.drawn())
            }
        }
    }

    pub(crate) fn call_decorated(&self, view: &ToolCallView<'_>) -> (Vec<LookLine>, Decorated) {
        let input = ToolRender {
            call: view.call.clone(),
            result: view.result.cloned(),
            state: view.state.clone(),
            theme: view.theme,
            facts: view.facts.cloned(),
            took: view.took,
            output: view.output.map(str::to_owned),
            children: view.children.to_vec(),
        };
        let key = match self.place.get() {
            Place::Message { index, block } => RenderKey::Tool(index, block),
            Place::Outside(at) => RenderKey::Streaming(at),
        };
        self.output(key, RenderInput::Tool(input), view.default_lines())
    }

    /// A delivered text's lines; only messages hold one.
    pub(crate) fn entry(&self, view: &EntryView<'_>) -> Vec<LookLine> {
        match self.in_message() {
            Some((index, block)) => self.entry_at(RenderKey::Entry(index, block), view),
            None => view
                .default_lines()
                .into_iter()
                .map(LookLine::from)
                .collect(),
        }
    }

    fn entry_at(&self, key: RenderKey, view: &EntryView<'_>) -> Vec<LookLine> {
        let input = EntryRender {
            id: view.id,
            plugin: view.plugin.to_owned(),
            title: view.title.to_owned(),
            text: view.text.to_owned(),
        };
        let lines = view.default_lines().into_iter().map(LookLine::from);
        self.output(key, RenderInput::Entry(input), lines.collect())
            .0
    }

    /// A content block's lines; only messages hold one.
    pub(crate) fn content(&self, view: &ContentView<'_>) -> Vec<LookLine> {
        let lines = view
            .default_lines()
            .into_iter()
            .map(LookLine::from)
            .collect();
        let Some((index, _)) = self.in_message() else {
            return lines;
        };
        let content = match view.block {
            ContentBlock::Image(v) => RenderContent::Image(v.clone()),
            ContentBlock::Audio(v) => RenderContent::Audio(v.clone()),
            ContentBlock::Video(v) => RenderContent::Video(v.clone()),
            ContentBlock::Document(v) => RenderContent::Document(v.clone()),
        };
        let input = ContentRender {
            message: view.message,
            index: view.index,
            content,
            assistant: view.assistant,
        };
        self.output(
            RenderKey::Content(index, view.index),
            RenderInput::Content(input),
            lines,
        )
        .0
    }

    pub(crate) fn aside(&self, view: &AsideView<'_>) -> Option<Vec<LookLine>> {
        let key = RenderKey::Aside(view.id);
        if let Some(note) = Note::of(view.aside) {
            return Some(self.entry_at(
                key,
                &EntryView {
                    id: None,
                    plugin: plugin_name(&view.aside.author),
                    title: &note.title,
                    text: &note.text,
                },
            ));
        }
        let input = AsideRender {
            id: view.id,
            aside: view.aside.clone(),
            theme: view.theme,
        };
        Some(self.output(key, RenderInput::Aside(input), Vec::new()).0)
    }

    /// A message's decorations; none outside a message.
    pub(crate) fn message_decorations(
        &self,
        view: &DecoratedMessage<'_>,
    ) -> (Vec<LookLine>, Decorated) {
        let Some((index, _)) = self.in_message() else {
            return (Vec::new(), Vec::new());
        };
        let input = MessageRender {
            message: view.message.clone(),
            id: view.id,
            author: view.author.cloned(),
        };
        self.output(
            RenderKey::Message(index),
            RenderInput::Message(input),
            Vec::new(),
        )
    }
}

#[cfg(test)]
mod tests;
