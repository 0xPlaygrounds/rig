//! Code in the terminal's own colors: bat's grammars (`two-face`) and its
//! `ansi` theme, which names colors of the terminal's palette rather than
//! colors of its own, so highlighted code follows the terminal's theme.
//! The grammars load once, on a thread of their own when the terminal
//! opens, or when code is first drawn.
//!
//! A [`Code`] is laid out as plain text, which its colors never change, and
//! its lines are colored when they are first drawn and kept colored, so the
//! work follows the screen rather than the transcript. A long text that
//! extends one made lately, as a file being written streams in, goes on
//! from that one's colored lines.
//!
//! ```
//! use rig_tui::highlight::{Language, code_rows};
//!
//! let rows = code_rows("fn main() {}\n", Language::File("src/main.rs"));
//! assert_eq!(rows.len(), 1);
//! assert_eq!(rows[0].colored().to_string(), "   fn main() {}");
//! ```

use std::cell::RefCell;
use std::ops::Range;
use std::path::Path;
use std::sync::{Arc, LazyLock, Mutex, MutexGuard, PoisonError};

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use syntect::easy::HighlightLines;
use syntect::highlighting::{FontStyle, HighlightState};
use syntect::parsing::{ParseState, SyntaxReference, SyntaxSet};
use syntect::util::LinesWithEndings;

use crate::{LookLine, OUTPUT_INDENT, text_rows};

static SYNTAXES: LazyLock<SyntaxSet> = LazyLock::new(two_face::syntax::extra_newlines);

static THEME: LazyLock<syntect::highlighting::Theme> = LazyLock::new(|| {
    let themes = two_face::theme::extra();
    themes.get(two_face::theme::EmbeddedThemeName::Ansi).clone()
});

/// What code is in.
#[derive(Clone, Copy, Debug)]
pub enum Language<'a> {
    /// A language's name or file extension, as a code block's fence says
    /// it: `rust`, `py`, `bash`.
    Named(&'a str),
    /// A file's path, by its extension or its name (`Makefile`).
    File(&'a str),
}

/// Loads the grammars and the theme on a thread of their own, so the
/// first code drawn does not wait for them.
pub(crate) fn warm_up() {
    std::thread::spawn(|| {
        LazyLock::force(&SYNTAXES);
        LazyLock::force(&THEME);
    });
}

/// Texts shorter than this are not kept for a longer one to go on from:
/// they cost little to color again.
const KEPT_FROM: usize = 4 * 1024;

/// How many codes are kept at most, so finding the one a text extends
/// stays quick.
const KEPT_COUNT: usize = 8;

/// About how much memory the kept codes take at most, counted when one is
/// kept.
const KEPT_BYTES: usize = 8 * 1024 * 1024;

thread_local! {
    /// The long codes made lately on this thread, oldest first. The agent
    /// lays out and draws on one thread; tests on others keep their own.
    static KEPT: RefCell<Vec<Code>> = const { RefCell::new(Vec::new()) };
}

/// A text in a language. Its lines are plain text until they are drawn;
/// each is colored then, once, by parsing on from the lines colored before
/// it, so a block is parsed top to bottom once however it is scrolled.
#[derive(Clone)]
pub struct Code(Arc<Inner>);

struct Inner {
    text: String,
    /// `None` for a language with no grammar: the lines stay plain.
    syntax: Option<&'static SyntaxReference>,
    /// Each line's bytes in `text`, with its end.
    lines: Vec<Range<usize>>,
    colors: Mutex<Colors>,
}

/// A code's colored lines so far.
#[derive(Default)]
struct Colors {
    /// The first lines' spans, each a line with its end.
    lines: Vec<Vec<Span<'static>>>,
    /// The parser's state after them, once any is colored.
    state: Option<(HighlightState, ParseState)>,
    /// The last line's spans, when it has no end yet, as a text streaming
    /// in: colored from a copy of the state, which it does not move on.
    last: Option<Vec<Span<'static>>>,
    /// About how much memory `lines` take.
    bytes: usize,
}

impl Code {
    /// `text` in `language`. A text of [`KEPT_FROM`] bytes or more is
    /// kept for a while: the same text in the same grammar again is that
    /// code, with its colors, and a text extending it goes on from its
    /// colored lines.
    pub fn new(text: &str, language: Language<'_>) -> Self {
        let syntax = syntax(language);
        if syntax.is_none() || text.len() < KEPT_FROM {
            return Self::made(text, syntax);
        }
        let kept = KEPT.try_with(|kept| {
            let mut kept = kept.try_borrow_mut().ok()?;
            let same = |code: &Self| same_syntax(code.0.syntax, syntax) && code.0.text == text;
            let code = match kept.iter().position(same) {
                Some(at) => kept.remove(at),
                None => Self::made(text, syntax).extending(&mut kept),
            };
            code.keep(&mut kept);
            Some(code)
        });
        kept.ok()
            .flatten()
            .unwrap_or_else(|| Self::made(text, syntax))
    }

    /// `text` in `syntax`, with no line colored.
    fn made(text: &str, syntax: Option<&'static SyntaxReference>) -> Self {
        let mut lines = Vec::new();
        let mut start = 0;
        for line in LinesWithEndings::from(text) {
            lines.push(start..start + line.len());
            start += line.len();
        }
        Self(Arc::new(Inner {
            text: text.to_owned(),
            syntax,
            lines,
            colors: Mutex::default(),
        }))
    }

    /// Whether its language has a grammar, so its lines get colors.
    pub fn colored(&self) -> bool {
        self.0.syntax.is_some()
    }

    /// How many lines it has.
    pub fn len(&self) -> usize {
        self.0.lines.len()
    }

    /// Whether it has no lines.
    pub fn is_empty(&self) -> bool {
        self.0.lines.is_empty()
    }

    /// Line `index` after `prefix`, such as an indent or a line number: the
    /// line without its end, tabs as four spaces, plain until it is drawn
    /// ([`LookLine::colored`]). Past the last line, just the prefix.
    pub fn line(&self, index: usize, prefix: Vec<Span<'static>>) -> LookLine {
        let at = prefix.iter().map(|span| span.content.len()).sum();
        let mut spans = prefix;
        let text = self.raw(index).map(|line| plain(line, 0..line.len()));
        spans.extend(text.map(Span::raw));
        LookLine {
            line: Line::from(spans),
            code: self.colored().then(|| crate::renderers::CodeLine {
                code: self.clone(),
                index,
                at,
            }),
        }
    }

    /// Its colors so far, locked.
    fn colors(&self) -> MutexGuard<'_, Colors> {
        self.0.colors.lock().unwrap_or_else(PoisonError::into_inner)
    }

    /// Line `index`, with its end.
    fn raw(&self, index: usize) -> Option<&str> {
        self.0.text.get(self.0.lines.get(index)?.clone())
    }

    /// Line `index`'s spans in color, their texts joined as
    /// [`line`](Self::line) draws it; `None` with no grammar, past the last
    /// line, or if the grammar fails on it.
    pub(crate) fn spans(&self, index: usize) -> Option<Vec<Span<'static>>> {
        let syntax = self.0.syntax?;
        let line = self.raw(index)?;
        let mut colors = self.colors();
        if let Some(spans) = colors.lines.get(index) {
            return Some(spans.clone());
        }
        let ended = line.ends_with('\n');
        if !ended && let Some(spans) = &colors.last {
            return Some(spans.clone());
        }
        let through = if ended { index + 1 } else { index };
        let colored = self.color(&mut colors, syntax, through);
        if colored.is_none() {
            // A grammar failing leaves a state it cannot go on from.
            *colors = Colors::default();
            return None;
        }
        if ended {
            return colors.lines.get(index).cloned();
        }
        let (highlight, parse) = colors
            .state
            .clone()
            .unwrap_or_else(|| HighlightLines::new(syntax, &THEME).state());
        let mut parser = HighlightLines::from_state(&THEME, highlight, parse);
        let spans = spans(&mut parser, line)?;
        colors.last = Some(spans.clone());
        Some(spans)
    }

    /// Colors the lines before `through` that `colors` has not.
    fn color(
        &self,
        colors: &mut Colors,
        syntax: &'static SyntaxReference,
        through: usize,
    ) -> Option<()> {
        if colors.lines.len() >= through {
            return Some(());
        }
        let (highlight, parse) = colors
            .state
            .take()
            .unwrap_or_else(|| HighlightLines::new(syntax, &THEME).state());
        let mut parser = HighlightLines::from_state(&THEME, highlight, parse);
        for index in colors.lines.len()..through {
            let line = self.raw(index)?;
            let spans = spans(&mut parser, line)?;
            colors.bytes +=
                size_of::<Vec<Span>>() + spans.capacity() * size_of::<Span>() + line.len();
            colors.lines.push(spans);
        }
        colors.state = Some(parser.state());
        Some(())
    }

    /// The bytes of its text that its colored lines span.
    fn colored_end(&self, colors: &Colors) -> usize {
        let last = colors.lines.len().checked_sub(1);
        let range = last.and_then(|last| self.0.lines.get(last));
        range.map_or(0, |range| range.end)
    }

    /// About how much memory it takes: its text, its lines' places and
    /// their colors.
    fn bytes(&self) -> usize {
        let colors = self.colors();
        let places = self.0.lines.capacity() * size_of::<Range<usize>>();
        let last = colors.last.as_ref().map_or(0, |last| {
            last.capacity() * size_of::<Span>()
                + last.iter().map(|span| span.content.len()).sum::<usize>()
        });
        self.0.text.len() + places + colors.bytes + last
    }

    /// This code, gone on from the colored lines of the code of `kept`
    /// whose colored lines are the longest start of its text, in the same
    /// grammar. That code is dropped from `kept` and its colors taken, so
    /// it colors itself again if it is drawn again.
    fn extending(self, kept: &mut Vec<Self>) -> Self {
        let start = |code: &Self, colors: &Colors| {
            let end = code.colored_end(colors);
            let start = code.0.text.get(..end)?;
            (end > 0 && self.0.text.starts_with(start)).then_some(end)
        };
        let extended = kept
            .iter()
            .enumerate()
            .filter(|(_, code)| same_syntax(code.0.syntax, self.0.syntax))
            .filter_map(|(at, code)| Some((at, start(code, &code.colors())?)))
            .max_by_key(|(_, end)| *end);
        let Some((at, _)) = extended else {
            return self;
        };
        let old = kept.remove(at);
        let mut colors = old.colors();
        // Checked again under its lock: it may have been colored since.
        if start(&old, &colors).is_some() {
            let mut taken = std::mem::take(&mut *colors);
            taken.last = None;
            *self.colors() = taken;
        }
        drop(colors);
        self
    }

    /// Keeps it in `kept` as the most recent, unless it alone passes the
    /// bounds, and drops the oldest codes past them. The bounds are counted
    /// each time a code is kept, so a kept code colored since counts then.
    fn keep(&self, kept: &mut Vec<Self>) {
        if self.bytes() > KEPT_BYTES {
            return;
        }
        kept.push(self.clone());
        let mut total: usize = kept.iter().map(Self::bytes).sum();
        while kept.len() > KEPT_COUNT || total > KEPT_BYTES {
            let Some(dropped) = kept.first().map(Self::bytes) else {
                break;
            };
            kept.remove(0);
            total -= dropped;
        }
    }
}

impl PartialEq for Code {
    fn eq(&self, other: &Self) -> bool {
        Arc::ptr_eq(&self.0, &other.0)
            || same_syntax(self.0.syntax, other.0.syntax) && self.0.text == other.0.text
    }
}

impl std::fmt::Debug for Code {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Code")
            .field("syntax", &self.0.syntax.map(|syntax| &syntax.name))
            .field("lines", &self.0.lines.len())
            .finish_non_exhaustive()
    }
}

/// Whether two codes are in the same grammar, or both in none.
fn same_syntax(one: Option<&SyntaxReference>, other: Option<&SyntaxReference>) -> bool {
    match (one, other) {
        (Some(one), Some(other)) => std::ptr::eq(one, other),
        (one, other) => one.is_none() && other.is_none(),
    }
}

/// The bytes `range` of `line` as drawn: without the line's end, tabs as
/// four spaces.
fn plain(line: &str, range: Range<usize>) -> String {
    let end = line.trim_end_matches(['\n', '\r']).len();
    let text = line.get(range.start.min(end)..range.end.min(end));
    text.unwrap_or_default().replace('\t', "    ")
}

/// `line`'s spans, the parser carried on past it, cut from the pieces the
/// grammar marks so their texts join to the line as [`plain`] draws it.
fn spans(parser: &mut HighlightLines<'_>, line: &str) -> Option<Vec<Span<'static>>> {
    let pieces = parser.highlight_line(line, &SYNTAXES).ok()?;
    let mut start = 0;
    let spans = pieces.into_iter().filter_map(|(style, piece)| {
        let range = start..start + piece.len();
        start = range.end;
        let text = plain(line, range);
        (!text.is_empty()).then(|| Span::styled(text, style_of(style)))
    });
    Some(spans.collect())
}

/// `code`'s rows as a tool's result draws them, indented by
/// [`OUTPUT_INDENT`], in `language`'s colors once drawn, or dimmed for a
/// language with no grammar.
pub fn code_rows(code: &str, language: Language<'_>) -> Vec<LookLine> {
    if syntax(language).is_none() {
        let rows = text_rows(code, Style::new().dim());
        return rows.into_iter().map(LookLine::from).collect();
    }
    let code = Code::new(code, language);
    let rows = (0..code.len()).map(|index| code.line(index, vec![Span::from(OUTPUT_INDENT)]));
    rows.collect()
}

/// The grammar of `language`, if one is known; plain text is none.
fn syntax(language: Language<'_>) -> Option<&'static SyntaxReference> {
    let syntaxes: &'static SyntaxSet = &SYNTAXES;
    let found = match language {
        Language::Named(name) => syntaxes.find_syntax_by_token(name.trim()),
        Language::File(path) => {
            let path = Path::new(path);
            let by =
                |name: Option<&std::ffi::OsStr>| syntaxes.find_syntax_by_extension(name?.to_str()?);
            by(path.extension()).or_else(|| by(path.file_name()))
        }
    };
    found.filter(|syntax| syntax.name != "Plain Text")
}

/// A piece's style: its color and its weight, slant and underline.
fn style_of(style: syntect::highlighting::Style) -> Style {
    let mut drawn = Style::new();
    if let Some(color) = color(style.foreground) {
        drawn = drawn.fg(color);
    }
    let font = style.font_style;
    if font.contains(FontStyle::BOLD) {
        drawn = drawn.bold();
    }
    if font.contains(FontStyle::ITALIC) {
        drawn = drawn.italic();
    }
    if font.contains(FontStyle::UNDERLINE) {
        drawn = drawn.underlined();
    }
    drawn
}

/// A theme color as the `ansi` theme writes it: alpha 0 names a color of
/// the terminal's palette by its red, alpha 1 the terminal's own text
/// color (`None`), any other alpha a 24-bit color.
fn color(color: syntect::highlighting::Color) -> Option<Color> {
    match color.a {
        0 => Some(Color::Indexed(color.r)),
        1 => None,
        _ => Some(Color::Rgb(color.r, color.g, color.b)),
    }
}

#[cfg(test)]
mod tests;
