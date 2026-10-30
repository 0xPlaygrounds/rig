//! Code in the terminal's own colors: bat's grammars (`two-face`) and its
//! `ansi` theme, which names colors of the terminal's palette rather than
//! colors of its own, so highlighted code follows the terminal's theme.
//! The grammars load once, on a thread of their own when the terminal
//! opens, or when code is first drawn. A long text that extends one
//! highlighted lately, as a file being written streams in, is highlighted
//! only past the lines highlighted before.
//!
//! ```
//! use rig_tui::highlight::{Language, code_rows};
//!
//! let rows = code_rows("fn main() {}\n", Language::File("src/main.rs"));
//! assert_eq!(rows.len(), 1);
//! ```

use std::cell::RefCell;
use std::mem::size_of;
use std::path::Path;
use std::sync::LazyLock;

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use syntect::easy::HighlightLines;
use syntect::highlighting::{FontStyle, HighlightState};
use syntect::parsing::{ParseState, SyntaxReference, SyntaxSet};
use syntect::util::LinesWithEndings;

use crate::{OUTPUT_INDENT, text_rows};

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

/// Texts shorter than this are highlighted afresh and not kept: they cost
/// little to highlight again.
const KEPT_FROM: usize = 4 * 1024;

/// How many texts are kept at most, so finding the one a text extends
/// stays quick.
const KEPT_COUNT: usize = 8;

/// About how much memory the kept texts take at most, on each thread.
const KEPT_BYTES: usize = 8 * 1024 * 1024;

thread_local! {
    /// The long texts highlighted lately on this thread, oldest first.
    /// Syntect's parser state cannot cross threads; the agent draws on
    /// one.
    static KEPT: RefCell<Vec<Kept>> = const { RefCell::new(Vec::new()) };
}

/// A text's complete lines as highlighted, and the parser's state after
/// them.
struct Kept {
    syntax: &'static SyntaxReference,
    /// Ends with a line's end, or is empty.
    text: String,
    lines: Vec<Vec<Span<'static>>>,
    state: (HighlightState, ParseState),
    /// About how much memory it takes.
    bytes: usize,
}

/// `code`'s lines in `language`'s colors, each as its spans, tabs made
/// spaces; `None` for a language with no grammar.
pub fn highlight(code: &str, language: Language<'_>) -> Option<Vec<Vec<Span<'static>>>> {
    let syntax = syntax(language)?;
    if code.len() < KEPT_FROM {
        return fresh(code, syntax);
    }
    KEPT.with_borrow_mut(|kept| grown(kept, code, syntax))
}

/// `code` highlighted from its first line.
fn fresh(code: &str, syntax: &SyntaxReference) -> Option<Vec<Vec<Span<'static>>>> {
    let mut lines = HighlightLines::new(syntax, &THEME);
    LinesWithEndings::from(code)
        .map(|line| spans(&mut lines, line))
        .collect()
}

/// `code` highlighted on from the longest text of `kept` it extends in the
/// same grammar, else afresh. Its complete lines are kept in that text's
/// place, as the most recent, and the oldest texts are dropped past the
/// bounds.
fn grown(
    kept: &mut Vec<Kept>,
    code: &str,
    syntax: &'static SyntaxReference,
) -> Option<Vec<Vec<Span<'static>>>> {
    let extended = kept
        .iter()
        .enumerate()
        .filter(|(_, text)| std::ptr::eq(text.syntax, syntax) && code.starts_with(&text.text))
        .max_by_key(|(_, text)| text.text.len())
        .map(|(at, _)| at);
    let Kept {
        mut text,
        mut lines,
        state: (highlight, parse),
        mut bytes,
        ..
    } = match extended {
        Some(at) => kept.remove(at),
        None => Kept {
            syntax,
            text: String::new(),
            lines: Vec::new(),
            state: HighlightLines::new(syntax, &THEME).state(),
            bytes: 0,
        },
    };
    let rest = code.get(text.len()..)?;
    let (whole, last) = rest.split_at(rest.rfind('\n').map_or(0, |at| at + 1));
    let mut parser = HighlightLines::from_state(&THEME, highlight, parse);
    for line in LinesWithEndings::from(whole) {
        let spans = spans(&mut parser, line)?;
        // The line is kept twice: in the text and in its spans.
        bytes += size_of::<Vec<Span>>() + spans.capacity() * size_of::<Span>() + 2 * line.len();
        lines.push(spans);
    }
    text.push_str(whole);
    let state = parser.state();
    let mut drawn = lines.clone();
    if !last.is_empty() {
        let (highlight, parse) = state.clone();
        let mut parser = HighlightLines::from_state(&THEME, highlight, parse);
        drawn.push(spans(&mut parser, last)?);
    }
    kept.push(Kept {
        syntax,
        text,
        lines,
        state,
        bytes,
    });
    let mut total: usize = kept.iter().map(|text| text.bytes).sum();
    while kept.len() > KEPT_COUNT || total > KEPT_BYTES {
        let Some(dropped) = kept.first().map(|text| text.bytes) else {
            break;
        };
        kept.remove(0);
        total -= dropped;
    }
    Some(drawn)
}

/// `line`'s spans, the parser carried on past it.
fn spans(parser: &mut HighlightLines<'_>, line: &str) -> Option<Vec<Span<'static>>> {
    let pieces = parser.highlight_line(line, &SYNTAXES).ok()?;
    let spans = pieces.into_iter().filter_map(|(style, text)| {
        let text = text.trim_end_matches(['\n', '\r']).replace('\t', "    ");
        (!text.is_empty()).then(|| Span::styled(text, style_of(style)))
    });
    Some(spans.collect())
}

/// `code`'s rows as a tool's result draws them, indented by
/// [`OUTPUT_INDENT`],
/// in `language`'s colors, or dimmed for a language with no grammar.
pub fn code_rows(code: &str, language: Language<'_>) -> Vec<Line<'static>> {
    let Some(lines) = highlight(code, language) else {
        return text_rows(code, Style::new().dim());
    };
    let rows = lines
        .into_iter()
        .map(|spans| Line::from([vec![Span::from(OUTPUT_INDENT)], spans].concat()));
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
