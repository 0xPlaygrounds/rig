//! Code in the terminal's own colors: bat's grammars (`two-face`) and its
//! `ansi` theme, which names colors of the terminal's palette rather than
//! colors of its own, so highlighted code follows the terminal's theme.
//! The grammars load once, on a thread of their own when the terminal
//! opens, or when code is first drawn.
//!
//! ```
//! use rig_tui::highlight::{Language, code_rows};
//!
//! let rows = code_rows("fn main() {}\n", Language::File("src/main.rs"));
//! assert_eq!(rows.len(), 1);
//! ```

use std::path::Path;
use std::sync::LazyLock;

use ratatui::style::{Color, Style};
use ratatui::text::{Line, Span};
use syntect::easy::HighlightLines;
use syntect::highlighting::FontStyle;
use syntect::parsing::{SyntaxReference, SyntaxSet};
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

/// `code`'s lines in `language`'s colors, each as its spans, tabs made
/// spaces; `None` for a language with no grammar.
pub fn highlight(code: &str, language: Language<'_>) -> Option<Vec<Vec<Span<'static>>>> {
    let syntax = syntax(language)?;
    let mut lines = HighlightLines::new(syntax, &THEME);
    LinesWithEndings::from(code)
        .map(|line| {
            let pieces = lines.highlight_line(line, &SYNTAXES).ok()?;
            let spans = pieces.into_iter().filter_map(|(style, text)| {
                let text = text.trim_end_matches(['\n', '\r']).replace('\t', "    ");
                (!text.is_empty()).then(|| Span::styled(text, style_of(style)))
            });
            Some(spans.collect())
        })
        .collect()
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
