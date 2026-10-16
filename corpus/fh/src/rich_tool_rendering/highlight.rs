//! A small line highlighter: keywords, strings, numbers and comments of a
//! few languages, told by the file's extension. A line is highlighted on
//! its own, so a string or comment spanning lines is not followed.

use rig_tui::ratatui::style::{Color, Style};
use rig_tui::ratatui::text::Span;

/// A language the highlighter knows.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Lang {
    Rust,
    Python,
    Script,
    Go,
    Shell,
    Toml,
    Plain,
}

impl Lang {
    /// The language of the file at `path`, by its extension.
    pub(crate) fn of(path: &str) -> Self {
        let extension = path.rsplit_once('.').map(|(_, extension)| extension);
        match extension.unwrap_or_default() {
            "rs" => Self::Rust,
            "py" => Self::Python,
            "js" | "jsx" | "ts" | "tsx" | "mjs" => Self::Script,
            "go" => Self::Go,
            "sh" | "bash" | "zsh" => Self::Shell,
            "toml" => Self::Toml,
            _ => Self::Plain,
        }
    }

    /// What starts a comment that runs to the end of the line.
    fn comment(self) -> Option<&'static str> {
        match self {
            Self::Rust | Self::Script | Self::Go => Some("//"),
            Self::Python | Self::Shell | Self::Toml => Some("#"),
            Self::Plain => None,
        }
    }

    /// Whether `'` quotes a string (in Rust it also marks a lifetime).
    fn single_quotes(self) -> bool {
        !matches!(self, Self::Rust | Self::Plain)
    }

    fn keywords(self) -> &'static [&'static str] {
        match self {
            Self::Rust => &[
                "as", "async", "await", "break", "const", "continue", "crate", "else", "enum",
                "fn", "for", "if", "impl", "in", "let", "loop", "match", "mod", "move", "mut",
                "pub", "ref", "return", "self", "Self", "static", "struct", "trait", "type",
                "unsafe", "use", "where", "while", "true", "false",
            ],
            Self::Python => &[
                "and", "as", "class", "def", "elif", "else", "except", "False", "for", "from",
                "if", "import", "in", "is", "lambda", "None", "not", "or", "pass", "raise",
                "return", "True", "try", "while", "with", "yield",
            ],
            Self::Script => &[
                "async",
                "await",
                "class",
                "const",
                "else",
                "export",
                "false",
                "for",
                "function",
                "if",
                "import",
                "let",
                "new",
                "null",
                "return",
                "this",
                "true",
                "type",
                "undefined",
                "var",
                "while",
            ],
            Self::Go => &[
                "func",
                "package",
                "import",
                "var",
                "const",
                "type",
                "struct",
                "interface",
                "if",
                "else",
                "for",
                "range",
                "return",
                "go",
                "defer",
                "map",
                "chan",
                "nil",
            ],
            Self::Shell => &[
                "if", "then", "else", "fi", "for", "do", "done", "while", "case", "esac",
                "function", "export", "local", "return",
            ],
            Self::Toml | Self::Plain => &["true", "false"],
        }
    }
}

/// `line` as spans coloured for `lang`.
pub(crate) fn spans(line: &str, lang: Lang) -> Vec<Span<'static>> {
    if lang == Lang::Plain {
        return vec![Span::from(line.to_owned())];
    }
    let mut spans = Vec::new();
    let mut plain = String::new();
    let mut rest = line;
    let flush = |plain: &mut String, spans: &mut Vec<Span<'static>>| {
        if !plain.is_empty() {
            spans.push(Span::from(std::mem::take(plain)));
        }
    };
    while let Some(character) = rest.chars().next() {
        if lang.comment().is_some_and(|start| rest.starts_with(start)) {
            flush(&mut plain, &mut spans);
            spans.push(Span::styled(
                rest.to_owned(),
                Style::new().fg(Color::DarkGray).italic(),
            ));
            return spans;
        }
        let quote = character == '"' || (character == '\'' && lang.single_quotes());
        let taken = if quote {
            Some((string_len(rest, character), Style::new().fg(Color::Yellow)))
        } else if character.is_ascii_digit() {
            let len = rest
                .find(|next: char| !next.is_ascii_alphanumeric() && next != '_' && next != '.')
                .unwrap_or(rest.len());
            Some((len, Style::new().fg(Color::Cyan)))
        } else if character.is_alphabetic() || character == '_' {
            let len = rest
                .find(|next: char| !next.is_alphanumeric() && next != '_')
                .unwrap_or(rest.len());
            let word = rest.get(..len).unwrap_or_default();
            let style = lang
                .keywords()
                .contains(&word)
                .then(|| Style::new().fg(Color::Magenta).bold());
            match style {
                Some(style) => Some((len, style)),
                None => {
                    plain.push_str(word);
                    rest = rest.get(len..).unwrap_or_default();
                    continue;
                }
            }
        } else {
            None
        };
        match taken {
            Some((len, style)) => {
                flush(&mut plain, &mut spans);
                let (token, after) = rest.split_at_checked(len).unwrap_or((rest, ""));
                spans.push(Span::styled(token.to_owned(), style));
                rest = after;
            }
            None => {
                plain.push(character);
                rest = rest.get(character.len_utf8()..).unwrap_or_default();
            }
        }
    }
    flush(&mut plain, &mut spans);
    spans
}

/// The length in bytes of the string `text` starts with, quoted by `quote`:
/// to its closing quote, or the end of the line.
fn string_len(text: &str, quote: char) -> usize {
    let mut escaped = false;
    for (at, character) in text.char_indices().skip(1) {
        match character {
            '\\' if !escaped => escaped = true,
            closing if closing == quote && !escaped => return at + closing.len_utf8(),
            _ => escaped = false,
        }
    }
    text.len()
}
