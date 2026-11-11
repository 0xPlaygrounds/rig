//! What plain `-p` writes to a terminal: the answer's markdown drawn as
//! styled text and word-wrapped to the width of the terminal stdout is, code
//! lines kept whole, and dim lines on stderr, wrapped to its own terminal,
//! for the narration and each tool call.
//! Styles are SGR escapes: bold, dim, italic, underline and such always,
//! foreground colors unless `NO_COLOR` is set, backgrounds never.

use std::io::IsTerminal;

use ratatui_core::style::{Color, Modifier, Style};
use ratatui_core::text::Line;
use rig_harness::rig_core::message::ToolCall;
use rig_harness::rig_core::tool::Tool;
use rig_markdown::markdown::{self, Kind, Palette, Plain};
use rig_markdown::wrap::wrap;
use rig_tools::{Edit, Read, Shell, Write};

use rig_markdown::code;

use crate::coding_tools::COMMAND_INDENT;

/// A terminal that stdout or stderr is: how text written to it looks.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Terminal {
    /// Its width in columns, when known: `COLUMNS`, else the size of the
    /// terminal the stream itself is. Text is not wrapped without it.
    pub(super) width: Option<usize>,
    /// Whether it shows colors: `NO_COLOR` is unset or empty.
    pub(super) color: bool,
}

impl Terminal {
    /// The terminal `stream` is, if it is one.
    pub(super) fn of(stream: &impl Stream) -> Option<Self> {
        stream.is_terminal().then(|| Self {
            width: width(stream),
            color: std::env::var_os("NO_COLOR").is_none_or(|value| value.is_empty()),
        })
    }

    /// `text` as markdown, each line wrapped to the width but lines of
    /// code, which are kept whole for copying.
    pub(super) fn markdown(&self, text: &str) -> String {
        let palette = Palette {
            plain: true,
            ..Palette::default()
        };
        let lines = markdown::render(text, &palette, self.width, &Plain);
        let rows = lines.into_iter().flat_map(|(line, kind)| match kind {
            Kind::Code(_) => vec![line],
            Kind::Text => self.wrapped(line),
        });
        self.escaped(rows)
    }

    /// `text`'s lines in `style`, each wrapped to the width.
    pub(super) fn text(&self, text: &str, style: Style) -> String {
        let lines = text
            .lines()
            .map(|line| Line::styled(line.to_owned(), style));
        self.escaped(lines.flat_map(|line| self.wrapped(line)))
    }

    /// The lines saying a call of `call` started, dim: a shell call's
    /// whole command laid out as its look lays it out, each line wrapped
    /// to the width; for another tool, one row with what its look's header
    /// shows of it.
    pub(super) fn call(&self, call: &ToolCall) -> String {
        let dim = Style::new().dim();
        let name = call.function.name.as_str();
        let argument = |key: &str| {
            let value = call.function.arguments.get(key);
            value.and_then(|value| value.as_str()).unwrap_or_default()
        };
        if name == Shell::NAME {
            let command = argument("command");
            let mut lines: Vec<String> = (code::lines(command))
                .map(|line| code::shown(line, 0..line.len()))
                .collect();
            if lines.is_empty() {
                lines.push(String::new());
            }
            let lines = lines.into_iter().enumerate();
            let lines = lines.map(|(index, line)| match index {
                0 => format!("· $ {line}"),
                _ => format!("{COMMAND_INDENT}{line}"),
            });
            let rows = lines.flat_map(|line| self.wrapped(Line::styled(line, dim)));
            return self.escaped(rows);
        }
        let summary = match name {
            Read::NAME | Write::NAME | Edit::NAME => format!("· {name} {}", argument("path")),
            _ => format!("· {name}"),
        };
        let summary = match self.width {
            Some(width) => rig_tools::shorten(&summary, width),
            None => summary,
        };
        self.escaped([Line::styled(summary, dim)])
    }

    /// `line` wrapped to the width, each tab in it a space first, as the
    /// wrap drops control characters.
    fn wrapped(&self, mut line: Line<'static>) -> Vec<Line<'static>> {
        for span in &mut line.spans {
            if span.content.contains('\t') {
                span.content = span.content.replace('\t', " ").into();
            }
        }
        match self.width {
            Some(width) => wrap(&line, width).into_iter().map(|(row, _)| row).collect(),
            None => vec![line],
        }
    }

    /// `lines` one per row, each span's text after the SGR escape of its
    /// style and before a reset. Control characters are left out, so the
    /// text cannot move the cursor or restyle the terminal.
    fn escaped(&self, lines: impl IntoIterator<Item = Line<'static>>) -> String {
        let rows = lines.into_iter().map(|line| {
            let mut row = String::new();
            for span in &line.spans {
                let text: String = span.content.chars().filter(|c| !c.is_control()).collect();
                match sgr(line.style.patch(span.style), self.color) {
                    Some(codes) => row.push_str(&format!("\x1b[{codes}m{text}\x1b[0m")),
                    None => row.push_str(&text),
                }
            }
            row
        });
        rows.collect::<Vec<_>>().join("\n")
    }
}

/// A stream that may be a terminal, whose width can be asked.
pub(super) trait Stream: IsTerminal {
    /// The width in columns of the terminal this stream is, if known.
    fn columns(&self) -> Option<usize>;
}

#[cfg(unix)]
impl<S: IsTerminal + std::os::fd::AsFd> Stream for S {
    fn columns(&self) -> Option<usize> {
        use std::os::fd::AsRawFd;

        let mut size = libc::winsize {
            ws_row: 0,
            ws_col: 0,
            ws_xpixel: 0,
            ws_ypixel: 0,
        };
        // SAFETY: TIOCGWINSZ writes one `winsize` to the pointer, which
        // outlives the call, and the descriptor is open while `self` is.
        let read =
            unsafe { libc::ioctl(self.as_fd().as_raw_fd(), libc::TIOCGWINSZ, &raw mut size) };
        (read == 0).then_some(usize::from(size.ws_col))
    }
}

#[cfg(not(unix))]
impl<S: IsTerminal> Stream for S {
    fn columns(&self) -> Option<usize> {
        let size = crossterm::terminal::size().ok();
        size.map(|(width, _)| usize::from(width))
    }
}

/// The width of the terminal `stream` is: `COLUMNS` when it is a positive
/// number of columns, else the terminal's own size.
fn width(stream: &impl Stream) -> Option<usize> {
    let columns = std::env::var("COLUMNS").ok();
    let columns = columns.and_then(|columns| columns.trim().parse().ok());
    let positive = |width: &usize| *width > 0;
    columns
        .filter(positive)
        .or_else(|| stream.columns().filter(positive))
}

/// The SGR parameters of `style`: its modifiers, and its foreground
/// color with `color`; `None` when it has neither.
fn sgr(style: Style, color: bool) -> Option<String> {
    const MODIFIERS: [(Modifier, &str); 6] = [
        (Modifier::BOLD, "1"),
        (Modifier::DIM, "2"),
        (Modifier::ITALIC, "3"),
        (Modifier::UNDERLINED, "4"),
        (Modifier::REVERSED, "7"),
        (Modifier::CROSSED_OUT, "9"),
    ];
    let mut codes: Vec<String> = MODIFIERS
        .iter()
        .filter(|(modifier, _)| style.add_modifier.contains(*modifier))
        .map(|(_, code)| (*code).to_owned())
        .collect();
    codes.extend(style.fg.filter(|_| color).and_then(foreground));
    (!codes.is_empty()).then(|| codes.join(";"))
}

/// The SGR parameter of `color` as the foreground, `None` for the
/// terminal's own.
fn foreground(color: Color) -> Option<String> {
    let code = match color {
        Color::Reset => return None,
        Color::Black => "30",
        Color::Red => "31",
        Color::Green => "32",
        Color::Yellow => "33",
        Color::Blue => "34",
        Color::Magenta => "35",
        Color::Cyan => "36",
        Color::Gray => "37",
        Color::DarkGray => "90",
        Color::LightRed => "91",
        Color::LightGreen => "92",
        Color::LightYellow => "93",
        Color::LightBlue => "94",
        Color::LightMagenta => "95",
        Color::LightCyan => "96",
        Color::White => "97",
        Color::Indexed(index) => return Some(format!("38;5;{index}")),
        Color::Rgb(r, g, b) => return Some(format!("38;2;{r};{g};{b}")),
    };
    Some(code.to_owned())
}

#[cfg(test)]
mod tests;
