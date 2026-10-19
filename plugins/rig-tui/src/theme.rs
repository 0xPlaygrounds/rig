//! The view's colors, by what each thing is: a band behind what the user
//! wrote, one behind tool calls and what plugins put in the conversation,
//! the model's replies on the terminal's own background, red and green
//! behind a diff's removed and added lines, stronger behind the words that
//! changed, a tool call's row tinted by where it stands, and the surfaces
//! of panels and of boxes over the screen. Each
//! is mixed from the terminal's background, asked of the terminal once it
//! is opened, so it reads as a tint on a dark or a light terminal alike; a
//! terminal without 24-bit color gets fixed shades of its 256.
//!
//! Nothing is drawn with box-drawing lines: a surface is a flat tint, the
//! higher the lighter (on a dark terminal), with a bold title on its first
//! row and a cell of padding around its content.

use bevy_ecs::prelude::*;
use ratatui::style::{Color, Stylize};
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Padding};

/// The background assumed when the terminal does not say its own.
const DARK: Rgb = (0x1c, 0x1c, 0x1c);

/// The colors each band leans to, and how far.
const ADDED: (Rgb, f32) = ((0x3f, 0xb9, 0x50), 0.16);
const ADDED_WORD: (Rgb, f32) = ((0x3f, 0xb9, 0x50), 0.38);
const REMOVED: (Rgb, f32) = ((0xf8, 0x51, 0x49), 0.16);
const REMOVED_WORD: (Rgb, f32) = ((0xf8, 0x51, 0x49), 0.38);
const SELECTED: (Rgb, f32) = ((0xd2, 0x99, 0x22), 0.30);
const CALL_RUNNING: (Rgb, f32) = ((0xd2, 0x99, 0x22), 0.14);
const CALL_DONE: (Rgb, f32) = ((0x3f, 0xb9, 0x50), 0.10);
const CALL_FAILED: (Rgb, f32) = ((0xf8, 0x51, 0x49), 0.24);
/// How far the user's and the tool band lean from the background to the
/// foreground: gray, the user's the lighter.
const USER: f32 = 0.12;
const TOOL: f32 = 0.06;
/// How far a panel's, the input's and a box over the screen's surface lean
/// the same way: a panel just off the background, the input a step above
/// what the user sent, a box above everything else.
const PANEL: f32 = 0.04;
const INPUT: f32 = 0.17;
const RAISED: f32 = 0.24;
/// How far a call not yet running leans to the foreground: a step above
/// the tool band its output is on.
const CALL_WAITING: f32 = 0.10;

/// The bands of a terminal without 24-bit color, of its 256: on a dark
/// background and on a light one. A tint mixed as above would fall on
/// the same few grays.
const DARK_256: Theme = indexed([238, 236, 22, 28, 52, 88, 58, 235, 242, 237, 94, 23, 89, 240]);
const LIGHT_256: Theme = indexed([
    252, 254, 194, 157, 224, 217, 230, 255, 248, 251, 223, 193, 218, 250,
]);

type Rgb = (u8, u8, u8);

/// The view's colors, a resource. Plugins' looks and panels may use them
/// too, such as [`added`](Self::added) behind a line a tool added, or
/// [`panel`](Self::panel) for a panel's surface.
#[derive(Resource, Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct Theme {
    /// Behind what the user wrote.
    pub user: Color,
    /// Behind tool calls, their results, and what plugins put in the
    /// conversation, such as notes and a compaction's summary.
    pub tool: Color,
    /// Behind a diff's added lines.
    pub added: Color,
    /// Behind the words an added line changed.
    pub added_word: Color,
    /// Behind a diff's removed lines.
    pub removed: Color,
    /// Behind the words a removed line changed.
    pub removed_word: Color,
    /// Behind the item the user selected, in the transcript and in lists.
    pub selected: Color,
    /// The surface of a panel beside the transcript.
    pub panel_surface: Color,
    /// The surface of a box over the screen, such as a picker.
    pub raised_surface: Color,
    /// Behind the input, a step off what the user sent.
    pub input: Color,
    /// Behind a tool call's row while it streams, waits or is held.
    pub call_waiting: Color,
    /// Behind a running call's row.
    pub call_running: Color,
    /// Behind an answered call's row.
    pub call_done: Color,
    /// Behind the row of a call that failed.
    pub call_failed: Color,
}

impl Default for Theme {
    /// The colors for a dark terminal, in 24-bit color when `COLORTERM`
    /// says the terminal has it.
    fn default() -> Self {
        Self::on(DARK, truecolor())
    }
}

impl Theme {
    /// The colors on a terminal whose background is `background`, in
    /// 24-bit color or the nearest of 256.
    pub fn on(background: Rgb, truecolor: bool) -> Self {
        let dark = lightness(background) < 0.5;
        if !truecolor {
            return if dark { DARK_256 } else { LIGHT_256 };
        }
        let contrast = if dark { (0xff, 0xff, 0xff) } else { (0, 0, 0) };
        let lean = |(to, by): (Rgb, f32)| {
            let (r, g, b) = mix(background, to, by);
            Color::Rgb(r, g, b)
        };
        Self {
            user: lean((contrast, USER)),
            tool: lean((contrast, TOOL)),
            added: lean(ADDED),
            added_word: lean(ADDED_WORD),
            removed: lean(REMOVED),
            removed_word: lean(REMOVED_WORD),
            selected: lean(SELECTED),
            panel_surface: lean((contrast, PANEL)),
            raised_surface: lean((contrast, RAISED)),
            input: lean((contrast, INPUT)),
            call_waiting: lean((contrast, CALL_WAITING)),
            call_running: lean(CALL_RUNNING),
            call_done: lean(CALL_DONE),
            call_failed: lean(CALL_FAILED),
        }
    }

    /// A panel's surface titled `title`: a flat
    /// [`panel_surface`](Self::panel_surface) with the title in bold on its
    /// first row, and a cell of padding left, right and below. Its
    /// [`inner`](Block::inner) area is the one a bordered block leaves.
    pub fn panel<'a>(&self, title: impl Into<Line<'a>>) -> Block<'a> {
        surface(self.panel_surface, title.into())
    }

    /// A box over the screen titled `title`, as [`panel`](Self::panel) but
    /// on the [`raised_surface`](Self::raised_surface).
    pub fn raised<'a>(&self, title: impl Into<Line<'a>>) -> Block<'a> {
        surface(self.raised_surface, title.into())
    }

    /// The colors on the terminal the view runs in, asking it for its
    /// background; the dark ones when it does not answer.
    pub(crate) fn of_terminal() -> Self {
        let options = terminal_colorsaurus::QueryOptions::default();
        match terminal_colorsaurus::background_color(options) {
            Ok(background) => Self::on(background.scale_to_8bit(), truecolor()),
            Err(_) => Self::default(),
        }
    }
}

/// A flat surface of `color` titled `title`, padded to leave what a
/// border would.
fn surface(color: Color, mut title: Line<'_>) -> Block<'_> {
    title.spans.insert(0, Span::from(" "));
    Block::new()
        .bg(color)
        .title(title.bold())
        .padding(Padding::new(1, 1, 0, 1))
}

/// Whether `COLORTERM` says the terminal shows 24-bit color.
fn truecolor() -> bool {
    std::env::var("COLORTERM").is_ok_and(|term| term == "truecolor" || term == "24bit")
}

/// `from` moved `by` (0 to 1) of the way to `to`.
fn mix(from: Rgb, to: Rgb, by: f32) -> Rgb {
    let channel = |from: u8, to: u8| {
        let mixed = f32::from(from) + (f32::from(to) - f32::from(from)) * by;
        mixed.round().clamp(0.0, 255.0) as u8
    };
    (
        channel(from.0, to.0),
        channel(from.1, to.1),
        channel(from.2, to.2),
    )
}

/// How light `rgb` looks, from 0 to 1.
fn lightness((r, g, b): Rgb) -> f32 {
    (0.2126 * f32::from(r) + 0.7152 * f32::from(g) + 0.0722 * f32::from(b)) / 255.0
}

/// The bands of 256-color indices, in the order of [`Theme`]'s fields.
const fn indexed(
    [
        user,
        tool,
        added,
        added_word,
        removed,
        removed_word,
        selected,
        panel_surface,
        raised_surface,
        call_waiting,
        call_running,
        call_done,
        call_failed,
        input,
    ]: [u8; 14],
) -> Theme {
    Theme {
        user: Color::Indexed(user),
        tool: Color::Indexed(tool),
        added: Color::Indexed(added),
        added_word: Color::Indexed(added_word),
        removed: Color::Indexed(removed),
        removed_word: Color::Indexed(removed_word),
        selected: Color::Indexed(selected),
        panel_surface: Color::Indexed(panel_surface),
        raised_surface: Color::Indexed(raised_surface),
        call_waiting: Color::Indexed(call_waiting),
        call_running: Color::Indexed(call_running),
        call_done: Color::Indexed(call_done),
        call_failed: Color::Indexed(call_failed),
        input: Color::Indexed(input),
    }
}

#[cfg(test)]
mod tests;
