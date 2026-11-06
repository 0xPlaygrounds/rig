//! Text dragged over with the mouse, as a terminal that reports the mouse
//! gives up its own selection, which Shift and a drag still reach.

use ratatui::buffer::{Buffer, Cell, CellWidth};
use ratatui::layout::{Position, Rect};
use ratatui::style::Modifier;

/// The cells between two places, in reading order, as a terminal selects.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) struct Selection {
    /// Where the button went down.
    pub(crate) from: Position,
    /// Where the mouse is, or the button came up.
    pub(crate) to: Position,
    /// Whether the button came up and the next frame copies the text.
    pub(crate) up: bool,
    /// Whether the button is still down, so a drag or release moves `to`.
    pub(crate) held: bool,
}

impl Selection {
    /// Shows the selected symbols of `buffer` reversed and returns their
    /// text: a line for each row, without the blanks that end it. A row is
    /// read from its left edge, each symbol over the cells its width
    /// covers, as ratatui draws it, so a symbol any cell of which is
    /// selected is selected whole and taken once.
    pub(crate) fn mark(self, buffer: &mut Buffer) -> String {
        let (from, to) = (self.from, self.to);
        let forward = (from.y, from.x) <= (to.y, to.x);
        let (start, end) = if forward { (from, to) } else { (to, from) };
        let (left, right) = (buffer.area.left(), buffer.area.right());
        let mut lines = Vec::new();
        for y in start.y..=end.y {
            let first = if y == start.y { start.x } else { left };
            let last = if y == end.y { end.x } else { right };
            let (mut line, mut x) = (String::new(), left);
            while x < right {
                let width = buffer
                    .cell((x, y))
                    .map_or(1, |cell| cell.cell_width().max(1));
                let next = x.saturating_add(width).min(right);
                if x <= last && next > first {
                    line.push_str(buffer.cell((x, y)).map_or("", Cell::symbol));
                    buffer.set_style(Rect::new(x, y, next - x, 1), Modifier::REVERSED);
                }
                x = next;
            }
            lines.push(line.trim_end().to_owned());
        }
        lines.join("\n").trim_end().to_owned()
    }
}
