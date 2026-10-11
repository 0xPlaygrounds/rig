//! The input editor: multiline text with a cursor and emacs-style editing
//! keys. It is the [`TuiInput`] resource, so a plugin reads and edits the
//! draft as the keys do (a template, a dictated message, the text an
//! editor wrote), and `is_changed` on it says the draft was edited. Up and
//! Down move between lines and, from the first or last line, through the
//! [`PromptHistory`](crate::PromptHistory), saving the draft as pi's
//! editor does
//! (`references/pi/packages/tui/src/components/editor.ts:344-347,457-470`).

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use ratatui::style::Stylize;
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthChar;

/// The prompt before the first line of the input.
const PROMPT: &str = "› ";

/// The terminal view's input: the text being typed, with its cursor, and
/// where it is in the prompt history while Up and Down browse it. It is
/// there in any run, empty unless the view runs. Enter sends what it holds
/// to the agent shown.
#[derive(Resource, Reflect, Default)]
#[reflect(Resource, Default)]
pub struct TuiInput {
    text: String,
    /// A byte offset in `text`, on a character boundary.
    cursor: usize,
    /// The history entry shown, while browsing it.
    browsing: Option<usize>,
    /// What was typed before browsing started.
    draft: String,
}

/// The input laid out to a width: its rows and where the cursor is.
pub(crate) struct Layout {
    pub(crate) rows: Vec<Line<'static>>,
    pub(crate) cursor_row: usize,
    pub(crate) cursor_column: usize,
}

impl TuiInput {
    /// The text.
    pub fn text(&self) -> &str {
        &self.text
    }

    /// The cursor: a byte offset in [`Self::text`], on a character
    /// boundary.
    pub fn cursor(&self) -> usize {
        self.cursor
    }

    /// Whether nothing is typed.
    pub fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Replaces the text, with the cursor at its end.
    pub fn set(&mut self, text: impl Into<String>) {
        self.browsing = None;
        self.show(text.into());
    }

    /// Puts the cursor at byte `offset`, or the nearest character boundary
    /// before it.
    pub fn set_cursor(&mut self, offset: usize) {
        let mut offset = offset.min(self.text.len());
        while !self.text.is_char_boundary(offset) {
            offset -= 1;
        }
        self.cursor = offset;
    }

    /// Shows `text`, with the cursor at its end, keeping the history's
    /// place.
    fn show(&mut self, text: String) {
        self.text = text;
        self.cursor = self.text.len();
    }

    /// Empties the editor and stops browsing the history.
    pub fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.browsing = None;
    }

    /// Takes the text out, leaving the input empty.
    pub fn take(&mut self) -> String {
        self.cursor = 0;
        self.browsing = None;
        std::mem::take(&mut self.text)
    }

    /// Types `text` at the cursor, which ends up after it.
    pub fn insert(&mut self, text: &str) {
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        self.text.insert_str(self.cursor, &text);
        self.cursor += text.len();
    }

    /// Types `character` at the cursor, which ends up after it.
    pub fn insert_char(&mut self, character: char) {
        self.text.insert(self.cursor, character);
        self.cursor += character.len_utf8();
    }

    /// Replaces the bytes `start..self.cursor()` with `text`.
    pub(crate) fn replace_before_cursor(&mut self, start: usize, text: &str) {
        if start <= self.cursor && self.text.is_char_boundary(start) {
            self.text.replace_range(start..self.cursor, text);
            self.cursor = start + text.len();
        }
    }

    /// Enter with a `\` before the cursor: the `\` becomes a newline.
    /// Returns whether it did.
    pub(crate) fn continue_line(&mut self) -> bool {
        if self
            .text
            .get(..self.cursor)
            .is_some_and(|before| before.ends_with('\\'))
        {
            self.cursor -= 1;
            self.text.remove(self.cursor);
            self.insert_char('\n');
            return true;
        }
        false
    }

    fn previous_boundary(&self, from: usize) -> usize {
        self.text
            .get(..from)
            .and_then(|before| before.char_indices().next_back())
            .map_or(0, |(index, _)| index)
    }

    fn next_boundary(&self, from: usize) -> usize {
        self.text
            .get(from..)
            .and_then(|after| after.chars().next())
            .map_or(self.text.len(), |character| from + character.len_utf8())
    }

    /// A character back.
    pub fn left(&mut self) {
        self.cursor = self.previous_boundary(self.cursor);
    }

    /// A character forward.
    pub fn right(&mut self) {
        self.cursor = self.next_boundary(self.cursor);
    }

    /// The start of the word before the cursor.
    fn word_start(&self) -> usize {
        let before = self.text.get(..self.cursor).unwrap_or_default();
        let trimmed = before.trim_end_matches(|c: char| !c.is_alphanumeric() && c != '_');
        trimmed
            .rfind(|c: char| !c.is_alphanumeric() && c != '_')
            .map_or(0, |index| {
                index
                    + trimmed
                        .get(index..)
                        .and_then(|rest| rest.chars().next())
                        .map_or(1, char::len_utf8)
            })
    }

    /// The end of the word after the cursor.
    fn word_end(&self) -> usize {
        let after = self.text.get(self.cursor..).unwrap_or_default();
        let skipped = after.len()
            - after
                .trim_start_matches(|c: char| !c.is_alphanumeric() && c != '_')
                .len();
        let rest = after.get(skipped..).unwrap_or_default();
        let word = rest
            .find(|c: char| !c.is_alphanumeric() && c != '_')
            .unwrap_or(rest.len());
        self.cursor + skipped + word
    }

    /// The start of the word before the cursor.
    pub fn word_left(&mut self) {
        self.cursor = self.word_start();
    }

    /// The end of the word after the cursor.
    pub fn word_right(&mut self) {
        self.cursor = self.word_end();
    }

    /// The start of the cursor's line.
    fn line_start(&self) -> usize {
        self.text
            .get(..self.cursor)
            .and_then(|before| before.rfind('\n'))
            .map_or(0, |index| index + 1)
    }

    /// The end of the cursor's line.
    fn line_end(&self) -> usize {
        self.text
            .get(self.cursor..)
            .and_then(|after| after.find('\n'))
            .map_or(self.text.len(), |index| self.cursor + index)
    }

    /// The start of the cursor's line.
    pub fn home(&mut self) {
        self.cursor = self.line_start();
    }

    /// The end of the cursor's line.
    pub fn end(&mut self) {
        self.cursor = self.line_end();
    }

    /// Deletes the character before the cursor.
    pub fn backspace(&mut self) {
        let start = self.previous_boundary(self.cursor);
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    /// Deletes the character after the cursor.
    pub fn delete(&mut self) {
        let end = self.next_boundary(self.cursor);
        self.text.replace_range(self.cursor..end, "");
    }

    /// Deletes the word before the cursor.
    pub fn delete_word_before(&mut self) {
        let start = self.word_start();
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    /// Deletes the word after the cursor.
    pub fn delete_word_after(&mut self) {
        let end = self.word_end();
        self.text.replace_range(self.cursor..end, "");
    }

    /// Deletes to the start of the line, or the newline before it when the
    /// cursor is at the start already.
    pub fn delete_to_line_start(&mut self) {
        let start = match self.line_start() {
            start if start == self.cursor => self.previous_boundary(start),
            start => start,
        };
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    /// Deletes to the end of the line, or the newline after it when the
    /// cursor is at the end already.
    pub fn delete_to_line_end(&mut self) {
        let end = match self.line_end() {
            end if end == self.cursor => self.next_boundary(end),
            end => end,
        };
        self.text.replace_range(self.cursor..end, "");
    }

    /// The cursor's column on its line, in characters.
    fn column(&self) -> usize {
        self.text
            .get(self.line_start()..self.cursor)
            .map_or(0, |line| line.chars().count())
    }

    /// Moves to `column` of the line starting at `start`, or its end.
    fn move_to_column(&mut self, start: usize, column: usize) {
        let line = self.text.get(start..).unwrap_or_default();
        let line = line.split('\n').next().unwrap_or_default();
        self.cursor = start
            + line
                .char_indices()
                .nth(column)
                .map_or(line.len(), |(index, _)| index);
    }

    /// Up: the line above, else the previous prompt of `history` (oldest
    /// first).
    pub fn up(&mut self, history: &[String]) {
        let start = self.line_start();
        if start == 0 {
            self.history_back(history);
            return;
        }
        let column = self.column();
        let above = self
            .text
            .get(..start - 1)
            .and_then(|before| before.rfind('\n'))
            .map_or(0, |index| index + 1);
        self.move_to_column(above, column);
    }

    /// Down: the line below, else the next prompt of `history`, or the
    /// draft after the newest.
    pub fn down(&mut self, history: &[String]) {
        let end = self.line_end();
        if end == self.text.len() {
            self.history_forward(history);
            return;
        }
        let column = self.column();
        self.move_to_column(end + 1, column);
    }

    fn history_back(&mut self, history: &[String]) {
        let index = match self.browsing {
            Some(0) => return,
            Some(index) => index.min(history.len()).saturating_sub(1),
            None if history.is_empty() => return,
            None => {
                self.draft = self.text.clone();
                history.len() - 1
            }
        };
        self.browsing = Some(index);
        let text = history.get(index).cloned().unwrap_or_default();
        self.show(text);
    }

    fn history_forward(&mut self, history: &[String]) {
        let Some(index) = self.browsing else {
            return;
        };
        let text = if index + 1 < history.len() {
            self.browsing = Some(index + 1);
            history.get(index + 1).cloned().unwrap_or_default()
        } else {
            self.browsing = None;
            std::mem::take(&mut self.draft)
        };
        self.show(text);
    }

    /// The text laid out in rows `width` columns wide, broken anywhere, the
    /// first row after the prompt and the others indented under it.
    pub(crate) fn layout(&self, width: u16) -> Layout {
        // The prompt is two columns wide.
        let width = usize::from(width).max(3);
        let mut rows = Vec::new();
        let mut row = String::new();
        let mut row_width = 2;
        let mut cursor_row = 0;
        let mut cursor_column = 2;
        let mut first = true;
        let finish = |row: &mut String, first: &mut bool, rows: &mut Vec<Line<'static>>| {
            let prefix = if *first { PROMPT } else { "  " };
            *first = false;
            rows.push(Line::from(vec![
                Span::from(prefix).cyan(),
                Span::from(std::mem::take(row)),
            ]));
        };
        for (index, character) in self.text.char_indices() {
            if index == self.cursor {
                cursor_row = rows.len();
                cursor_column = row_width;
            }
            if character == '\n' {
                finish(&mut row, &mut first, &mut rows);
                row_width = 2;
                continue;
            }
            let cell = character.width().unwrap_or(0);
            if row_width + cell > width {
                finish(&mut row, &mut first, &mut rows);
                row_width = 2;
                if index == self.cursor {
                    cursor_row = rows.len();
                    cursor_column = row_width;
                }
            }
            row.push(if character == '\t' { ' ' } else { character });
            row_width += cell;
        }
        if self.cursor >= self.text.len() {
            if row_width >= width {
                finish(&mut row, &mut first, &mut rows);
                row_width = 2;
            }
            cursor_row = rows.len();
            cursor_column = row_width;
        }
        finish(&mut row, &mut first, &mut rows);
        Layout {
            rows,
            cursor_row,
            cursor_column,
        }
    }
}
