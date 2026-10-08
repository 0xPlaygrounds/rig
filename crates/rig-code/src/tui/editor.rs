//! The input editor: multiline text with a cursor, emacs-style editing
//! keys and the prompt history, which is kept in `RIG_HOME` across
//! sessions. Up and Down move between lines and, from the first or last
//! line, through the history, saving the draft as pi's editor does
//! (`references/pi/packages/tui/src/components/editor.ts:344-347,457-470`).

use std::fs;
use std::io::Write as _;
use std::path::PathBuf;

use bevy_log::warn;
use ratatui::style::{Style, Stylize};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthChar;

/// Prompts kept in the history.
const HISTORY: usize = 500;

/// The prompt before the first line of the input.
const PROMPT: &str = "› ";

/// The text being typed and the prompts typed before.
#[derive(Default)]
pub(crate) struct Editor {
    text: String,
    /// A byte offset in `text`, on a character boundary.
    cursor: usize,
    /// Earlier prompts, oldest first.
    history: Vec<String>,
    /// The history entry shown, while browsing it.
    browsing: Option<usize>,
    /// What was typed before browsing started.
    draft: String,
    /// Where the history is kept.
    file: Option<PathBuf>,
}

/// The input laid out to a width: its rows and where the cursor is.
pub(crate) struct Layout {
    pub(crate) rows: Vec<Line<'static>>,
    pub(crate) cursor_row: usize,
    pub(crate) cursor_column: usize,
}

impl Editor {
    /// An empty editor with the history kept in `file`.
    pub(crate) fn with_history(file: PathBuf) -> Self {
        let history = load_history(&file);
        Self {
            history,
            file: Some(file),
            ..Self::default()
        }
    }

    pub(crate) fn text(&self) -> &str {
        &self.text
    }

    pub(crate) fn cursor(&self) -> usize {
        self.cursor
    }

    pub(crate) fn is_empty(&self) -> bool {
        self.text.is_empty()
    }

    /// Replaces the text, with the cursor at its end.
    pub(crate) fn set(&mut self, text: String) {
        self.text = text;
        self.cursor = self.text.len();
    }

    /// Empties the editor and stops browsing the history.
    pub(crate) fn clear(&mut self) {
        self.text.clear();
        self.cursor = 0;
        self.browsing = None;
    }

    /// Takes the text out, leaving the editor empty.
    pub(crate) fn take(&mut self) -> String {
        self.cursor = 0;
        self.browsing = None;
        std::mem::take(&mut self.text)
    }

    pub(crate) fn insert(&mut self, text: &str) {
        let text = text.replace("\r\n", "\n").replace('\r', "\n");
        self.text.insert_str(self.cursor, &text);
        self.cursor += text.len();
    }

    pub(crate) fn insert_char(&mut self, character: char) {
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

    pub(crate) fn left(&mut self) {
        self.cursor = self.previous_boundary(self.cursor);
    }

    pub(crate) fn right(&mut self) {
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

    pub(crate) fn word_left(&mut self) {
        self.cursor = self.word_start();
    }

    pub(crate) fn word_right(&mut self) {
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

    pub(crate) fn home(&mut self) {
        self.cursor = self.line_start();
    }

    pub(crate) fn end(&mut self) {
        self.cursor = self.line_end();
    }

    pub(crate) fn backspace(&mut self) {
        let start = self.previous_boundary(self.cursor);
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    pub(crate) fn delete(&mut self) {
        let end = self.next_boundary(self.cursor);
        self.text.replace_range(self.cursor..end, "");
    }

    pub(crate) fn delete_word_before(&mut self) {
        let start = self.word_start();
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    pub(crate) fn delete_word_after(&mut self) {
        let end = self.word_end();
        self.text.replace_range(self.cursor..end, "");
    }

    /// Deletes to the start of the line, or the newline before it when the
    /// cursor is at the start already.
    pub(crate) fn delete_to_line_start(&mut self) {
        let start = match self.line_start() {
            start if start == self.cursor => self.previous_boundary(start),
            start => start,
        };
        self.text.replace_range(start..self.cursor, "");
        self.cursor = start;
    }

    /// Deletes to the end of the line, or the newline after it when the
    /// cursor is at the end already.
    pub(crate) fn delete_to_line_end(&mut self) {
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

    /// Up: the line above, else the previous prompt of the history.
    pub(crate) fn up(&mut self) {
        let start = self.line_start();
        if start == 0 {
            self.history_back();
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

    /// Down: the line below, else the next prompt of the history.
    pub(crate) fn down(&mut self) {
        let end = self.line_end();
        if end == self.text.len() {
            self.history_forward();
            return;
        }
        let column = self.column();
        self.move_to_column(end + 1, column);
    }

    fn history_back(&mut self) {
        let index = match self.browsing {
            Some(0) => return,
            Some(index) => index - 1,
            None if self.history.is_empty() => return,
            None => {
                self.draft = self.text.clone();
                self.history.len() - 1
            }
        };
        self.browsing = Some(index);
        let text = self.history.get(index).cloned().unwrap_or_default();
        self.set(text);
    }

    fn history_forward(&mut self) {
        let Some(index) = self.browsing else {
            return;
        };
        let text = if index + 1 < self.history.len() {
            self.browsing = Some(index + 1);
            self.history.get(index + 1).cloned().unwrap_or_default()
        } else {
            self.browsing = None;
            std::mem::take(&mut self.draft)
        };
        self.set(text);
    }

    /// Adds a sent prompt to the history and its file, unless it repeats
    /// the last one.
    pub(crate) fn remember(&mut self, text: &str) {
        let text = text.trim();
        if text.is_empty() || self.history.last().is_some_and(|last| last == text) {
            return;
        }
        self.history.push(text.to_owned());
        let excess = self.history.len().saturating_sub(HISTORY);
        self.history.drain(..excess);
        if let Some(file) = &self.file {
            let appended = serde_json::to_string(text)
                .map_err(std::io::Error::from)
                .and_then(|line| {
                    if let Some(parent) = file.parent() {
                        fs::create_dir_all(parent)?;
                    }
                    let mut out = fs::OpenOptions::new()
                        .create(true)
                        .append(true)
                        .open(file)?;
                    writeln!(out, "{line}")
                });
            if let Err(failure) = appended {
                warn!("could not save the prompt history: {failure}");
            }
        }
    }

    /// The text laid out in rows `width` columns wide, broken anywhere, the
    /// first row after the prompt and the others indented under it.
    pub(crate) fn layout(&self, width: u16, style: Style) -> Layout {
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
                Span::styled(std::mem::take(row), style),
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

/// The last [`HISTORY`] prompts of `file`, rewriting it when it grew past
/// twice that.
fn load_history(file: &PathBuf) -> Vec<String> {
    let Ok(text) = fs::read_to_string(file) else {
        return Vec::new();
    };
    let mut history: Vec<String> = text
        .lines()
        .filter_map(|line| serde_json::from_str(line).ok())
        .collect();
    let total = history.len();
    history.drain(..total.saturating_sub(HISTORY));
    if total > 2 * HISTORY {
        let kept: String = history
            .iter()
            .filter_map(|prompt| serde_json::to_string(prompt).ok())
            .map(|line| line + "\n")
            .collect();
        let temporary = file.with_extension("jsonl.new");
        if let Err(failure) =
            fs::write(&temporary, kept).and_then(|()| fs::rename(&temporary, file))
        {
            warn!("could not trim the prompt history: {failure}");
        }
    }
    history
}
