//! The input editor.

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use ratatui::layout::Position;
use ratatui::style::Stylize;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Padding, Paragraph};
use rig_ecs::agent::ActiveTurn;
use rig_harness::main_agent::ShownAgent;
use unicode_width::UnicodeWidthChar;

use super::input::InputContext;
use super::keys::KeyFocus;
use super::panel::PanelCanvas;
use super::pick::Choices;
use super::status::{StatusLine, StatusSources};
use super::theme::Theme;

/// The prompt before the first line of the input.
const PROMPT: &str = "› ";

/// Most lines an editor that fits its text shows; past them it scrolls to
/// keep the cursor in sight.
const MOST_LINES: usize = 10;

/// A text being typed: multiline, with a cursor and emacs-style editing
/// keys. The keys type into the [`MainEditor`], or into the editor of the
/// panel with the [`KeyFocus`](crate::KeyFocus); Enter sends the main
/// one's text to the shown agent. A plugin reads and edits a draft as the
/// keys do (a template, a dictated message, the text an editor wrote), the
/// main one through [`MainInput`]; `is_changed` on it says the draft was
/// edited. Up and Down move between lines and, from the first or last
/// line, through the [`PromptHistory`](crate::PromptHistory), keeping the
/// draft, which Down past the newest prompt brings back. In a panel placed
/// at the top, the bottom or below the input, it is as tall as its text, up
/// to ten lines.
///
/// ```no_run
/// use rig_harness::prelude::*;
/// use rig_tui::MainInput;
///
/// /// Puts a template in the input.
/// fn template(mut input: MainInput) {
///     input.set("Review the diff for:\n- ");
/// }
/// ```
#[derive(Component, Reflect, Clone, Debug, Default)]
#[reflect(Component, Default)]
#[require(Caret)]
pub struct Editor {
    text: String,
    /// A byte offset in `text`, on a character boundary.
    cursor: usize,
    /// The history entry shown, while browsing it.
    browsing: Option<usize>,
    /// What was typed before browsing started.
    draft: String,
}

/// Marks the editor the keys type into while no panel has the
/// [`KeyFocus`](crate::KeyFocus), whose text Enter sends to the shown
/// agent, and which a recalled message or a reload's draft comes back to:
/// the default screen's. An app that builds its own screen marks its own.
#[derive(Component, Clone, Copy, Debug, Default)]
pub struct MainEditor;

/// The main editor's text, to read or edit. A system with it is skipped
/// when there is no such editor, as in an app without the default screen.
pub type MainInput<'w, 's> = Single<'w, 's, &'static mut Editor, With<MainEditor>>;

/// Where an editor drew the cursor of the keys, on screen, while it has
/// them and no picker does.
#[derive(Component, Default)]
pub(crate) struct Caret(pub(crate) Option<Position>);

/// The input laid out to a width: its rows and where the cursor is.
pub(crate) struct Layout {
    pub(crate) rows: Vec<Line<'static>>,
    pub(crate) cursor_row: usize,
    pub(crate) cursor_column: usize,
}

impl Editor {
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

    /// The rows it fits its text in, `width` columns wide with its border:
    /// one line at least, [`MOST_LINES`] at most.
    pub(crate) fn fit_rows(&self, width: u16) -> u16 {
        let lines = self.layout(width.saturating_sub(2)).rows.len();
        u16::try_from(lines.clamp(1, MOST_LINES) + 2).unwrap_or(u16::MAX)
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
                Span::from(prefix).dim(),
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

/// The keys of an empty editor that has them, over where its text goes:
/// a message's, with Enter steering and Tab and Esc for the turn while one
/// runs.
fn hint(turn_running: bool, panel_keys: bool) -> &'static str {
    match (panel_keys, turn_running) {
        (true, _) => " A panel has the keys · Esc gives them back ",
        (false, true) => {
            " Enter steers · Tab sends after the turn · Esc stops · Shift+Enter or Ctrl+J new line · Ctrl+V image · / commands · @ files "
        }
        (false, false) => {
            " Enter sends · Shift+Enter or Ctrl+J new line · Ctrl+V image · / commands · @ files "
        }
    }
}

/// Draws each editor with room into its canvas, on the input's band with a
/// row above and below and a cell each side, scrolled to
/// keep its cursor in sight. The one the keys type into is titled by the
/// question it answers, if any, and while nothing is typed says over its
/// text what the keys do; it places the caret, unless a picker or a panel
/// without an editor has the keys. An editor with a [`StatusLine`] draws
/// it in its bottom row while nothing is typed.
pub(crate) fn draw_editors(
    mut editors: Query<(
        Entity,
        &Editor,
        Option<&StatusLine>,
        &mut PanelCanvas,
        &mut Caret,
    )>,
    (state, choices, focus): (Res<InputContext>, Choices, Res<KeyFocus>),
    (shown, active): (Res<ShownAgent>, Query<(), With<ActiveTurn>>),
    (theme, status): (Res<Theme>, StatusSources),
) {
    let answering = state.answering().and_then(|ask| choices.shown(ask));
    let busy = shown.0.is_some_and(|agent| active.contains(agent));
    for (entity, editor, line, mut canvas, mut caret) in &mut editors {
        let caret = &mut caret.bypass_change_detection().0;
        *caret = None;
        let area = canvas.area();
        if area.width < 3 || area.height < 3 {
            continue;
        }
        let layout = editor.layout(area.width - 2);
        let top = (layout.cursor_row + 1).saturating_sub(usize::from(area.height - 2));
        let (mut title, mut keys) = (Line::default(), "");
        if state.editor() == Some(entity) {
            let elsewhere = focus.0.is_some_and(|focused| focused != entity);
            keys = match &answering {
                Some(ask) => {
                    title = Line::from(format!(" {} ", ask.title)).bold();
                    " Enter answers · Shift+Enter or Ctrl+J new line · Esc refuses "
                }
                None => hint(busy, elsewhere),
            };
            if !editor.is_empty() {
                keys = "";
            }
            if state.picking().is_none() && !elsewhere {
                let column = u16::try_from(layout.cursor_column).unwrap_or(u16::MAX);
                let row = u16::try_from(layout.cursor_row - top).unwrap_or(0);
                let x = area.x.saturating_add(1).saturating_add(column);
                *caret = Some(Position::new(
                    x,
                    area.y.saturating_add(1).saturating_add(row),
                ));
            }
        }
        let top = u16::try_from(top).unwrap_or(u16::MAX);
        let (left, right) = match line.filter(|_| editor.is_empty()) {
            Some(line) => {
                let (left, right) = status.line(line, area.width - 2);
                let pad = |mut line: Line<'static>, at: usize| {
                    line.spans.insert(at.min(line.spans.len()), Span::from(" "));
                    line
                };
                (pad(left, 0), pad(right, usize::MAX))
            }
            None => (Line::default(), Line::default()),
        };
        // Titles, empty or not, keep the rows above and below the text.
        let block = Block::new()
            .bg(theme.input)
            .padding(Padding::horizontal(1))
            .title(title.left_aligned())
            .title(Line::from(keys).dim().right_aligned())
            .title_bottom(left.left_aligned())
            .title_bottom(right.right_aligned());
        canvas.render(Paragraph::new(layout.rows).scroll((top, 0)).block(block));
    }
}
