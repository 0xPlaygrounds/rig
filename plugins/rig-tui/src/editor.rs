//! The input editor.

use bevy_ecs::prelude::*;
use bevy_reflect::prelude::*;
use ratatui::layout::Position;
use ratatui::style::Stylize;
use ratatui::text::{Line, Span};
use ratatui::widgets::{Block, Padding, Paragraph};
use rig_ecs::agent::ActiveTurn;
use rig_harness::main_agent::ShownAgent;
use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};

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
/// edited. Up and Down move between lines; Up on the first line goes to
/// its start, and from there to the [`PromptHistory`](crate::PromptHistory),
/// keeping the draft, which Down past the newest prompt brings back; Ctrl+P
/// goes there from anywhere on the first line. While a prompt of it is
/// shown unchanged with the cursor at its start or end, Up and Down walk
/// the history. Text of the person's that the editor drops unsent goes to
/// the history, so Up brings it back: the draft behind a prompt shown when
/// another text replaces it, and what [`clear`](Self::clear) empties; a
/// question's answer never does. In a panel placed at the top, the bottom
/// or below the input, it is as tall as its text, up to ten lines.
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
    /// The history entry shown, while browsing it: its index, and its
    /// text, which finds it again when the entries move.
    browsing: Option<(usize, String)>,
    /// What was typed before browsing started; empty while not browsing.
    draft: String,
    /// The person's text the editor dropped unsent, which the history
    /// records.
    kept: Vec<String>,
    /// While it holds a question's answer, which is not a prompt to keep,
    /// the person's text taken out for it.
    answering: Option<String>,
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

    /// Replaces the text, with the cursor at its end. The text replaced is
    /// not kept, as a plugin edits with it too: one that discards the
    /// person's text calls [`clear`](Self::clear) first. Another text ends
    /// browsing the history, keeping the draft behind it there.
    pub fn set(&mut self, text: impl Into<String>) {
        let text = text.into();
        if text != self.text {
            self.stop_browsing();
        }
        self.show(text);
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

    /// Empties the editor and stops browsing the history. The person's
    /// text, and the draft behind a prompt of the history shown, go to
    /// the history.
    pub fn clear(&mut self) {
        let recalled = self.recalled();
        self.stop_browsing();
        let text = std::mem::take(&mut self.text);
        if !recalled {
            self.keep(text);
        }
        self.cursor = 0;
    }

    /// Takes the text out, leaving the input empty, for the caller to send
    /// or keep. The draft behind a prompt of the history shown goes to the
    /// history.
    pub fn take(&mut self) -> String {
        self.cursor = 0;
        self.stop_browsing();
        std::mem::take(&mut self.text)
    }

    /// Gives the input to a question answered in it, with its `initial`
    /// answer, and returns the person's text taken out: the draft behind a
    /// prompt of the history shown, else the text. Until
    /// [`take_answer`](Self::take_answer), nothing it drops is kept.
    pub(crate) fn answer(&mut self, initial: impl Into<String>) -> String {
        let draft = match self.recalled() {
            true => {
                self.browsing = None;
                std::mem::take(&mut self.draft)
            }
            false => self.take(),
        };
        self.set(initial);
        self.answering = Some(draft.clone());
        draft
    }

    /// Whether it holds a question's answer.
    pub(crate) fn answering(&self) -> bool {
        self.answering.is_some()
    }

    /// Takes a question's answer out when the question closes, leaving the
    /// input empty, and drops the history's place and the draft behind it.
    pub(crate) fn take_answer(&mut self) -> String {
        self.browsing = None;
        self.draft.clear();
        self.answering = None;
        self.cursor = 0;
        std::mem::take(&mut self.text)
    }

    /// For a question gone without closing: drops its answer and puts back
    /// the person's text taken out for it.
    pub(crate) fn drop_question(&mut self) {
        let draft = self.answering.take().unwrap_or_default();
        self.take_answer();
        self.set(draft);
    }

    /// Whether a prompt of the history is shown unchanged.
    pub(crate) fn recalled(&self) -> bool {
        self.browsing
            .as_ref()
            .is_some_and(|(_, entry)| *entry == self.text)
    }

    /// The person's own text: the draft behind a prompt of the history
    /// shown unchanged, else the text.
    pub(crate) fn unsent(&self) -> &str {
        match self.recalled() {
            true => &self.draft,
            false => &self.text,
        }
    }

    /// The texts dropped since the last call, for the history. An edited
    /// prompt of the history is the person's own text from now on.
    pub(crate) fn take_dropped(&mut self) -> Vec<String> {
        if !self.recalled() {
            self.stop_browsing();
        }
        std::mem::take(&mut self.kept)
    }

    /// Stops browsing the history, keeping the draft there: a prompt of it
    /// shown is the person's own text from now on, as if typed.
    pub(crate) fn stop_browsing(&mut self) {
        if self.browsing.take().is_some() {
            let draft = std::mem::take(&mut self.draft);
            self.keep(draft);
        }
    }

    /// Keeps `text` for the history as the newest, unless it is blank or an
    /// answer.
    fn keep(&mut self, text: String) {
        if self.answering.is_none() && !text.trim().is_empty() {
            self.kept.retain(|kept| *kept != text);
            self.kept.push(text);
        }
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

    /// Whether Up and Down walk the history: a prompt of it is shown
    /// unchanged, the cursor at its start or end.
    fn walking(&self) -> bool {
        self.recalled() && (self.cursor == 0 || self.cursor == self.text.len())
    }

    /// Up: the previous prompt of `history` (oldest first) while walking
    /// it, else the line above, else the start of the first line, else
    /// the newest prompt, keeping the text as the draft.
    pub fn up(&mut self, history: &[String]) {
        self.up_from(history, true);
    }

    /// Ctrl+P: as [`up`](Self::up), but from anywhere on the first line
    /// straight to the history.
    pub(crate) fn line_up(&mut self, history: &[String]) {
        self.up_from(history, false);
    }

    /// Up, going first to the start of the first line if `to_start`.
    fn up_from(&mut self, history: &[String], to_start: bool) {
        let start = self.line_start();
        if self.walking() || (start == 0 && (self.cursor == 0 || !to_start)) {
            self.history_back(history);
            return;
        }
        if start == 0 {
            self.cursor = 0;
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

    /// Down: the next prompt of `history`, or the draft after the newest,
    /// while walking it, else the line below, else the end of the last
    /// line.
    pub fn down(&mut self, history: &[String]) {
        if self.walking() {
            self.history_forward(history);
            return;
        }
        let end = self.line_end();
        if end == self.text.len() {
            self.cursor = end;
            return;
        }
        let column = self.column();
        self.move_to_column(end + 1, column);
    }

    fn history_back(&mut self, history: &[String]) {
        if !self.recalled() {
            self.stop_browsing();
        }
        let index = match &self.browsing {
            Some(shown) => match place(shown, history) {
                0 => return,
                index => index - 1,
            },
            None if history.is_empty() => return,
            None => {
                self.draft = self.text.clone();
                history.len() - 1
            }
        };
        self.browse(history, index);
    }

    fn history_forward(&mut self, history: &[String]) {
        let Some(shown) = &self.browsing else {
            return;
        };
        let index = place(shown, history) + 1;
        if index < history.len() {
            self.browse(history, index);
        } else {
            self.browsing = None;
            let draft = std::mem::take(&mut self.draft);
            self.show(draft);
        }
    }

    /// Shows the history entry at `index`.
    fn browse(&mut self, history: &[String], index: usize) {
        let text = history.get(index).cloned().unwrap_or_default();
        self.browsing = Some((index, text.clone()));
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

/// Where the history entry `shown` is now: at its index while it is still
/// there, else where its text moved, else past the newest.
fn place((index, text): &(usize, String), history: &[String]) -> usize {
    if history.get(*index) == Some(text) {
        return *index;
    }
    history
        .iter()
        .rposition(|entry| entry == text)
        .unwrap_or(history.len())
}

/// The keys of an empty editor that has them, over where its text goes:
/// a message's, with Enter steering and Tab and Esc for the turn while one
/// runs.
fn hint(turn_running: bool, panel_keys: bool) -> &'static str {
    match (panel_keys, turn_running) {
        (true, _) => " A panel has the keys · Esc gives them back ",
        (false, true) => {
            " Enter steers · Tab sends after the turn · Esc stops · Shift+Enter or Ctrl+J new line · Ctrl+V image · / commands · @ files · Ctrl+D quits "
        }
        (false, false) => {
            " Enter sends · Shift+Enter or Ctrl+J new line · Ctrl+V image · / commands · @ files · Ctrl+D quits "
        }
    }
}

/// `keys`, items split by ` · `, in `width` columns: the first items that
/// fit, whole.
fn fit_keys(keys: &str, width: usize) -> String {
    let mut fitted = String::new();
    for item in keys.split(" · ").map(str::trim) {
        let next = match fitted.is_empty() {
            true => format!(" {item} "),
            false => format!("{} · {item} ", fitted.trim_end()),
        };
        if next.width() > width {
            break;
        }
        fitted = next;
    }
    fitted
}

/// Draws each editor with room into its canvas, on the input's band with a
/// row above and below and a cell each side, scrolled to
/// keep its cursor in sight. The one the keys type into is titled by the
/// question it answers, if any, and while nothing is typed says over its
/// text what the keys do, or while it shows a prompt of the history, that
/// Down goes back; it places the caret, unless a picker or a panel
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
            if editor.recalled() && answering.is_none() {
                keys = match editor.draft.trim().is_empty() {
                    true => " An earlier prompt · Down back to an empty input ",
                    false => " An earlier prompt · Down back to your draft ",
                };
            } else if !editor.is_empty() {
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
        let keys = fit_keys(keys, usize::from(area.width).saturating_sub(title.width()));
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
