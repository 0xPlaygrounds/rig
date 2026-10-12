//! The modal editor itself: vim's modes, motions, operators, registers and
//! undo over a text and a cursor, with no terminal and no ECS, so the
//! plugin only feeds it keys and copies the text back into the input.

use bevy_reflect::Reflect;

/// The mode the input is in.
#[derive(Reflect, Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Mode {
    /// Keys type text, as without the plugin.
    #[default]
    Insert,
    /// Keys move, operate and switch modes.
    Normal,
    /// Motions extend a selection that `d`, `y` and `c` act on.
    Visual,
}

impl Mode {
    /// What the status line says in this mode.
    pub fn label(self) -> &'static str {
        match self {
            Self::Insert => "-- INSERT --",
            Self::Normal => "-- NORMAL --",
            Self::Visual => "-- VISUAL --",
        }
    }
}

/// A key the editor acts on.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Key {
    /// A printable character.
    Char(char),
    /// Escape.
    Esc,
    /// Ctrl-r, redo.
    Redo,
}

/// What the editor asks of whoever hosts it after a key.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Effect {
    /// `"+y`: put this text on the system clipboard.
    Copy(String),
    /// `"+p` or `"+P`: paste the system clipboard after the cursor or
    /// before it, through [`Editor::put_text`].
    PasteClipboard {
        /// After the cursor (`p`) rather than before it (`P`).
        after: bool,
    },
    /// Esc in normal mode with nothing to cancel.
    BareEsc,
}

/// The text being edited and its cursor, a byte offset on a character
/// boundary.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Buffer {
    /// The text.
    pub text: String,
    /// The cursor.
    pub cursor: usize,
}

/// What `y` and `d` keep for `p`.
#[derive(Clone, Debug, Default)]
struct Register {
    text: String,
    linewise: bool,
}

/// Vim's modal editing over a [`Buffer`].
#[derive(Debug, Default)]
pub struct Editor {
    mode: Mode,
    /// The keys of a command not complete yet, such as `2d`.
    pending: Vec<char>,
    register: Register,
    /// Where the visual selection started.
    anchor: usize,
    undo: Vec<Buffer>,
    redo: Vec<Buffer>,
    /// The text as insert mode was entered, for one undo step per insert.
    insert_from: Option<Buffer>,
}

impl Editor {
    /// An editor starting in `mode`.
    pub fn new(mode: Mode) -> Self {
        Self {
            mode,
            ..Self::default()
        }
    }

    /// The mode.
    pub fn mode(&self) -> Mode {
        self.mode
    }

    /// Back to `mode` with nothing pending, as after a message was sent.
    pub fn reset(&mut self, mode: Mode) {
        self.mode = mode;
        self.pending.clear();
        self.insert_from = None;
    }

    /// Acts on `key`, editing `buffer`.
    pub fn key(&mut self, key: Key, buffer: &mut Buffer) -> Option<Effect> {
        match (key, self.mode) {
            (Key::Esc, _) => self.escape(buffer),
            (Key::Char(character), Mode::Insert) => {
                insert(buffer, &character.to_string());
                None
            }
            (Key::Redo, Mode::Normal) => {
                self.pending.clear();
                if let Some(next) = self.redo.pop() {
                    self.undo.push(std::mem::replace(buffer, next));
                    clamp(buffer);
                }
                None
            }
            (Key::Redo, _) => None,
            (Key::Char(character), Mode::Normal | Mode::Visual) => {
                self.pending.push(character);
                match parse(&self.pending, self.mode == Mode::Visual) {
                    Parse::More => None,
                    Parse::Bad => {
                        self.pending.clear();
                        None
                    }
                    Parse::Done(command) => {
                        self.pending.clear();
                        self.run(command, buffer)
                    }
                }
            }
        }
    }

    /// Pastes `text` from outside, such as the clipboard, as `p` would.
    pub fn put_text(&mut self, buffer: &mut Buffer, text: &str, after: bool) {
        let before = buffer.clone();
        put(buffer, text, false, after, 1);
        self.changed(before, buffer);
    }

    fn escape(&mut self, buffer: &mut Buffer) -> Option<Effect> {
        let pending = !self.pending.is_empty();
        self.pending.clear();
        match self.mode {
            Mode::Insert => {
                self.mode = Mode::Normal;
                if let Some(from) = self.insert_from.take() {
                    self.changed(from, buffer);
                }
                if buffer.cursor > line_start(&buffer.text, buffer.cursor) {
                    buffer.cursor = prev(&buffer.text, buffer.cursor);
                }
                None
            }
            Mode::Visual => {
                self.mode = Mode::Normal;
                clamp(buffer);
                None
            }
            Mode::Normal => (!pending).then_some(Effect::BareEsc),
        }
    }

    /// Keeps `before` as an undo step if `buffer`'s text changed.
    fn changed(&mut self, before: Buffer, buffer: &Buffer) {
        if before.text != buffer.text {
            self.undo.push(before);
            self.redo.clear();
        }
    }

    fn run(&mut self, command: Command, buffer: &mut Buffer) -> Option<Effect> {
        let before = buffer.clone();
        let count = command.count.unwrap_or(1);
        let mut effect = None;
        match command.action {
            Action::Undo => {
                for _ in 0..count {
                    if let Some(earlier) = self.undo.pop() {
                        self.redo.push(std::mem::replace(buffer, earlier));
                    }
                }
                clamp(buffer);
                return None;
            }
            Action::Visual => {
                self.mode = match self.mode {
                    Mode::Visual => Mode::Normal,
                    _ => Mode::Visual,
                };
                self.anchor = buffer.cursor;
            }
            Action::Move(motion) => {
                if let Some(target) = target(buffer, motion, command.count, false) {
                    buffer.cursor = match target.kind {
                        Kind::Linewise => first_non_blank(&buffer.text, target.pos),
                        _ => target.pos,
                    };
                }
            }
            Action::Insert(at) => {
                buffer.cursor = match at {
                    InsertAt::Cursor => buffer.cursor,
                    InsertAt::After => match char_at(&buffer.text, buffer.cursor) {
                        Some('\n') | None => buffer.cursor,
                        Some(_) => next(&buffer.text, buffer.cursor),
                    },
                    InsertAt::LineStart => first_non_blank(&buffer.text, buffer.cursor),
                    InsertAt::LineEnd => line_end(&buffer.text, buffer.cursor),
                    InsertAt::Below => {
                        buffer.cursor = line_end(&buffer.text, buffer.cursor);
                        insert(buffer, "\n");
                        buffer.cursor
                    }
                    InsertAt::Above => {
                        let start = line_start(&buffer.text, buffer.cursor);
                        buffer.cursor = start;
                        insert(buffer, "\n");
                        start
                    }
                };
                self.mode = Mode::Insert;
            }
            Action::Put { after } if command.clipboard => {
                return Some(Effect::PasteClipboard { after });
            }
            Action::Put { after } => {
                let Register { text, linewise } = self.register.clone();
                put(buffer, &text, linewise, after, count);
            }
            Action::Operate(operator, span) => {
                effect = self.operate(operator, span, &command, buffer);
            }
        }
        if self.mode == Mode::Insert {
            self.insert_from = Some(before);
        } else {
            self.changed(before, buffer);
            if self.mode == Mode::Normal {
                clamp(buffer);
            }
        }
        effect
    }

    fn operate(
        &mut self,
        operator: Operator,
        span: Span,
        command: &Command,
        buffer: &mut Buffer,
    ) -> Option<Effect> {
        let text = &buffer.text;
        let cursor = buffer.cursor;
        let (start, end, linewise) = match span {
            Span::Lines => {
                let first = line_start(text, cursor);
                let last = lines_down(text, first, command.count.unwrap_or(1) - 1);
                (first, line_end(text, last), true)
            }
            Span::Selection => {
                let (low, high) = (self.anchor.min(cursor), self.anchor.max(cursor));
                (low, next(text, high), false)
            }
            Span::Motion(motion) => {
                // `cw` on a word changes to its end, as `ce`.
                let on_word = char_at(text, cursor).is_some_and(|c| !c.is_whitespace());
                let motion = match (operator, motion) {
                    (Operator::Change, Motion::WordStart) if on_word => Motion::WordEnd,
                    _ => motion,
                };
                let target = target(buffer, motion, command.count, true)?;
                let (low, high) = (cursor.min(target.pos), cursor.max(target.pos));
                match target.kind {
                    Kind::Exclusive => {
                        // `dw` over the last word of a line stops at its end.
                        let high = match (motion, text.get(low..high)) {
                            (Motion::WordStart, Some(moved)) => {
                                moved.rfind('\n').map_or(high, |at| low + at)
                            }
                            _ => high,
                        };
                        (low, high, false)
                    }
                    Kind::Inclusive => (low, next(text, high), false),
                    Kind::Linewise => (line_start(text, low), line_end(text, high), true),
                }
            }
        };
        let taken = buffer.text.get(start..end).unwrap_or_default().to_owned();
        self.register = Register {
            text: taken.clone(),
            linewise,
        };
        if self.mode == Mode::Visual {
            self.mode = Mode::Normal;
        }
        match operator {
            Operator::Yank => {
                if !linewise {
                    buffer.cursor = start;
                }
                return command.clipboard.then_some(Effect::Copy(taken));
            }
            Operator::Delete => {
                let len = buffer.text.len();
                let (from, to) = match (linewise, end < len, start > 0) {
                    (true, true, _) => (start, end + 1),
                    (true, false, true) => (start - 1, end),
                    _ => (start, end),
                };
                splice(buffer, from, to, "");
                buffer.cursor = match linewise {
                    true => first_non_blank(&buffer.text, from.min(buffer.text.len())),
                    false => from,
                };
            }
            Operator::Change => {
                splice(buffer, start, end, "");
                buffer.cursor = start;
                self.mode = Mode::Insert;
            }
        }
        command.clipboard.then_some(Effect::Copy(taken))
    }
}

/// A complete command.
#[derive(Debug)]
struct Command {
    /// `"+` was given: the system clipboard is the register.
    clipboard: bool,
    count: Option<usize>,
    action: Action,
}

#[derive(Clone, Copy, Debug)]
enum Action {
    Move(Motion),
    Operate(Operator, Span),
    Put { after: bool },
    Insert(InsertAt),
    Undo,
    Visual,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Operator {
    Delete,
    Change,
    Yank,
}

/// What an operator acts on.
#[derive(Clone, Copy, Debug)]
enum Span {
    /// `dd`: whole lines.
    Lines,
    /// The visual selection.
    Selection,
    Motion(Motion),
}

#[derive(Clone, Copy, Debug)]
enum InsertAt {
    Cursor,
    After,
    LineStart,
    LineEnd,
    Below,
    Above,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Motion {
    Left,
    Right,
    Down,
    Up,
    WordStart,
    WordBack,
    WordEnd,
    LineStart,
    FirstNonBlank,
    LineEnd,
    FirstLine,
    LastLine,
    Find(char),
    Till(char),
}

enum Parse {
    More,
    Bad,
    Done(Command),
}

/// Reads the keys typed so far as a command: `["x] [count] action`, an
/// operator taking its own count and a motion, or itself doubled.
fn parse(keys: &[char], visual: bool) -> Parse {
    let mut rest = keys;
    let mut clipboard = false;
    if let Some(('"', after)) = rest.split_first() {
        let Some((register, after)) = after.split_first() else {
            return Parse::More;
        };
        clipboard = *register == '+';
        rest = after;
    }
    let (first, after) = count(rest);
    let Some((&key, after)) = after.split_first() else {
        return Parse::More;
    };
    let done = |count: Option<usize>, action| {
        Parse::Done(Command {
            clipboard,
            count,
            action,
        })
    };
    let operator = match key {
        'd' | 'x' => Some(Operator::Delete),
        'c' => Some(Operator::Change),
        'y' => Some(Operator::Yank),
        _ => None,
    };
    if let (true, Some(operator)) = (visual, operator) {
        return done(first, Action::Operate(operator, Span::Selection));
    }
    let simple = match key {
        'x' => Some(Action::Operate(
            Operator::Delete,
            Span::Motion(Motion::Right),
        )),
        'D' => Some(Action::Operate(
            Operator::Delete,
            Span::Motion(Motion::LineEnd),
        )),
        'C' => Some(Action::Operate(
            Operator::Change,
            Span::Motion(Motion::LineEnd),
        )),
        'p' => Some(Action::Put { after: true }),
        'P' => Some(Action::Put { after: false }),
        'i' => Some(Action::Insert(InsertAt::Cursor)),
        'a' => Some(Action::Insert(InsertAt::After)),
        'I' => Some(Action::Insert(InsertAt::LineStart)),
        'A' => Some(Action::Insert(InsertAt::LineEnd)),
        'o' => Some(Action::Insert(InsertAt::Below)),
        'O' => Some(Action::Insert(InsertAt::Above)),
        'u' => Some(Action::Undo),
        'v' => Some(Action::Visual),
        _ => None,
    };
    if let Some(action) = simple {
        return done(first, action);
    }
    if let Some(operator) = operator {
        let (second, after) = count(after);
        let count = match (first, second) {
            (None, None) => None,
            (first, second) => Some(first.unwrap_or(1) * second.unwrap_or(1)),
        };
        return match after.split_first() {
            None => Parse::More,
            Some((&doubled, _)) if doubled == key => {
                done(count, Action::Operate(operator, Span::Lines))
            }
            Some((&key, after)) => match motion(key, after) {
                Parse::Done(Command { action, .. }) => match action {
                    Action::Move(motion) => {
                        done(count, Action::Operate(operator, Span::Motion(motion)))
                    }
                    _ => Parse::Bad,
                },
                other => other,
            },
        };
    }
    match motion(key, after) {
        Parse::Done(command) => done(first, command.action),
        other => other,
    }
}

/// The digits a command starts with, unless the first is `0`, which is a
/// motion.
fn count(keys: &[char]) -> (Option<usize>, &[char]) {
    let digits = match keys.first() {
        Some('1'..='9') => keys.iter().take_while(|key| key.is_ascii_digit()).count(),
        _ => 0,
    };
    let (number, rest) = keys.split_at(digits.min(keys.len()));
    let value = number
        .iter()
        .try_fold(0usize, |total, digit| {
            total
                .checked_mul(10)?
                .checked_add(digit.to_digit(10)? as usize)
        })
        .filter(|_| digits > 0)
        .map(|value| value.min(10_000));
    (value, rest)
}

/// A motion from `key` and the keys after it.
fn motion(key: char, rest: &[char]) -> Parse {
    let motion = match (key, rest.first()) {
        ('h', _) => Motion::Left,
        ('l', _) => Motion::Right,
        ('j', _) => Motion::Down,
        ('k', _) => Motion::Up,
        ('w', _) => Motion::WordStart,
        ('b', _) => Motion::WordBack,
        ('e', _) => Motion::WordEnd,
        ('0', _) => Motion::LineStart,
        ('^', _) => Motion::FirstNonBlank,
        ('$', _) => Motion::LineEnd,
        ('G', _) => Motion::LastLine,
        ('g' | 'f' | 't', None) => return Parse::More,
        ('g', Some('g')) => Motion::FirstLine,
        ('f', Some(&target)) => Motion::Find(target),
        ('t', Some(&target)) => Motion::Till(target),
        _ => return Parse::Bad,
    };
    Parse::Done(Command {
        clipboard: false,
        count: None,
        action: Action::Move(motion),
    })
}

/// How an operator takes the text up to a motion's target.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Kind {
    /// Up to the target, not including it.
    Exclusive,
    /// Up to and including the target's character.
    Inclusive,
    /// Every line from the cursor's to the target's.
    Linewise,
}

struct Target {
    pos: usize,
    kind: Kind,
}

/// Where `motion` repeated `count` times goes from the cursor; `None`
/// where it cannot go. An operator's `l` may go past the line's last
/// character.
fn target(buffer: &Buffer, motion: Motion, count: Option<usize>, operator: bool) -> Option<Target> {
    let text = buffer.text.as_str();
    let cursor = buffer.cursor;
    let times = count.unwrap_or(1);
    let start = line_start(text, cursor);
    let end = line_end(text, cursor);
    let repeat = |step: fn(&str, usize) -> usize| (0..times).fold(cursor, |pos, _| step(text, pos));
    let (pos, kind) = match motion {
        Motion::Left => (
            (0..times).fold(cursor, |pos, _| prev(text, pos).max(start)),
            Kind::Exclusive,
        ),
        Motion::Right => {
            let last = if operator {
                end
            } else {
                prev(text, end).max(start)
            };
            (
                (0..times).fold(cursor, |pos, _| next(text, pos).min(last)),
                Kind::Exclusive,
            )
        }
        Motion::Down | Motion::Up => {
            let column = text
                .get(start..cursor)
                .map_or(0, |before| before.chars().count());
            let line = match motion {
                Motion::Down => lines_down(text, start, times),
                _ => lines_up(text, start, times),
            };
            let line_text = text.get(line..line_end(text, line)).unwrap_or_default();
            let offset = line_text
                .char_indices()
                .nth(column)
                .map_or(line_text.len(), |(at, _)| at);
            (line + offset, Kind::Linewise)
        }
        Motion::WordStart => (repeat(word_start), Kind::Exclusive),
        Motion::WordBack => (repeat(word_back), Kind::Exclusive),
        Motion::WordEnd => (repeat(word_end), Kind::Inclusive),
        Motion::LineStart => (start, Kind::Exclusive),
        Motion::FirstNonBlank => (first_non_blank(text, cursor), Kind::Exclusive),
        Motion::LineEnd => {
            let line = lines_down(text, start, times - 1);
            let end = line_end(text, line);
            match end > line {
                true => (prev(text, end), Kind::Inclusive),
                false => (line, Kind::Exclusive),
            }
        }
        Motion::FirstLine | Motion::LastLine => {
            let number = match (motion, count) {
                (_, Some(number)) => number,
                (Motion::FirstLine, None) => 1,
                _ => usize::MAX,
            };
            (
                lines_down(text, 0, number.saturating_sub(1)),
                Kind::Linewise,
            )
        }
        Motion::Find(wanted) | Motion::Till(wanted) => {
            let from = next(text, cursor);
            let line = text.get(from..end).unwrap_or_default();
            let (at, _) = line.match_indices(wanted).nth(times - 1)?;
            let found = from + at;
            match motion {
                Motion::Till(_) => (prev(text, found).max(cursor), Kind::Inclusive),
                _ => (found, Kind::Inclusive),
            }
        }
    };
    Some(Target { pos, kind })
}

/// Pastes `text` `count` times: whole lines below or above the cursor's,
/// else after or before the cursor's character.
fn put(buffer: &mut Buffer, text: &str, linewise: bool, after: bool, count: usize) {
    let many = text.repeat(count);
    if many.is_empty() {
        return;
    }
    if linewise {
        let line = if after {
            let end = line_end(&buffer.text, buffer.cursor);
            buffer.cursor = end;
            insert(buffer, &format!("\n{many}"));
            end + 1
        } else {
            let start = line_start(&buffer.text, buffer.cursor);
            buffer.cursor = start;
            insert(buffer, &format!("{many}\n"));
            start
        };
        buffer.cursor = first_non_blank(&buffer.text, line);
        return;
    }
    if after && matches!(char_at(&buffer.text, buffer.cursor), Some(c) if c != '\n') {
        buffer.cursor = next(&buffer.text, buffer.cursor);
    }
    insert(buffer, &many);
    buffer.cursor = prev(&buffer.text, buffer.cursor);
}

/// Types `text` at the cursor, which ends up after it.
fn insert(buffer: &mut Buffer, text: &str) {
    let at = buffer.cursor;
    splice(buffer, at, at, text);
    buffer.cursor = at + text.len();
}

/// Replaces `start..end` with `with`, if those are character boundaries.
fn splice(buffer: &mut Buffer, start: usize, end: usize, with: &str) {
    let text = &mut buffer.text;
    if start <= end
        && end <= text.len()
        && text.is_char_boundary(start)
        && text.is_char_boundary(end)
    {
        text.replace_range(start..end, with);
    }
}

/// Keeps a normal-mode cursor on a character: never past a line's last.
fn clamp(buffer: &mut Buffer) {
    let text = &buffer.text;
    let cursor = buffer.cursor.min(text.len());
    let start = line_start(text, cursor);
    buffer.cursor = match cursor == line_end(text, cursor) && cursor > start {
        true => prev(text, cursor),
        false => cursor,
    };
}

fn char_at(text: &str, pos: usize) -> Option<char> {
    text.get(pos..)?.chars().next()
}

fn next(text: &str, pos: usize) -> usize {
    char_at(text, pos).map_or(pos, |c| pos + c.len_utf8())
}

fn prev(text: &str, pos: usize) -> usize {
    text.get(..pos)
        .and_then(|before| before.chars().next_back())
        .map_or(pos, |c| pos - c.len_utf8())
}

fn line_start(text: &str, pos: usize) -> usize {
    text.get(..pos)
        .and_then(|before| before.rfind('\n'))
        .map_or(0, |at| at + 1)
}

fn line_end(text: &str, pos: usize) -> usize {
    text.get(pos..)
        .and_then(|after| after.find('\n'))
        .map_or(text.len(), |at| pos + at)
}

/// The start of the line `count` lines below the one starting at `line`,
/// or of the last.
fn lines_down(text: &str, line: usize, count: usize) -> usize {
    (0..count).fold(line, |line, _| {
        let end = line_end(text, line);
        if end < text.len() { end + 1 } else { line }
    })
}

/// The start of the line `count` lines above the one starting at `line`,
/// or of the first.
fn lines_up(text: &str, line: usize, count: usize) -> usize {
    (0..count).fold(line, |line, _| match line {
        0 => 0,
        line => line_start(text, line - 1),
    })
}

fn first_non_blank(text: &str, pos: usize) -> usize {
    let start = line_start(text, pos);
    let line = text.get(start..line_end(text, start)).unwrap_or_default();
    start + (line.len() - line.trim_start_matches([' ', '\t']).len())
}

/// Blanks, word characters and other characters, as vim's `w` tells
/// words apart.
fn class(character: char) -> u8 {
    match character {
        c if c.is_whitespace() => 0,
        c if c.is_alphanumeric() || c == '_' => 1,
        _ => 2,
    }
}

fn word_start(text: &str, pos: usize) -> usize {
    let mut at = pos;
    if let Some(kind) = char_at(text, at).map(class).filter(|kind| *kind != 0) {
        while char_at(text, at).map(class) == Some(kind) {
            at = next(text, at);
        }
    }
    while char_at(text, at).is_some_and(char::is_whitespace) {
        at = next(text, at);
    }
    at
}

fn word_back(text: &str, pos: usize) -> usize {
    let mut at = pos;
    while at > 0 && char_at(text, prev(text, at)).is_some_and(char::is_whitespace) {
        at = prev(text, at);
    }
    let kind = char_at(text, prev(text, at)).map(class);
    while at > 0 && char_at(text, prev(text, at)).map(class) == kind {
        at = prev(text, at);
    }
    at
}

fn word_end(text: &str, pos: usize) -> usize {
    let mut at = next(text, pos);
    while char_at(text, at).is_some_and(char::is_whitespace) {
        at = next(text, at);
    }
    let Some(kind) = char_at(text, at).map(class) else {
        return pos;
    };
    while char_at(text, next(text, at)).map(class) == Some(kind) {
        at = next(text, at);
    }
    at
}
