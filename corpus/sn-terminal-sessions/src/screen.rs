//! The screen a program draws on a terminal: the escape sequences that
//! move the cursor and clear are interpreted, so a progress bar shows its
//! last state; colours and modes are read and left out. Lines that scroll
//! off the top go to the scrollback.

/// Lines kept above the screen.
const SCROLLBACK: usize = 2000;

/// A terminal screen and its scrollback.
#[derive(Debug)]
pub struct Screen {
    cols: usize,
    rows: Vec<Vec<char>>,
    row: usize,
    col: usize,
    saved: (usize, usize),
    scrollback: Vec<String>,
    parser: Parser,
    /// Bytes of a character not complete yet.
    partial: Vec<u8>,
}

/// Where the parser is within an escape sequence.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
enum Parser {
    #[default]
    Ground,
    Escape,
    /// `ESC (` and the like: one more byte names a character set.
    Charset,
    /// A control sequence's parameters so far.
    Csi(String),
    /// An operating system command, up to BEL or `ESC \`.
    Osc {
        escape: bool,
    },
}

impl Default for Screen {
    fn default() -> Self {
        Self::new(80, 24)
    }
}

impl Screen {
    /// A blank screen of `cols` × `rows`.
    pub fn new(cols: u16, rows: u16) -> Self {
        let (cols, height) = (usize::from(cols.max(1)), usize::from(rows.max(1)));
        Self {
            cols,
            rows: vec![vec![' '; cols]; height],
            row: 0,
            col: 0,
            saved: (0, 0),
            scrollback: Vec::new(),
            parser: Parser::Ground,
            partial: Vec::new(),
        }
    }

    /// Its width.
    pub fn cols(&self) -> u16 {
        u16::try_from(self.cols).unwrap_or(u16::MAX)
    }

    /// The row the cursor is on.
    pub fn cursor_row(&self) -> usize {
        self.row
    }

    /// Its lines as drawn, trailing blanks cut.
    pub fn lines(&self) -> Vec<String> {
        self.rows.iter().map(|row| trimmed(row)).collect()
    }

    /// The screen as text, without the blank lines at its end.
    pub fn text(&self) -> String {
        let mut lines = self.lines();
        while lines.last().is_some_and(String::is_empty) {
            lines.pop();
        }
        lines.join("\n")
    }

    /// The last `count` lines of the scrollback and the screen.
    pub fn tail(&self, count: usize) -> String {
        let mut lines = self.scrollback.clone();
        lines.extend(self.lines());
        while lines.last().is_some_and(String::is_empty) {
            lines.pop();
        }
        let from = lines.len().saturating_sub(count);
        lines.get(from..).unwrap_or_default().join("\n")
    }

    /// `bytes` as a screen `cols` wide would show them, read from a blank
    /// one tall enough for all of them: output as text.
    pub fn plain(bytes: &[u8], cols: u16) -> String {
        let mut screen = Self::new(cols, 1);
        screen.feed(bytes);
        screen.tail(usize::MAX)
    }

    /// Draws `bytes`, as they come.
    pub fn feed(&mut self, bytes: &[u8]) {
        self.partial.extend_from_slice(bytes);
        let pending = std::mem::take(&mut self.partial);
        let mut rest: &[u8] = &pending;
        while !rest.is_empty() {
            match std::str::from_utf8(rest) {
                Ok(text) => {
                    text.chars().for_each(|c| self.put(c));
                    rest = &[];
                }
                Err(error) => {
                    let (good, bad) = rest.split_at(error.valid_up_to());
                    String::from_utf8_lossy(good)
                        .chars()
                        .for_each(|c| self.put(c));
                    match error.error_len() {
                        Some(len) => {
                            self.put(char::REPLACEMENT_CHARACTER);
                            rest = bad.get(len..).unwrap_or_default();
                        }
                        None => {
                            self.partial = bad.to_vec();
                            rest = &[];
                        }
                    }
                }
            }
        }
    }

    fn put(&mut self, c: char) {
        match std::mem::take(&mut self.parser) {
            Parser::Ground => self.ground(c),
            Parser::Escape => self.escape(c),
            Parser::Charset => {}
            Parser::Csi(mut params) => {
                if ('\u{40}'..='\u{7e}').contains(&c) {
                    self.csi(&params, c);
                } else if c == '\u{1b}' {
                    self.parser = Parser::Escape;
                } else {
                    params.push(c);
                    self.parser = Parser::Csi(params);
                }
            }
            Parser::Osc { escape } => {
                if c == '\u{7}' || (escape && c == '\\') {
                    return;
                }
                self.parser = Parser::Osc {
                    escape: c == '\u{1b}',
                };
            }
        }
    }

    fn ground(&mut self, c: char) {
        match c {
            '\u{1b}' => self.parser = Parser::Escape,
            '\r' => self.col = 0,
            '\n' | '\u{b}' | '\u{c}' => self.line_feed(),
            '\u{8}' => self.col = self.col.min(self.cols - 1).saturating_sub(1),
            '\t' => self.col = ((self.col / 8 + 1) * 8).min(self.cols - 1),
            c if c.is_control() => {}
            c => {
                if self.col >= self.cols {
                    self.col = 0;
                    self.line_feed();
                }
                if let Some(cell) = self
                    .rows
                    .get_mut(self.row)
                    .and_then(|row| row.get_mut(self.col))
                {
                    *cell = c;
                }
                self.col += 1;
            }
        }
    }

    fn escape(&mut self, c: char) {
        match c {
            '[' => self.parser = Parser::Csi(String::new()),
            ']' | 'P' | '_' | '^' | 'X' => self.parser = Parser::Osc { escape: false },
            '(' | ')' | '*' | '+' | '#' | '%' => self.parser = Parser::Charset,
            '7' => self.saved = (self.row, self.col),
            '8' => (self.row, self.col) = self.saved,
            'D' => self.line_feed(),
            'E' => {
                self.col = 0;
                self.line_feed();
            }
            'M' => {
                if self.row == 0 {
                    self.rows.pop();
                    self.rows.insert(0, vec![' '; self.cols]);
                } else {
                    self.row -= 1;
                }
            }
            'c' => *self = Self::new(self.cols(), self.height()),
            _ => {}
        }
    }

    fn height(&self) -> u16 {
        u16::try_from(self.rows.len()).unwrap_or(u16::MAX)
    }

    fn csi(&mut self, params: &str, last: char) {
        // Private modes (`?25l`) and the like change nothing drawn.
        if params.starts_with(['?', '>', '<', '=']) {
            return;
        }
        let numbers: Vec<usize> = params
            .split(';')
            .map(|number| number.parse().unwrap_or(0))
            .collect();
        let at = |index: usize| numbers.get(index).copied().unwrap_or(0);
        let count = at(0).max(1);
        let last_row = self.rows.len() - 1;
        match last {
            'A' => self.row = self.row.saturating_sub(count),
            'B' | 'e' => self.row = (self.row + count).min(last_row),
            'C' | 'a' => self.col = (self.col + count).min(self.cols - 1),
            'D' => self.col = self.col.min(self.cols - 1).saturating_sub(count),
            'E' => (self.row, self.col) = ((self.row + count).min(last_row), 0),
            'F' => (self.row, self.col) = (self.row.saturating_sub(count), 0),
            'G' | '`' => self.col = (count - 1).min(self.cols - 1),
            'd' => self.row = (count - 1).min(last_row),
            'H' | 'f' => {
                self.row = (at(0).max(1) - 1).min(last_row);
                self.col = (at(1).max(1) - 1).min(self.cols - 1);
            }
            'J' => self.erase_display(at(0)),
            'K' => self.erase_line(at(0)),
            'X' => self.blank(self.row, self.col, self.col + count),
            'P' => {
                let (col, cols) = (self.col, self.cols);
                if let Some(row) = self.rows.get_mut(self.row) {
                    for _ in 0..count.min(cols.saturating_sub(col)) {
                        if col < row.len() {
                            row.remove(col);
                            row.push(' ');
                        }
                    }
                }
            }
            '@' => {
                let (col, cols) = (self.col, self.cols);
                if let Some(row) = self.rows.get_mut(self.row) {
                    for _ in 0..count.min(cols.saturating_sub(col)) {
                        row.insert(col.min(row.len()), ' ');
                        row.truncate(cols);
                    }
                }
            }
            'L' => {
                for _ in 0..count {
                    self.rows.insert(self.row, vec![' '; self.cols]);
                    self.rows.pop();
                }
            }
            'M' => {
                for _ in 0..count {
                    if self.row < self.rows.len() {
                        self.rows.remove(self.row);
                        self.rows.push(vec![' '; self.cols]);
                    }
                }
            }
            's' => self.saved = (self.row, self.col),
            'u' => (self.row, self.col) = self.saved,
            _ => {}
        }
    }

    fn erase_display(&mut self, how: usize) {
        let height = self.rows.len();
        match how {
            0 => {
                self.erase_line(0);
                for row in self.row + 1..height {
                    self.blank(row, 0, self.cols);
                }
            }
            1 => {
                self.erase_line(1);
                for row in 0..self.row {
                    self.blank(row, 0, self.cols);
                }
            }
            _ => {
                for row in 0..height {
                    self.blank(row, 0, self.cols);
                }
            }
        }
    }

    fn erase_line(&mut self, how: usize) {
        let (row, col, cols) = (self.row, self.col, self.cols);
        match how {
            0 => self.blank(row, col, cols),
            1 => self.blank(row, 0, col + 1),
            _ => self.blank(row, 0, cols),
        }
    }

    fn blank(&mut self, row: usize, from: usize, to: usize) {
        if let Some(cells) = self.rows.get_mut(row) {
            cells
                .iter_mut()
                .take(to)
                .skip(from)
                .for_each(|cell| *cell = ' ');
        }
    }

    fn line_feed(&mut self) {
        if self.row + 1 < self.rows.len() {
            self.row += 1;
            return;
        }
        if !self.rows.is_empty() {
            let gone = self.rows.remove(0);
            self.scrollback.push(trimmed(&gone));
            if self.scrollback.len() > SCROLLBACK {
                self.scrollback.remove(0);
            }
        }
        self.rows.push(vec![' '; self.cols]);
    }
}

fn trimmed(row: &[char]) -> String {
    row.iter().collect::<String>().trim_end().to_owned()
}
