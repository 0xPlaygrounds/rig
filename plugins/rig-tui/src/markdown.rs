//! Markdown in the model's answers drawn as styled lines: headings,
//! emphasis, inline code, fenced code, lists, quotes, links, rules and
//! tables. Parsing is pulldown-cmark's, with strikethrough, tables and task
//! lists on.

use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

use crate::highlight::{Language, highlight};

/// The style of inline code and code blocks.
fn code_style() -> Style {
    Style::new().fg(Color::LightYellow)
}

/// `text` as markdown, in lines not yet wrapped.
pub fn render(text: &str) -> Vec<Line<'static>> {
    let mut options = Options::empty();
    options.insert(Options::ENABLE_STRIKETHROUGH);
    options.insert(Options::ENABLE_TABLES);
    options.insert(Options::ENABLE_TASKLISTS);
    let mut writer = Writer::default();
    for event in Parser::new_ext(text, options) {
        writer.event(event);
    }
    writer.finish()
}

/// The first of the private-use characters [`render_marked`] stands
/// marks in with.
const SENTINEL: u32 = 0xE000;

/// The most marks one text takes: the private-use area's size.
const MOST_MARKS: usize = 0x1900;

/// `text` as markdown, as [`render`] draws it, with each of `marks` drawn
/// before its byte: past the end at the end, inside a character before
/// it. Each mark stands in the text as a private-use character while it
/// is rendered, so markdown on both sides of it is kept.
pub fn render_marked(text: &str, marks: &[(usize, Span<'static>)]) -> Vec<Line<'static>> {
    if marks.is_empty() {
        return render(text);
    }
    let mut placed: Vec<(usize, usize)> = marks
        .iter()
        .take(MOST_MARKS)
        .enumerate()
        .map(|(index, (at, _))| {
            let mut at = (*at).min(text.len());
            while !text.is_char_boundary(at) {
                at = at.saturating_sub(1);
            }
            (at, index)
        })
        .collect();
    placed.sort_by_key(|(at, _)| *at);
    let mut marked = String::with_capacity(text.len() + 3 * placed.len());
    let mut from = 0;
    for (at, index) in &placed {
        marked.push_str(text.get(from..*at).unwrap_or_default());
        let code = SENTINEL.saturating_add(u32::try_from(*index).unwrap_or(u32::MAX));
        marked.extend(char::from_u32(code));
        from = *at;
    }
    marked.push_str(text.get(from..).unwrap_or_default());
    let mark_of = |ch: char| {
        let index = u32::from(ch).checked_sub(SENTINEL)?;
        marks
            .get(usize::try_from(index).ok()?)
            .map(|(_, span)| span)
    };
    let swap = |span: Span<'static>| -> Vec<Span<'static>> {
        if !span.content.chars().any(|ch| mark_of(ch).is_some()) {
            return vec![span];
        }
        let mut spans = Vec::new();
        let mut run = String::new();
        for ch in span.content.chars() {
            match mark_of(ch) {
                Some(mark) => {
                    if !run.is_empty() {
                        spans.push(Span::styled(std::mem::take(&mut run), span.style));
                    }
                    spans.push(mark.clone());
                }
                None => run.push(ch),
            }
        }
        if !run.is_empty() {
            spans.push(Span::styled(run, span.style));
        }
        spans
    };
    render(&marked)
        .into_iter()
        .map(|line| {
            let spans = line.spans.into_iter().flat_map(swap).collect::<Vec<_>>();
            Line { spans, ..line }
        })
        .collect()
}

/// A list being written: the next item's number, or `None` for bullets.
struct List {
    next: Option<u64>,
}

/// A table being written: its rows of cells, each cell its spans.
#[derive(Default)]
struct Table {
    rows: Vec<Vec<Vec<Span<'static>>>>,
    header_rows: usize,
}

#[derive(Default)]
struct Writer {
    lines: Vec<Line<'static>>,
    /// The line being written.
    line: Vec<Span<'static>>,
    /// Inline styles open, innermost last.
    styles: Vec<Style>,
    lists: Vec<List>,
    /// How deep in block quotes.
    quotes: usize,
    /// Inside a fenced or indented code block: its fence's language, and
    /// its text so far, drawn whole at its end.
    code: Option<(String, String)>,
    /// The prefix of the current item's first line, until it is written.
    item_marker: Option<String>,
    /// Link targets of the open links, written after their text.
    links: Vec<String>,
    table: Option<Table>,
    /// A blank line goes before the next block.
    gap: bool,
}

impl Writer {
    fn style(&self) -> Style {
        self.styles
            .iter()
            .fold(Style::new(), |style, inner| style.patch(*inner))
    }

    /// The indent of the current line: the quote depth and the list depth.
    fn prefix(&mut self) -> Vec<Span<'static>> {
        let mut prefix = vec![Span::from("  ".repeat(self.quotes))];
        let depth = self.lists.len();
        if depth > 0 {
            match self.item_marker.take() {
                Some(marker) => {
                    prefix.push(Span::from("  ".repeat(depth - 1)));
                    prefix.push(Span::from(marker).cyan());
                }
                None => prefix.push(Span::from("  ".repeat(depth))),
            }
        }
        prefix
    }

    /// Starts a block: a blank line after the previous one.
    fn block(&mut self) {
        self.flush();
        if self.gap && !self.lines.is_empty() && self.lists.is_empty() {
            self.lines.push(Line::default());
        }
        self.gap = false;
    }

    /// Ends the current line, if it has text.
    fn flush(&mut self) {
        if self.line.is_empty() {
            return;
        }
        let mut spans = self.prefix();
        spans.append(&mut self.line);
        self.lines.push(Line::from(spans));
    }

    /// Writes `span` on the current line, or in the current table cell.
    fn push(&mut self, span: Span<'static>) {
        match &mut self.table {
            Some(table) => {
                if let Some(cell) = table.rows.last_mut().and_then(|row| row.last_mut()) {
                    cell.push(span);
                }
            }
            None => self.line.push(span),
        }
    }

    fn text(&mut self, text: &str) {
        if let Some((_, code)) = &mut self.code {
            code.push_str(text);
            return;
        }
        self.push(Span::styled(text.to_owned(), self.style()));
    }

    /// Writes a code block's `code`, in the colors of its fence's
    /// `language` when it has a grammar, each line kept, empty ones too.
    fn code_block(&mut self, language: &str, code: &str) {
        let code = code.strip_suffix('\n').unwrap_or(code);
        let plain = || {
            let lines = code.split('\n');
            let line = |line: &str| vec![Span::styled(line.replace('\t', "    "), code_style())];
            lines.map(line).collect()
        };
        let lines = match language {
            "" => plain(),
            language => highlight(code, Language::Named(language)).unwrap_or_else(plain),
        };
        for line in lines {
            let mut spans = self.prefix();
            spans.push(Span::from("  "));
            spans.extend(line);
            self.lines.push(Line::from(spans));
        }
    }

    fn event(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(text) => self.text(&text),
            Event::Code(code) => self.push(Span::styled(code.to_string(), code_style())),
            Event::Html(html) | Event::InlineHtml(html) => {
                let style = self.style().dim();
                for (index, line) in html.trim_end_matches('\n').split('\n').enumerate() {
                    if index > 0 {
                        self.flush();
                    }
                    self.line.push(Span::styled(line.to_owned(), style));
                }
            }
            Event::SoftBreak => self.line.push(Span::from(" ")),
            Event::HardBreak => self.flush(),
            Event::Rule => {
                self.block();
                // A hairline: blank cells underlined.
                self.lines
                    .push(Line::from(" ".repeat(40)).dark_gray().underlined());
                self.gap = true;
            }
            Event::TaskListMarker(done) => {
                let mark = if done { "[x] " } else { "[ ] " };
                self.line.push(Span::from(mark).cyan());
            }
            // Math and footnotes, which `render` does not turn on.
            _ => {}
        }
    }

    fn start(&mut self, tag: Tag<'_>) {
        match tag {
            Tag::Paragraph => self.block(),
            Tag::Heading { level, .. } => {
                self.block();
                let style = match level {
                    HeadingLevel::H1 => Style::new().bold().underlined().cyan(),
                    HeadingLevel::H2 => Style::new().bold().cyan(),
                    _ => Style::new().bold(),
                };
                self.styles.push(style);
            }
            Tag::BlockQuote(_) => {
                self.block();
                self.quotes += 1;
                self.styles.push(Style::new().italic().dark_gray());
            }
            // The fence's info string (the language) is not content: it
            // colors the block's code.
            Tag::CodeBlock(kind) => {
                self.block();
                let language = match kind {
                    CodeBlockKind::Fenced(info) => {
                        info.split_whitespace().next().map(str::to_owned)
                    }
                    CodeBlockKind::Indented => None,
                };
                self.code = Some((language.unwrap_or_default(), String::new()));
            }
            Tag::List(start) => {
                if self.lists.is_empty() {
                    self.block();
                } else {
                    self.flush();
                }
                self.lists.push(List { next: start });
            }
            Tag::Item => {
                self.flush();
                let marker = match self.lists.last_mut() {
                    Some(List { next: Some(number) }) => {
                        let marker = format!("{number}. ");
                        *number += 1;
                        marker
                    }
                    _ => "• ".to_owned(),
                };
                self.item_marker = Some(marker);
            }
            Tag::Emphasis => self.styles.push(Style::new().italic()),
            Tag::Strong => self.styles.push(Style::new().bold()),
            Tag::Strikethrough => self.styles.push(Style::new().crossed_out()),
            // An image is drawn as a link to it.
            Tag::Link { dest_url, .. } | Tag::Image { dest_url, .. } => {
                self.styles.push(Style::new().underlined().blue());
                self.links.push(dest_url.to_string());
            }
            Tag::Table(_) => {
                self.block();
                self.table = Some(Table::default());
            }
            Tag::TableHead | Tag::TableRow => {
                if let Some(table) = &mut self.table {
                    table.rows.push(Vec::new());
                }
            }
            Tag::TableCell => {
                if let Some(row) = self.table.as_mut().and_then(|table| table.rows.last_mut()) {
                    row.push(Vec::new());
                }
            }
            _ => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph | TagEnd::HtmlBlock => {
                self.flush();
                self.gap = true;
            }
            TagEnd::Heading(_) => {
                self.styles.pop();
                self.flush();
                self.gap = true;
            }
            TagEnd::BlockQuote(_) => {
                self.flush();
                self.quotes = self.quotes.saturating_sub(1);
                self.styles.pop();
                self.gap = true;
            }
            TagEnd::CodeBlock => {
                if let Some((language, code)) = self.code.take() {
                    self.code_block(&language, &code);
                }
                self.gap = true;
            }
            TagEnd::List(_) => {
                self.flush();
                self.lists.pop();
                self.gap = true;
            }
            TagEnd::Item => self.flush(),
            TagEnd::Emphasis | TagEnd::Strong | TagEnd::Strikethrough => {
                self.styles.pop();
            }
            TagEnd::Link | TagEnd::Image => {
                self.styles.pop();
                if let Some(url) = self.links.pop() {
                    let shown = self
                        .line
                        .last()
                        .is_some_and(|span| span.content.as_ref() == url);
                    if !shown && !url.is_empty() {
                        self.line.push(Span::from(format!(" ({url})")).dark_gray());
                    }
                }
            }
            TagEnd::Table => {
                if let Some(table) = self.table.take() {
                    self.table_lines(table);
                }
                self.gap = true;
            }
            TagEnd::TableHead => {
                if let Some(table) = &mut self.table {
                    table.header_rows = table.rows.len();
                }
            }
            _ => {}
        }
    }

    /// Writes a table with its columns padded to their widest cell.
    fn table_lines(&mut self, table: Table) {
        let columns = table.rows.iter().map(Vec::len).max().unwrap_or(0);
        let width_of =
            |cell: &Vec<Span<'static>>| cell.iter().map(|span| span.content.width()).sum::<usize>();
        let widths: Vec<usize> = (0..columns)
            .map(|column| {
                table
                    .rows
                    .iter()
                    .filter_map(|row| row.get(column))
                    .map(width_of)
                    .max()
                    .unwrap_or(0)
            })
            .collect();
        for (index, row) in table.rows.into_iter().enumerate() {
            let header = index < table.header_rows;
            let mut spans = self.prefix();
            for (column, width) in widths.iter().enumerate() {
                if column > 0 {
                    spans.push(Span::from("   "));
                }
                let cell = row.get(column).cloned().unwrap_or_default();
                let padding = width.saturating_sub(width_of(&cell));
                for span in cell {
                    let style = if header {
                        span.style
                            .add_modifier(Modifier::BOLD | Modifier::UNDERLINED)
                    } else {
                        span.style
                    };
                    spans.push(Span::styled(span.content, style));
                }
                spans.push(Span::from(" ".repeat(padding)));
            }
            self.lines.push(Line::from(spans));
        }
    }

    fn finish(mut self) -> Vec<Line<'static>> {
        self.flush();
        self.lines
    }
}

#[cfg(test)]
mod tests;
