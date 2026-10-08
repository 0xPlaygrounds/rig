//! Markdown in the model's answers drawn as styled lines: headings,
//! emphasis, inline code, fenced code, lists, quotes, links, rules and
//! tables. Parsing is pulldown-cmark's, with the options codex's terminal
//! view turns on (`references/codex/codex-rs/tui/src/markdown_render.rs:400-403`).

use pulldown_cmark::{CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd};
use ratatui::style::{Color, Modifier, Style, Stylize};
use ratatui::text::{Line, Span};
use unicode_width::UnicodeWidthStr;

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
    /// Inside a fenced or indented code block.
    code: bool,
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

    /// The indent of the current line: the quote bars and the list depth.
    fn prefix(&mut self) -> Vec<Span<'static>> {
        let mut prefix = Vec::new();
        for _ in 0..self.quotes {
            prefix.push(Span::from("│ ").dark_gray());
        }
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

    fn text(&mut self, text: &str) {
        if let Some(table) = &mut self.table {
            let style = self.styles.iter().fold(Style::new(), |s, i| s.patch(*i));
            if let Some(cell) = table.rows.last_mut().and_then(|row| row.last_mut()) {
                cell.push(Span::styled(text.to_owned(), style));
            }
            return;
        }
        if self.code {
            for (index, line) in text.split('\n').enumerate() {
                if index > 0 {
                    self.end_code_line();
                }
                if !line.is_empty() {
                    self.line
                        .push(Span::styled(line.replace('\t', "    "), code_style()));
                }
            }
            return;
        }
        let style = self.style();
        self.line.push(Span::styled(text.to_owned(), style));
    }

    /// Ends a line of a code block, keeping empty ones.
    fn end_code_line(&mut self) {
        let mut spans = self.prefix();
        spans.push(Span::from("  "));
        spans.append(&mut self.line);
        self.lines.push(Line::from(spans));
    }

    fn event(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(text) => self.text(&text),
            Event::Code(code) => {
                if let Some(table) = &mut self.table {
                    if let Some(cell) = table.rows.last_mut().and_then(|row| row.last_mut()) {
                        cell.push(Span::styled(code.to_string(), code_style()));
                    }
                } else {
                    self.line.push(Span::styled(code.to_string(), code_style()));
                }
            }
            Event::InlineMath(math) | Event::DisplayMath(math) => {
                self.line.push(Span::styled(math.to_string(), code_style()));
            }
            Event::Html(html) | Event::InlineHtml(html) => {
                let style = self.style().dim();
                for (index, line) in html.trim_end_matches('\n').split('\n').enumerate() {
                    if index > 0 {
                        self.flush();
                    }
                    self.line.push(Span::styled(line.to_owned(), style));
                }
            }
            Event::FootnoteReference(name) => {
                self.line.push(Span::from(format!("[^{name}]")).dim());
            }
            Event::SoftBreak => self.line.push(Span::from(" ")),
            Event::HardBreak => self.flush(),
            Event::Rule => {
                self.block();
                self.lines.push(Line::from("─".repeat(40)).dark_gray());
                self.gap = true;
            }
            Event::TaskListMarker(done) => {
                let mark = if done { "[x] " } else { "[ ] " };
                self.line.push(Span::from(mark).cyan());
            }
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
            }
            Tag::CodeBlock(kind) => {
                self.block();
                self.code = true;
                if let CodeBlockKind::Fenced(language) = kind
                    && !language.is_empty()
                {
                    let mut spans = self.prefix();
                    spans.push(Span::from(format!("  {language}")).dim().italic());
                    self.lines.push(Line::from(spans));
                }
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
            Tag::Link { dest_url, .. } => {
                self.styles.push(Style::new().underlined().blue());
                self.links.push(dest_url.to_string());
            }
            Tag::Image { dest_url, .. } => {
                self.styles.push(Style::new().italic());
                self.line.push(Span::from("[image: ").dim());
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
            Tag::FootnoteDefinition(name) => {
                self.block();
                self.line.push(Span::from(format!("[^{name}]: ")).dim());
            }
            Tag::DefinitionListDefinition => {
                self.flush();
                self.line.push(Span::from("  "));
            }
            Tag::HtmlBlock
            | Tag::DefinitionList
            | Tag::DefinitionListTitle
            | Tag::Superscript
            | Tag::Subscript
            | Tag::MetadataBlock(_) => {}
        }
    }

    fn end(&mut self, tag: TagEnd) {
        match tag {
            TagEnd::Paragraph | TagEnd::HtmlBlock | TagEnd::FootnoteDefinition => {
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
                self.gap = true;
            }
            TagEnd::CodeBlock => {
                if !self.line.is_empty() {
                    self.end_code_line();
                }
                self.code = false;
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
            TagEnd::Link => {
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
            TagEnd::Image => {
                self.styles.pop();
                let url = self.links.pop().unwrap_or_default();
                self.line.push(Span::from(format!(" {url}]")).dim());
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
            TagEnd::DefinitionListTitle | TagEnd::DefinitionListDefinition => self.flush(),
            TagEnd::TableRow
            | TagEnd::TableCell
            | TagEnd::DefinitionList
            | TagEnd::Superscript
            | TagEnd::Subscript
            | TagEnd::MetadataBlock(_) => {}
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
        let border = Style::new().dark_gray();
        for (index, row) in table.rows.into_iter().enumerate() {
            let header = index < table.header_rows;
            let mut spans = self.prefix();
            for (column, width) in widths.iter().enumerate() {
                if column > 0 {
                    spans.push(Span::styled(" │ ", border));
                }
                let cell = row.get(column).cloned().unwrap_or_default();
                let padding = width.saturating_sub(width_of(&cell));
                for span in cell {
                    let style = if header {
                        span.style.add_modifier(Modifier::BOLD)
                    } else {
                        span.style
                    };
                    spans.push(Span::styled(span.content, style));
                }
                spans.push(Span::from(" ".repeat(padding)));
            }
            self.lines.push(Line::from(spans));
            if header && index + 1 == table.header_rows {
                let rule: Vec<String> = widths.iter().map(|width| "─".repeat(*width)).collect();
                self.lines.push(Line::styled(rule.join("─┼─"), border));
            }
        }
    }

    fn finish(mut self) -> Vec<Line<'static>> {
        self.flush();
        self.lines
    }
}
