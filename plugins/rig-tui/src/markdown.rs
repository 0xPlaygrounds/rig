//! Markdown in the model's answers drawn as styled lines: headings,
//! emphasis, inline code, fenced code, lists, quotes and alerts, links,
//! rules and tables. Parsing is pulldown-cmark's, with strikethrough,
//! tables, task lists and GitHub's alerts on. Code sits on the theme's code
//! tint, with a fenced block's language over it when it has a grammar, and
//! an alert (`> [!NOTE]`) on the tint of what it is, under its name in bold.

use pulldown_cmark::{
    BlockQuoteKind, CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd,
};
use ratatui::style::{Color, Style, Stylize};
use ratatui::text::{Line, Span};

use crate::LookLine;
use crate::Theme;
use crate::highlight::{Code, Language};
use crate::renderers::CodeLine;

use table::Table;

mod table;

/// The extensions parsed: strikethrough, tables, task lists and GitHub's
/// alerts.
fn options() -> Options {
    Options::ENABLE_STRIKETHROUGH
        | Options::ENABLE_TABLES
        | Options::ENABLE_TASKLISTS
        | Options::ENABLE_GFM
}

/// `text` as markdown in `theme`'s tints, in lines not yet wrapped, its
/// tables fit to `width` (at their widest cells with `None`).
pub fn render(text: &str, theme: &Theme, width: Option<usize>) -> Vec<LookLine> {
    let mut writer = Writer {
        width,
        theme: *theme,
        ..Writer::default()
    };
    for event in Parser::new_ext(text, options()) {
        writer.event(event);
    }
    writer.finish()
}

/// Whether `text` as markdown has a table, so its lines depend on the
/// width they are rendered at.
pub(crate) fn has_table(text: &str) -> bool {
    Parser::new_ext(text, options()).any(|event| matches!(event, Event::Start(Tag::Table(_))))
}

/// The first of the private-use characters [`render_marked`] stands
/// marks in with.
const SENTINEL: u32 = 0xE000;

/// The most marks one text takes: the private-use area's size.
const MOST_MARKS: usize = 0x1900;

/// `text` as markdown at `width`, as [`render`] draws it, with each of `marks` drawn
/// before its byte: past the end at the end, inside a character before
/// it. Each mark stands in the text as a private-use character while it
/// is rendered, so markdown on both sides of it is kept.
pub fn render_marked(
    text: &str,
    marks: &[(usize, Span<'static>)],
    theme: &Theme,
    width: Option<usize>,
) -> Vec<LookLine> {
    if marks.is_empty() {
        return render(text, theme, width);
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
    let marked_in = |line: &Line<'_>| {
        let mut chars = line.spans.iter().flat_map(|span| span.content.chars());
        chars.any(|ch| mark_of(ch).is_some())
    };
    render(&marked, theme, width)
        .into_iter()
        .map(|look| {
            // A mark in a line of code is swapped into its colored spans.
            let (line, code) = match look.code {
                Some(_) if marked_in(&look.line) => (look.colored(), None),
                code => (look.line, code),
            };
            let spans = line.spans.into_iter().flat_map(swap).collect::<Vec<_>>();
            let line = Line { spans, ..line };
            LookLine { line, code }
        })
        .collect()
}

/// A list being written: the next item's number, or `None` for bullets.
struct List {
    next: Option<u64>,
}

#[derive(Default)]
struct Writer {
    lines: Vec<Line<'static>>,
    /// The lines of `lines` that are lines of code, by their place.
    code_lines: Vec<(usize, CodeLine)>,
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
    /// What tables are fit to.
    width: Option<usize>,
    theme: Theme,
    /// The open alerts: the first of their lines and their tint, `None`
    /// for a plain quote.
    alerts: Vec<Option<(usize, Color)>>,
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
                let cell = table.rows.last_mut().and_then(|row| row.last_mut());
                if let Some(line) = cell.and_then(|cell| cell.last_mut()) {
                    line.push(span);
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

    /// Writes a code block's `code` on the code tint, each line kept, empty
    /// ones too. When its fence's `language` has a grammar, the code is in
    /// that language's colors under the language's name, once drawn;
    /// otherwise it is plain and unlabelled.
    fn code_block(&mut self, language: &str, code: &str) {
        let code = code.strip_suffix('\n').unwrap_or(code);
        let tint = self.theme.code;
        let colored = match language {
            "" => None,
            language => Some(Code::new(code, Language::Named(language))),
        };
        let Some(colored) = colored.filter(Code::has_grammar) else {
            for line in code.split('\n') {
                let mut spans = self.prefix();
                spans.push(Span::from("  "));
                spans.push(Span::from(line.replace('\t', "    ")));
                self.lines.push(Line::from(spans).bg(tint));
            }
            return;
        };
        let mut spans = self.prefix();
        spans.push(Span::from(format!("  {language}")).dim().italic());
        self.lines.push(Line::from(spans).bg(tint));
        for index in 0..colored.len() {
            let mut prefix = self.prefix();
            prefix.push(Span::from("  "));
            let look = colored.line(index, prefix);
            if let Some(code) = look.code {
                self.code_lines.push((self.lines.len(), code));
            }
            self.lines.push(look.line.bg(tint));
        }
    }

    /// Inline code: on the code tint, or in its backticks with no colors.
    fn inline_code(&mut self, code: &str) {
        let span = match self.theme.plain {
            true => Span::from(format!("`{code}`")),
            false => Span::styled(code.to_owned(), Style::new().bg(self.theme.code)),
        };
        self.push(span.patch_style(self.style()));
    }

    fn event(&mut self, event: Event<'_>) {
        match event {
            Event::Start(tag) => self.start(tag),
            Event::End(tag) => self.end(tag),
            Event::Text(text) => self.text(&text),
            Event::Code(code) => self.inline_code(&code),
            Event::Html(html) | Event::InlineHtml(html) if is_break(&html) => self.hard_break(),
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
            Event::HardBreak => self.hard_break(),
            Event::Rule => {
                self.block();
                // A hairline: blank cells underlined.
                self.lines
                    .push(Line::from(" ".repeat(40)).dark_gray().underlined());
                self.gap = true;
            }
            Event::TaskListMarker(done) => {
                let mark = match done {
                    true => Span::from("✓ ").green().bold(),
                    false => Span::from("○ ").dim(),
                };
                self.line.push(mark);
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
            Tag::BlockQuote(kind) => {
                self.block();
                let alert = kind.map(|kind| self.alert(kind));
                self.quotes += 1;
                // A plain quote is set apart in gray italics; an alert by its tint.
                let style = match alert {
                    Some(_) => Style::new(),
                    None => Style::new().italic().dark_gray(),
                };
                self.styles.push(style);
                self.alerts.push(alert);
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
                    row.push(vec![Vec::new()]);
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
                if let Some(Some((first, tint))) = self.alerts.pop() {
                    for line in self.lines.iter_mut().skip(first) {
                        line.style.bg = line.style.bg.or(Some(tint));
                    }
                }
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

    /// Writes a table fit to the width, in the block it is in.
    fn table_lines(&mut self, table: Table) {
        let prefixes = (self.prefix(), self.prefix());
        let lines = table::lines(table, prefixes, self.width);
        self.lines.extend(lines);
    }

    /// Starts an alert of `kind`: its name in bold on its first line; the
    /// first of its lines and its tint.
    fn alert(&mut self, kind: BlockQuoteKind) -> (usize, Color) {
        let theme = self.theme;
        let (name, tint) = match kind {
            BlockQuoteKind::Note => ("Note", theme.note),
            BlockQuoteKind::Tip => ("Tip", theme.call_done),
            BlockQuoteKind::Important => ("Important", theme.important),
            BlockQuoteKind::Warning => ("Warning", theme.call_running),
            BlockQuoteKind::Caution => ("Caution", theme.call_failed),
        };
        let first = self.lines.len();
        let mut spans = self.prefix();
        spans.push(Span::from(name).bold());
        self.lines.push(Line::from(spans));
        (first, tint)
    }

    /// Ends the line here: in a table cell, starts the cell's next line.
    fn hard_break(&mut self) {
        match &mut self.table {
            Some(table) => {
                if let Some(cell) = table.rows.last_mut().and_then(|row| row.last_mut()) {
                    cell.push(Vec::new());
                }
            }
            None => self.flush(),
        }
    }

    fn finish(mut self) -> Vec<LookLine> {
        self.flush();
        let mut code = self.code_lines.into_iter().peekable();
        let lines = self.lines.into_iter().enumerate().map(|(at, line)| {
            let code = code.next_if(|(line, _)| *line == at).map(|(_, code)| code);
            LookLine { line, code }
        });
        lines.collect()
    }
}

/// Whether `html` is a line break: `<br>`, `<br/>` or `<br />`, in any case.
fn is_break(html: &str) -> bool {
    let tag: String = html
        .chars()
        .filter(|c| !c.is_whitespace() && *c != '/')
        .collect();
    tag.eq_ignore_ascii_case("<br>")
}

#[cfg(test)]
mod tests;
