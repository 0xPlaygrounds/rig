//! Markdown drawn as styled lines: headings, emphasis, inline code, fenced
//! code, lists, quotes and alerts, links, rules and tables. Parsing is
//! pulldown-cmark's, with strikethrough, tables, task lists and GitHub's
//! alerts on. Code sits on the palette's code tint, with a fenced block's
//! language over it when the [`Highlight`] colors it, and an alert
//! (`> [!NOTE]`) on the tint of what it is, under its name in bold.
//!
//! ```
//! use rig_markdown::markdown::{Palette, Plain, render};
//!
//! let lines = render("Some *markdown*.", &Palette::default(), None, &Plain);
//! assert_eq!(lines.len(), 1);
//! ```

use pulldown_cmark::{
    BlockQuoteKind, CodeBlockKind, Event, HeadingLevel, Options, Parser, Tag, TagEnd,
};
use ratatui_core::style::{Color, Style, Stylize};
use ratatui_core::text::{Line, Span};

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

/// The tints markdown is drawn on.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Palette {
    /// Behind code: a fenced block, and inline code.
    pub code: Color,
    /// Behind a note (`> [!NOTE]`).
    pub note: Color,
    /// Behind a tip (`> [!TIP]`).
    pub tip: Color,
    /// Behind something important (`> [!IMPORTANT]`).
    pub important: Color,
    /// Behind a warning (`> [!WARNING]`).
    pub warning: Color,
    /// Behind a caution (`> [!CAUTION]`).
    pub caution: Color,
    /// No colors: inline code keeps its backticks instead of a tint.
    pub plain: bool,
}

/// Colors a fenced code block's lines, such as by its language's grammar.
pub trait Highlight {
    /// What a colored line carries beside its text, such as where its
    /// colors are found.
    type Mark;

    /// The lines of `code` in `language`, each after `prefix`, or `None`
    /// when it has no colors for `language`, so the block is drawn plain
    /// and unlabelled. `language` is never empty.
    fn block(
        &self,
        code: &str,
        language: &str,
        prefix: &[Span<'static>],
    ) -> Option<Vec<(Line<'static>, Option<Self::Mark>)>>;
}

/// No colors for any code: every block is plain and unlabelled.
#[derive(Clone, Copy, Debug, Default)]
pub struct Plain;

impl Highlight for Plain {
    type Mark = ();

    fn block(
        &self,
        _: &str,
        _: &str,
        _: &[Span<'static>],
    ) -> Option<Vec<(Line<'static>, Option<()>)>> {
        None
    }
}

/// `text` as markdown on `palette`, in lines not yet wrapped, its tables
/// fit to `width` (at their widest cells with `None`), its fenced code
/// colored by `highlight`. Each line has its mark from `highlight`, if
/// any.
pub fn render<H: Highlight>(
    text: &str,
    palette: &Palette,
    width: Option<usize>,
    highlight: &H,
) -> Vec<(Line<'static>, Option<H::Mark>)> {
    let mut writer = Writer {
        lines: Vec::new(),
        code_lines: Vec::new(),
        line: Vec::new(),
        styles: Vec::new(),
        lists: Vec::new(),
        quotes: 0,
        code: None,
        item_marker: None,
        links: Vec::new(),
        table: None,
        gap: false,
        width,
        palette: *palette,
        alerts: Vec::new(),
        highlight,
    };
    for event in Parser::new_ext(text, options()) {
        writer.event(event);
    }
    writer.finish()
}

/// Whether `text` as markdown has a table, so its lines depend on the
/// width they are rendered at.
pub fn has_table(text: &str) -> bool {
    Parser::new_ext(text, options()).any(|event| matches!(event, Event::Start(Tag::Table(_))))
}

/// A list being written: the next item's number, or `None` for bullets.
struct List {
    next: Option<u64>,
}

struct Writer<'h, H: Highlight> {
    lines: Vec<Line<'static>>,
    /// The marks of the lines of `lines` that are colored code, by their
    /// place.
    code_lines: Vec<(usize, H::Mark)>,
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
    palette: Palette,
    /// The open alerts: the first of their lines and their tint, `None`
    /// for a plain quote.
    alerts: Vec<Option<(usize, Color)>>,
    highlight: &'h H,
}

impl<H: Highlight> Writer<'_, H> {
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
    /// ones too. When the highlight colors its fence's `language`, the code
    /// is in its colors under the language's name; otherwise it is plain
    /// and unlabelled.
    fn code_block(&mut self, language: &str, code: &str) {
        let code = code.strip_suffix('\n').unwrap_or(code);
        let tint = self.palette.code;
        let first = self.prefix();
        let mut rest = self.prefix();
        rest.push(Span::from("  "));
        let colored = match language {
            "" => None,
            language => self.highlight.block(code, language, &rest),
        };
        let Some(colored) = colored else {
            let prefixes = std::iter::once(first).chain(std::iter::repeat(self.prefix()));
            for (line, mut spans) in code.split('\n').zip(prefixes) {
                spans.push(Span::from("  "));
                spans.push(Span::from(line.replace('\t', "    ")));
                self.lines.push(Line::from(spans).bg(tint));
            }
            return;
        };
        let mut spans = first;
        spans.push(Span::from(format!("  {language}")).dim().italic());
        self.lines.push(Line::from(spans).bg(tint));
        for (line, mark) in colored {
            if let Some(mark) = mark {
                self.code_lines.push((self.lines.len(), mark));
            }
            self.lines.push(line.bg(tint));
        }
    }

    /// Inline code: on the code tint, or in its backticks with no colors.
    fn inline_code(&mut self, code: &str) {
        let span = match self.palette.plain {
            true => Span::from(format!("`{code}`")),
            false => Span::styled(code.to_owned(), Style::new().bg(self.palette.code)),
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
        let palette = self.palette;
        let (name, tint) = match kind {
            BlockQuoteKind::Note => ("Note", palette.note),
            BlockQuoteKind::Tip => ("Tip", palette.tip),
            BlockQuoteKind::Important => ("Important", palette.important),
            BlockQuoteKind::Warning => ("Warning", palette.warning),
            BlockQuoteKind::Caution => ("Caution", palette.caution),
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

    fn finish(mut self) -> Vec<(Line<'static>, Option<H::Mark>)> {
        self.flush();
        let mut code = self.code_lines.into_iter().peekable();
        let lines = self.lines.into_iter().enumerate().map(|(at, line)| {
            let code = code.next_if(|(line, _)| *line == at).map(|(_, code)| code);
            (line, code)
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
