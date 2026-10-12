//! HTML to Markdown on html5ever's tokenizer: headings, paragraphs, lists,
//! links, code blocks, emphasis and tables are kept; scripts, styles,
//! navigation, footers, forms and the document's head are dropped.

use std::cell::RefCell;

use html5ever::tendril::StrTendril;
use html5ever::tokenizer::states::RawKind;
use html5ever::tokenizer::{
    BufferQueue, Tag, TagKind, Token, TokenSink, TokenSinkResult, Tokenizer, TokenizerOpts,
};
use reqwest::Url;

/// Elements dropped with everything in them.
const DROPPED: &[&str] = &[
    "script", "style", "noscript", "template", "svg", "nav", "footer", "head", "iframe", "form",
    "aside", "button", "select", "canvas", "object",
];

/// Elements that start and end a paragraph of their own.
const BLOCKS: &[&str] = &[
    "p",
    "div",
    "section",
    "article",
    "main",
    "header",
    "blockquote",
    "figure",
    "figcaption",
    "dl",
    "dt",
    "dd",
    "address",
    "details",
    "summary",
];

/// Elements that never have an end tag.
const VOID: &[&str] = &[
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta", "source", "track",
    "wbr",
];

/// `html` as Markdown, its relative links resolved against `base`.
pub(crate) fn to_markdown(html: &str, base: Option<&Url>) -> String {
    let sink = Sink(RefCell::new(Markdown {
        base: base.cloned(),
        ..Markdown::default()
    }));
    let input = BufferQueue::default();
    input.push_back(StrTendril::from_slice(html));
    let tokenizer = Tokenizer::new(sink, TokenizerOpts::default());
    let _ = tokenizer.feed(&input);
    tokenizer.end();
    let text = tokenizer
        .sink
        .0
        .try_borrow()
        .map(|markdown| markdown.out.clone())
        .unwrap_or_default();
    tidy(&text)
}

/// Trailing spaces off every line, at most one blank line in a row.
fn tidy(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut blank = 0;
    for line in text.trim().lines() {
        let line = line.trim_end();
        if line.is_empty() {
            blank += 1;
            if blank > 1 {
                continue;
            }
        } else {
            blank = 0;
        }
        out.push_str(line);
        out.push('\n');
    }
    out.trim_end().to_owned()
}

/// The tokenizer's sink.
struct Sink(RefCell<Markdown>);

impl TokenSink for Sink {
    type Handle = ();

    fn process_token(&self, token: Token, _line: u64) -> TokenSinkResult<()> {
        let Ok(mut markdown) = self.0.try_borrow_mut() else {
            return TokenSinkResult::Continue;
        };
        match token {
            Token::TagToken(tag) => match tag.kind {
                TagKind::StartTag => {
                    markdown.start(&tag);
                    raw_kind(&tag.name).map_or(TokenSinkResult::Continue, TokenSinkResult::RawData)
                }
                TagKind::EndTag => {
                    markdown.end(&tag.name);
                    TokenSinkResult::Continue
                }
            },
            Token::CharacterTokens(text) => {
                markdown.text(&text);
                TokenSinkResult::Continue
            }
            _ => TokenSinkResult::Continue,
        }
    }
}

/// How the tokenizer reads what an element holds, as a tree builder would
/// tell it: a script's or a style's text is no markup.
fn raw_kind(name: &str) -> Option<RawKind> {
    match name {
        "script" => Some(RawKind::ScriptData),
        "style" | "xmp" | "iframe" | "noembed" | "noframes" => Some(RawKind::Rawtext),
        "textarea" | "title" => Some(RawKind::Rcdata),
        _ => None,
    }
}

/// The Markdown written so far, and where in the document it is.
#[derive(Default)]
struct Markdown {
    out: String,
    base: Option<Url>,
    /// The dropped element being skipped, and how deep in it.
    skip: Option<(String, usize)>,
    /// How many `pre` elements are open.
    pre: usize,
    /// Where an open fence's language goes.
    fence_at: Option<usize>,
    /// Open lists: the next number of an ordered one, `None` for bullets.
    lists: Vec<Option<u64>>,
    /// Open links: where their text starts and their target, `None` for
    /// one that is no link worth keeping.
    links: Vec<Option<(usize, String)>>,
    /// Inside a table cell.
    cell: bool,
    /// Cells of the open row.
    row_cells: usize,
    /// Rows written of each open table.
    tables: Vec<usize>,
    /// Whitespace seen and not written yet.
    space: bool,
}

impl Markdown {
    fn start(&mut self, tag: &Tag) {
        let name: &str = &tag.name;
        if let Some((skipped, depth)) = &mut self.skip {
            if skipped == name {
                *depth += 1;
            }
            return;
        }
        let role = attribute(tag, "role");
        if DROPPED.contains(&name) || matches!(role.as_deref(), Some("navigation" | "contentinfo"))
        {
            if !VOID.contains(&name) && !tag.self_closing {
                self.skip = Some((name.to_owned(), 1));
            }
            return;
        }
        match name {
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => {
                self.block();
                let level = name.trim_start_matches('h').parse().unwrap_or(1);
                self.out.push_str(&"#".repeat(level));
                self.out.push(' ');
            }
            "br" if self.pre > 0 => self.out.push('\n'),
            "br" => self.line(),
            "hr" => {
                self.block();
                self.out.push_str("---");
                self.block();
            }
            "ul" | "ol" => {
                if self.lists.is_empty() {
                    self.block();
                } else {
                    self.line();
                }
                let start = attribute(tag, "start").and_then(|start| start.parse().ok());
                self.lists
                    .push((name == "ol").then_some(start.unwrap_or(1)));
            }
            "li" => {
                self.line();
                let indent = "  ".repeat(self.lists.len().saturating_sub(1));
                let marker = match self.lists.last_mut() {
                    Some(Some(next)) => {
                        let marker = format!("{next}. ");
                        *next += 1;
                        marker
                    }
                    _ => "- ".to_owned(),
                };
                self.out.push_str(&indent);
                self.out.push_str(&marker);
            }
            "a" => {
                let href = attribute(tag, "href").unwrap_or_default();
                let keep =
                    !href.is_empty() && !href.starts_with('#') && !href.starts_with("javascript:");
                if keep && self.pre == 0 {
                    let target = self
                        .base
                        .as_ref()
                        .and_then(|base| base.join(&href).ok())
                        .map_or(href, String::from);
                    self.open("[");
                    self.links.push(Some((self.out.len(), target)));
                } else {
                    self.links.push(None);
                }
            }
            "pre" => {
                self.block();
                self.out.push_str("```");
                self.fence_at = Some(self.out.len());
                self.out.push('\n');
                self.pre += 1;
            }
            "code" if self.pre > 0 => self.fence_language(tag),
            "code" | "kbd" | "samp" => self.open("`"),
            "strong" | "b" => self.open("**"),
            "em" | "i" => self.open("*"),
            "img" => {
                let alt = attribute(tag, "alt").unwrap_or_default();
                if !alt.trim().is_empty() {
                    self.open(&format!("[image: {}]", alt.trim()));
                }
            }
            "table" => {
                self.block();
                self.tables.push(0);
            }
            "tr" => {
                self.line();
                self.out.push('|');
                self.row_cells = 0;
            }
            "td" | "th" => {
                self.cell = true;
                self.space = false;
                self.out.push(' ');
            }
            _ if BLOCKS.contains(&name) => self.block(),
            _ => {}
        }
    }

    fn end(&mut self, name: &str) {
        if let Some((skipped, depth)) = &mut self.skip {
            if skipped == name {
                *depth -= 1;
                if *depth == 0 {
                    self.skip = None;
                }
            }
            return;
        }
        match name {
            "h1" | "h2" | "h3" | "h4" | "h5" | "h6" => self.block(),
            "ul" | "ol" => {
                self.lists.pop();
                if self.lists.is_empty() {
                    self.block();
                } else {
                    self.line();
                }
            }
            "li" => self.line(),
            "a" => self.close_link(),
            "pre" => {
                self.pre = self.pre.saturating_sub(1);
                if !self.out.ends_with('\n') {
                    self.out.push('\n');
                }
                self.out.push_str("```");
                self.fence_at = None;
                self.block();
            }
            "code" if self.pre > 0 => {}
            "code" | "kbd" | "samp" => self.close("`"),
            "strong" | "b" => self.close("**"),
            "em" | "i" => self.close("*"),
            "td" | "th" => {
                self.trim_end();
                self.out.push_str(" |");
                self.cell = false;
                self.space = false;
                self.row_cells += 1;
            }
            "tr" => {
                let cells = self.row_cells;
                if let Some(rows) = self.tables.last_mut() {
                    if *rows == 0 && cells > 0 {
                        self.out.push_str("\n|");
                        self.out.push_str(&" --- |".repeat(cells));
                    }
                    *rows += 1;
                }
                self.line();
            }
            "table" => {
                self.tables.pop();
                self.block();
            }
            _ if BLOCKS.contains(&name) => self.block(),
            _ => {}
        }
    }

    fn text(&mut self, text: &str) {
        if self.skip.is_some() {
            return;
        }
        if self.pre > 0 {
            // The newline right after `<pre>` is not content.
            let fresh = self.fence_at.is_some_and(|at| self.out.len() == at + 1);
            let text = if fresh {
                text.strip_prefix('\n').unwrap_or(text)
            } else {
                text
            };
            self.out.push_str(text);
            return;
        }
        for word in text.split_inclusive(char::is_whitespace) {
            let trimmed = word.trim_end();
            if !trimmed.is_empty() {
                self.flush_space();
                self.out.push_str(trimmed);
            }
            if trimmed.len() != word.len() {
                self.space = true;
            }
        }
    }

    /// The fence's language from the code's `language-x` class.
    fn fence_language(&mut self, tag: &Tag) {
        let class = attribute(tag, "class").unwrap_or_default();
        let language = class.split_whitespace().find_map(|class| {
            class
                .strip_prefix("language-")
                .or_else(|| class.strip_prefix("lang-"))
        });
        if let (Some(language), Some(at)) = (language, self.fence_at)
            && self.out.len() == at + 1
            && self.out.is_char_boundary(at)
        {
            self.out.insert_str(at, language);
            self.fence_at = Some(at + language.len());
        }
    }

    fn close_link(&mut self) {
        let Some(Some((at, target))) = self.links.pop() else {
            return;
        };
        let has_text = self
            .out
            .get(at..)
            .is_some_and(|text| !text.trim().is_empty());
        if has_text {
            self.close(&format!("]({target})"));
        } else if let Some(bracket) = at.checked_sub(1)
            && self.out.get(bracket..at) == Some("[")
        {
            self.out.truncate(bracket);
        }
    }

    /// Writes an opening marker after any whitespace seen before it.
    fn open(&mut self, marker: &str) {
        self.flush_space();
        self.out.push_str(marker);
    }

    /// Writes a closing marker against the text before it.
    fn close(&mut self, marker: &str) {
        self.out.push_str(marker);
    }

    fn flush_space(&mut self) {
        if self.space && !self.out.is_empty() && !self.out.ends_with([' ', '\n']) {
            self.out.push(' ');
        }
        self.space = false;
    }

    fn trim_end(&mut self) {
        let kept = self.out.trim_end_matches(' ').len();
        self.out.truncate(kept);
    }

    /// Ends the line, unless it just ended; a space in a table cell.
    fn line(&mut self) {
        if self.cell {
            self.space = true;
            return;
        }
        self.space = false;
        self.trim_end();
        if !self.out.is_empty() && !self.out.ends_with('\n') {
            self.out.push('\n');
        }
    }

    /// Ends the paragraph; only the line inside a list or a cell.
    fn block(&mut self) {
        if self.cell || !self.lists.is_empty() {
            self.line();
            return;
        }
        self.line();
        if !self.out.is_empty() && !self.out.ends_with("\n\n") {
            self.out.push('\n');
        }
    }
}

/// The value of `tag`'s attribute `name`.
fn attribute(tag: &Tag, name: &str) -> Option<String> {
    tag.attrs
        .iter()
        .find(|attribute| &*attribute.name.local == name)
        .map(|attribute| attribute.value.to_string())
}
