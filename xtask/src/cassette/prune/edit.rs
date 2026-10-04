//! Take deleted tests out of a provider source file: a test function with
//! its attributes and doc comments, or a matrix row with its attributes; a
//! matrix left with no row goes whole. The comment lines directly above a
//! deleted item go with it, and so does a doc table row that names a
//! deleted test and no kept one.

#[cfg(test)]
mod tests;

use std::collections::BTreeSet;

use proc_macro2::{Delimiter, LineColumn, TokenStream, TokenTree};
use syn::spanned::Spanned as _;
use syn::visit::{self, Visit};
use syn::{ItemFn, ItemMacro, ItemMod};

use super::names::is_row_body;

/// `source` without the tests whose qualified names are in `deleted`, and
/// how many it took out. `module` is the file's module path.
pub(crate) fn remove_tests(
    source: &str,
    module: &str,
    deleted: &BTreeSet<String>,
) -> Result<(String, usize), String> {
    let file = syn::parse_file(source).map_err(|error| error.to_string())?;
    let mut finder = Finder {
        modules: vec![module.to_owned()],
        deleted,
        ranges: Vec::new(),
        removed: 0,
        gone: BTreeSet::new(),
        kept: BTreeSet::new(),
    };
    finder.visit_file(&file);
    if finder.ranges.is_empty() {
        return Ok((source.to_owned(), 0));
    }
    let lines: Vec<&str> = source.split_inclusive('\n').collect();
    let mut drop: Vec<bool> = lines
        .iter()
        .map(|line| stale_table_row(line, &finder.gone, &finder.kept))
        .collect();
    for (start, end) in &finder.ranges {
        let mut first = start.saturating_sub(1);
        // The comments directly above the item describe it.
        while first > 0
            && lines.get(first - 1).is_some_and(|line| {
                let line = line.trim_start();
                line.starts_with("//") && !line.starts_with("//!")
            })
        {
            first -= 1;
        }
        for slot in drop.iter_mut().take(*end).skip(first) {
            *slot = true;
        }
    }
    let mut out = String::with_capacity(source.len());
    let mut previous_blank = false;
    for (line, dropped) in lines.iter().zip(&drop) {
        if *dropped {
            continue;
        }
        let blank = line.trim().is_empty();
        if blank && previous_blank {
            continue;
        }
        previous_blank = blank;
        out.push_str(line);
    }
    Ok((out, finder.removed))
}

struct Finder<'a> {
    modules: Vec<String>,
    deleted: &'a BTreeSet<String>,
    /// Line ranges to delete, 1-based and inclusive.
    ranges: Vec<(usize, usize)>,
    removed: usize,
    /// The unqualified names of the tests taken out, and of those kept.
    gone: BTreeSet<String>,
    kept: BTreeSet<String>,
}

/// Whether `line` is a doc table row that names a test taken out and no
/// test kept.
fn stale_table_row(line: &str, gone: &BTreeSet<String>, kept: &BTreeSet<String>) -> bool {
    let text = line.trim();
    let Some(row) = text
        .strip_prefix("//!")
        .or_else(|| text.strip_prefix("///"))
        .map(str::trim)
    else {
        return false;
    };
    if !(row.starts_with('|') && row.ends_with('|')) {
        return false;
    }
    let named: BTreeSet<&str> = row.split('`').skip(1).step_by(2).collect();
    named.iter().any(|name| gone.contains(*name)) && !named.iter().any(|name| kept.contains(*name))
}

impl Finder<'_> {
    /// `name` under the current module path; a file at the target root has
    /// an empty one.
    fn qualified(&self, name: &str) -> String {
        self.modules
            .iter()
            .filter(|module| !module.is_empty())
            .map(String::as_str)
            .chain([name])
            .collect::<Vec<_>>()
            .join("::")
    }
}

fn is_test_fn(node: &ItemFn) -> bool {
    node.attrs.iter().any(|attr| {
        attr.path()
            .segments
            .last()
            .is_some_and(|segment| segment.ident == "test")
    })
}

fn line(at: LineColumn) -> usize {
    at.line
}

/// The rows of a matrix: each `;`-separated segment, its row name when it is
/// a row, and its first and last line.
fn rows(tokens: &TokenStream) -> Vec<(Option<String>, usize, usize)> {
    let trees: Vec<TokenTree> = tokens.clone().into_iter().collect();
    let mut out = Vec::new();
    let mut segment: Vec<&TokenTree> = Vec::new();
    let mut flush = |segment: &mut Vec<&TokenTree>, terminator: Option<&TokenTree>| {
        let Some(first) = segment.first() else {
            return;
        };
        let start = line(first.span().start());
        let end = terminator
            .or(segment.last().copied())
            .map_or(start, |tree| line(tree.span().end()));
        let mut rest: &[&TokenTree] = segment;
        while let [TokenTree::Punct(hash), TokenTree::Group(group), tail @ ..] = rest
            && hash.as_char() == '#'
            && group.delimiter() == Delimiter::Bracket
        {
            rest = tail;
        }
        let name = match rest {
            [TokenTree::Ident(name), TokenTree::Punct(colon), body @ ..]
                if colon.as_char() == ':'
                    && !matches!(body.first(), Some(TokenTree::Punct(p)) if p.as_char() == ':')
                    && is_row_body(body) =>
            {
                Some(name.to_string())
            }
            _ => None,
        };
        out.push((name, start, end));
        segment.clear();
    };
    for tree in &trees {
        if matches!(tree, TokenTree::Punct(p) if p.as_char() == ';') {
            flush(&mut segment, Some(tree));
        } else {
            segment.push(tree);
        }
    }
    flush(&mut segment, None);
    out
}

impl<'ast> Visit<'ast> for Finder<'_> {
    fn visit_item_mod(&mut self, node: &'ast ItemMod) {
        let inline = node.content.is_some();
        if inline {
            self.modules.push(node.ident.to_string());
        }
        visit::visit_item_mod(self, node);
        if inline {
            self.modules.pop();
        }
    }

    fn visit_item_fn(&mut self, node: &'ast ItemFn) {
        let name = node.sig.ident.to_string();
        let test = is_test_fn(node);
        if test && !self.deleted.contains(&self.qualified(&name)) {
            self.kept.insert(name.clone());
        }
        if test && self.deleted.contains(&self.qualified(&name)) {
            let start = node
                .attrs
                .iter()
                .map(|attr| line(attr.span().start()))
                .chain([line(node.sig.span().start())])
                .min()
                .unwrap_or(0);
            let end = line(node.block.span().end());
            self.ranges.push((start, end));
            self.removed += 1;
            self.gone.insert(name);
            return;
        }
        visit::visit_item_fn(self, node);
    }

    fn visit_item_macro(&mut self, node: &'ast ItemMacro) {
        let rows = rows(&node.mac.tokens);
        let named: Vec<&(Option<String>, usize, usize)> =
            rows.iter().filter(|(name, _, _)| name.is_some()).collect();
        let gone: Vec<&(Option<String>, usize, usize)> = named
            .iter()
            .copied()
            .filter(|(name, _, _)| {
                name.as_ref()
                    .is_some_and(|name| self.deleted.contains(&self.qualified(name)))
            })
            .collect();
        for (name, _, _) in &named {
            if let Some(name) = name {
                if gone
                    .iter()
                    .any(|(other, _, _)| other.as_ref() == Some(name))
                {
                    self.gone.insert(name.clone());
                } else {
                    self.kept.insert(name.clone());
                }
            }
        }
        if gone.is_empty() {
            return;
        }
        self.removed += gone.len();
        if gone.len() == named.len() {
            let start = node
                .attrs
                .iter()
                .map(|attr| line(attr.span().start()))
                .chain([line(node.mac.path.span().start())])
                .min()
                .unwrap_or(0);
            let end = node.semi_token.map_or_else(
                || line(node.mac.delimiter.span().close().end()),
                |semi| line(semi.span().end()),
            );
            self.ranges.push((start, end));
        } else {
            for (_, start, end) in gone {
                self.ranges.push((*start, *end));
            }
        }
    }
}
