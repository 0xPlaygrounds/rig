//! `check-sorted-blocks`: lists that grow with every provider, crate or
//! dependency stay in byte order, so two PRs that each add one entry touch
//! different lines instead of the same hunk.
//!
//! A block is the text between a line containing `sorted: start` and one
//! containing `sorted: end` (inside whatever comment syntax the file uses:
//! `#`, `//`, or `<!-- -->`). Inside a block every entry's key must be
//! greater than the previous entry's, comparing ASCII-lowercased bytes (so a
//! Markdown table reads `ScyllaDB, SQLite`, not `SQLite, ScyllaDB`) with the
//! original bytes as the tie-break. An entry is one of:
//!
//! * a TOML key (`name = ...`), with a multi-line inline table or array
//!   counted as part of the entry until its brackets balance;
//! * a Rust `mod name;` / `pub mod name;` declaration;
//! * a Markdown table row, keyed on its first cell.
//!
//! Blank lines, comment lines and the `| --- |` table separator are skipped.
//! The check is textual on purpose: it is about line order, which a TOML or
//! Markdown parser would discard. Every file listed in `CHECKED_FILES` must
//! contain at least one block, so removing the markers cannot pass silently.

use std::path::Path;

/// Files that carry `sorted: start` / `sorted: end` blocks, relative to the
/// workspace root.
const CHECKED_FILES: &[&str] = &[
    "Cargo.toml",
    "crates/rig-core/src/providers/mod.rs",
    "README.md",
];

pub(crate) fn check(workspace: &Path) -> Result<(), String> {
    let mut failures = Vec::new();
    let mut blocks = 0;
    for relative in CHECKED_FILES {
        let path = workspace.join(relative);
        let source = std::fs::read_to_string(&path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        let found = check_source(relative, &source, &mut failures);
        if found == 0 {
            failures.push(format!(
                "{relative}: no `sorted: start` / `sorted: end` block found; the markers were \
                 removed or the file no longer carries a sorted list"
            ));
        }
        blocks += found;
    }
    if failures.is_empty() {
        println!(
            "ok: {blocks} sorted block(s) in {} file(s) are in byte order",
            CHECKED_FILES.len()
        );
        return Ok(());
    }
    let mut message = String::from(
        "sorted blocks out of order; reorder the entries so each key is greater than the one \
         before it:\n",
    );
    for failure in &failures {
        message.push_str("  ");
        message.push_str(failure);
        message.push('\n');
    }
    Err(message)
}

/// Check every block in `source`, appending failures; returns how many blocks
/// were found (a block that never closes counts and is reported).
pub(crate) fn check_source(name: &str, source: &str, failures: &mut Vec<String>) -> usize {
    let mut blocks = 0;
    let mut open: Option<(usize, Option<(usize, String)>)> = None;
    let mut depth: i32 = 0;

    let lines: Vec<&str> = source.lines().collect();
    for (index, line) in lines.iter().enumerate() {
        let number = index + 1;
        let trimmed = line.trim();

        if trimmed.contains("sorted: start") {
            if let Some((start, _)) = open {
                failures.push(format!(
                    "{name}:{number}: `sorted: start` while the block from line {start} is still open"
                ));
            }
            open = Some((number, None));
            depth = 0;
            blocks += 1;
            continue;
        }
        if trimmed.contains("sorted: end") {
            if open.take().is_none() {
                failures.push(format!(
                    "{name}:{number}: `sorted: end` without a matching start"
                ));
            }
            continue;
        }
        let Some((_, previous)) = open.as_mut() else {
            continue;
        };

        if depth > 0 {
            depth += bracket_delta(line);
            continue;
        }
        let next = lines.get(index + 1).map(|next| next.trim()).unwrap_or("");
        let Some(key) = entry_key(trimmed, next) else {
            continue;
        };
        depth = bracket_delta(line).max(0);

        if let Some((previous_line, previous_key)) = previous
            && sort_key(&key) <= sort_key(previous_key)
        {
            failures.push(format!(
                "{name}:{number}: `{key}` sorts before `{previous_key}` (line {previous_line})"
            ));
        }
        *previous = Some((number, key));
    }

    if let Some((start, _)) = open {
        failures.push(format!("{name}:{start}: `sorted: start` is never closed"));
    }
    blocks
}

/// The key an entry line sorts on, or `None` for a line that is not an entry.
/// `next` is the following line: a table row directly above the `| --- |`
/// separator is the header, not an entry.
fn entry_key(trimmed: &str, next: &str) -> Option<String> {
    if trimmed.is_empty()
        || trimmed.starts_with('#')
        || trimmed.starts_with("//")
        || trimmed.starts_with("<!--")
    {
        return None;
    }
    if let Some(row) = trimmed.strip_prefix('|') {
        let first = row.split('|').next()?.trim();
        if is_table_separator(first) || next.strip_prefix('|').is_some_and(is_table_separator) {
            return None;
        }
        return Some(first.to_string());
    }
    if let Some(rest) = trimmed
        .strip_prefix("pub mod ")
        .or_else(|| trimmed.strip_prefix("mod "))
    {
        return Some(rest.trim_end_matches(';').trim().to_string());
    }
    let (key, _) = trimmed.split_once('=')?;
    Some(key.trim().to_string())
}

/// Case-insensitive first, exact bytes second, so `a` and `A` are distinct
/// but adjacent.
fn sort_key(key: &str) -> (Vec<u8>, &[u8]) {
    (key.to_ascii_lowercase().into_bytes(), key.as_bytes())
}

fn is_table_separator(cell: &str) -> bool {
    let cell = cell.trim_start().split('|').next().unwrap_or("").trim();
    !cell.is_empty() && cell.chars().all(|ch| ch == '-' || ch == ':')
}

/// Net bracket depth change of a line, ignoring brackets inside string
/// literals and after a `#` comment.
fn bracket_delta(line: &str) -> i32 {
    let mut delta = 0;
    let mut in_string = false;
    for ch in line.chars() {
        match ch {
            '"' => in_string = !in_string,
            '#' if !in_string => break,
            '[' | '{' if !in_string => delta += 1,
            ']' | '}' if !in_string => delta -= 1,
            _ => {}
        }
    }
    delta
}

#[cfg(test)]
mod tests;
