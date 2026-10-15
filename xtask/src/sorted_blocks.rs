//! `check-sorted-blocks`: lists that grow with every provider, crate or
//! dependency stay in order, so two changes that each add an entry touch
//! different lines.
//!
//! A block is the lines between one containing `sorted: start` and one
//! containing `sorted: end`, inside whatever comment syntax the file uses.
//! Each entry's key must sort after the previous one's, comparing
//! ASCII-lowercased bytes first and the original bytes second. An entry is a
//! TOML key (a multi-line inline table or array belongs to it until its
//! brackets balance), a Rust `mod` declaration, or a Markdown table row keyed
//! on its first cell. Blank lines, comment lines, a table header and its
//! separator are skipped.
//!
//! The check is textual because line order is the point and a TOML or
//! Markdown parser discards it. Every file in `CHECKED_FILES` must hold at
//! least one block, so deleting the markers cannot pass silently.

use std::path::Path;

/// Files that carry sorted blocks, relative to the workspace root.
const CHECKED_FILES: &[&str] = &[
    "Cargo.toml",
    "README.md",
    "crates/rig-core/src/providers/mod.rs",
];

/// Check every block in `CHECKED_FILES`.
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
                "{relative}: no `sorted: start` / `sorted: end` block found"
            ));
        }
        blocks += found;
    }
    if failures.is_empty() {
        println!(
            "ok: {blocks} sorted block(s) in {} file(s)",
            CHECKED_FILES.len()
        );
        return Ok(());
    }
    Err(format!(
        "sorted blocks out of order; insert each entry where it sorts:\n  {}",
        failures.join("\n  ")
    ))
}

/// Check every block in `source`, appending failures named after `name`.
/// Returns how many blocks were found, an unclosed one included.
pub(crate) fn check_source(name: &str, source: &str, failures: &mut Vec<String>) -> usize {
    let mut blocks = 0;
    let mut open: Option<usize> = None;
    let mut previous: Option<(usize, String)> = None;
    let mut depth = 0;

    let lines: Vec<&str> = source.lines().collect();
    for (index, line) in lines.iter().enumerate() {
        let number = index + 1;
        let trimmed = line.trim();

        if trimmed.contains("sorted: start") {
            if let Some(start) = open {
                failures.push(format!(
                    "{name}:{number}: `sorted: start` while the block from line {start} is open"
                ));
            }
            open = Some(number);
            previous = None;
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
        if open.is_none() {
            continue;
        }
        if depth > 0 {
            depth += bracket_delta(line);
            continue;
        }
        let next = lines.get(index + 1).map_or("", |next| next.trim());
        let Some(key) = entry_key(trimmed, next) else {
            continue;
        };
        depth = bracket_delta(line).max(0);
        if let Some((previous_line, previous_key)) = &previous
            && sort_key(&key) <= sort_key(previous_key)
        {
            failures.push(format!(
                "{name}:{number}: `{key}` sorts before `{previous_key}` (line {previous_line})"
            ));
        }
        previous = Some((number, key));
    }

    if let Some(start) = open {
        failures.push(format!("{name}:{start}: `sorted: start` is never closed"));
    }
    blocks
}

/// The key an entry line sorts on, or `None` for a line that is not an entry.
/// A table row directly above the `| --- |` separator is the header.
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
        return Some(first.to_owned());
    }
    if let Some(rest) = trimmed
        .strip_prefix("pub mod ")
        .or_else(|| trimmed.strip_prefix("mod "))
    {
        return Some(rest.trim_end_matches(';').trim().to_owned());
    }
    let (key, _) = trimmed.split_once('=')?;
    Some(key.trim().to_owned())
}

/// Case-insensitive first and exact bytes second, so `a` and `A` differ but
/// sort together.
fn sort_key(key: &str) -> (Vec<u8>, &[u8]) {
    (key.to_ascii_lowercase().into_bytes(), key.as_bytes())
}

fn is_table_separator(cells: &str) -> bool {
    let cell = cells.split('|').next().unwrap_or("").trim();
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
