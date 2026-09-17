//! `check-dead`: nothing in rig-core is kept for a caller that does not
//! exist.
//!
//! Two gates, and they answer different questions:
//!
//! | gate | what it finds |
//! |---|---|
//! | a `pub fn`/`pub trait` in `crates/rig-core/src` no **non-test** file outside its own mentions | surface with no caller this workspace can see |
//! | a `pub` struct field that no shipped code reads **and** no cassette or golden carries | a document rig models but nobody reads |
//!
//! The second gate is two conditions on purpose. "No shipped reader" alone
//! is not enough to delete a field: most unread fields *are* in the
//! recordings, which means a provider really sends them and a caller
//! reading `CompletionResponse::raw` sees them. Deleting one of those
//! narrows what rig can be asked about a reply. Both conditions together
//! mean the field is a shape nobody has ever observed, and those are the
//! only ones worth removing.
//!
//! **Both gates report; neither fails the build.** The first one used to
//! fail, and it could not carry that weight. Three reasons, each a fact
//! about Rust rather than about this implementation:
//!
//! - A caller can be a **string**. `#[serde(serialize_with =
//!   "crate::json_utils::serialize_map_sorted")]` at
//!   `providers/cohere/completion.rs:145` is the only non-test reference to
//!   that function, so a scan that ignores string literals calls it dead
//!   and licenses a deletion that would silently change what cohere sends.
//!   A scan that reads them lets prose and a checker's own name table
//!   vouch for a symbol instead. There is no third option that is a word
//!   count.
//! - A caller can be **generated**. A macro body spells a method name that
//!   no call site contains.
//! - A caller can be **downstream**. rig-core is published; a user's call
//!   is a call, and this workspace is not the user.
//!
//! So the scan is a heuristic, and a heuristic must not be able to stop a
//! release. What it is good for is a list a human reads: 199 functions with
//! no caller outside their own file is a real finding about the surface,
//! and it is the list the follow-up works through. What CI enforces is the
//! part that is a fact — `check-wires`, `check-test-layout`, and the
//! suites.
//!
//! Test files no longer vouch for anything: [`occurrences`] is built from
//! non-test files only, which is what [`is_test`] always claimed and the
//! code did not do. A symbol kept alive solely by its own test is exactly
//! the thing worth reporting.

use std::collections::{HashMap, HashSet};
use std::path::Path;

/// The crates whose public surface this check holds to "someone calls it".
///
/// rig-core only: a companion crate's surface is called by its users, and
/// this workspace is not them.
const CHECKED: &str = "crates/rig-core/src";

/// What a green run inspected, so a scan that matched nothing cannot pass
/// for a scan that found nothing.
#[derive(Default)]
struct Report {
    files: usize,
    functions: usize,
    fields: usize,
    dead_functions: Vec<String>,
    module_private: Vec<String>,
    unread_fields: Vec<String>,
}

/// Run both gates over the workspace.
pub(crate) fn check(workspace: &Path) -> Result<(), String> {
    let files = sources(workspace)?;
    // Non-test files only: a symbol its own test is the sole mention of is
    // the finding, not a symbol with a caller.
    let occurrences = occurrences(&files);
    let recorded = recorded_keys(workspace)?;

    let mut report = Report::default();
    for (path, source) in &files {
        let relative = relative(workspace, path);
        if !relative.starts_with(CHECKED) || is_test(&relative) {
            continue;
        }
        report.files += 1;
        let own = words(source);
        for (kind, name) in declarations(source) {
            report.functions += 1;
            let total = occurrences.get(&name).copied().unwrap_or_default();
            let here = own.get(&name).copied().unwrap_or_default();
            if total != here {
                continue;
            }
            // Its own file is the only place the name occurs. One
            // occurrence is the declaration alone — nothing calls it.
            // More means it is called, but only from inside: real
            // behaviour behind an unnecessarily public door.
            if here <= 1 {
                report.dead_functions.push(format!(
                    "  {relative}: `pub {kind} {name}` is mentioned by no other non-test file"
                ));
            } else {
                report.module_private.push(format!(
                    "  {relative}: `pub {kind} {name}` is called only by its own module"
                ));
            }
        }
        for (name, wire) in fields(source) {
            report.fields += 1;
            let total = occurrences.get(&name).copied().unwrap_or_default();
            let here = own.get(&name).copied().unwrap_or_default();
            // `here > 1` is the field's own module reading it — the
            // declaration is one occurrence, a read is another — which is a
            // reader like any other.
            if total != here || here > 1 || recorded.contains(&wire) || recorded.contains(&name) {
                continue;
            }
            report
                .unread_fields
                .push(format!("  {relative}: `{name}` (wire `{wire}`)"));
        }
    }

    println!(
        "ok: {} rig-core files, {} public functions and traits, {} fields inspected\n\
         no caller outside their own file: {}{}\n\
         called only by their own module: {}{}\n\
         fields no code reads and no recording carries: {}{}",
        report.files,
        report.functions,
        report.fields,
        report.dead_functions.len(),
        listing(&report.dead_functions),
        report.module_private.len(),
        listing(&report.module_private),
        report.unread_fields.len(),
        listing(&report.unread_fields)
    );
    Ok(())
}

/// A count's detail, when there is any.
fn listing(items: &[String]) -> String {
    if items.is_empty() {
        String::new()
    } else {
        format!("\n{}", items.join("\n"))
    }
}

/// Every Rust source file in the workspace, as text.
fn sources(workspace: &Path) -> Result<Vec<(std::path::PathBuf, String)>, String> {
    let mut out = Vec::new();
    // `src` is the root facade crate, which re-exports rig-core's surface
    // item by item: without it a re-export-only symbol has no caller.
    for root in [
        "crates",
        "src",
        "tests",
        "examples",
        "xtask",
        "test-support",
    ] {
        walk(&workspace.join(root), &mut out)?;
    }
    Ok(out)
}

fn walk(dir: &Path, out: &mut Vec<(std::path::PathBuf, String)>) -> Result<(), String> {
    if !dir.is_dir() {
        return Ok(());
    }
    let entries = std::fs::read_dir(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("{}: {error}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            if path.file_name().is_some_and(|name| name == "target") {
                continue;
            }
            walk(&path, out)?;
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            let source = std::fs::read_to_string(&path)
                .map_err(|error| format!("{}: {error}", path.display()))?;
            out.push((path, source));
        }
    }
    Ok(())
}

/// The path as the report names it.
fn relative(workspace: &Path, path: &Path) -> String {
    path.strip_prefix(workspace)
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Whether the file is test code, whose callers do not count as readers.
fn is_test(relative: &str) -> bool {
    relative.ends_with("tests.rs") || relative.contains("/tests/") || relative.starts_with("tests/")
}

/// Every identifier in `source`, with how often it occurs.
fn words(source: &str) -> HashMap<String, usize> {
    let mut counts: HashMap<String, usize> = HashMap::new();
    let mut word = String::new();
    for character in source.chars() {
        if character.is_alphanumeric() || character == '_' {
            word.push(character);
            continue;
        }
        if !word.is_empty() {
            *counts.entry(std::mem::take(&mut word)).or_default() += 1;
        }
    }
    if !word.is_empty() {
        *counts.entry(word).or_default() += 1;
    }
    counts
}

/// Every identifier in the workspace, with how often it occurs.
fn occurrences(files: &[(std::path::PathBuf, String)]) -> HashMap<String, usize> {
    let mut total: HashMap<String, usize> = HashMap::new();
    for (path, source) in files {
        // A test is not a caller. `is_test` has always said so; counting
        // every walked file said otherwise, and 63 rig-core functions
        // passed on a test-file mention alone.
        if is_test(&path.to_string_lossy().replace('\\', "/")) {
            continue;
        }
        for (word, count) in words(source) {
            *total.entry(word).or_default() += count;
        }
    }
    total
}

/// Every JSON key any cassette or golden carries.
fn recorded_keys(workspace: &Path) -> Result<HashSet<String>, String> {
    let mut keys = HashSet::new();
    for root in ["tests/cassettes", "crates/rig-verify/fixtures"] {
        let mut files = Vec::new();
        collect(&workspace.join(root), &mut files)?;
        for source in files {
            // A cassette stores a JSON body as a YAML scalar, so most recordings
            // carry `\"model\"` rather than `"model"`. Unescaping is half the
            // job: the scalar's own opening quote shifts the quote parity, so a
            // scan that always resumes past a closing quote reads `:` and `,` as
            // the keys and finds none of the real ones -- the gate would then
            // report "no recording carries this field" for files that do.
            // Resuming past the *opening* quote instead tries both parities.
            let source = source.replace("\\\"", "\"");
            let bytes = source.as_bytes();
            let mut rest = source.as_str();
            let mut offset = 0usize;
            while let Some(start) = rest.find('"') {
                let open = offset + start;
                rest = &rest[start + 1..];
                offset = open + 1;
                let Some(end) = rest.find('"') else { break };
                let (candidate, after) = (&rest[..end], &rest[end + 1..]);
                // A key opens where a key can open: at the start of an
                // object, after a comma, or after the enclosing scalar's own
                // quote. Accepting any quote-delimited run followed by a
                // colon minted `ent`, `elo`, `lue` and `Tok` out of the
                // middles of values, and a junk key that happens to match a
                // short field name reads as "a recording carries this".
                let opens_a_key = bytes
                    .get(..open)
                    .unwrap_or_default()
                    .iter()
                    .rev()
                    .find(|byte| !byte.is_ascii_whitespace())
                    .is_none_or(|byte| matches!(byte, b'{' | b',' | b'"' | b':' | b'['));
                if opens_a_key
                    && after.trim_start().starts_with(':')
                    && !candidate.is_empty()
                    && candidate
                        .chars()
                        .all(|c| c.is_alphanumeric() || c == '_' || c == '-')
                {
                    keys.insert(candidate.to_owned());
                }
            }
        }
    }
    Ok(keys)
}

/// Every recorded file's text, whatever its extension.
fn collect(dir: &Path, out: &mut Vec<String>) -> Result<(), String> {
    if !dir.is_dir() {
        return Ok(());
    }
    let entries = std::fs::read_dir(dir).map_err(|error| format!("{}: {error}", dir.display()))?;
    for entry in entries {
        let entry = entry.map_err(|error| format!("{}: {error}", dir.display()))?;
        let path = entry.path();
        if path.is_dir() {
            collect(&path, out)?;
        } else if let Ok(source) = std::fs::read_to_string(&path) {
            out.push(source);
        }
    }
    Ok(())
}

/// Every `pub fn` / `pub trait` declared in `source`, by kind and name.
fn declarations(source: &str) -> Vec<(&'static str, String)> {
    let mut out = Vec::new();
    for line in source.lines() {
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("pub ") else {
            continue;
        };
        let rest = rest
            .strip_prefix("const ")
            .or_else(|| rest.strip_prefix("async "))
            .unwrap_or(rest);
        let (kind, rest) = if let Some(rest) = rest.strip_prefix("fn ") {
            ("fn", rest)
        } else if let Some(rest) = rest.strip_prefix("trait ") {
            ("trait", rest)
        } else {
            continue;
        };
        let name: String = rest
            .chars()
            .take_while(|c| c.is_alphanumeric() || *c == '_')
            .collect();
        if !name.is_empty() {
            out.push((kind, name));
        }
    }
    out
}

/// Every `pub` field declared in `source`, with the wire key it carries.
///
/// The key is the field's `rename` when it has one, else its container's
/// `rename_all` applied to the name, else the name. Getting this wrong in
/// either direction breaks the gate: a `camelCase` container whose key is
/// looked up in snake_case reports every recorded field as unrecorded.
fn fields(source: &str) -> Vec<(String, String)> {
    let mut out = Vec::new();
    let lines: Vec<&str> = source.lines().collect();
    let mut container = None;
    for (index, line) in lines.iter().enumerate() {
        // A container's `rename_all` governs that container only. Carrying
        // the first one seen through the rest of the file keyed 18
        // rig-core containers off a neighbour's casing -- `Schema`'s
        // `max_items` looked up as `maxItems` -- so every one of their
        // fields was compared against a key no recording could hold.
        let trimmed_line = line.trim_start();
        if trimmed_line.starts_with("pub struct ")
            || trimmed_line.starts_with("pub enum ")
            || trimmed_line.starts_with("struct ")
            || trimmed_line.starts_with("enum ")
        {
            // Read this container's own casing from its own attributes.
            // Carrying the first `rename_all` in the file through the rest
            // of it keyed 18 rig-core containers off a neighbour's casing
            // -- `Schema`'s `max_items` looked up as `maxItems` -- so every
            // field of theirs was compared against a key no recording could
            // hold.
            container = None;
            for back in 1..6 {
                let Some(above) = index.checked_sub(back).and_then(|i| lines.get(i)) else {
                    break;
                };
                if let Some(start) = above.find("rename_all = \"") {
                    let after = &above[start + "rename_all = \"".len()..];
                    if let Some(end) = after.find('"') {
                        container = Some(after[..end].to_owned());
                    }
                    break;
                }
                let above = above.trim_start();
                if !above.starts_with('#') && !above.starts_with("///") && !above.starts_with("//")
                {
                    break;
                }
            }
        }
        let trimmed = line.trim_start();
        let Some(rest) = trimmed.strip_prefix("pub ") else {
            continue;
        };
        let name: String = rest
            .chars()
            .take_while(|c| c.is_lowercase() || c.is_numeric() || *c == '_')
            .collect();
        if name.is_empty() || !rest[name.len()..].starts_with(':') {
            continue;
        }
        let mut wire = match container.as_deref() {
            Some("camelCase") => camel_case(&name),
            _ => name.clone(),
        };
        // A field rig *writes* when a caller sets it is a request option, and a
        // recording of rig's own past traffic cannot speak to it: nothing
        // in-tree constructs `FileSearchTool`, yet dropping the store names
        // leaves a search of nowhere. `skip_serializing_if` is exactly that
        // marker, so such a field is never reported as unread.
        let mut request_side = false;
        for back in 1..4 {
            let Some(above) = index.checked_sub(back).and_then(|i| lines.get(i)) else {
                break;
            };
            if above.contains("skip_serializing_if") {
                request_side = true;
            }
            if let Some(start) = above.find("rename = \"") {
                let after = &above[start + "rename = \"".len()..];
                if let Some(end) = after.find('"') {
                    wire = after[..end].to_owned();
                }
                break;
            }
            if !above.trim_start().starts_with('#') {
                break;
            }
        }
        if request_side {
            continue;
        }
        out.push((name, wire));
    }
    out
}

/// `inference_queue_time` as `inferenceQueueTime`.
fn camel_case(name: &str) -> String {
    let mut out = String::with_capacity(name.len());
    let mut capitalize = false;
    for character in name.chars() {
        if character == '_' {
            capitalize = true;
            continue;
        }
        if capitalize {
            out.extend(character.to_uppercase());
            capitalize = false;
        } else {
            out.push(character);
        }
    }
    out
}

#[cfg(test)]
mod tests;
