//! The driver-adoption source guards, run by `verify --check source-guards`.
//!
//! The driver's read loop in `rig-core/src/driver.rs` is the only place that
//! decides what happens to `WireEvent::Unknown` and `WireEvent::Corrupt`
//! frames. Provider code may classify a frame but never triage it, so a
//! mention of either variant outside the driver, the classify layers and test
//! code is a second policy table. The same walk keeps raw serde parsing out
//! of streaming modules (exceptions live in `serde_policy_allowlist.txt`),
//! keeps wire payloads out of WARN logs, keeps runtimes from building
//! assistant turns by hand, and keeps encoders reading provider items only
//! through `AssistantContent::replay`.
//!
//! Every scan reads the shipped portion of each `.rs` file under `crates/`:
//! `#[cfg(test)]` items are blanked and test-only sibling modules skipped.
//! The scans are textual tripwires against drift, not security boundaries.

use std::path::Path;

/// The serde wall's exceptions: `path suffix | line snippet | justification`.
const SERDE_POLICY_ALLOWLIST: &str = include_str!("driver_adoption/serde_policy_allowlist.txt");
const SERDE_POLICY_ALLOWLIST_PATH: &str = "xtask/src/driver_adoption/serde_policy_allowlist.txt";

/// Path suffixes of the only files where the policy table and classifiers
/// name the triage variants. Matching the full suffix, not the basename,
/// keeps a foreign `adapter.rs` in scope.
const ALLOWED_POLICY_HOMES: &[&str] = &[
    // The driver's read loop and its HTTP framing half.
    "rig-core/src/driver.rs",
    "rig-core/src/driver/http_transport.rs",
    // The `Decoder` contract and `WireFrame`, which the variants are about.
    "rig-core/src/wire.rs",
    "rig-core/src/providers/internal/wire.rs",
];

/// Whether `path` is one of [`ALLOWED_POLICY_HOMES`]: the exemption both
/// guards share.
fn is_named_policy_home(path: &str) -> bool {
    ALLOWED_POLICY_HOMES
        .iter()
        .any(|suffix| path.ends_with(suffix))
}

/// Whether `path` may name the triage variants, for the triage guard only.
///
/// A provider's own `wire.rs`, and the modules under its `wire/`, is its
/// classify layer. Naming the variants there is not licence to raw-parse the
/// wire, so [`is_serde_wall_target`] does not consult this predicate.
fn is_policy_home(path: &str) -> bool {
    if let Some((_, providers)) = path.split_once("rig-core/src/providers/")
        && (providers.ends_with("/wire.rs") || providers.contains("/wire/"))
    {
        return true;
    }
    is_named_policy_home(path)
}

/// Directories that hold test harness code rather than shipped policy.
const SKIPPED_DIRS: &[&str] = &["tests", "test_utils", "fixtures", "target"];

/// Blanks the contents of comments and string and char literals with spaces,
/// keeping newlines, so line numbers and line count are unchanged.
///
/// Structural decisions run on the mask, so a `{` inside a JSON fixture
/// string or a `#[cfg(test)]` quoted in a block comment cannot steer them.
fn mask_literals_and_comments(source: &str) -> String {
    enum State {
        Code,
        LineComment,
        /// Block comments nest; the payload is the open depth.
        BlockComment(usize),
        Str,
        /// `r##"…"##`; the payload is the hash count that closes it.
        RawStr(usize),
        Char,
    }

    let mut masked = String::with_capacity(source.len());
    let mut state = State::Code;
    let mut chars = source.chars().peekable();
    let mut escaped = false;

    let emit = |masked: &mut String, ch: char, keep: bool| {
        if ch == '\n' || keep {
            masked.push(ch);
        } else {
            masked.push(' ');
        }
    };

    while let Some(ch) = chars.next() {
        match state {
            State::Code => match ch {
                '/' if chars.peek() == Some(&'/') => {
                    state = State::LineComment;
                    emit(&mut masked, ch, false);
                }
                '/' if chars.peek() == Some(&'*') => {
                    state = State::BlockComment(1);
                    emit(&mut masked, ch, false);
                }
                '"' => {
                    state = State::Str;
                    emit(&mut masked, ch, false);
                }
                'r' if matches!(chars.peek(), Some('"') | Some('#')) => {
                    let mut hashes = 0usize;
                    let mut lookahead = chars.clone();
                    while lookahead.peek() == Some(&'#') {
                        let _ = lookahead.next();
                        hashes += 1;
                    }
                    if lookahead.peek() == Some(&'"') {
                        for _ in 0..=hashes {
                            if let Some(consumed) = chars.next() {
                                emit(&mut masked, consumed, false);
                            }
                        }
                        state = State::RawStr(hashes);
                        emit(&mut masked, ch, false);
                    } else {
                        emit(&mut masked, ch, true);
                    }
                }
                '\'' => {
                    // A lifetime (`'a`) is code; only a char literal has its
                    // closing quote within the next two characters.
                    let mut lookahead = chars.clone();
                    let first = lookahead.next();
                    let second = lookahead.next();
                    let is_char_literal = first == Some('\\')
                        || (first.is_some() && second == Some('\''))
                        || first == Some('\'');
                    if is_char_literal {
                        state = State::Char;
                        emit(&mut masked, ch, false);
                    } else {
                        emit(&mut masked, ch, true);
                    }
                }
                _ => emit(&mut masked, ch, true),
            },
            State::LineComment => {
                if ch == '\n' {
                    state = State::Code;
                }
                emit(&mut masked, ch, false);
            }
            State::BlockComment(depth) => {
                if ch == '/' && chars.peek() == Some(&'*') {
                    state = State::BlockComment(depth.saturating_add(1));
                } else if ch == '*' && chars.peek() == Some(&'/') {
                    if depth <= 1 {
                        // Consume the `/` so the comment cannot re-open.
                        if let Some(slash) = chars.next() {
                            emit(&mut masked, ch, false);
                            emit(&mut masked, slash, false);
                            state = State::Code;
                            continue;
                        }
                    }
                    state = State::BlockComment(depth.saturating_sub(1));
                }
                emit(&mut masked, ch, false);
            }
            State::Str | State::Char => {
                let closing = if matches!(state, State::Str) {
                    '"'
                } else {
                    '\''
                };
                if escaped {
                    escaped = false;
                } else if ch == '\\' {
                    escaped = true;
                } else if ch == closing {
                    state = State::Code;
                }
                emit(&mut masked, ch, false);
            }
            State::RawStr(hashes) => {
                if ch == '"' {
                    let mut lookahead = chars.clone();
                    let mut seen = 0usize;
                    while seen < hashes && lookahead.peek() == Some(&'#') {
                        let _ = lookahead.next();
                        seen += 1;
                    }
                    if seen == hashes {
                        for _ in 0..hashes {
                            if let Some(consumed) = chars.next() {
                                emit(&mut masked, consumed, false);
                            }
                        }
                        state = State::Code;
                    }
                }
                emit(&mut masked, ch, false);
            }
        }
    }

    masked
}

/// Index one past the last line of the `#[cfg(test)]`-gated item starting at
/// `start`, given the masked lines of a file.
///
/// The item ends when its brace depth returns to zero or, for a brace-less
/// item (`#[cfg(test)] use foo;`), at the first top-level `;`.
fn end_of_gated_item(masked_lines: &[&str], start: usize) -> usize {
    let mut depth: isize = 0;
    let mut seen_brace = false;
    let mut index = start;
    while let Some(line) = masked_lines.get(index) {
        for ch in line.chars() {
            match ch {
                '{' => {
                    depth += 1;
                    seen_brace = true;
                }
                '}' => depth -= 1,
                ';' if !seen_brace && depth == 0 => return index + 1,
                _ => {}
            }
        }
        index += 1;
        if seen_brace && depth <= 0 {
            return index;
        }
    }
    masked_lines.len()
}

/// The shipped portion of a source file: each `#[cfg(test)]`-gated item is
/// replaced by empty lines, so line numbers still match the file on disk.
///
/// Blanking is item-scoped, so shipped code after a gated helper is scanned.
/// Attribute position and item extent are decided on the mask.
fn shipped_portion(source: &str) -> String {
    let masked = mask_literals_and_comments(source);
    let lines: Vec<&str> = source.split_inclusive('\n').collect();
    let masked_lines: Vec<&str> = masked.split_inclusive('\n').collect();

    let mut shipped = String::with_capacity(source.len());
    let mut index = 0usize;
    while let Some(line) = lines.get(index) {
        let is_gate = masked_lines
            .get(index)
            .is_some_and(|masked| masked.trim_start().starts_with("#[cfg(test)]"));
        if is_gate {
            let end = end_of_gated_item(&masked_lines, index);
            for blanked in index..end {
                if lines.get(blanked).is_some_and(|l| l.ends_with('\n')) {
                    shipped.push('\n');
                }
            }
            index = end;
            continue;
        }
        shipped.push_str(line);
        index += 1;
    }
    shipped
}

/// Whether `text` has a whole-line `mod <name>;` declaration of any
/// visibility.
fn declares_module(text: &str, name: &str) -> bool {
    text.lines().any(|line| {
        let line = line.trim();
        let line = line.strip_prefix("pub").map_or(line, |rest| {
            // `pub(crate)`, `pub(super)`, `pub(in path)`.
            let rest = rest.trim_start();
            rest.strip_prefix('(')
                .and_then(|r| r.split_once(')'))
                .map_or(rest, |(_, after)| after)
                .trim_start()
        });
        line.strip_prefix("mod ")
            .and_then(|rest| rest.trim_start().strip_suffix(';'))
            .is_some_and(|declared| declared.trim() == name)
    })
}

/// Whether the file at `path` is reachable only through a `#[cfg(test)]`
/// `mod <name>;` in its parent module file.
///
/// The parent is found as rustc finds it: `dir/mod.rs`, `dir.rs` beside the
/// directory, or `lib.rs`/`main.rs` when the directory is `src`. A file
/// whose parent declares it ungated, or has no parent, is shipped.
fn is_test_only_module(path: &Path) -> bool {
    let Some(stem) = path.file_stem().and_then(|s| s.to_str()) else {
        return false;
    };
    let Some(dir) = path.parent() else {
        return false;
    };
    // `dir/mod.rs` is declared as `mod <dir>;` by its own parent.
    let (name, dir) = if stem == "mod" {
        match (dir.file_name().and_then(|s| s.to_str()), dir.parent()) {
            (Some(name), Some(parent)) => (name.to_owned(), parent),
            _ => return false,
        }
    } else {
        (stem.to_owned(), dir)
    };

    let mut candidates = vec![dir.join("mod.rs"), dir.with_extension("rs")];
    if dir.file_name().is_some_and(|d| d == "src") {
        candidates.push(dir.join("lib.rs"));
        candidates.push(dir.join("main.rs"));
    }
    candidates.into_iter().any(|parent| {
        let Ok(source) = std::fs::read_to_string(&parent) else {
            return false;
        };
        let masked = mask_literals_and_comments(&source);
        declares_module(&masked, &name) && !declares_module(&shipped_portion(&source), &name)
    })
}

/// One shipped source file: its workspace-relative path with `/` separators,
/// and its text with test-gated items blanked.
struct Source {
    path: String,
    shipped: String,
}

/// Every shipped `.rs` file under `workspace/crates`, skipping
/// [`SKIPPED_DIRS`] and test-only modules.
fn shipped_sources(workspace: &Path) -> Result<Vec<Source>, String> {
    let crates = workspace.join("crates");
    let mut sources = Vec::new();
    for path in crate::support::files_under(&crates, Some("rs"))? {
        let relative = path.strip_prefix(&crates).unwrap_or(&path);
        let skipped = relative.parent().is_some_and(|dir| {
            dir.components()
                .any(|part| SKIPPED_DIRS.iter().any(|skip| part.as_os_str() == *skip))
        });
        if skipped || is_test_only_module(&path) {
            continue;
        }
        let source = std::fs::read_to_string(&path)
            .map_err(|error| format!("could not read {}: {error}", path.display()))?;
        let path = path
            .strip_prefix(workspace)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        sources.push(Source {
            path,
            shipped: shipped_portion(&source),
        });
    }
    Ok(sources)
}

/// Files the walk must find, or it has collapsed (a moved crate, a renamed
/// directory) and every scan passes vacuously.
const WALK_FLOOR_FILES: &[&str] = &[
    "rig-core/src/operation/completion.rs",
    "rig-core/src/providers/internal/wire.rs",
    "rig-core/src/providers/anthropic/streaming.rs",
    "rig-core/src/providers/openai/responses_api/streaming.rs",
    "rig-bedrock/src/streaming.rs",
];

fn walk_floor(sources: &[Source]) -> Result<(), String> {
    let missing: Vec<&&str> = WALK_FLOOR_FILES
        .iter()
        .filter(|suffix| !sources.iter().any(|source| source.path.ends_with(*suffix)))
        .collect();
    if missing.is_empty() {
        return Ok(());
    }
    Err(format!(
        "the source walk found nothing at {missing:?} — a collapsed walk \
         (moved crate, renamed directory) must fail loudly rather than let \
         every scan pass vacuously; walked {} files",
        sources.len()
    ))
}

/// Run every driver-adoption guard over the shipped sources under
/// `workspace/crates`, reporting each failing guard's message.
pub(crate) fn check(workspace: &Path) -> Result<(), String> {
    let sources = shipped_sources(workspace)?;
    walk_floor(&sources)?;
    let failures: Vec<String> = [
        every_triage_site_runs_on_the_single_policy_driver(&sources),
        streaming_modules_never_debug_print_wire_payloads_in_warn_logs(&sources),
        provider_streaming_modules_never_raw_parse_the_wire(&sources, SERDE_POLICY_ALLOWLIST),
        runtime_turns_come_from_folds(&sources),
        encoders_read_provider_items_only_through_replay(&sources),
    ]
    .into_iter()
    .filter_map(Result::err)
    .collect();
    if failures.is_empty() {
        Ok(())
    } else {
        Err(failures.join("\n\n"))
    }
}

/// No provider restates the driver's Unknown/Corrupt policy table.
fn every_triage_site_runs_on_the_single_policy_driver(sources: &[Source]) -> Result<(), String> {
    let mut violations = Vec::new();
    let mut policy_home_mentions = 0usize;
    for Source { path, shipped } in sources {
        if is_policy_home(path) {
            // The driver and classifiers do name the variants, so zero
            // mentions there means the marker or the mask broke.
            policy_home_mentions += shipped.matches("WireEvent::").count();
            continue;
        }
        for (index, line) in shipped.lines().enumerate() {
            if line.contains("WireEvent::Unknown") || line.contains("WireEvent::Corrupt") {
                violations.push(format!("{path}:{}: {}", index + 1, line.trim()));
            }
        }
    }
    if policy_home_mentions == 0 {
        return Err(
            "the policy homes no longer mention `WireEvent::` — the marker this \
             scan greps for cannot occur, so the scan is vacuous"
                .to_owned(),
        );
    }
    if !violations.is_empty() {
        return Err(format!(
            "Unknown/Corrupt triage restated outside the driver (driver.rs) and \
             classify layer (wire.rs) — route it through the driver's read \
             loop / `triage` instead:\n{}",
            violations.join("\n")
        ));
    }
    Ok(())
}

/// The syntactic markers of raw wire decoding. An untagged fallback variant
/// decides what happens to frames the classifier never saw, as
/// `#[serde(other)]` does.
const RAW_SERDE_MARKERS: &[&str] = &[
    "serde_json::from_str",
    "serde_json::from_slice",
    "serde_json::from_value",
    "#[serde(other)]",
    "#[serde(untagged)]",
];

/// Single-file providers whose streaming code sits in files the basename
/// pattern misses.
const SINGLE_FILE_STREAMING_MODULES: &[&str] =
    &["providers/copilot/mod.rs", "providers/chatgpt/mod.rs"];

/// Identifiers a file cannot mention without taking part in wire handling.
const WIRE_MACHINERY_MARKERS: &[&str] = &["WireEvent", "WireFrame", "triage("];

/// Whether the serde wall scans `path`: its basename says streaming or
/// websocket, it is in [`SINGLE_FILE_STREAMING_MODULES`], or its shipped
/// content names the wire machinery, so a new helper is scanned the moment
/// it touches a frame. [`ALLOWED_POLICY_HOMES`], `rig-agent` (consumer-side)
/// and `test_utils` are exempt.
fn is_serde_wall_target(path: &str, shipped: &str) -> bool {
    if path.contains("/rig-agent/") || path.contains("/test_utils/") {
        return false;
    }
    if is_named_policy_home(path) {
        return false;
    }
    if SINGLE_FILE_STREAMING_MODULES
        .iter()
        .any(|suffix| path.ends_with(suffix))
    {
        return true;
    }
    let name = path.rsplit('/').next().unwrap_or(path);
    if name.contains("streaming") || name.contains("websocket") {
        return true;
    }
    // A wire declaring the frame type its transport delivers names the type
    // without handling a frame.
    let machinery: String = shipped
        .lines()
        .filter(|line| !line.trim_start().starts_with("type Frame ="))
        .collect::<Vec<_>>()
        .join("\n");
    WIRE_MACHINERY_MARKERS
        .iter()
        .any(|marker| machinery.contains(marker))
}

/// One allowlist entry: `path suffix | line snippet | justification`.
struct AllowlistEntry {
    path_suffix: String,
    snippet: String,
    used: bool,
}

fn parse_allowlist(raw: &str) -> Result<Vec<AllowlistEntry>, String> {
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let mut fields = line.splitn(3, '|').map(str::trim);
            let path_suffix = fields.next().unwrap_or_default().to_string();
            let snippet = fields.next().unwrap_or_default().to_string();
            let justification = fields.next().unwrap_or_default();
            if path_suffix.is_empty() || snippet.is_empty() || justification.is_empty() {
                return Err(format!(
                    "malformed serde_policy_allowlist.txt entry (need `path | snippet | justification`): {line}"
                ));
            }
            Ok(AllowlistEntry {
                path_suffix,
                snippet,
                used: false,
            })
        })
        .collect()
}

/// Scans one shipped source for raw serde markers, marking the allowlist
/// entries that cover them used. Returns the uncovered violations.
fn scan_streaming_source(
    path: &str,
    shipped: &str,
    allowlist: &mut [AllowlistEntry],
) -> Vec<String> {
    let mut violations = Vec::new();
    for (index, line) in shipped.lines().enumerate() {
        // Comments may discuss the markers; only code counts.
        if line.trim_start().starts_with("//") {
            continue;
        }
        if !RAW_SERDE_MARKERS.iter().any(|marker| line.contains(marker)) {
            continue;
        }
        let mut covered = false;
        for entry in allowlist.iter_mut() {
            if path.ends_with(entry.path_suffix.as_str()) && line.contains(&entry.snippet) {
                entry.used = true;
                covered = true;
            }
        }
        if !covered {
            violations.push(format!("{path}:{}: {}", index + 1, line.trim()));
        }
    }
    violations
}

/// The whole macro invocation starting at `start` (the macro name), through
/// its matching close paren, with comments elided so a stray `)` or
/// capture-like prose neither truncates nor pollutes it. When the parens
/// never balance, the rest of the file is returned, so the scan fails closed.
fn macro_body(source: &str, start: usize) -> String {
    let rest = source.get(start..).unwrap_or_default();
    let Some((head, body)) = rest.find('(').and_then(|open| rest.split_at_checked(open)) else {
        return rest.to_owned();
    };
    let bytes = body.as_bytes();
    // Comment byte ranges within `body`, elided from the returned text.
    let mut comments: Vec<(usize, usize)> = Vec::new();
    let assemble = |end: usize, comments: &[(usize, usize)]| {
        let mut kept = head.to_owned();
        let mut cursor = 0usize;
        for &(from, to) in comments {
            kept.push_str(body.get(cursor..from.min(end)).unwrap_or_default());
            cursor = to.min(end);
        }
        kept.push_str(body.get(cursor..end).unwrap_or_default());
        kept
    };
    let mut depth = 0usize;
    let mut index = 0usize;
    while let Some(&byte) = bytes.get(index) {
        match byte {
            // String literals are skipped whole: a stray `"` or `)` inside
            // would desync the paren tracking and truncate the body.
            b'"' => {
                index += 1;
                while let Some(&inner) = bytes.get(index) {
                    match inner {
                        b'\\' => index += 1,
                        b'"' => break,
                        _ => {}
                    }
                    index += 1;
                }
            }
            b'r' if matches!(bytes.get(index + 1), Some(b'"' | b'#')) => {
                let mut hashes = 0usize;
                let mut probe = index + 1;
                while bytes.get(probe) == Some(&b'#') {
                    hashes += 1;
                    probe += 1;
                }
                if bytes.get(probe) == Some(&b'"') {
                    index = probe + 1;
                    while let Some(&inner) = bytes.get(index) {
                        if inner == b'"'
                            && bytes
                                .get(index + 1..index + 1 + hashes)
                                .is_some_and(|tail| tail.iter().all(|b| *b == b'#'))
                        {
                            index += hashes;
                            break;
                        }
                        index += 1;
                    }
                }
            }
            // A char literal (`'"'`, `'('`); a lifetime has no closing quote
            // in the next two bytes and falls through.
            b'\'' => {
                if bytes.get(index + 1) == Some(&b'\\') && bytes.get(index + 3) == Some(&b'\'') {
                    index += 3;
                } else if bytes.get(index + 2) == Some(&b'\'') {
                    index += 2;
                }
            }
            b'/' if bytes.get(index + 1) == Some(&b'/') => {
                let from = index;
                while let Some(&inner) = bytes.get(index) {
                    if inner == b'\n' {
                        break;
                    }
                    index += 1;
                }
                comments.push((from, index));
                continue;
            }
            b'/' if bytes.get(index + 1) == Some(&b'*') => {
                let from = index;
                let mut comment_depth = 1usize;
                index += 2;
                while let Some(&inner) = bytes.get(index) {
                    if inner == b'/' && bytes.get(index + 1) == Some(&b'*') {
                        comment_depth += 1;
                        index += 1;
                    } else if inner == b'*' && bytes.get(index + 1) == Some(&b'/') {
                        comment_depth = comment_depth.saturating_sub(1);
                        index += 1;
                        if comment_depth == 0 {
                            index += 1;
                            break;
                        }
                    }
                    index += 1;
                }
                comments.push((from, index));
                continue;
            }
            b'(' => depth += 1,
            b')' => {
                depth = depth.saturating_sub(1);
                if depth == 0 {
                    return assemble(index + 1, &comments);
                }
            }
            _ => {}
        }
        index += 1;
    }
    assemble(body.len(), &comments)
}

/// Whether a warn-macro body Debug-captures a value: a `?ident` field
/// capture (`warn!(?frame)`, `warn!(payload = ?frame)`) or any Debug format
/// spec (`{:?}`, `{frame:#?}`, `{frame:x?}`). A capture through
/// `std::mem::discriminant` prints only the variant and is allowed.
fn body_debug_captures(body: &str) -> bool {
    // Every Debug format spec ends `?}`; an escaped `{{x:?}}` can only
    // over-flag, which fails closed.
    if body.contains("?}") {
        return true;
    }
    let bytes = body.as_bytes();
    for (index, _) in body.match_indices('?') {
        // A capture sigil is `?` directly before an identifier...
        if !bytes
            .get(index + 1)
            .is_some_and(|next| next.is_ascii_alphabetic() || *next == b'_')
        {
            continue;
        }
        // ...after `(`, `,` or `=`, never a question mark in prose.
        let preceding = body
            .get(..index)
            .unwrap_or_default()
            .chars()
            .rev()
            .find(|ch| !ch.is_whitespace());
        if !matches!(preceding, Some('(' | ',' | '=')) {
            continue;
        }
        // The capture runs to the next top-level `,` or the closing `)`.
        let capture = body.get(index..).unwrap_or_default();
        let mut depth = 0usize;
        let mut end = capture.len();
        for (offset, ch) in capture.char_indices() {
            match ch {
                '(' => depth += 1,
                ')' | ',' if depth == 0 => {
                    end = offset;
                    break;
                }
                ')' => depth -= 1,
                _ => {}
            }
        }
        if capture
            .get(..end)
            .is_some_and(|capture| capture.contains("std::mem::discriminant"))
        {
            continue;
        }
        return true;
    }
    false
}

/// Shipped streaming code never Debug-prints a wire payload into a WARN log.
/// Unmodeled frames can carry model output, and the one redaction policy
/// (kind and byte size only) lives in `driver::warn_unmodeled`. The scan
/// matters for payload types rig does not own (SDK frames, protos), whose
/// derived `Debug` prints wire content.
fn streaming_modules_never_debug_print_wire_payloads_in_warn_logs(
    sources: &[Source],
) -> Result<(), String> {
    let mut violations = Vec::new();
    let mut scanned_targets = 0usize;
    for Source { path, shipped } in sources {
        if !is_serde_wall_target(path, shipped) {
            continue;
        }
        scanned_targets += 1;
        // A `use` of `tracing` that renames the macro hides its call sites:
        // an `… as` whose statement (back to the previous `;`) is such a use.
        for aliased in ["warn as", "event as"] {
            for (at, _) in shipped.match_indices(aliased) {
                let before = shipped.get(..at).unwrap_or_default();
                let statement_start = before.rfind(';').map_or(0, |semi| semi + 1);
                let statement = before.get(statement_start..).unwrap_or_default();
                if statement.contains("use") && statement.contains("tracing") {
                    let line_number = before.matches('\n').count() + 1;
                    violations.push(format!(
                        "{path}:{line_number}: a `use … {aliased} …` alias hides WARN call sites from this scan",
                    ));
                }
            }
        }
        let warn_sites = shipped
            .match_indices("warn!")
            .map(|(start, _)| (start, false));
        let event_sites = shipped
            .match_indices("event!")
            .map(|(start, _)| (start, true));
        for (start, is_event) in warn_sites.chain(event_sites) {
            let before = shipped.get(..start).unwrap_or_default();
            // A mid-identifier match (`it_would_warn!`) is not tracing's.
            if before
                .chars()
                .next_back()
                .is_some_and(|ch| ch.is_ascii_alphanumeric() || ch == '_')
            {
                continue;
            }
            let line_number = before.matches('\n').count() + 1;
            let line = before.rsplit('\n').next().unwrap_or(before);
            if line.trim_start().starts_with("//") {
                continue;
            }
            let body = macro_body(shipped, start);
            // Only `event!`'s WARN spelling is in scope: lower levels are
            // compiled out of release telemetry and ERROR is reviewed by hand.
            if is_event && !body.contains("Level::WARN") {
                continue;
            }
            if body_debug_captures(&body) {
                violations.push(format!(
                    "{path}:{line_number}: {}",
                    body.lines().next().unwrap_or(&body).trim()
                ));
            }
        }
    }
    if scanned_targets <= 5 {
        return Err(format!(
            "the payload-warn scan scoped almost nothing ({scanned_targets} files) — vacuous"
        ));
    }
    if !violations.is_empty() {
        return Err(format!(
            "a WARN log Debug-captures a wire payload — route it through \
             `driver::warn_unmodeled` (kind + byte size only):\n{}",
            violations.join("\n")
        ));
    }
    Ok(())
}

/// Streaming modules the serde wall must scan, or its scoping collapsed. A
/// provider's classify layer is the triage guard's policy home and still a
/// serde wall target.
const SERDE_WALL_FLOOR: &[&str] = &[
    "rig-core/src/providers/anthropic/streaming.rs",
    "rig-core/src/providers/openai/responses_api/streaming.rs",
    "rig-bedrock/src/streaming.rs",
    "rig-core/src/providers/openai/wire/chat.rs",
];

/// Shipped provider streaming code never raw-parses the wire: decoding goes
/// through the `wire.rs` classifiers, and every exception is in `allowlist`
/// with a justification. An entry that covers nothing is stale.
fn provider_streaming_modules_never_raw_parse_the_wire(
    sources: &[Source],
    allowlist: &str,
) -> Result<(), String> {
    let mut allowlist = parse_allowlist(allowlist)?;
    let mut violations = Vec::new();
    let mut scanned_targets = Vec::new();
    for Source { path, shipped } in sources {
        if !is_serde_wall_target(path, shipped) {
            continue;
        }
        scanned_targets.push(path.as_str());
        violations.extend(scan_streaming_source(path, shipped, &mut allowlist));
    }

    for suffix in SERDE_WALL_FLOOR {
        if !scanned_targets.iter().any(|path| path.ends_with(suffix)) {
            return Err(format!(
                "the serde wall no longer scans {suffix} — its scoping collapsed; \
                 scanned {} targets",
                scanned_targets.len()
            ));
        }
    }

    if !violations.is_empty() {
        return Err(format!(
            "raw serde parsing in a provider streaming module — route wire decoding \
             through the `wire.rs` classify layer, or (for a genuine non-triage use) \
             add a `path | snippet | justification` entry to \
             {SERDE_POLICY_ALLOWLIST_PATH}:\n{}",
            violations.join("\n")
        ));
    }

    let stale: Vec<&str> = allowlist
        .iter()
        .filter(|entry| !entry.used)
        .map(|entry| entry.snippet.as_str())
        .collect();
    if !stale.is_empty() {
        return Err(format!(
            "stale serde_policy_allowlist.txt entries (the code they covered is gone \
             — delete them): {stale:?}"
        ));
    }
    Ok(())
}

/// Runtimes never build an assistant turn by hand: a turn that enters
/// history comes from a fold (`CompletionResponse::message`, `head`,
/// `continued`), so it keeps its origin and provider items.
fn runtime_turns_come_from_folds(sources: &[Source]) -> Result<(), String> {
    if !sources
        .iter()
        .any(|source| source.path.ends_with("rig-agent/src/run/streamed.rs"))
    {
        return Err("the walk covers the runtime".to_owned());
    }
    let offenders: Vec<&str> = sources
        .iter()
        .filter(|source| {
            source.path.contains("/rig-agent/src/")
                && mask_literals_and_comments(&source.shipped).contains("AssistantMessage::new(")
        })
        .map(|source| source.path.as_str())
        .collect();
    if !offenders.is_empty() {
        return Err(format!(
            "runtime code builds assistant turns by hand: {offenders:?}"
        ));
    }
    Ok(())
}

/// Provider files that still read provider items directly. Each family
/// moves its encoder onto `AssistantContent::replay` and deletes its entry;
/// the list only shrinks.
const PENDING_REPLAY_ADOPTION: &[&str] = &[];

/// The encoder roots whose provider-item reads go through `replay`.
const ENCODER_ROOTS: &[&str] = &[
    "/rig-core/src/providers/",
    "/rig-bedrock/src/",
    "/rig-vertexai/src/",
    "/rig-gemini-grpc/src/",
    "/rig-candle/src/",
];

/// Encoders read provider items only through `AssistantContent::replay`, so
/// the core alone decides what a current, edited or foreign item becomes.
fn encoders_read_provider_items_only_through_replay(sources: &[Source]) -> Result<(), String> {
    let mut offenders = Vec::new();
    for Source { path, shipped } in sources {
        if !ENCODER_ROOTS.iter().any(|root| path.contains(root)) {
            continue;
        }
        let masked = mask_literals_and_comments(shipped);
        let reads = masked.contains("native_item(")
            || masked.contains("stale_item(")
            || masked.match_indices(".native").any(|(at, _)| {
                let rest = masked.get(at + ".native".len()..).unwrap_or_default();
                !rest.starts_with('_')
                    && !rest.trim_start().starts_with('=')
                    && !rest.starts_with("::")
            });
        if reads {
            offenders.push(path.as_str());
        }
    }
    let unlisted: Vec<&&str> = offenders
        .iter()
        .filter(|path| {
            !PENDING_REPLAY_ADOPTION
                .iter()
                .any(|entry| path.ends_with(entry))
        })
        .collect();
    if !unlisted.is_empty() {
        return Err(format!(
            "encoders read provider items directly instead of through `replay`: {unlisted:?}"
        ));
    }
    let stale: Vec<&&str> = PENDING_REPLAY_ADOPTION
        .iter()
        .filter(|entry| !offenders.iter().any(|path| path.ends_with(*entry)))
        .collect();
    if !stale.is_empty() {
        return Err(format!(
            "these files no longer read provider items; delete their entries: {stale:?}"
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests;
