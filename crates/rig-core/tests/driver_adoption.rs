//! Structural guard: every streaming triage site runs on the single-policy
//! driver.
//!
//! The `run_wire_stream`/`run_wire_buffered` driver (and its factored
//! `triage_frame` helper) in `providers/internal/adapter.rs` is the ONLY place
//! allowed to decide what happens to `WireEvent::Unknown` / `WireEvent::Corrupt`
//! frames. The websocket divergence fixed on this branch is the standing proof
//! that hand-copied triage tables drift; this test makes reintroducing one a CI
//! failure.
//!
//! Mechanism: non-test provider source may *classify* (produce a `WireEvent`)
//! but never *triage* it — and triage requires matching the `Unknown`/`Corrupt`
//! variants. So any mention of `WireEvent::Unknown` or `WireEvent::Corrupt`
//! outside the driver (`adapter.rs`), the classify layer (`wire.rs`), and test
//! code is a restated policy table.

#![allow(clippy::expect_used)]

use std::path::PathBuf;

/// Full path suffixes of the ONLY files where the policy table and
/// classifiers legitimately name the triage variants. Matching by full path
/// (not basename) means a future `rig-bedrock/src/streaming/adapter.rs` that
/// hand-rolls a `WireEvent` policy table is scanned like any other file
/// instead of inheriting the core driver's exemption.
const ALLOWED_POLICY_HOMES: &[&str] = &[
    "rig-core/src/providers/internal/adapter.rs",
    "rig-core/src/providers/internal/wire.rs",
];

/// Whether `path` is one of the two files allowed to state triage policy.
fn is_policy_home(path: &std::path::Path) -> bool {
    let unix_path = path.to_string_lossy().replace('\\', "/");
    ALLOWED_POLICY_HOMES
        .iter()
        .any(|suffix| unix_path.ends_with(suffix))
}

/// Directories that hold test harness code rather than shipped policy.
const SKIPPED_DIRS: &[&str] = &["tests", "test_utils", "fixtures", "target"];

/// Blanks the *contents* of comments and string/char literals, replacing every
/// masked character with a space and preserving newlines (so line numbering and
/// line count are unchanged).
///
/// Structural decisions — "is this line an attribute?", "where does this item
/// end?" — run on the mask, never on the raw text, so a `{` inside a JSON
/// fixture string, or a `#[cfg(test)]` quoted inside a `/* … */` block, cannot
/// steer the scan. Raw strings (`r#"…"#`) are handled because inline test
/// modules are full of them. Marker matching still runs on the ORIGINAL text,
/// so violation messages quote real source.
fn mask_literals_and_comments(source: &str) -> String {
    /// What the scanner is currently inside.
    enum State {
        Code,
        LineComment,
        /// Block comments nest in Rust; the payload is the open depth.
        BlockComment(usize),
        Str,
        /// `r##"…"##` — the payload is the hash count that closes it.
        RawStr(usize),
        Char,
    }

    let mut masked = String::with_capacity(source.len());
    let mut state = State::Code;
    let mut chars = source.chars().peekable();
    // Set while consuming a `\`-escaped pair inside a string/char literal.
    let mut escaped = false;

    // Pushes `ch` verbatim, or a space if it is masked; newlines always pass
    // through so the mask stays line-aligned with the source.
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
                    // `r"…"` / `r#"…"#`, but also plain identifiers starting
                    // with `r` followed by `#` are impossible, so counting the
                    // hashes and requiring a `"` is unambiguous.
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
                    // A lifetime (`'a`) is code; a char literal (`'x'`, `'\n'`)
                    // is masked. Only the literal forms have a closing quote
                    // within the next two characters.
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
/// The item ends when its brace depth returns to zero (`mod tests { … }`,
/// `impl`, `fn`), or — for a brace-less item (`#[cfg(test)] use foo;`) — at the
/// first top-level `;`. Extra attributes between the `#[cfg(test)]` and the
/// item itself are consumed on the way, since neither carries braces or a
/// top-level semicolon.
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

/// Returns the shipped portion of a source file: the whole file with each
/// `#[cfg(test)]`-gated ITEM blanked out (its lines replaced by empty lines, so
/// reported line numbers still match the file on disk).
///
/// Item-scoped, not truncate-at-first-marker, for two reasons found by review
/// (#2258):
///
/// 1. **Shipped code after an inline test module is scanned.** A `#[cfg(test)]`
///    *helper* midway through a file (live example:
///    `providers/openrouter/completion.rs`, a gated `final_request_body` at
///    ~line 1370 followed by 2600 more lines of shipped code) used to hide
///    everything below it from both guards.
/// 2. **Content scoping sees the whole file.** [`is_serde_wall_target`] opts a
///    file in when its shipped text names the wire machinery; under truncation,
///    machinery named only *after* a gated item could not opt its file in.
///
/// Attribute position and item extent are decided on
/// [`mask_literals_and_comments`]'s output, so a doc comment or trailing
/// comment merely *mentioning* `#[cfg(test)]`, an attribute quoted inside a
/// `/* … */` block, and a stray brace inside a JSON fixture string are all
/// inert (self-tests `shipped_portion_ignores_cfg_test_mentions_in_comments`
/// and `shipped_portion_is_item_scoped`).
fn shipped_portion(source: &str) -> String {
    let masked = mask_literals_and_comments(source);
    let lines: Vec<&str> = source.split_inclusive('\n').collect();
    let masked_lines: Vec<&str> = masked.split_inclusive('\n').collect();
    debug_assert_eq!(lines.len(), masked_lines.len());

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

/// Walks every `.rs` file under the workspace `crates/` directory (skipping
/// [`SKIPPED_DIRS`]) and calls `visit(path, shipped_source)`, where
/// `shipped_source` is the file content with `#[cfg(test)]`-gated items blanked
/// by [`shipped_portion`].
fn for_each_shipped_source(mut visit: impl FnMut(&std::path::Path, &str)) {
    // rig-core/tests -> workspace crates/ directory, so the guards also cover
    // the out-of-core adapter crates (bedrock, candle, gemini-grpc).
    let crates_dir: PathBuf = [env!("CARGO_MANIFEST_DIR"), ".."].iter().collect();

    let mut pending = vec![crates_dir];
    while let Some(dir) = pending.pop() {
        let entries = std::fs::read_dir(&dir).expect("workspace directory should be readable");
        for entry in entries {
            let entry = entry.expect("directory entry should be readable");
            let path = entry.path();
            let name = entry.file_name();
            let name = name.to_string_lossy().into_owned();

            if path.is_dir() {
                if !SKIPPED_DIRS.contains(&name.as_str()) {
                    pending.push(path);
                }
                continue;
            }

            if path.extension().is_none_or(|ext| ext != "rs") {
                continue;
            }

            let source = std::fs::read_to_string(&path).expect("source file should be readable");
            visit(&path, &shipped_portion(&source));
        }
    }
}

/// Files the source walk must find, or it has collapsed (a moved crate, a
/// renamed directory) and every scan built on it passes vacuously. The floor
/// is asserted by both guards below — pydantic-ai's
/// `test_public_interface_contracts` rule: a walk that finds nothing to
/// check must fail loudly, never pass green.
const WALK_FLOOR_FILES: &[&str] = &[
    "rig-core/src/providers/internal/adapter.rs",
    "rig-core/src/providers/internal/wire.rs",
    "rig-core/src/providers/anthropic/streaming.rs",
    "rig-core/src/providers/openai/responses_api/streaming.rs",
    "rig-bedrock/src/streaming.rs",
    "rig-gemini-grpc/src/streaming.rs",
];

/// Assert the walk saw every floor file; `walked` holds normalized paths.
fn assert_walk_floor(walked: &[String]) {
    let missing: Vec<&&str> = WALK_FLOOR_FILES
        .iter()
        .filter(|suffix| !walked.iter().any(|path| path.ends_with(*suffix)))
        .collect();
    assert!(
        missing.is_empty(),
        "the source walk found nothing at {missing:?} — a collapsed walk \
         (moved crate, renamed directory) must fail loudly rather than let \
         every scan pass vacuously; walked {} files",
        walked.len()
    );
}

/// No provider restates the driver's Unknown/Corrupt policy table.
#[test]
fn every_triage_site_runs_on_the_single_policy_driver() {
    let mut violations = Vec::new();
    let mut walked = Vec::new();
    let mut policy_home_mentions = 0usize;

    for_each_shipped_source(|path, shipped| {
        walked.push(path.to_string_lossy().replace('\\', "/"));
        if is_policy_home(path) {
            // The floor for the scan itself: the driver and classifier DO
            // name the variants, so finding zero mentions there means the
            // marker (or the mask) broke and the provider scan below is
            // checking for a string that can no longer occur.
            policy_home_mentions += shipped.matches("WireEvent::").count();
            return;
        }

        for (index, line) in shipped.lines().enumerate() {
            if line.contains("WireEvent::Unknown") || line.contains("WireEvent::Corrupt") {
                violations.push(format!("{}:{}: {}", path.display(), index + 1, line.trim()));
            }
        }
    });

    assert_walk_floor(&walked);
    assert!(
        policy_home_mentions > 0,
        "the policy homes no longer mention `WireEvent::` — the marker this \
         scan greps for cannot occur, so the scan is vacuous"
    );
    assert!(
        violations.is_empty(),
        "Unknown/Corrupt triage restated outside the driver (adapter.rs) and \
         classify layer (wire.rs) — route it through run_wire_stream / \
         run_wire_buffered / triage_frame instead:\n{}",
        violations.join("\n")
    );
}

// ---------------------------------------------------------------------------
// Guard 2: the serde policy wall.
//
// Raw serde parsing inside a provider streaming module is how policy tables
// escape the classify layer: a hand-rolled `from_str` (or a `#[serde(other)]`
// catch-all) silently decides what happens to frames the classifier never saw
// — exactly the websocket divergence this branch fixed. Shipped code in a
// streaming module must delegate wire decoding to `wire.rs` classifiers; the
// few legitimate exceptions (documented envelope pre-dispatch, content
// assembly, classifier-internal helpers) live in the committed allowlist,
// each with a one-line justification.
// ---------------------------------------------------------------------------

/// The syntactic markers of raw wire decoding.
const RAW_SERDE_MARKERS: &[&str] = &[
    "serde_json::from_str",
    "serde_json::from_slice",
    "serde_json::from_value",
    "#[serde(other)]",
    // An untagged fallback variant is the same policy hazard as
    // `#[serde(other)]`: it decides what happens to frames the classifier
    // never saw (and on an internally-tagged enum it also swallows a known
    // tag with a defective payload). Legitimate uses are allowlisted.
    "#[serde(untagged)]",
];

/// A file is in scope for the serde policy wall when ANY of:
///
/// 1. its basename says it is streaming code (`streaming`/`websocket`);
/// 2. it is a single-file provider named in
///    [`SINGLE_FILE_STREAMING_MODULES`] (those keep streaming code in files
///    the basename pattern misses — extend the list when a new provider
///    adopts that layout);
/// 3. **fail-closed content scoping**: its shipped content references the
///    wire/classify/adapter machinery ([`WIRE_MACHINERY_MARKERS`]). This is
///    what closes the "compat helper" hole: a helper like
///    `internal/openai_chat_completions_compatible.rs` (or any future
///    `compat.rs`/`sse.rs`) has a basename the pattern misses, but it cannot
///    participate in wire handling without naming the machinery, so touching
///    the machinery is what opts a file into the scan. A helper that decodes
///    the wire WITHOUT touching the machinery is a driver-adoption problem
///    (guard 1 / review), not a scoping problem — content scoping was chosen
///    over an ever-growing explicit path list precisely so future helpers are
///    scanned the moment they are written, with no list to forget to update.
///
/// The classify layer (`wire.rs`) and driver (`adapter.rs`) are exempt by
/// FULL path suffix ([`ALLOWED_POLICY_HOMES`]), not by basename, so a foreign
/// `adapter.rs` elsewhere in the workspace is scanned like any other file.
/// `rig-agent`'s streaming modules are consumer-side (no wire decoding) and
/// are excluded by path, as is `test_utils`.
///
/// Both this scan and the driver-adoption scan are textual tripwires against
/// drift, not security boundaries: an aliased import could evade them, and
/// that aliasing would itself be reviewable. AST-grade enforcement is
/// deliberately not attempted.
const SINGLE_FILE_STREAMING_MODULES: &[&str] = &[
    "providers/ollama.rs",
    "providers/copilot/mod.rs",
    "providers/chatgpt/mod.rs",
];

/// Identifiers a file cannot mention without participating in wire handling.
const WIRE_MACHINERY_MARKERS: &[&str] = &[
    "WireEvent",
    "WireAdapter",
    "WireFrame",
    "run_wire_stream",
    "run_wire_buffered",
    "triage_frame",
];

fn is_serde_wall_target(path: &std::path::Path, shipped: &str) -> bool {
    let unix_path = path.to_string_lossy().replace('\\', "/");
    if unix_path.contains("/rig-agent/") || unix_path.contains("/test_utils/") {
        return false;
    }
    if is_policy_home(path) {
        return false;
    }
    if SINGLE_FILE_STREAMING_MODULES
        .iter()
        .any(|suffix| unix_path.ends_with(suffix))
    {
        return true;
    }
    if path
        .file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .is_some_and(|name| name.contains("streaming") || name.contains("websocket"))
    {
        return true;
    }
    WIRE_MACHINERY_MARKERS
        .iter()
        .any(|marker| shipped.contains(marker))
}

/// One allowlist entry: `path suffix | line snippet | justification`.
struct AllowlistEntry {
    path_suffix: String,
    snippet: String,
    used: bool,
}

fn parse_allowlist(raw: &str) -> Vec<AllowlistEntry> {
    raw.lines()
        .map(str::trim)
        .filter(|line| !line.is_empty() && !line.starts_with('#'))
        .map(|line| {
            let mut fields = line.splitn(3, '|').map(str::trim);
            let path_suffix = fields.next().unwrap_or_default().to_string();
            let snippet = fields.next().unwrap_or_default().to_string();
            let justification = fields.next().unwrap_or_default();
            assert!(
                !path_suffix.is_empty() && !snippet.is_empty() && !justification.is_empty(),
                "malformed serde_policy_allowlist.txt entry (need `path | snippet | justification`): {line}"
            );
            AllowlistEntry {
                path_suffix,
                snippet,
                used: false,
            }
        })
        .collect()
}

/// Scans one shipped source for raw serde markers, consuming allowlist
/// entries that cover them. Returns the uncovered violations.
fn scan_streaming_source(
    path_label: &str,
    shipped: &str,
    allowlist: &mut [AllowlistEntry],
) -> Vec<String> {
    let mut violations = Vec::new();
    for (index, line) in shipped.lines().enumerate() {
        // Comments may discuss the markers (e.g. "there is no #[serde(other)]
        // fallback"); only code counts.
        if line.trim_start().starts_with("//") {
            continue;
        }
        if !RAW_SERDE_MARKERS.iter().any(|marker| line.contains(marker)) {
            continue;
        }
        let mut covered = false;
        for entry in allowlist.iter_mut() {
            if path_label.ends_with(entry.path_suffix.as_str()) && line.contains(&entry.snippet) {
                entry.used = true;
                covered = true;
            }
        }
        if !covered {
            violations.push(format!("{}:{}: {}", path_label, index + 1, line.trim()));
        }
    }
    violations
}

/// The whole `warn!(...)` invocation starting at `start` (the index of the
/// macro name), through its matching close paren — rustfmt-wrapped
/// multi-line bodies included, comments elided (a stray `)` or capture-like
/// text inside one must neither truncate nor pollute the scanned body).
/// Falls back to the rest of the file when the parens never balance (fail
/// closed: an unparseable body is scanned whole).
fn macro_body(source: &str, start: usize) -> String {
    let Some(open) = source[start..].find('(') else {
        return source[start..].to_owned();
    };
    let body = &source[start + open..];
    let bytes = body.as_bytes();
    // Comment byte ranges within `body`, elided from the returned text.
    let mut comments: Vec<(usize, usize)> = Vec::new();
    let assemble = |end: usize, comments: &[(usize, usize)]| {
        let mut kept = source[start..start + open].to_owned();
        let mut cursor = 0usize;
        for &(from, to) in comments {
            kept.push_str(&body[cursor..from.min(end)]);
            cursor = to.min(end);
        }
        kept.push_str(&body[cursor..end]);
        kept
    };
    let mut depth = 0usize;
    let mut index = 0usize;
    while let Some(&byte) = bytes.get(index) {
        match byte {
            // Ordinary string literal: skip to its closing quote, honoring
            // escapes. A stray `"` inside would otherwise desync the paren
            // tracking and truncate the body (fail OPEN), so string forms
            // are handled explicitly.
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
            // Raw string literal `r"…"` / `r#"…"#`: no escapes; runs to a
            // `"` followed by the same number of `#`s.
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
            // Char literal (`'"'`, `'('`, `'\''`): a quote or paren inside
            // would desync the tracking. Lifetimes (`'a`) have no closing
            // quote in the next two bytes and fall through untouched.
            b'\'' => {
                if bytes.get(index + 1) == Some(&b'\\') && bytes.get(index + 3) == Some(&b'\'') {
                    index += 3;
                } else if bytes.get(index + 2) == Some(&b'\'') {
                    index += 2;
                }
            }
            // Line comment: an unbalanced `)` or `"` inside would desync
            // the tracking (a stray `)` truncates the body — fail OPEN),
            // and comment prose can imitate captures — so comments are
            // skipped AND elided from the returned body.
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
            // Block comment, nesting honored (Rust block comments nest).
            b'/' if bytes.get(index + 1) == Some(&b'*') => {
                let from = index;
                let mut comment_depth = 1usize;
                index += 2;
                while let Some(&inner) = bytes.get(index) {
                    if inner == b'/' && bytes.get(index + 1) == Some(&b'*') {
                        comment_depth += 1;
                        index += 1;
                    } else if inner == b'*' && bytes.get(index + 1) == Some(&b'/') {
                        comment_depth -= 1;
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
                depth -= 1;
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
/// capture (positional `warn!(?frame)`, named `warn!(payload = ?frame)`),
/// a `{:?}`/`{:#?}` Debug placeholder, or a Rust-2021 inline format capture
/// (`{frame:?}` / `{frame:#?}`) in the format string. A capture whose
/// expression goes through `std::mem::discriminant` is structural by
/// construction (the variant tag Debug-prints as a kind, never the payload)
/// and is not a violation.
fn body_debug_captures(body: &str) -> bool {
    // Every Debug format spec ends `?}` — `{:?}`, `{frame:?}`, `{:#?}`,
    // and the less common specs (`{frame:x?}`, `{frame:>10?}`,
    // `{frame:.3?}`) alike — so the sigil-agnostic suffix is the check
    // (an escaped-brace literal like `{{x:?}}` can only over-flag, which
    // fails closed).
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
        // ...introduced by `(`, `,` or `=` (skipping whitespace backwards) —
        // never a question mark inside prose.
        let preceding = body[..index].chars().rev().find(|ch| !ch.is_whitespace());
        if !matches!(preceding, Some('(' | ',' | '=')) {
            continue;
        }
        // The capture expression runs to the next top-level `,` or the
        // closing `)` — nested call parens (`.as_ref().map(...)`) included.
        let mut depth = 0usize;
        let mut end = body.len();
        for (offset, ch) in body[index..].char_indices() {
            match ch {
                '(' => depth += 1,
                ')' if depth == 0 => {
                    end = index + offset;
                    break;
                }
                ')' => depth -= 1,
                ',' if depth == 0 => {
                    end = index + offset;
                    break;
                }
                _ => {}
            }
        }
        if body[index..end].contains("std::mem::discriminant") {
            continue;
        }
        return true;
    }
    false
}

/// Shipped streaming code never Debug-prints a wire payload into a WARN log:
/// unmodeled frames and parts can carry model output, and the one redaction
/// policy (kind + byte size only) lives in `adapter::warn_unmodeled`. The
/// scan covers the whole macro invocation — multi-line bodies, positional
/// (`warn!(?frame)`) and named (`warn!(payload = ?frame)`) captures, and
/// `{:?}` format-string Debug prints — so no spelling of a direct payload
/// capture bypasses the helper without failing CI.
///
/// Scope note: the JSON passthrough channel is now type-walled —
/// `streaming::UnknownPayload` Debug-redacts itself, so `warn!(?value)` on
/// that channel leaks nothing by construction. This scan's remaining
/// load-bearing scope is payloads rig does not own the type of: SDK frame
/// enums and typed events (bedrock's Converse types, gemini-grpc's protos)
/// whose derived `Debug` prints wire content. The type system cannot
/// withhold `Debug` on foreign types, so the scan stays for them.
#[test]
fn streaming_modules_never_debug_print_wire_payloads_in_warn_logs() {
    let mut violations = Vec::new();
    let mut walked = Vec::new();
    let mut scanned_targets = 0usize;
    for_each_shipped_source(|path, shipped| {
        walked.push(path.to_string_lossy().replace('\\', "/"));
        if !is_serde_wall_target(path, shipped) {
            return;
        }
        scanned_targets += 1;
        // Renaming the macro would move its call sites out of the scan's
        // sight; the alias itself is the violation. The path-qualified
        // spelling (`use tracing::warn as w`) and the brace-grouped one
        // (`use tracing::{warn as w}`, possibly multi-line) both count: an
        // `… as` occurrence whose enclosing statement (back to the previous
        // `;`) is a `use` of `tracing` is an alias.
        for aliased in ["warn as", "event as"] {
            for (at, _) in shipped.match_indices(aliased) {
                let statement_start = shipped[..at].rfind(';').map_or(0, |semi| semi + 1);
                let statement = &shipped[statement_start..at];
                if statement.contains("use") && statement.contains("tracing") {
                    let line_number = shipped[..at].matches('\n').count() + 1;
                    violations.push(format!(
                        "{}:{}: a `use … {aliased} …` alias hides WARN call sites from this scan",
                        path.display(),
                        line_number,
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
            // Mid-identifier matches (e.g. a `it_would_warn!` test helper)
            // are not the tracing macros.
            if shipped[..start]
                .chars()
                .next_back()
                .is_some_and(|ch| ch.is_ascii_alphanumeric() || ch == '_')
            {
                continue;
            }
            let line_number = shipped[..start].matches('\n').count() + 1;
            let line = shipped[..start]
                .rfind('\n')
                .map_or(&shipped[..start], |at| &shipped[at + 1..start]);
            if line.trim_start().starts_with("//") {
                continue;
            }
            let body = macro_body(shipped, start);
            // `event!` is level-parameterized; only its WARN spelling is in
            // scope (lower levels are compiled out of release telemetry and
            // ERROR sites are the panic path, reviewed by hand).
            if is_event && !body.contains("Level::WARN") {
                continue;
            }
            if body_debug_captures(&body) {
                violations.push(format!(
                    "{}:{}: {}",
                    path.display(),
                    line_number,
                    body.lines().next().unwrap_or(&body).trim()
                ));
            }
        }
    });
    assert_walk_floor(&walked);
    assert!(
        scanned_targets > 5,
        "the payload-warn scan scoped almost nothing ({scanned_targets} files) — vacuous"
    );
    assert!(
        violations.is_empty(),
        "a WARN log Debug-captures a wire payload — route it through \
         `adapter::warn_unmodeled` (kind + byte size only):\n{}",
        violations.join("\n")
    );
}

#[test]
fn the_warn_scan_catches_every_capture_spelling() {
    // Positional, named, format-string, and rustfmt-wrapped multi-line
    // captures are all flagged...
    for leaking in [
        "tracing::warn!(?frame, \"skipping\");",
        "tracing::warn!(payload = ?frame, \"skipping\");",
        "tracing::warn!(\"bad frame: {:?}\", frame);",
        "tracing::warn!(\n    ?frame,\n    \"skipping\"\n);",
        "tracing::warn!(\n    payload =\n        ?frame,\n    \"skipping\"\n);",
        // Rust-2021 inline format captures Debug-print just as loudly.
        "tracing::warn!(\"bad frame: {frame:?}\");",
        "tracing::warn!(\"bad frame: {frame:#?}\");",
        // Exotic Debug specs (hex, width, precision) still Debug-print
        // the payload.
        "tracing::warn!(\"bad frame: {frame:x?}\");",
        "tracing::warn!(\"bad frame: {frame:X?}\");",
        "tracing::warn!(\"bad frame: {frame:>10?}\");",
        "tracing::warn!(\"bad frame: {frame:.3?}\");",
        // A stray `)` in a comment must not truncate the scanned body
        // ahead of the capture.
        "tracing::warn!(\n    // see step 3)\n    ?frame,\n    \"skipping\"\n);",
        "tracing::warn!(\n    /* note (a) */\n    ?frame,\n    \"skipping\"\n);",
        // `event!` at WARN level is the same macro in a longer coat.
        "tracing::event!(tracing::Level::WARN, ?frame, \"skipping\");",
    ] {
        let start = leaking
            .find("warn!")
            .or_else(|| leaking.find("event!"))
            .expect("fixture contains a warn-level macro");
        assert!(
            body_debug_captures(&macro_body(leaking, start)),
            "must flag: {leaking}"
        );
    }
    // ...while structural logging, discriminant captures, and prose
    // question marks are not.
    for clean in [
        "tracing::warn!(kind, payload_bytes = size, \"skipping unmodeled wire payload\");",
        "tracing::warn!(step_index, \"arguments_delta for an unopened step?\");",
        "tracing::warn!(\n    kind,\n    payload_bytes = bytes,\n    \"skipping\"\n);",
        "tracing::warn!(\n    delta = ?std::mem::discriminant(&unknown),\n    \"skipping\"\n);",
        // Display-formatted inline captures and string forms that could
        // desync a naive parser stay clean.
        "tracing::warn!(\"dropped {count} frames\");",
        "tracing::warn!(r#\"marker \"quoted\" text\"#, count);",
    ] {
        assert!(
            !body_debug_captures(&macro_body(
                clean,
                clean.find("warn!").expect("fixture contains warn!")
            )),
            "must not flag: {clean}"
        );
    }
}

/// Shipped provider streaming code never raw-parses the wire: decoding goes
/// through the `wire.rs` classifiers, and every exception is allowlisted with
/// a justification in `serde_policy_allowlist.txt`.
#[test]
fn provider_streaming_modules_never_raw_parse_the_wire() {
    let allowlist_path: PathBuf = [
        env!("CARGO_MANIFEST_DIR"),
        "tests",
        "serde_policy_allowlist.txt",
    ]
    .iter()
    .collect();
    let raw = std::fs::read_to_string(&allowlist_path).expect("allowlist file should be readable");
    let mut allowlist = parse_allowlist(&raw);

    let mut violations = Vec::new();
    let mut walked = Vec::new();
    let mut scanned_targets = Vec::new();
    for_each_shipped_source(|path, shipped| {
        walked.push(path.to_string_lossy().replace('\\', "/"));
        if !is_serde_wall_target(path, shipped) {
            return;
        }
        let label = path.to_string_lossy().replace('\\', "/");
        scanned_targets.push(label.clone());
        violations.extend(scan_streaming_source(&label, shipped, &mut allowlist));
    });

    assert_walk_floor(&walked);
    // Scope floor: the wall must actually cover the streaming modules it
    // exists for — an in-scope set that collapses to nothing (a renamed
    // basename pattern, broken content scoping) is a vacuous pass.
    for suffix in [
        "rig-core/src/providers/anthropic/streaming.rs",
        "rig-core/src/providers/ollama.rs",
        "rig-core/src/providers/openai/responses_api/streaming.rs",
        "rig-bedrock/src/streaming.rs",
    ] {
        assert!(
            scanned_targets.iter().any(|label| label.ends_with(suffix)),
            "the serde wall no longer scans {suffix} — its scoping collapsed; \
             scanned {} targets",
            scanned_targets.len()
        );
    }

    assert!(
        violations.is_empty(),
        "raw serde parsing in a provider streaming module — route wire decoding \
         through the `wire.rs` classify layer, or (for a genuine non-triage use) \
         add a `path | snippet | justification` entry to \
         crates/rig-core/tests/serde_policy_allowlist.txt:\n{}",
        violations.join("\n")
    );

    let stale: Vec<&str> = allowlist
        .iter()
        .filter(|entry| !entry.used)
        .map(|entry| entry.snippet.as_str())
        .collect();
    assert!(
        stale.is_empty(),
        "stale serde_policy_allowlist.txt entries (the code they covered is gone \
         — delete them): {stale:?}"
    );
}

/// The scanner itself flags new raw serde: a synthetic streaming-module
/// source with an unlisted `serde_json::from_str` (and a `#[serde(other)]`)
/// fails, and the allowlist covers exactly what it names.
#[test]
fn serde_policy_scanner_catches_raw_parses() {
    let bad_source = r#"
        fn sneak_a_policy_site(data: &str) {
            let value = serde_json::from_str::<serde_json::Value>(data);
        }
        #[serde(other)]
        struct Marker;
    "#;

    let violations = scan_streaming_source(
        "crates/rig-core/src/providers/fake/streaming.rs",
        bad_source,
        &mut [],
    );
    assert_eq!(
        violations.len(),
        2,
        "the scanner must flag both the raw parse and the serde(other) fallback: {violations:?}"
    );

    // An allowlist entry covers its named line and nothing else.
    let mut allowlist = parse_allowlist(
        "providers/fake/streaming.rs | serde_json::from_str::<serde_json::Value>(data) | synthetic",
    );
    let violations = scan_streaming_source(
        "crates/rig-core/src/providers/fake/streaming.rs",
        bad_source,
        &mut allowlist,
    );
    assert_eq!(
        violations.len(),
        1,
        "only the serde(other) line stays flagged"
    );
    assert!(allowlist.iter().all(|entry| entry.used));

    // The classify layer's own file is out of scope by construction, even
    // though its content mentions the machinery.
    assert!(!is_serde_wall_target(
        std::path::Path::new("crates/rig-core/src/providers/internal/wire.rs"),
        "fn classify(frame: WireFrame) -> WireEvent { todo!() }",
    ));
    assert!(is_serde_wall_target(
        std::path::Path::new("crates/rig-core/src/providers/openai/responses_api/websocket.rs"),
        "",
    ));
}

/// Fail-closed content scoping: a helper whose basename the streaming
/// pattern misses (compat/sse helpers) is still scanned once its shipped
/// content touches the wire machinery, and stays out of scope while it
/// doesn't.
#[test]
fn serde_wall_scopes_by_machinery_content() {
    let compat = std::path::Path::new(
        "crates/rig-core/src/providers/internal/openai_chat_completions_compatible.rs",
    );
    assert!(
        is_serde_wall_target(compat, "use super::adapter::run_wire_stream;"),
        "a compat helper referencing the machinery must be scanned"
    );
    let future_helper = std::path::Path::new("crates/rig-core/src/providers/somegateway/sse.rs");
    assert!(
        is_serde_wall_target(
            future_helper,
            "let out = run_wire_buffered(adapter, frames);"
        ),
        "any future compat/sse helper opts in the moment it names the machinery"
    );
    assert!(
        !is_serde_wall_target(future_helper, "fn plain_request_builder() {}"),
        "machinery-free helpers stay out of scope"
    );
}

/// The wire.rs/adapter.rs exemption is by full path suffix: a foreign
/// `adapter.rs` (e.g. a hand-rolled rig-bedrock driver) is scanned by both
/// guards instead of inheriting the core driver's exemption.
#[test]
fn foreign_adapter_files_are_not_exempt() {
    let foreign = std::path::Path::new("crates/rig-bedrock/src/streaming/adapter.rs");
    assert!(
        !is_policy_home(foreign),
        "guard 1 must scan a foreign adapter.rs for restated policy tables"
    );
    assert!(
        is_serde_wall_target(foreign, "match event { WireEvent::Unknown(_) => {} }"),
        "guard 2 must scan a foreign adapter.rs that touches the machinery"
    );
    assert!(is_policy_home(std::path::Path::new(
        "crates/rig-core/src/providers/internal/adapter.rs"
    )));
    assert!(is_policy_home(std::path::Path::new(
        "crates/rig-core/src/providers/internal/wire.rs"
    )));
}

/// Gating at `#[cfg(test)]` is line-anchored: only an attribute in attribute
/// position removes an item. A doc-comment (or trailing comment) merely
/// mentioning the attribute must not exempt subsequent code.
#[test]
fn shipped_portion_ignores_cfg_test_mentions_in_comments() {
    let source = "\
/// This helper is exercised under #[cfg(test)] elsewhere.
fn shipped_code() { let _ = WireEvent::Unknown; }
#[cfg(test)]
mod tests {
    fn test_only() { let _ = WireEvent::Corrupt; }
}
";
    let shipped = shipped_portion(source);
    assert!(
        shipped.contains("WireEvent::Unknown"),
        "code after a doc-comment mention still ships"
    );
    assert!(
        !shipped.contains("WireEvent::Corrupt"),
        "the real attribute-position marker still removes its item"
    );

    // Indented attribute position (inside an impl block) removes its item.
    let indented = "fn a() {}\n    #[cfg(test)]\n    fn b() {}\n";
    assert_eq!(shipped_portion(indented), "fn a() {}\n\n\n");

    // No marker at all: the whole file ships.
    let plain = "fn a() {}\n";
    assert_eq!(shipped_portion(plain), plain);

    // A block comment quoting the attribute at column 0 is inert — the
    // residue that a truncating scanner could not see past.
    let block_commented = "\
/*
#[cfg(test)]
*/
fn shipped_code() { let _ = WireEvent::Unknown; }
";
    assert!(
        shipped_portion(block_commented).contains("WireEvent::Unknown"),
        "an attribute inside a block comment must not gate the code below it"
    );
}

/// Gating is ITEM-scoped: a `#[cfg(test)]` helper mid-file removes only that
/// helper, and everything after it is scanned again. This is the
/// `providers/openrouter/completion.rs` shape (a gated `final_request_body`
/// followed by thousands of shipped lines) that a truncating scanner hid.
#[test]
fn shipped_portion_is_item_scoped() {
    let source = "\
fn before() { let _ = WireEvent::Unknown; }

#[cfg(test)]
pub(super) fn gated_helper() -> u8 {
    let braces_in_a_string = \"unbalanced { brace\";
    let raw = r#\"also unbalanced } here\"#;
    let _ = (braces_in_a_string, raw);
    7
}

fn after() { let _ = run_wire_stream(); }

#[cfg(test)]
mod tests {
    fn test_only() { let _ = WireEvent::Corrupt; }
}

fn last() { let _ = triage_frame(); }
";
    let shipped = shipped_portion(source);
    assert!(shipped.contains("WireEvent::Unknown"), "{shipped}");
    assert!(
        shipped.contains("run_wire_stream"),
        "shipped code AFTER a gated helper must stay visible: {shipped}"
    );
    assert!(
        shipped.contains("triage_frame"),
        "shipped code after a gated `mod tests` must stay visible too: {shipped}"
    );
    assert!(
        !shipped.contains("gated_helper") && !shipped.contains("WireEvent::Corrupt"),
        "gated items themselves must be blanked: {shipped}"
    );
    assert_eq!(
        shipped.lines().count(),
        source.lines().count(),
        "blanking must preserve line numbering so violations cite real lines"
    );

    // Content scoping now sees machinery named only after a gated item — the
    // truncation-ordering half of the residue.
    assert!(
        is_serde_wall_target(
            std::path::Path::new("crates/rig-core/src/providers/somegateway/compat.rs"),
            &shipped,
        ),
        "a file whose only machinery reference sits after a gated helper must still be scanned"
    );

    // A brace-less gated item ends at its semicolon, not at end of file.
    let brace_less = "#[cfg(test)]\nuse std::fmt;\nfn after() { let _ = WireEvent::Unknown; }\n";
    assert!(shipped_portion(brace_less).contains("WireEvent::Unknown"));
}
