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

/// Returns the shipped portion of a source file: everything before the first
/// line whose (trimmed) content *starts with* `#[cfg(test)]` — i.e. an actual
/// attribute in attribute position. Inline unit-test modules may exercise wire
/// shapes freely; only the code before the test module ships. Line-anchoring
/// matters: a doc comment or trailing comment merely *mentioning*
/// `#[cfg(test)]` must not exempt the shipped code below it (see the
/// `shipped_portion_ignores_cfg_test_mentions_in_comments` self-test).
fn shipped_portion(source: &str) -> &str {
    let mut offset = 0usize;
    for line in source.split_inclusive('\n') {
        if line.trim_start().starts_with("#[cfg(test)]") {
            break;
        }
        offset += line.len();
    }
    source
        .get(..offset)
        .expect("offset lies on a line boundary")
}

/// Walks every `.rs` file under the workspace `crates/` directory (skipping
/// [`SKIPPED_DIRS`]) and calls `visit(path, shipped_source)`, where
/// `shipped_source` is the file content truncated by [`shipped_portion`].
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
            visit(&path, shipped_portion(&source));
        }
    }
}

/// No provider restates the driver's Unknown/Corrupt policy table.
#[test]
fn every_triage_site_runs_on_the_single_policy_driver() {
    let mut violations = Vec::new();

    for_each_shipped_source(|path, shipped| {
        if is_policy_home(path) {
            return;
        }

        for (index, line) in shipped.lines().enumerate() {
            if line.contains("WireEvent::Unknown") || line.contains("WireEvent::Corrupt") {
                violations.push(format!("{}:{}: {}", path.display(), index + 1, line.trim()));
            }
        }
    });

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
    for_each_shipped_source(|path, shipped| {
        if !is_serde_wall_target(path, shipped) {
            return;
        }
        let label = path.to_string_lossy().replace('\\', "/");
        violations.extend(scan_streaming_source(&label, shipped, &mut allowlist));
    });

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

/// Truncation at `#[cfg(test)]` is line-anchored: only an attribute in
/// attribute position ends the shipped portion. A doc-comment (or trailing
/// comment) merely mentioning the attribute must not exempt subsequent code.
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
        "the real attribute-position marker still truncates"
    );

    // Indented attribute position (inside an impl block) also truncates.
    let indented = "fn a() {}\n    #[cfg(test)]\n    fn b() {}\n";
    assert_eq!(shipped_portion(indented), "fn a() {}\n");

    // No marker at all: the whole file ships.
    let plain = "fn a() {}\n";
    assert_eq!(shipped_portion(plain), plain);
}
