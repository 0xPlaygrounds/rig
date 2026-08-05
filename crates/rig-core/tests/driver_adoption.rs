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

/// File basenames where the policy table and classifiers legitimately name the
/// triage variants.
const ALLOWED_FILES: &[&str] = &["adapter.rs", "wire.rs"];

/// Directories that hold test harness code rather than shipped policy.
const SKIPPED_DIRS: &[&str] = &["tests", "test_utils", "fixtures", "target"];

/// Walks every `.rs` file under the workspace `crates/` directory (skipping
/// [`SKIPPED_DIRS`]) and calls `visit(path, shipped_source)`, where
/// `shipped_source` is the file content truncated at the first
/// `#[cfg(test)]` marker: inline unit-test modules may exercise wire shapes
/// freely, only the code before the test module ships.
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
            let shipped = source
                .split("#[cfg(test)]")
                .next()
                .expect("split always yields at least one part");
            visit(&path, shipped);
        }
    }
}

/// No provider restates the driver's Unknown/Corrupt policy table.
#[test]
fn every_triage_site_runs_on_the_single_policy_driver() {
    let mut violations = Vec::new();

    for_each_shipped_source(|path, shipped| {
        let name = path
            .file_name()
            .map(|name| name.to_string_lossy().into_owned())
            .unwrap_or_default();
        if ALLOWED_FILES.contains(&name.as_str()) {
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

/// A file is a provider streaming module when its basename says so. `wire.rs`
/// (the classify layer) and `adapter.rs` (the driver) are different basenames
/// and therefore never scanned; `rig-agent`'s streaming modules are
/// consumer-side (no wire decoding) and are excluded by path.
///
/// Single-file providers keep their streaming code in files the basename
/// pattern misses, so those are scanned by explicit path suffix — extend
/// `SINGLE_FILE_STREAMING_MODULES` when a new provider adopts that layout.
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

fn is_provider_streaming_module(path: &std::path::Path) -> bool {
    let unix_path = path.to_string_lossy().replace('\\', "/");
    if unix_path.contains("/rig-agent/") || unix_path.contains("/test_utils/") {
        return false;
    }
    if SINGLE_FILE_STREAMING_MODULES
        .iter()
        .any(|suffix| unix_path.ends_with(suffix))
    {
        return true;
    }
    path.file_name()
        .map(|name| name.to_string_lossy().into_owned())
        .is_some_and(|name| name.contains("streaming") || name.contains("websocket"))
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
        if !is_provider_streaming_module(path) {
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

    // The classify layer's own file is out of scope by construction.
    assert!(!is_provider_streaming_module(std::path::Path::new(
        "crates/rig-core/src/providers/internal/wire.rs"
    )));
    assert!(is_provider_streaming_module(std::path::Path::new(
        "crates/rig-core/src/providers/openai/responses_api/websocket.rs"
    )));
}
