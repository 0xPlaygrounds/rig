//! Workspace registry for the wire-conformance corpus: every streaming wire
//! family in
//! [`WIRE_FAMILIES`](rig_core::test_utils::streaming_conformance::WIRE_FAMILIES)
//! must carry a `streaming_conformance_suite!` invocation, or CI fails (#2258,
//! Part III (c) — permanently closes the fixtures-covered-a-subset gap).
//!
//! **Compile-linked, not text-scanned (#2258 F3).** The first version of this
//! registry read the workspace's `.rs` files and matched the `provider: "…"`
//! literal of each `streaming_conformance_suite!` invocation. That made the
//! guard a paper claim: commenting a suite out, or wrapping it in a
//! `#[cfg(any())]`, removed its 11 tests and still matched the text, so the
//! registry reported the family covered. It now reads `SUITE_FAMILIES` consts
//! built out of the `WIRE_FAMILY` const the macro emits *inside each expanded
//! suite*, so a disabled suite is a compile error in the file that hosts it.
//!
//! The remaining seam is binary boundaries: four suites compile into test
//! binaries other than this one and therefore cannot be linked here. They are
//! enumerated in [`OUT_OF_BINARY_FAMILIES`], each naming the verification
//! check that actually executes it — and
//! [`out_of_binary_families_name_a_live_check`] resolves those annotations
//! against the check definitions CI runs, so they cannot decay into the same
//! kind of paper claim.

use std::collections::BTreeSet;
use std::path::Path;

use rig_core::test_utils::streaming_conformance::WIRE_FAMILIES;

use super::{streaming_conformance, streaming_conformance_suites};

/// A wire family whose suite compiles into a different test binary than this
/// registry, so it cannot be compile-linked here.
struct OutOfBinaryFamily {
    /// The `WIRE_FAMILIES` entry.
    family: &'static str,
    /// Where the suite is invoked (workspace-relative).
    suite_file: &'static str,
    /// Id of the `cargo xtask verify --check` that runs it (ci.yaml runs
    /// every check by id).
    ci_check: &'static str,
    /// The nextest predicate inside that check which selects the binary,
    /// when the check uses one.
    ci_selector: Option<&'static str>,
    /// `-p` package the check must compile for this suite to exist at all,
    /// when the suite lives outside the facade package.
    ci_package: Option<&'static str>,
    /// Why it cannot live in the `core` binary.
    reason: &'static str,
}

/// The check added for the out-of-facade suites (#2258 N1). Before it
/// existed, `cargo nextest run --all-features` resolved the workspace's
/// *default members* — just the `rig` facade — so every binary below was
/// compiled by nobody and ran nowhere.
const GUARD_CHECK: &str = "conformance";

/// The default-member sweep, which does cover targets inside `rig`.
const FACADE_CHECK: &str = "default-tests";

const OUT_OF_BINARY_FAMILIES: &[OutOfBinaryFamily] = &[
    OutOfBinaryFamily {
        family: "openai_responses_websocket",
        suite_file: "crates/rig-tungstenite/tests/streaming_conformance_websocket.rs",
        ci_check: GUARD_CHECK,
        ci_selector: Some("binary(streaming_conformance_websocket)"),
        ci_package: Some("rig-tungstenite"),
        reason: "drives a real `ResponsesWebSocketSession` over the bundled tungstenite backend \
                 against a local ws server; rig-tungstenite is a separate package the facade \
                 pulls in only behind its `websocket` feature",
    },
    OutOfBinaryFamily {
        family: "candle",
        suite_file: "crates/rig-candle/tests/streaming_conformance.rs",
        ci_check: GUARD_CHECK,
        ci_selector: Some("binary(streaming_conformance)"),
        ci_package: Some("rig-candle"),
        reason: "rig-candle is a separate package; the facade does not depend on it",
    },
    OutOfBinaryFamily {
        family: "gemini_grpc",
        suite_file: "crates/rig-gemini-grpc/tests/streaming_conformance.rs",
        ci_check: GUARD_CHECK,
        ci_selector: Some("binary(streaming_conformance)"),
        ci_package: Some("rig-gemini-grpc"),
        reason: "rig-gemini-grpc is a separate package; the facade does not depend on it",
    },
    OutOfBinaryFamily {
        family: "bedrock",
        suite_file: "tests/providers/bedrock/streaming_conformance.rs",
        ci_check: FACADE_CHECK,
        // The PR gate's sweep runs `--features bedrock`, not `--all-features`,
        // so this suite's existence depends on that one flag: without it the
        // `#[cfg(feature = "bedrock")] mod bedrock` in `tests/bedrock.rs` is
        // cfg-ed out and all 11 conformance tests silently stop compiling.
        // (Do not gate the whole file: its cassette-safety scan is
        // deliberately ungated.)
        ci_selector: Some("--features bedrock"),
        ci_package: None,
        reason: "lives in the `rig` facade but behind the `bedrock` feature, so it compiles into \
                 the `bedrock` test binary rather than `core`; the workspace sweep enables \
                 `--features bedrock` specifically so that check keeps executing it",
    },
];

/// Union of the suite families compiled into THIS binary.
fn linked_families() -> BTreeSet<&'static str> {
    streaming_conformance_suites::SUITE_FAMILIES
        .iter()
        .chain(streaming_conformance::SUITE_FAMILIES)
        .copied()
        .collect()
}

#[test]
fn all_wire_families_have_conformance_suites() {
    let mut covered = linked_families();
    covered.extend(OUT_OF_BINARY_FAMILIES.iter().map(|entry| entry.family));

    let missing: Vec<&&str> = WIRE_FAMILIES
        .iter()
        .filter(|family| !covered.contains(**family))
        .collect();
    assert!(
        missing.is_empty(),
        "wire families without a streaming_conformance_suite! invocation: {missing:?} \
         (covered: {covered:?})",
    );

    // A family name outside the canonical list is a typo or an unregistered
    // family; both must be fixed at the WIRE_FAMILIES source of truth.
    let unknown: Vec<&&str> = covered
        .iter()
        .filter(|family| !WIRE_FAMILIES.contains(*family))
        .collect();
    assert!(
        unknown.is_empty(),
        "suite invocations name families missing from WIRE_FAMILIES: {unknown:?}",
    );
}

/// An `OUT_OF_BINARY_FAMILIES` entry is an admission that this binary cannot
/// prove the suite exists, so each one must be a genuine gap: a family that IS
/// linkable here belongs in a `SUITE_FAMILIES` const instead, where the
/// compiler enforces it.
#[test]
fn out_of_binary_families_are_genuinely_out_of_binary() {
    let linked = linked_families();
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    for entry in OUT_OF_BINARY_FAMILIES {
        assert!(
            !linked.contains(entry.family),
            "{} is compile-linked into this binary — drop it from OUT_OF_BINARY_FAMILIES",
            entry.family,
        );
        assert!(
            root.join(entry.suite_file).is_file(),
            "OUT_OF_BINARY_FAMILIES points {} at {}, which does not exist",
            entry.family,
            entry.suite_file,
        );
        assert!(
            !entry.reason.is_empty(),
            "{} needs a reason it cannot live in the core binary",
            entry.family,
        );
    }
}

/// The "which check runs it" annotations above are the only thing standing
/// between an out-of-binary suite and never executing again (#2258 N1: three
/// of these four families were compiled by no CI step at all while this
/// registry reported them covered). Resolve each against the check
/// definitions and confirm ci.yaml invokes that check.
#[test]
fn out_of_binary_families_name_a_live_check() {
    let workflow = Path::new(env!("CARGO_MANIFEST_DIR")).join(".github/workflows/ci.yaml");
    let text = std::fs::read_to_string(&workflow).expect("ci.yaml should be readable");
    // A YAML parse, not a text search: a `run:` left behind in a comment
    // must not satisfy this.
    let doc: serde_yaml::Value = serde_yaml::from_str(&text).expect("ci.yaml should parse as YAML");
    let runs: Vec<String> = doc
        .get("jobs")
        .and_then(serde_yaml::Value::as_mapping)
        .expect("ci.yaml should have a jobs mapping")
        .values()
        .filter_map(|job| job.get("steps").and_then(serde_yaml::Value::as_sequence))
        .flatten()
        .filter_map(|step| step.get("run").and_then(serde_yaml::Value::as_str))
        .map(str::to_owned)
        .collect();
    let checks = verification_checks::all();
    for entry in OUT_OF_BINARY_FAMILIES {
        let check = checks
            .iter()
            .find(|check| check.id == entry.ci_check)
            .unwrap_or_else(|| {
                panic!(
                    "{} names check {:?}, which does not exist",
                    entry.family, entry.ci_check
                )
            });
        assert!(
            runs.iter()
                .any(|run| run.trim() == format!("cargo xtask verify --check {}", entry.ci_check)),
            "{} names check {:?}, which no ci.yaml step runs",
            entry.family,
            entry.ci_check,
        );
        let run = check
            .steps
            .iter()
            .map(|step| format!("{} {}", step.program, step.args.join(" ")))
            .collect::<Vec<_>>()
            .join("\n");
        if let Some(selector) = entry.ci_selector {
            // A `--features X` selector asserts an outcome, not a spelling:
            // `--all-features` enables strictly more.
            let satisfied = run.contains(selector)
                || (selector.starts_with("--features ") && run.contains("--all-features"));
            assert!(
                satisfied,
                "{}'s check no longer carries the {:?} predicate, so its binary is selected by \
                 nothing — a nextest filter matching zero tests still exits 0",
                entry.family, selector,
            );
        }
        if let Some(package) = entry.ci_package {
            assert!(
                run.contains(&format!("-p {package}")),
                "{}'s check no longer compiles `-p {package}`, so its suite binary does not build",
                entry.family,
            );
        }
    }
}

// Compile the same definitions used by xtask; comments or stale command copies
// cannot satisfy the live CI selector assertions above.
#[allow(dead_code)]
#[path = "../../xtask/src/verify/checks.rs"]
mod verification_checks;
