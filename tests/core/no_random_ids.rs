//! Nothing the engine mints is random.
//!
//! A run replays from its effect log only if every value the program
//! produces is a function of what it received (the record-replay rule:
//! every nondeterministic input is in the record, or replay is fiction).
//! Tool-call handles for wires that carry no id used to draw from
//! `fastrand`; they now derive from the block that assembled the call or
//! the call's index in the response. This guard pins the remaining random
//! sources to the two transport headers and the LSH index, none of which
//! can reach a request, a message or an effect record.

use std::path::{Path, PathBuf};

fn source_files(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            if path.is_dir() {
                stack.push(path);
            } else if path.extension().is_some_and(|ext| ext == "rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

fn relative(path: &Path) -> String {
    path.strip_prefix(Path::new(env!("CARGO_MANIFEST_DIR")))
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Where randomness is allowed to be drawn, and why.
const FASTRAND_SITES: &[(&str, &str)] = &[
    ("crates/rig-core/src/id.rs", "the id generator itself"),
    (
        "crates/rig-core/src/vector_store/lsh.rs",
        "LSH hyperplanes; an index, never a request or a record",
    ),
];

/// Where `id::generate` may be called, and why.
const GENERATE_SITES: &[(&str, &str)] = &[
    (
        "crates/rig-core/src/id.rs",
        "the generator's own entry point",
    ),
    (
        "crates/rig-core/src/providers/copilot/mod.rs",
        "an `x-request-id` transport header",
    ),
    (
        "crates/rig-core/src/providers/chatgpt/mod.rs",
        "a `session_id` transport header",
    ),
];

fn offenders(needle: &str, allowed: &[(&str, &str)]) -> Vec<String> {
    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("crates");
    let mut offenders = Vec::new();
    for path in source_files(&root) {
        let relative = relative(&path);
        if relative.contains("/tests/") || relative.ends_with("tests.rs") {
            continue;
        }
        if allowed.iter().any(|(site, _)| *site == relative) {
            continue;
        }
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("{relative} is readable: {err}"));
        for (number, line) in text.lines().enumerate() {
            if line.trim_start().starts_with("//") {
                continue;
            }
            if line.contains(needle) {
                offenders.push(format!("{relative}:{}: {}", number + 1, line.trim()));
            }
        }
    }
    offenders
}

#[test]
fn randomness_is_drawn_only_where_it_cannot_reach_a_record() {
    let fastrand = offenders("fastrand::", FASTRAND_SITES);
    assert!(
        fastrand.is_empty(),
        "`fastrand` is drawn outside its allowed sites:\n{}",
        fastrand.join("\n")
    );
    let generate = offenders("id::generate(", GENERATE_SITES);
    assert!(
        generate.is_empty(),
        "`id::generate` is called outside the two transport headers:\n{}",
        generate.join("\n")
    );
}

#[test]
fn the_allowed_sites_are_live() {
    for (site, reason) in FASTRAND_SITES.iter().chain(GENERATE_SITES) {
        let path = Path::new(env!("CARGO_MANIFEST_DIR")).join(site);
        assert!(
            path.exists(),
            "allowed site `{site}` ({reason}) no longer exists; drop it"
        );
    }
}

#[test]
fn tool_call_ids_have_no_random_constructor() {
    let path =
        Path::new(env!("CARGO_MANIFEST_DIR")).join("crates/rig-core/src/completion/message.rs");
    let text = std::fs::read_to_string(path).expect("message.rs is readable");
    assert!(
        !text.contains("fn mint()") && !text.contains("id::generate"),
        "ToolCallId must not mint from the random generator"
    );
}
