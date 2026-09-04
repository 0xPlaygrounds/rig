//! The `rig_ecs::bus` module is written as if it were already its own
//! crate, so that the extraction to `rig-bevy` is a `git mv`:
//!
//! - nothing under `crates/rig-ecs/src/bus/` has a crate-scoped visibility
//!   — no `pub(crate)`, no `pub(super)`, no `pub(in ..)`: an item is `pub`
//!   (a downstream crate could use it) or private to its file;
//! - no `use crate::` in `bus/` names a module other than `bus`;
//! - no file under `bus/`, and no `tests/bus_*` file, mentions an
//!   agent-shaped identifier;
//! - and the discipline the shape depends on: nothing in the crate blocks
//!   (`block_on`), holds an `Entity` inside a serde type, forks on the
//!   target (`cfg(target`), draws a clock or a random number, or reaches
//!   for a side channel of the bus runtime (the names rig-bus lost when
//!   its host became a world).

use std::path::{Path, PathBuf};

fn crate_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("crates/rig-ecs")
}

fn rust_files(root: &Path) -> Vec<PathBuf> {
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

fn read(path: &Path) -> String {
    std::fs::read_to_string(path).unwrap_or_else(|err| panic!("{}: {err}", path.display()))
}

fn relative(path: &Path) -> String {
    path.strip_prefix(crate_root())
        .unwrap_or(path)
        .to_string_lossy()
        .replace('\\', "/")
}

/// Code lines: comments and doc comments skipped.
fn code_lines(text: &str) -> impl Iterator<Item = (usize, &str)> {
    text.lines()
        .enumerate()
        .map(|(number, line)| (number + 1, line))
        .filter(|(_, line)| !line.trim_start().starts_with("//"))
}

#[test]
fn nothing_in_the_bus_module_is_crate_scoped() {
    let mut offenders = Vec::new();
    for path in rust_files(&crate_root().join("src/bus")) {
        let text = read(&path);
        for (number, line) in code_lines(&text) {
            let trimmed = line.trim_start();
            if trimmed.contains("pub(crate)")
                || trimmed.contains("pub(super)")
                || trimmed.contains("pub(in ")
            {
                offenders.push(format!("{}:{number}: {}", relative(&path), line.trim()));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "a crate-scoped item in rig_ecs::bus (the module is the future rig-bevy crate: pub, or private to its file):\n{}",
        offenders.join("\n")
    );
}

#[test]
fn the_bus_module_imports_no_sibling() {
    let mut offenders = Vec::new();
    for path in rust_files(&crate_root().join("src/bus")) {
        let text = read(&path);
        for (number, line) in code_lines(&text) {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("use crate::")
                && !rest.starts_with("bus")
            {
                offenders.push(format!("{}:{number}: {trimmed}", relative(&path)));
            }
            if trimmed.contains("crate::")
                && !trimmed.contains("crate::bus")
                && !trimmed.starts_with("use ")
            {
                offenders.push(format!("{}:{number}: {trimmed}", relative(&path)));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "rig_ecs::bus reaches a sibling module:\n{}",
        offenders.join("\n")
    );
}

/// Identifiers the substrate must not know. Matched as whole words on code
/// lines.
const AGENT_SHAPED: [&str; 6] = ["Agent", "Run", "Turn", "Utterance", "Conversation", "Hook"];

fn mentions_word(line: &str, word: &str) -> bool {
    let bytes = line.as_bytes();
    let mut search = 0;
    while let Some(found) = line[search..].find(word) {
        let at = search + found;
        let end = at + word.len();
        let before = at == 0 || !(bytes[at - 1].is_ascii_alphanumeric() || bytes[at - 1] == b'_');
        let after =
            end >= bytes.len() || !(bytes[end].is_ascii_alphanumeric() || bytes[end] == b'_');
        if before && after {
            return true;
        }
        search = end;
    }
    false
}

#[test]
fn the_bus_module_and_its_suites_are_agent_free() {
    let mut files = rust_files(&crate_root().join("src/bus"));
    files.extend(
        rust_files(&crate_root().join("tests"))
            .into_iter()
            .filter(|path| {
                path.file_name()
                    .is_some_and(|name| name.to_string_lossy().starts_with("bus_"))
            }),
    );
    files.extend(rust_files(&crate_root().join("tests/bus_support")));
    let mut offenders = Vec::new();
    for path in files {
        let text = read(&path);
        for (number, line) in code_lines(&text) {
            for word in AGENT_SHAPED {
                if mentions_word(line, word) {
                    offenders.push(format!("{}:{number}: {word}", relative(&path)));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "an agent-shaped identifier in the substrate:\n{}",
        offenders.join("\n")
    );
}

/// What the crate must never do, matched on code lines of `src/` and of
/// the examples: block, fork on the target, draw a clock or a random
/// number, or use a side channel of the bus runtime.
const FORBIDDEN_IN_SOURCE: [(&str, &str); 12] = [
    (
        "block_on",
        "the crate never blocks; a system that needs an answer reads the component next pass",
    ),
    ("cfg(target", "one spelling on every target"),
    ("cfg(not(target", "one spelling on every target"),
    ("Instant", "nondeterminism is an effect or forbidden"),
    ("SystemTime", "nondeterminism is an effect or forbidden"),
    ("fastrand", "nondeterminism is an effect or forbidden"),
    ("rand::", "nondeterminism is an effect or forbidden"),
    (
        "take_resolved",
        "no completion inbox: readiness is the component",
    ),
    ("poll_outcome", "no future for a host to probe"),
    ("poll_item", "no future for a host to probe"),
    ("mint_id", "the world mints ids"),
    ("reopen", "nothing dies, nothing reopens"),
];

#[test]
fn the_crate_source_holds_no_blocking_forks_clocks_or_side_channels() {
    let mut files = rust_files(&crate_root().join("src"));
    files.extend(rust_files(&crate_root().join("examples")));
    let mut offenders = Vec::new();
    for path in files {
        let text = read(&path);
        for (number, line) in code_lines(&text) {
            for (needle, why) in FORBIDDEN_IN_SOURCE {
                if line.contains(needle) {
                    offenders.push(format!("{}:{number}: `{needle}` — {why}", relative(&path)));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "rig-ecs source breaks the discipline:\n{}",
        offenders.join("\n")
    );
}

/// No `Entity` inside a type that derives `Serialize`: a scene remaps
/// entities by position, never by id. The check is over the struct or
/// enum body that follows a `derive(...Serialize...)` line.
#[test]
fn no_serde_type_holds_an_entity() {
    let mut offenders = Vec::new();
    for path in rust_files(&crate_root().join("src")) {
        let text = read(&path);
        let lines: Vec<&str> = text.lines().collect();
        let mut index = 0;
        while index < lines.len() {
            let line = lines[index];
            if line.trim_start().starts_with("#[derive(") {
                // The derive may span lines (rustfmt splits long ones):
                // gather it up to its `)]`, then the attributes that follow,
                // then take the item's body up to the closing brace at the
                // item's indentation.
                let mut derive = String::new();
                let mut cursor = index;
                while cursor < lines.len() {
                    derive.push_str(lines[cursor]);
                    cursor += 1;
                    if derive.contains(")]") {
                        break;
                    }
                }
                if !derive.contains("Serialize") {
                    index = cursor;
                    continue;
                }
                while cursor < lines.len() && lines[cursor].trim_start().starts_with('#') {
                    cursor += 1;
                }
                let body_start = cursor;
                let mut depth = 0i32;
                let mut opened = false;
                while cursor < lines.len() {
                    let body_line = lines[cursor];
                    depth += body_line.matches('{').count() as i32;
                    depth -= body_line.matches('}').count() as i32;
                    if body_line.contains('{') {
                        opened = true;
                    }
                    if mentions_word(body_line, "Entity") {
                        offenders.push(format!(
                            "{}:{}: {}",
                            relative(&path),
                            cursor + 1,
                            body_line.trim()
                        ));
                    }
                    if (opened && depth <= 0) || (!opened && body_line.trim_end().ends_with(';')) {
                        break;
                    }
                    cursor += 1;
                }
                index = body_start.max(cursor);
            }
            index += 1;
        }
    }
    assert!(
        offenders.is_empty(),
        "an Entity inside a serde type (scenes remap by position):\n{}",
        offenders.join("\n")
    );
}

/// The crate's tests live where the module guard can find them: every
/// integration test file is a `bus_*` file (the substrate's own suite) or
/// one of the agent's — `run_*`, `tool_*`, `steer_*`, `memory_*`,
/// `reflect_*` — and no `bus_*` file imports an agent module: the
/// substrate's suite is the future rig-bevy suite verbatim.
#[test]
fn every_test_file_belongs_to_a_suite_and_the_bus_suite_is_agent_free() {
    const SUITES: [&str; 6] = ["bus_", "run_", "tool_", "steer_", "memory_", "reflect_"];
    let tests = crate_root().join("tests");
    let offenders: Vec<String> = std::fs::read_dir(&tests)
        .expect("rig-ecs has tests")
        .flatten()
        .map(|entry| entry.file_name().to_string_lossy().into_owned())
        .filter(|name| !SUITES.iter().any(|suite| name.starts_with(suite)))
        .collect();
    assert!(
        offenders.is_empty(),
        "a test file outside the bus and run suites: {offenders:?}"
    );
    let mut imports = Vec::new();
    for path in rust_files(&tests) {
        let name = path
            .file_name()
            .map(|n| n.to_string_lossy().into_owned())
            .unwrap_or_default();
        if !name.starts_with("bus_") && !path.to_string_lossy().contains("bus_support") {
            continue;
        }
        let text = read(&path);
        for (number, line) in code_lines(&text) {
            for module in [
                "rig_ecs::agent",
                "rig_ecs::systems",
                "rig_ecs::policy",
                "rig_ecs::replay",
            ] {
                if line.contains(module) {
                    imports.push(format!("{}:{number}: {module}", relative(&path)));
                }
            }
        }
    }
    assert!(
        imports.is_empty(),
        "the bus suite imports an agent module:\n{}",
        imports.join("\n")
    );
}

/// The agent modules' discipline: nothing from the frozen crate, no
/// history vector or document vector outside the fold, no
/// `CompletionRequest` built outside the fold, and (with the bus module's
/// rules) no clock, no random draw, no `Entity` in a serde payload.
#[test]
fn the_agent_modules_hold_the_discipline() {
    let src = crate_root().join("src");
    let mut offenders = Vec::new();
    for path in rust_files(&src) {
        let relative_path = relative(&path);
        let is_fold =
            relative_path == "src/policy/mod.rs" || relative_path == "src/policy/tests.rs";
        let text = read(&path);
        for (number, line) in code_lines(&text) {
            for needle in ["rig_agent", "rig::agent"] {
                if line.contains(needle) {
                    offenders.push(format!(
                        "{relative_path}:{number}: `{needle}` — nothing from the frozen crate"
                    ));
                }
            }
            if !is_fold {
                for needle in ["Vec<Message>", "Vec<Document>", "CompletionRequest {"] {
                    if line.contains(needle) {
                        offenders.push(format!(
                            "{relative_path}:{number}: `{needle}` — the fold is the only place a request or a history vector exists"
                        ));
                    }
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "the agent modules break the discipline:\n{}",
        offenders.join("\n")
    );
}
