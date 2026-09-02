//! The effect bus is rig-core's only erasure.
//!
//! After the bus, the only `dyn` over a behaviour trait stored anywhere in
//! rig-core is the handler table inside `bus/`. This guard scans rig-core's
//! non-test sources for `Arc<dyn …>` / `Box<dyn …>` over the five impl-side
//! traits — `CompletionModel`, `EmbeddingModel`, `Tool`, `ConversationMemory`,
//! `VectorStoreIndex` — or over any `Erased*` / `*Callback` trait, and for
//! stored `dyn Handler` outside `bus/`. `dyn Fn`, `dyn Error`, `dyn Future`,
//! `dyn Iterator`/`dyn Stream` boxes and `InMemoryConversationMemory`'s
//! `MessageFilter` are not erasures of a behaviour trait and are exempt.
//!
//! Review greps, pinned here too: no `OnceLock` and no `static` in `bus/` or
//! `effect/` — spawning is explicit, there is no ambient executor.

use std::path::{Path, PathBuf};

fn rig_core_src() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("crates/rig-core/src")
}

/// Every `.rs` file under `root`, skipping test modules (`tests.rs`,
/// `*_tests.rs`, and `tests/` directories) — tests may build erased values
/// to exercise the forwarding impls.
fn non_test_sources(root: &Path) -> Vec<PathBuf> {
    let mut out = Vec::new();
    let mut stack = vec![root.to_path_buf()];
    while let Some(dir) = stack.pop() {
        let Ok(entries) = std::fs::read_dir(&dir) else {
            continue;
        };
        for entry in entries.flatten() {
            let path = entry.path();
            let name = entry.file_name().to_string_lossy().into_owned();
            if path.is_dir() {
                if name == "tests" {
                    continue;
                }
                stack.push(path);
            } else if name.ends_with(".rs") && name != "tests.rs" && !name.ends_with("_tests.rs") {
                out.push(path);
            }
        }
    }
    out.sort();
    out
}

/// The behaviour traits nothing may erase outside the bus.
const BEHAVIOUR_TRAITS: [&str; 5] = [
    "CompletionModel",
    "EmbeddingModel",
    "Tool",
    "ConversationMemory",
    "VectorStoreIndex",
];

/// `dyn` targets that are not erasures of a behaviour trait.
const EXEMPT_DYN: [&str; 8] = [
    "Fn", "FnMut", "FnOnce", "Error", "Future", "Iterator", "Stream", "Any",
];

fn is_exempt(target: &str) -> bool {
    let head: String = target
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == ':')
        .collect();
    let head = head.rsplit("::").next().unwrap_or(&head).to_owned();
    EXEMPT_DYN.contains(&head.as_str())
        || head == "MessageFilter"
        || head == "StdError"
        || head == "ErasedHttpClient"
        || head == "HttpClient"
        || head == "WasmCompatSendStream"
        || head == "DynAgentHook"
        || head == "Subscriber"
}

/// Whether a `dyn <target>` is an erasure the guard forbids.
fn is_forbidden(target: &str, in_bus: bool) -> bool {
    let head: String = target
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == ':')
        .collect();
    let head = head.rsplit("::").next().unwrap_or(&head).to_owned();
    if head == "Handler" {
        return !in_bus;
    }
    if is_exempt(&head) {
        return false;
    }
    if BEHAVIOUR_TRAITS.contains(&head.as_str()) {
        return true;
    }
    // Rerank is not one of the five families and has no effect kind (the
    // transcription rule keeps the vocabulary to the five traits), so its
    // vtable handle stays — the recorded follow-up is a `Rerank` family.
    if head == "ErasedRerankModel" {
        return false;
    }
    (head.starts_with("Erased") && head != "ErasedHandler") || head.ends_with("Callback")
}

#[test]
fn rig_core_has_one_erasure() {
    let root = rig_core_src();
    let mut offenders = Vec::new();
    for path in non_test_sources(&root) {
        let relative = path
            .strip_prefix(&root)
            .unwrap_or(&path)
            .to_string_lossy()
            .replace('\\', "/");
        let in_bus = relative.starts_with("bus/");
        let text = std::fs::read_to_string(&path)
            .unwrap_or_else(|err| panic!("{relative} is readable: {err}"));
        for (line_number, line) in text.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            // Any `dyn <trait>` — directly under `Arc`/`Box`/`Rc`, or nested
            // inside a driver struct (`Arc<Driver<dyn Erased…>>`) — counts.
            let needle = "dyn ";
            let mut rest = line;
            while let Some(index) = rest.find(needle) {
                let target = &rest[index + needle.len()..];
                if is_forbidden(target, in_bus) {
                    offenders.push(format!("{relative}:{}: {}", line_number + 1, line.trim()));
                }
                rest = &rest[index + needle.len()..];
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "rig-core stores a `dyn` over a behaviour trait outside the bus:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn bus_and_effect_have_no_global_state() {
    let root = rig_core_src();
    let mut offenders = Vec::new();
    for module in ["bus", "effect"] {
        for path in non_test_sources(&root.join(module)) {
            let relative = path
                .strip_prefix(&root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("{relative} is readable: {err}"));
            for (line_number, line) in text.lines().enumerate() {
                let trimmed = line.trim_start();
                if trimmed.starts_with("//") {
                    continue;
                }
                let is_static = trimmed.starts_with("static ")
                    || trimmed.starts_with("pub static ")
                    || trimmed.starts_with("pub(crate) static ");
                if is_static || line.contains("OnceLock") || line.contains("thread_local!") {
                    offenders.push(format!("{relative}:{}: {}", line_number + 1, line.trim()));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "no global or ambient state in bus/ or effect/:\n{}",
        offenders.join("\n")
    );
}

/// The typed views implement none of the consumer-facing traits: a handle is
/// consumed through its inherent methods.
#[test]
fn typed_views_implement_no_consumer_facing_trait() {
    let text = std::fs::read_to_string(rig_core_src().join("bus/handle.rs"))
        .expect("bus/handle.rs is readable");
    for trait_name in [
        "CompletionModel",
        "EmbeddingModel",
        "ImageEmbeddingModel",
        "Tool",
        "ConversationMemory",
        "VectorStoreIndex",
    ] {
        let needle = format!("impl {trait_name} for");
        assert!(
            !text.contains(&needle),
            "bus/handle.rs implements `{trait_name}` for a typed view"
        );
        let needle = "impl<";
        let generic_impls = text
            .lines()
            .filter(|line| line.contains(needle) && line.contains(&format!("{trait_name} for")))
            .count();
        assert_eq!(
            generic_impls, 0,
            "bus/handle.rs implements `{trait_name}` for a typed view"
        );
    }
}
