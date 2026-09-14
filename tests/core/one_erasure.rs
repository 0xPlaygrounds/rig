//! The effect bus is rig-core's only erasure.
//!
//! After the bus, the only `dyn` over a behaviour trait stored anywhere in
//! rig-core or rig-agent is the handler table inside rig-agent's bus driver. This guard
//! scans both crates' non-test sources for `Arc<dyn …>` / `Box<dyn …>` over
//! the six impl-side traits — `CompletionModel`, `EmbeddingModel`, `Tool`,
//! `ConversationMemory`, `VectorStoreIndex`, `RerankModel` — or over any
//! `Erased*` /
//! `*Callback` trait, and for `dyn Handler` outside the newtype that holds it
//! (`serve/handler.rs`). `dyn Fn`, `dyn Error`, `dyn Future`, `dyn Iterator`/
//! `dyn Stream` boxes and the named non-behaviour erasures below are exempt;
//! every named exemption must still occur, so a renamed type fails the guard
//! instead of silently widening it.
//!
//! Review greps, pinned here too: no `OnceLock` and no `static` in `bus/` or
//! `effect/` — spawning is explicit, there is no ambient executor.

use std::path::{Path, PathBuf};

fn crate_src(name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("crates")
        .join(name)
        .join("src")
}

/// The crates the guard scans: rig-core (the vocabulary, the handler seam
/// and the one erasure), rig-effect-log (record and replay), rig-agent (the
/// bus runtime, `bus/`, and the engine over it) and rig-ecs (the second
/// runtime: the bus in a `World`). A crate that is not listed is not
/// scanned: a future runtime must be added here the commit it appears, or
/// a `dyn` it stores escapes the guard silently.
const SCANNED: [&str; 4] = ["rig-core", "rig-effect-log", "rig-agent", "rig-ecs"];

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

/// A scanned source: its crate-relative path and text.
struct Source {
    krate: &'static str,
    relative: String,
    text: String,
}

fn sources() -> Vec<Source> {
    let mut out = Vec::new();
    for krate in SCANNED {
        let root = crate_src(krate);
        for path in non_test_sources(&root) {
            let relative = path
                .strip_prefix(&root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("{krate}/{relative} is readable: {err}"));
            out.push(Source {
                krate,
                relative,
                text,
            });
        }
    }
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

/// `dyn` targets from std/futures that are not erasures of a behaviour trait.
const EXEMPT_DYN: [&str; 8] = [
    "Fn", "FnMut", "FnOnce", "Error", "Future", "Iterator", "Stream", "Any",
];

/// Named exemptions: erasures the scanned crates hold that are not of a
/// behaviour trait. Each must still occur somewhere in the scanned sources
/// (`named_exemptions_are_live`), so a rename fails here rather than
/// widening the guard.
const NAMED_EXEMPT: [(&str, &str); 5] = [
    // `InMemoryConversationMemory`'s message filter: a predicate, not a
    // behaviour.
    ("MessageFilter", "a predicate over messages"),
    // The HTTP client the providers share: a transport, not a family.
    ("ErasedHttpClient", "the erased HTTP transport"),
    // The wasm-compatible stream alias.
    ("WasmCompatSendStream", "the wasm-compatible boxed stream"),
    // rig-agent's hook stack: hooks are observers, not a family.
    ("DynAgentHook", "the agent's hook-stack entry"),
    // The one erasure: the handler table's entry type.
    ("Handler", "the bus's handler newtype"),
];

/// The file the one erasure lives in: `ErasedHandler`'s newtype.
const ERASURE_FILE: (&str, &str) = ("rig-core", "serve/handler.rs");

fn head_of(target: &str) -> String {
    let head: String = target
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_' || *c == ':')
        .collect();
    head.rsplit("::").next().unwrap_or(&head).to_owned()
}

/// Whether a `dyn <target>` in `source` is an erasure the guard forbids.
fn is_forbidden(source: &Source, target: &str) -> bool {
    let head = head_of(target);
    if head == "Handler" {
        return (source.krate, source.relative.as_str()) != ERASURE_FILE;
    }
    if EXEMPT_DYN.contains(&head.as_str()) || NAMED_EXEMPT.iter().any(|(name, _)| *name == head) {
        return false;
    }
    if BEHAVIOUR_TRAITS.contains(&head.as_str()) {
        return true;
    }
    head.starts_with("Erased") || head.ends_with("Callback")
}

/// Every `dyn <target>` in `text` with the line it starts on, whitespace
/// between `dyn` and its target normalised (a `dyn` at a line break is
/// `dyn `). Comment lines are skipped.
fn dyn_targets(text: &str) -> Vec<(usize, String)> {
    let mut out = Vec::new();
    let bytes = text.as_bytes();
    let mut search = 0;
    while let Some(found) = text[search..].find("dyn") {
        let index = search + found;
        search = index + 3;
        let before_ok =
            index == 0 || !(bytes[index - 1].is_ascii_alphanumeric() || bytes[index - 1] == b'_');
        let after = &text[index + 3..];
        let Some(first) = after.chars().next() else {
            break;
        };
        if !before_ok || !first.is_whitespace() {
            continue;
        }
        let line_start = text[..index].rfind('\n').map_or(0, |at| at + 1);
        let line = text[line_start..].lines().next().unwrap_or("");
        if line.trim_start().starts_with("//") {
            continue;
        }
        let target = after.trim_start();
        let line_number = text[..index].matches('\n').count() + 1;
        out.push((line_number, target.chars().take(80).collect()));
    }
    out
}

#[test]
fn rig_core_and_rig_agent_have_one_erasure() {
    let mut offenders = Vec::new();
    for source in sources() {
        for (line, target) in dyn_targets(&source.text) {
            if is_forbidden(&source, &target) {
                offenders.push(format!(
                    "{}/{}:{line}: dyn {}",
                    source.krate,
                    source.relative,
                    target.lines().next().unwrap_or("")
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "a `dyn` over a behaviour trait outside the bus's handler newtype:\n{}",
        offenders.join("\n")
    );
}

#[test]
fn named_exemptions_are_live() {
    let sources = sources();
    let mut dead = Vec::new();
    for (name, what) in NAMED_EXEMPT {
        let seen = sources.iter().any(|source| {
            dyn_targets(&source.text)
                .iter()
                .any(|(_, target)| head_of(target) == name)
        });
        if !seen {
            dead.push(format!("{name} ({what})"));
        }
    }
    assert!(
        dead.is_empty(),
        "an exemption names a `dyn` target that no longer occurs; drop it or \
         rename it with the type:\n{}",
        dead.join("\n")
    );
    let erasure_file = sources
        .iter()
        .find(|source| (source.krate, source.relative.as_str()) == ERASURE_FILE)
        .expect("the erasure file exists");
    assert!(
        erasure_file.text.contains("pub struct ErasedHandler("),
        "{}/{} holds the `ErasedHandler` newtype",
        ERASURE_FILE.0,
        ERASURE_FILE.1
    );
    // The boxed trait is the bus's own: authors implement `Serve`.
    assert!(
        erasure_file.text.contains("pub(crate) trait Handler"),
        "the boxed `Handler` trait is crate-private"
    );
    assert!(
        !erasure_file.text.contains("\npub trait Handler"),
        "the boxed `Handler` trait must not be public"
    );
}

/// No global or ambient state in the vocabulary (`rig-core/src/effect`) or
/// in the bus runtime (`rig-agent/src/bus`): no `static`, no `OnceLock`,
/// no `thread_local!` outside tests. (The bus half scanned `rig-core/src/
/// bus` — a directory that had not existed since the bus left rig-core —
/// until the fold re-keyed it here.)
#[test]
fn bus_and_effect_have_no_global_state() {
    let mut offenders = Vec::new();
    for (krate, module) in [("rig-core", "effect"), ("rig-agent", "bus")] {
        let root = crate_src(krate);
        let dir = root.join(module);
        assert!(dir.is_dir(), "{krate}/src/{module} exists");
        for path in non_test_sources(&dir) {
            let relative = path
                .strip_prefix(&root)
                .unwrap_or(&path)
                .to_string_lossy()
                .replace('\\', "/");
            let text = std::fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("{krate}/{relative} is readable: {err}"));
            for (line_number, line) in text.lines().enumerate() {
                let trimmed = line.trim_start();
                if trimmed.starts_with("//") {
                    continue;
                }
                let is_static = trimmed.starts_with("static ")
                    || trimmed.starts_with("pub static ")
                    || trimmed.starts_with("pub(crate) static ");
                if is_static || line.contains("OnceLock") || line.contains("thread_local!") {
                    offenders.push(format!(
                        "{krate}/{relative}:{}: {}",
                        line_number + 1,
                        line.trim()
                    ));
                }
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "no global or ambient state in the bus or in effect/:\n{}",
        offenders.join("\n")
    );
}

/// The consumer-facing traits a typed view must not implement.
const CONSUMER_TRAITS: [&str; 6] = [
    "CompletionModel",
    "EmbeddingModel",
    "ImageEmbeddingModel",
    "Tool",
    "ConversationMemory",
    "VectorStoreIndex",
];

/// The typed views.
const TYPED_VIEWS: [&str; 6] = [
    "Handle",
    "ModelHandle",
    "ToolHandle",
    "MemoryHandle",
    "IndexHandle",
    "EmbedHandle",
];

/// Strip one level of angle-bracketed generics from a path (`Handle<F>` →
/// `Handle`, `a::b<T>::C` → `a::b::C`).
fn without_generics(path: &str) -> String {
    let mut out = String::new();
    let mut depth = 0usize;
    for c in path.chars() {
        match c {
            '<' => depth += 1,
            '>' => depth = depth.saturating_sub(1),
            c if depth == 0 => out.push(c),
            _ => {}
        }
    }
    out
}

/// Every `impl … <Trait> for <Type>` header in `text`, whitespace
/// normalised, as `(trait head, type head)` — path-qualified names reduced
/// to their last segment.
fn impl_headers(text: &str) -> Vec<(String, String)> {
    let normalised: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
    let mut out = Vec::new();
    let mut search = 0;
    while let Some(found) = normalised[search..].find("impl") {
        let index = search + found;
        search = index + 4;
        let bytes = normalised.as_bytes();
        let before_ok =
            index == 0 || !(bytes[index - 1].is_ascii_alphanumeric() || bytes[index - 1] == b'_');
        let rest = &normalised[index + 4..];
        let Some(next) = rest.chars().next() else {
            break;
        };
        if !before_ok || !(next == ' ' || next == '<') {
            continue;
        }
        // The header runs to the impl body or a `where` clause.
        let end = rest
            .find(" {")
            .or_else(|| rest.find(" where "))
            .unwrap_or(rest.len());
        let header = &rest[..end];
        // Skip the impl's own generic parameters.
        let header = if header.starts_with('<') {
            let mut depth = 0usize;
            let mut cut = header.len();
            for (i, c) in header.char_indices() {
                match c {
                    '<' => depth += 1,
                    '>' => {
                        depth -= 1;
                        if depth == 0 {
                            cut = i + 1;
                            break;
                        }
                    }
                    _ => {}
                }
            }
            header[cut..].trim_start()
        } else {
            header.trim_start()
        };
        let Some((trait_part, type_part)) = header.split_once(" for ") else {
            continue;
        };
        let trait_head = head_of(&without_generics(trait_part.trim()));
        let type_head = head_of(&without_generics(type_part.trim()));
        out.push((trait_head, type_head));
    }
    out
}

/// The typed views implement none of the consumer-facing traits: a handle is
/// consumed through its inherent methods.
#[test]
fn typed_views_implement_no_consumer_facing_trait() {
    let mut offenders = Vec::new();
    for source in sources() {
        for (trait_head, type_head) in impl_headers(&source.text) {
            if CONSUMER_TRAITS.contains(&trait_head.as_str())
                && TYPED_VIEWS.contains(&type_head.as_str())
            {
                offenders.push(format!(
                    "{}/{}: impl {trait_head} for {type_head}",
                    source.krate, source.relative
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "a typed view implements a consumer-facing trait:\n{}",
        offenders.join("\n")
    );
    // The scan sees impls at all: the handle file's own inherent impls.
    let handle = sources()
        .into_iter()
        .find(|source| source.krate == "rig-agent" && source.relative == "bus/handle.rs")
        .expect("rig-agent's bus/handle.rs exists");
    assert!(
        impl_headers(&handle.text)
            .iter()
            .any(|(_, type_head)| TYPED_VIEWS.contains(&type_head.as_str())),
        "the impl scan finds the bus's typed-view impls"
    );
}

#[test]
fn the_impl_scan_reads_qualified_and_wrapped_headers() {
    let text = "impl<F: Family>\n    rig_core::completion::CompletionModel<X>\n    for crate::bus::Handle<F> where F: Family {}";
    assert_eq!(
        impl_headers(text),
        vec![("CompletionModel".to_owned(), "Handle".to_owned())]
    );
    assert_eq!(
        dyn_targets("let x: Box<\n    dyn\n    CompletionModel> = y;"),
        vec![(2, "CompletionModel> = y;".to_owned())]
    );
    assert!(dyn_targets("// dyn CompletionModel in a comment").is_empty());
    assert!(dyn_targets("let dynamic = 1;").is_empty());
}

/// The bus has no `unsafe`: the handler's thread affinity is carried by the
/// type that carries the handler (`Registrar`), so nothing in rig-agent's
/// `bus/` or in rig-core's handler seam needs to assert `Send` or `Sync`
/// by hand.
#[test]
fn bus_has_no_unsafe() {
    let mut offenders = Vec::new();
    for source in sources() {
        let in_scope = (source.krate == "rig-agent" && source.relative.starts_with("bus/"))
            || (source.krate == "rig-core" && source.relative.starts_with("serve/"));
        if !in_scope {
            continue;
        }
        for (line_number, line) in source.text.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            let has_token = line
                .split(|c: char| !(c.is_alphanumeric() || c == '_'))
                .any(|token| token == "unsafe");
            if has_token {
                offenders.push(format!(
                    "{}/{}:{}: {}",
                    source.krate,
                    source.relative,
                    line_number + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "`unsafe` in the bus:\n{}",
        offenders.join("\n")
    );
}

/// The bus reads no thread identity: re-entrancy is a chain of parent ids on
/// the commands (causal dispatch), so a nested dispatch made from a spawned
/// task is refused like one made inline, and nothing depends on which thread
/// polls the driver. `loom::thread` in the models is spawning, not identity.
#[test]
fn bus_reads_no_thread_identity() {
    let mut offenders = Vec::new();
    for source in sources() {
        if !(source.krate == "rig-agent" && source.relative.starts_with("bus/")) {
            continue;
        }
        for (line_number, line) in source.text.lines().enumerate() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            if line.contains("thread::current") || line.contains("ThreadId") {
                offenders.push(format!(
                    "{}/{}:{}: {}",
                    source.krate,
                    source.relative,
                    line_number + 1,
                    line.trim()
                ));
            }
        }
    }
    assert!(
        offenders.is_empty(),
        "thread identity read in the bus:\n{}",
        offenders.join("\n")
    );
}
