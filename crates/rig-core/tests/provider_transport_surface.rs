//! The two jobs the deleted `rig-reqwest::providers` alias tree used to do,
//! now guarded in the crate that owns the types.
//!
//! That tree mirrored every transport-generic provider type with `H` defaulted
//! to the bundled transport, *and* hoisted types declared in submodules up to
//! the provider root. `tests/providers_complete.rs` kept it complete. Folding
//! the transport into rig-core removed the mirror; without these tests nothing
//! would notice a provider type that quietly stops defaulting (`openai::Client`
//! needing an explicit `H` again) or one that is only reachable at
//! `huggingface::completion::CompletionModel` instead of the provider root.

#![allow(clippy::expect_used, clippy::indexing_slicing)]

use std::collections::{BTreeSet, HashMap};
use std::path::{Path, PathBuf};

fn providers_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("src/providers")
}

fn rust_files(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("readable dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            rust_files(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// The identifier a `pub type`/`pub struct` declaration names, plus what
/// follows it, for a line already known to start with one of those keywords.
fn declared_name(rest: &str) -> (String, &str) {
    let name: String = rest
        .chars()
        .take_while(|c| c.is_alphanumeric() || *c == '_')
        .collect();
    let after = &rest[name.len()..];
    (name, after)
}

/// Every transport-generic provider type, as `(provider module, type name,
/// file, declares the default)`.
fn transport_generic_types() -> Vec<(String, String, PathBuf, bool)> {
    let root = providers_dir();
    let mut files = Vec::new();
    rust_files(&root, &mut files);
    files.sort();

    let mut out = Vec::new();
    for file in files {
        let rel = file.strip_prefix(&root).expect("under providers/");
        let module = rel
            .components()
            .next()
            .map(|c| {
                c.as_os_str()
                    .to_string_lossy()
                    .trim_end_matches(".rs")
                    .to_string()
            })
            .unwrap_or_default();
        if module == "internal" || module == "mod" {
            continue;
        }
        let source = std::fs::read_to_string(&file).expect("readable source");
        for line in source.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with("//") {
                continue;
            }
            for prefix in ["pub type ", "pub struct "] {
                let Some(rest) = trimmed.strip_prefix(prefix) else {
                    continue;
                };
                let (name, after) = declared_name(rest);
                if name.is_empty() {
                    continue;
                }
                // Only the transport slot matters: `<H`/`<T` as the FIRST
                // parameter is the convention across the provider tree.
                if !(after.starts_with("<H") || after.starts_with("<T")) {
                    continue;
                }
                let defaulted = after.starts_with("<H = ") || after.starts_with("<T = ");
                // A builder's transport slot deliberately defaults to
                // `Missing` — it is filled by `.http_client(..)` or by the
                // `reqwest` feature's inherent `build`, so it is not part of
                // the bundled-transport surface.
                if after.contains("markers::Missing") {
                    continue;
                }
                out.push((module.clone(), name, file.clone(), defaulted));
            }
        }
    }
    out
}

/// Every transport-generic provider type defaults its transport, so
/// `openai::CompletionModel` keeps meaning `…<reqwest::Client>` with the
/// `reqwest` feature and callers never spell `H` for the bundled path.
#[test]
fn every_transport_generic_provider_type_defaults_its_transport() {
    let missing: Vec<String> = transport_generic_types()
        .into_iter()
        .filter(|(_, _, _, defaulted)| !defaulted)
        .map(|(module, name, file, _)| {
            format!(
                "{module}::{name} ({})",
                file.file_name().unwrap_or_default().to_string_lossy()
            )
        })
        .collect();
    assert!(
        missing.is_empty(),
        "transport-generic provider types without `H = crate::http_client::DefaultHttp`: \
         {missing:?}\nAdd the default so the type is nameable without a transport."
    );
}

/// Types declared in a provider's submodules stay reachable at the provider
/// root: `huggingface::CompletionModel`, not only
/// `huggingface::completion::CompletionModel`.
#[test]
fn transport_generic_types_are_reachable_at_the_provider_root() {
    let root = providers_dir();
    let mut by_module: HashMap<String, BTreeSet<(String, String)>> = HashMap::new();
    for (module, name, file, _) in transport_generic_types() {
        // Only submodule declarations need hoisting; a single-file provider
        // already declares at its root.
        let rel = file.strip_prefix(&root).expect("under providers/");
        if rel.components().count() < 2 {
            continue;
        }
        let submodule = file
            .file_stem()
            .map(|s| s.to_string_lossy().to_string())
            .unwrap_or_default();
        if submodule == "mod" {
            continue;
        }
        by_module
            .entry(module)
            .or_default()
            .insert((submodule, name));
    }

    let mut unreachable = Vec::new();
    for (module, types) in by_module {
        let mod_rs = root.join(&module).join("mod.rs");
        let Ok(source) = std::fs::read_to_string(&mod_rs) else {
            continue;
        };
        // `pub use <submodule>::*;` re-exports everything that submodule
        // declares; anything else must name the type.
        for (submodule, name) in types {
            let glob = format!("pub use {submodule}::*;");
            if source.contains(&glob) || source.contains(&name) {
                continue;
            }
            unreachable.push(format!("{module}::{name} (declared in {submodule}.rs)"));
        }
    }
    unreachable.sort();
    assert!(
        unreachable.is_empty(),
        "transport-generic provider types not re-exported at their provider root: \
         {unreachable:?}\nAdd a `pub use` in the provider's mod.rs."
    );
}
