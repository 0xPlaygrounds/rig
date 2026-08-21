//! `rig_reqwest::providers` must carry a defaulted alias for every
//! transport-generic type rig-core's providers export; a missing one is the
//! silent failure mode (a type that quietly stops defaulting to the bundled
//! transport). Cross-checked against the rig-core sources at test time.

#![allow(clippy::expect_used, clippy::indexing_slicing)]

use std::path::{Path, PathBuf};

fn rig_core_providers_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../rig-core/src/providers")
        .canonicalize()
        .expect("rig-core/src/providers exists next to rig-reqwest")
}

fn walk(dir: &Path, out: &mut Vec<PathBuf>) {
    for entry in std::fs::read_dir(dir).expect("readable dir") {
        let path = entry.expect("dir entry").path();
        if path.is_dir() {
            walk(&path, out);
        } else if path.extension().is_some_and(|ext| ext == "rs") {
            out.push(path);
        }
    }
}

/// `(provider module, type name)` for every `pub type X<H>` / `pub struct X<H>`
/// (or `<T>`) under rig-core's providers, excluding `internal/`.
fn expected_aliases() -> Vec<(String, String)> {
    let root = rig_core_providers_dir();
    let mut files = Vec::new();
    walk(&root, &mut files);
    let mut out = Vec::new();
    for file in files {
        let rel = file.strip_prefix(&root).expect("under root");
        let first = rel
            .components()
            .next()
            .map(|c| c.as_os_str().to_string_lossy().to_string())
            .unwrap_or_default();
        let module = first.trim_end_matches(".rs").to_string();
        if module == "internal" || module == "mod" {
            continue;
        }
        let source = std::fs::read_to_string(&file).expect("readable source");
        for line in source.lines() {
            let line = line.trim_start();
            for prefix in ["pub type ", "pub struct "] {
                if let Some(rest) = line.strip_prefix(prefix) {
                    let name: String = rest
                        .chars()
                        .take_while(|c| c.is_alphanumeric() || *c == '_')
                        .collect();
                    let after = &rest[name.len()..];
                    if after.starts_with("<H>") || after.starts_with("<T>") {
                        out.push((module.clone(), name));
                    }
                }
            }
            // `impl_model_lister!(Name, …)` generates `pub struct Name<H>`.
            if let Some(rest) = line.strip_prefix("impl_model_lister!(") {
                let name: String = rest
                    .chars()
                    .take_while(|c| c.is_alphanumeric() || *c == '_')
                    .collect();
                if !name.is_empty() {
                    out.push((module.clone(), name));
                }
            }
        }
    }
    // The dual-dialect macro providers generate `Client` / `AnthropicClient`.
    for module in ["minimax", "moonshot", "xiaomimimo", "zai"] {
        out.push((module.to_string(), "Client".to_string()));
        out.push((module.to_string(), "AnthropicClient".to_string()));
    }
    out.sort();
    out.dedup();
    out
}

#[test]
fn every_transport_generic_provider_type_has_a_defaulted_alias() {
    let aliases = include_str!("../src/providers.rs");
    let mut missing = Vec::new();
    let mut current_module = String::new();
    let mut present = std::collections::HashSet::new();
    for line in aliases.lines() {
        let trimmed = line.trim_start();
        if let Some(rest) = trimmed.strip_prefix("pub mod ") {
            current_module = rest.trim_end_matches(" {").to_string();
        } else if let Some(rest) = trimmed.strip_prefix("pub type ") {
            let name: String = rest
                .chars()
                .take_while(|c| c.is_alphanumeric() || *c == '_')
                .collect();
            present.insert((current_module.clone(), name));
        }
    }
    for (module, name) in expected_aliases() {
        if !present.contains(&(module.clone(), name.clone())) {
            missing.push(format!("{module}::{name}"));
        }
    }
    assert!(
        missing.is_empty(),
        "rig_reqwest::providers is missing defaulted aliases for: {missing:?}"
    );
}
