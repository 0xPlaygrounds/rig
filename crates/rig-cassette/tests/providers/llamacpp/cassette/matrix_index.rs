//! The guard that keeps this suite's cell documentation from drifting.
//!
//! Every `*_matrix.rs` module names each of its own test functions in its
//! module-level docs, and every module under `cassette/` appears in the suite
//! index in `mod.rs`. A cell added without being named reads as uncovered, and a
//! renamed cell leaves docs describing coverage that no longer exists.
//!
//! The check is source-level because `--list` is not reachable from inside a
//! test, so the cells are read from the same files their docs live in. Modules
//! that are not matrices are exempt from the first rule by name, in
//! `NOT_TABULATED`, because their coverage is indexed at the suite level.

use std::collections::BTreeSet;
use std::path::{Path, PathBuf};

/// Modules whose cells are indexed at the suite level rather than tabulated.
///
/// These are the migrated smoke cells: one or two straightforward cells each,
/// covering a surface the matrices then exercise in depth. `mod.rs`'s module
/// doc accounts for them.
const NOT_TABULATED: &[&str] = &[
    "agent",
    "embeddings",
    "image_tool_result",
    "matrix_index",
    "models",
    "multi_extract",
    "permission_control",
    "streaming",
    "streaming_tools",
    "structured_output",
    "typed_prompt_tools",
];

fn cassette_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/providers/llamacpp/cassette")
}

fn module_names() -> Vec<String> {
    let mut names = Vec::new();
    for entry in std::fs::read_dir(cassette_dir()).expect("cassette dir should be readable") {
        let path = entry.expect("dir entry").path();
        if path.extension().and_then(|extension| extension.to_str()) == Some("rs")
            && let Some(stem) = path.file_stem().and_then(|stem| stem.to_str())
        {
            names.push(stem.to_owned());
        }
    }
    names.sort();
    names
}

fn read(module: &str) -> String {
    let path = cassette_dir().join(format!("{module}.rs"));
    std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("{} should be readable: {error}", path.display()))
}

/// The module-level doc comment: every line starting `//!`, up to the first
/// item.
fn module_docs(source: &str) -> String {
    source
        .lines()
        .take_while(|line| {
            let trimmed = line.trim_start();
            trimmed.starts_with("//!") || trimmed.is_empty()
        })
        .collect::<Vec<_>>()
        .join("\n")
}

/// Test function names declared in a module.
///
/// Matches the `fn NAME(` on the line after a `#[tokio::test]` or `#[test]`
/// attribute, which is how every cell in this suite is written. Nested helper
/// functions inside a test body are not preceded by a test attribute and are
/// therefore not collected.
fn test_names(source: &str) -> BTreeSet<String> {
    let lines: Vec<&str> = source.lines().collect();
    let mut names = BTreeSet::new();
    for (index, line) in lines.iter().enumerate() {
        let trimmed = line.trim_start();
        if trimmed != "#[tokio::test]" && trimmed != "#[test]" {
            continue;
        }
        // The signature may be on the next line or the one after (an `async
        // fn` preceded by a doc-less attribute is the only shape here, but
        // scanning a couple of lines keeps this from being brittle).
        for candidate in lines.iter().skip(index + 1).take(3) {
            let candidate = candidate.trim_start();
            let after_fn = candidate
                .strip_prefix("async fn ")
                .or_else(|| candidate.strip_prefix("fn "));
            if let Some(rest) = after_fn
                && let Some((name, _)) = rest.split_once('(')
            {
                names.insert(name.to_owned());
                break;
            }
        }
    }
    names
}

/// Rule 1: every matrix module tabulates every cell it declares.
#[test]
fn every_matrix_module_names_all_of_its_cells_in_its_table() {
    let mut failures = Vec::new();
    let mut checked = 0usize;

    for module in module_names() {
        if NOT_TABULATED.contains(&module.as_str()) {
            continue;
        }
        let source = read(&module);
        let docs = module_docs(&source);
        let cells = test_names(&source);
        assert!(
            !cells.is_empty(),
            "{module}: has no test functions, so it should not be in the cassette \
             directory at all"
        );

        let missing: Vec<&String> = cells
            .iter()
            .filter(|cell| !docs.contains(cell.as_str()))
            .collect();
        if !missing.is_empty() {
            failures.push(format!(
                "  {module}: {} cell(s) not named in the module's table: {:?}",
                missing.len(),
                missing
            ));
        }
        checked += 1;
    }

    assert!(
        checked >= 10,
        "only {checked} matrix modules checked; the layout drifted and this guard \
         went vacuous"
    );
    assert!(
        failures.is_empty(),
        "a matrix cell exists that its own table does not mention. Add the row, or — \
         if the module is not a matrix — add it to NOT_TABULATED with the reason:\n{}",
        failures.join("\n")
    );
}

/// Rule 2: the suite index in `mod.rs` mentions every module.
#[test]
fn the_suite_index_mentions_every_cassette_module() {
    let mod_rs = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/providers/llamacpp/mod.rs");
    let source = std::fs::read_to_string(&mod_rs).expect("mod.rs should be readable");
    let docs = module_docs(&source);

    let missing: Vec<String> = module_names()
        .into_iter()
        // This module is the guard, not a dimension; it is named in `mod.rs`'s
        // declaration list but has nothing to index.
        .filter(|module| module != "matrix_index")
        .filter(|module| !docs.contains(module.as_str()))
        .collect();

    assert!(
        missing.is_empty(),
        "the suite index in tests/providers/llamacpp/mod.rs does not mention: {missing:?}. \
         Every module belongs in one of its tables, so a reader can find which dimension \
         it covers and which server it was recorded against."
    );

    // And the declaration list must match the directory, so a file cannot be
    // added without being compiled.
    for module in module_names() {
        assert!(
            source.contains(&format!("mod {module};")),
            "{module}.rs exists but `mod {module};` is missing from mod.rs, so it is \
             never compiled and never runs"
        );
    }
}

/// Every `NOT_TABULATED` entry names a module that exists.
#[test]
fn the_untabulated_list_has_no_stale_entries() {
    let modules = module_names();
    let stale: Vec<&&str> = NOT_TABULATED
        .iter()
        .filter(|name| !modules.iter().any(|module| module == *name))
        .collect();
    assert!(
        stale.is_empty(),
        "stale NOT_TABULATED entries (the module was renamed or deleted; delete the \
         entry): {stale:?}"
    );
    let _ = Path::new("");
}
