//! Migrated from `examples/loaders.rs`.

use rig::loaders::FileLoader;

#[test]
fn file_loader_reads_manifest() {
    let pattern = concat!(env!("CARGO_MANIFEST_DIR"), "/Cargo.toml");
    let results: Vec<_> = FileLoader::with_glob(pattern)
        .expect("glob should parse")
        .read()
        .into_iter()
        .collect();

    assert_eq!(results.len(), 1, "expected exactly one manifest");
    let content = results
        .into_iter()
        .next()
        .expect("expected a result")
        .expect("reading Cargo.toml should succeed");
    assert!(
        content.contains("[package]"),
        "manifest content should include the package section"
    );
}

#[test]
fn loader_fixture_glob_resolves_from_manifest_dir() {
    let results: Vec<_> = FileLoader::with_glob(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/tests/data/loaders/*.rs"
    ))
    .expect("glob should parse")
    .read_with_path()
    .ignore_errors()
    .into_iter()
    .collect();

    assert_eq!(results.len(), 3, "expected every loader fixture to resolve");
    assert!(
        results
            .iter()
            .any(|(path, _)| path.ends_with("agent_with_loaders.rs")),
        "expected the agent_with_loaders fixture to be included"
    );
}
