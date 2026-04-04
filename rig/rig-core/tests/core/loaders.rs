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
