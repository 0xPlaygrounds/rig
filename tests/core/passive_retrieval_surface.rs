use std::path::{Path, PathBuf};

fn scan_live_sources(path: &Path, violations: &mut Vec<PathBuf>) {
    if path
        .file_name()
        .is_some_and(|name| matches!(name.to_str(), Some(".git" | "target" | "CHANGELOG.md")))
    {
        return;
    }

    if path.is_dir() {
        for entry in std::fs::read_dir(path).expect("read repository source directory") {
            scan_live_sources(&entry.expect("read source entry").path(), violations);
        }
        return;
    }

    let Some(extension) = path.extension().and_then(|extension| extension.to_str()) else {
        return;
    };
    if !matches!(extension, "rs" | "md" | "toml") {
        return;
    }

    let source = std::fs::read_to_string(path).expect("read UTF-8 repository source");
    let removed_names = [
        concat!("dynamic_", "context"),
        concat!("DynamicContext", "Store"),
    ];
    if removed_names.iter().any(|name| source.contains(name)) {
        violations.push(path.to_path_buf());
    }
}

#[test]
fn removed_passive_retrieval_api_does_not_reappear_in_live_sources() {
    let root = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut violations = Vec::new();
    scan_live_sources(root, &mut violations);

    assert!(
        violations.is_empty(),
        "the removed hardcoded passive-retrieval API reappeared outside historical changelogs: \
         {violations:#?}"
    );
}
