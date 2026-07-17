//! Guards the removal of the former built-in passive retrieval API.

use std::{
    fs,
    path::{Path, PathBuf},
};

fn is_ignored(path: &Path) -> bool {
    path.file_name()
        .is_some_and(|name| matches!(name.to_str(), Some(".git" | "target" | "CHANGELOG.md")))
}

fn source_files(root: &Path, files: &mut Vec<PathBuf>) {
    if is_ignored(root) || !root.exists() {
        return;
    }

    for entry in fs::read_dir(root).expect("repository source directory should be readable") {
        let path = entry.expect("repository entry should be readable").path();
        if is_ignored(&path) {
            continue;
        }

        if path.is_dir() {
            source_files(&path, files);
            continue;
        }

        if matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("rs" | "md" | "toml" | "yml" | "yaml")
        ) {
            files.push(path);
        }
    }
}

#[test]
fn removed_passive_context_builder_api_stays_absent() {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for directory in ["crates", "examples", "src", "tests", ".github"] {
        source_files(&repository.join(directory), &mut files);
    }
    for file in ["Cargo.toml", "README.md", "AGENTS.md", "CONTRIBUTING.md"] {
        let path = repository.join(file);
        if path.exists() {
            files.push(path);
        }
    }

    let removed_names = [
        ["dynamic", "context"].join("_"),
        ["Dynamic", "Context", "Store"].concat(),
    ];
    let references = files
        .into_iter()
        .filter_map(|path| {
            let contents = fs::read_to_string(&path).ok()?;
            removed_names
                .iter()
                .any(|removed_name| contents.contains(removed_name))
                .then_some(path)
        })
        .collect::<Vec<_>>();

    assert!(
        references.is_empty(),
        "the removed passive retrieval API is still referenced in {references:#?}"
    );
}
