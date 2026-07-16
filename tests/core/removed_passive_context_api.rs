//! Guards the removal of the former built-in passive retrieval API.

use std::{fs, path::Path};

fn source_files(root: &Path, files: &mut Vec<std::path::PathBuf>) {
    for entry in fs::read_dir(root).expect("repository source directory should be readable") {
        let path = entry.expect("repository entry should be readable").path();
        if path.is_dir() {
            source_files(&path, files);
            continue;
        }

        if path.file_name().and_then(|name| name.to_str()) == Some("CHANGELOG.md") {
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
    for directory in ["crates", "examples", "src", "tests"] {
        source_files(&repository.join(directory), &mut files);
    }
    files.extend([repository.join("Cargo.toml"), repository.join("README.md")]);

    let removed_name = ["dynamic", "context"].join("_");
    let references = files
        .into_iter()
        .filter_map(|path| {
            let contents = fs::read_to_string(&path).ok()?;
            contents.contains(&removed_name).then_some(path)
        })
        .collect::<Vec<_>>();

    assert!(
        references.is_empty(),
        "the removed passive retrieval API is still referenced in {references:#?}"
    );
}
