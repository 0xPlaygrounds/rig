//! Guards the provider-independent managed agent hook API.

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
        } else if matches!(
            path.extension().and_then(|extension| extension.to_str()),
            Some("rs" | "md")
        ) {
            files.push(path);
        }
    }
}

fn has_multiple_type_arguments(contents: &str, type_name: &str) -> bool {
    let compact = contents
        .chars()
        .filter(|character| !character.is_whitespace())
        .collect::<String>();
    let needle = format!("{type_name}<");
    let mut remaining = compact.as_str();
    while let Some(start) = remaining.find(&needle) {
        let generic = &remaining[start + needle.len()..];
        let Some(end) = generic.find('>') else {
            return false;
        };
        if generic[..end].contains(',') {
            return true;
        }
        remaining = &generic[end + 1..];
    }
    false
}

#[test]
fn managed_agent_hook_interfaces_stay_non_generic() {
    let repository = Path::new(env!("CARGO_MANIFEST_DIR"));
    let mut files = Vec::new();
    for directory in ["crates", "examples", "src", "tests"] {
        source_files(&repository.join(directory), &mut files);
    }
    for file in ["README.md", "AGENTS.md", "CONTRIBUTING.md"] {
        let path = repository.join(file);
        if path.exists() {
            files.push(path);
        }
    }

    let trait_names = [["Agent", "Hook"].concat(), ["Hook", "Stack"].concat()];
    let response_event_names = [
        ["Completion", "Response"].concat(),
        ["Completion", "Response", "Event"].concat(),
        ["Stream", "Response", "Finish"].concat(),
    ];
    let references = files
        .into_iter()
        .filter_map(|path| {
            let contents = fs::read_to_string(&path).ok()?;
            let compact = contents
                .chars()
                .filter(|character| !character.is_whitespace())
                .collect::<String>();
            let generic_trait = trait_names
                .iter()
                .any(|name| compact.contains(&format!("{name}<")));
            let generic_response_event = response_event_names
                .iter()
                .any(|name| has_multiple_type_arguments(&contents, name));
            (generic_trait || generic_response_event).then_some(path)
        })
        .collect::<Vec<_>>();

    assert!(
        references.is_empty(),
        "managed agent hook interfaces still carry model parameters in {references:#?}"
    );
}
